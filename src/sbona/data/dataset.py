"""Prepared datasets, window datasets and collation.

The only tensor information that reaches a model is :class:`ModelInput`.
:class:`TargetBatch` never enters ``forward_candidates`` / ``predict_top1``.
Invalid target values are replaced by ``0.0`` *after* the validity mask is
recorded, so no downstream arithmetic can rely on ``NaN * 0 == 0``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sbona.config import Config
from sbona.contracts import GraphContext, ModelInput, ScalerStats, TargetBatch
from sbona.data.graph import build_graph_context, load_edge_list
from sbona.data.manifest import (
    DataError,
    DatasetManifest,
    audit_array,
    load_raw_array,
    load_timestamps_file,
    make_file_record,
    node_order_hash,
    read_node_ids,
    read_node_ids_file,
    sha256_array,
    sha256_file,
)
from sbona.data.scaler import fill_invalid_with_mean, fit_scaler, standardize_array
from sbona.data.windows import WindowScheme, build_window_scheme

PREPARED_SCHEMA_VERSION = 1


@dataclass
class PreparedData:
    dataset: str
    data: np.ndarray  # [M, N] raw float32
    valid: np.ndarray  # [M, N] bool
    timestamps: np.ndarray | None
    node_ids: np.ndarray
    scaler: ScalerStats
    scheme: WindowScheme
    graph: GraphContext
    manifest: DatasetManifest
    calendar_available: bool
    steps_per_day: int = 288
    topology: str = "published_binary_undirected"
    protocol_id: str = "flow_12x12_v1"

    @property
    def num_steps(self) -> int:
        return self.data.shape[0]

    @property
    def num_nodes(self) -> int:
        return self.data.shape[1]


# ---------------------------------------------------------------------------
# Synthetic fixture (explicitly labelled)
# ---------------------------------------------------------------------------


def generate_synthetic(
    num_nodes: int = 8,
    num_steps: int = 432,
    seed: int = 11,
    missing_fraction: float = 0.01,
    steps_per_day: int = 288,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a deterministic non-negative flow fixture.

    The last node is deliberately isolated in the returned edge list and one
    short block is set to ``NaN`` so tests cover missing data and isolated
    nodes.  Callers must label runs as synthetic.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(num_steps, dtype=np.float32)
    base = 180.0 + 120.0 * np.sin(2 * np.pi * t / steps_per_day - 0.7)
    weekly = 1.0 + 0.08 * np.sin(2 * np.pi * t / (steps_per_day * 7))
    node_scale = np.linspace(0.85, 1.15, num_nodes, dtype=np.float32)
    shift = rng.normal(0.0, 20.0, size=num_nodes).astype(np.float32)
    data = base[:, None] * weekly[:, None] * node_scale[None, :] + shift[None, :]
    for node in range(num_nodes):
        data[:, node] += 12.0 * np.sin(
            2 * np.pi * (t / 37.0) + node * 0.9
        ).astype(np.float32)
    data += rng.normal(0.0, 4.0, size=data.shape).astype(np.float32)
    data = np.clip(data, 0.0, None).astype(np.float32)

    if num_nodes > 1:
        edges = np.stack(
            [np.arange(num_nodes - 1), np.arange(1, num_nodes)], axis=1
        ).astype(np.int64)
    else:
        edges = np.zeros((0, 2), dtype=np.int64)

    n_missing = int(round(missing_fraction * data.size))
    if n_missing > 0:
        flat = rng.choice(data.size, size=n_missing, replace=False)
        data.reshape(-1)[flat] = np.nan
    return data, edges, np.arange(num_nodes, dtype=np.int64)


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def _resolve_path(root: Path, value: str | None, name: str) -> Path:
    if value is None:
        raise DataError(f"{name} is not configured")
    path = Path(value)
    return path if path.is_absolute() else root / path


def compute_prepare_signature(
    cfg: Config,
    data_root: str | Path,
    synthetic: bool | None = None,
) -> str:
    """Hash every input that changes the *prepared* arrays.

    The signature is stored in ``manifest.extra`` and compared on load, so a
    changed raw file, split/scaler setting or graph definition cannot silently
    reuse stale artifacts.  Training hyperparameters are deliberately excluded.
    """
    data_root = Path(data_root)
    use_synthetic = bool(cfg.data.synthetic if synthetic is None else synthetic)
    payload: dict[str, object] = {
        "dataset": cfg.data.dataset,
        "synthetic": use_synthetic,
        "input_steps": cfg.data.input_steps,
        "output_steps": cfg.data.output_steps,
        "split_ratios": list(cfg.data.split_ratios),
        "steps_per_day": cfg.data.steps_per_day,
        "missing_policy": cfg.data.missing_policy,
        "zeros_are_valid": cfg.data.zeros_are_valid,
        "envelope_quantile": cfg.data.envelope_quantile,
        "envelope_min_count": cfg.data.envelope_min_count,
        "mape_min_abs_target": cfg.data.mape_min_abs_target,
        "unit": cfg.data.unit,
        "flow_channel": cfg.data.flow_channel,
        "missing_sentinel": cfg.data.missing_sentinel,
        "timestamps_available": cfg.data.timestamps_available,
        "time_of_day": cfg.data.time_of_day,
        "day_of_week": cfg.data.day_of_week,
        "graph_topology": cfg.graph.topology,
        "graph_diffusion": cfg.graph.diffusion,
    }
    if use_synthetic:
        payload.update(
            {
                "seed": cfg.seed,
                "synthetic_num_nodes": cfg.data.synthetic_num_nodes,
                "synthetic_num_steps": cfg.data.synthetic_num_steps,
                "synthetic_missing_fraction": cfg.data.synthetic_missing_fraction,
            }
        )
    else:
        data_path = _resolve_path(data_root, cfg.data.data_file, "data.data_file")
        payload["data_sha256"] = sha256_file(data_path)
        graph_file = cfg.data.graph_file
        if cfg.graph.topology != "identity" and graph_file is None:
            raise DataError("data.graph_file is required for a real published graph")
        if graph_file is not None:
            payload["graph_sha256"] = sha256_file(_resolve_path(data_root, graph_file, "data.graph_file"))
        if cfg.data.node_ids_file is not None:
            payload["node_ids_sha256"] = sha256_file(
                _resolve_path(data_root, cfg.data.node_ids_file, "data.node_ids_file")
            )
        if cfg.data.timestamps_file is not None:
            payload["timestamps_sha256"] = sha256_file(
                _resolve_path(data_root, cfg.data.timestamps_file, "data.timestamps_file")
            )
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_dataset(
    cfg: Config,
    data_root: str | Path,
    output_root: str | Path,
    synthetic: bool | None = None,
    signature: str | None = None,
) -> tuple[PreparedData, Path]:
    """Load raw data, audit it, fit train-only statistics and persist artifacts."""
    data_root = Path(data_root)
    output_root = Path(output_root)
    use_synthetic = bool(cfg.data.synthetic if synthetic is None else synthetic)
    protocol_suffix = "_synthetic" if use_synthetic else ""
    extra_files: list = []
    calendar_record = None

    if use_synthetic:
        data, edges, node_ids = generate_synthetic(
            num_nodes=cfg.data.synthetic_num_nodes,
            num_steps=cfg.data.synthetic_num_steps,
            seed=cfg.seed,
            missing_fraction=cfg.data.synthetic_missing_fraction,
            steps_per_day=cfg.data.steps_per_day,
        )
        timestamps = None
        source_mask = None
        source = "synthetic fixture generated by sbona.data.dataset.generate_synthetic"
        license_note = "project-generated synthetic data (not PEMS)"
        source_dir = data_root / "synthetic" / cfg.data.dataset
        source_dir.mkdir(parents=True, exist_ok=True)
        data_path = source_dir / "data.npz"
        graph_path = source_dir / "graph.csv"
        np.savez_compressed(data_path, data=data, node_ids=node_ids)
        with graph_path.open("w", encoding="utf-8") as handle:
            handle.write("from,to,cost\n")
            for i, j in edges.tolist():
                handle.write(f"{i},{j},1.0\n")
        graph_source = str(graph_path)
    else:
        data_path = _resolve_path(data_root, cfg.data.data_file, "data.data_file")
        data, source_mask, timestamps = load_raw_array(
            data_path,
            flow_channel=cfg.data.flow_channel,
            missing_sentinel=cfg.data.missing_sentinel,
        )
        if cfg.data.node_ids_file is not None:
            node_ids_path = _resolve_path(
                data_root, cfg.data.node_ids_file, "data.node_ids_file"
            )
            node_ids = read_node_ids_file(node_ids_path, data.shape[1])
            extra_files.append(make_file_record(node_ids_path, role="node_ids"))
        else:
            node_ids = read_node_ids(data_path, data.shape[1])

        if cfg.data.timestamps_file is not None:
            timestamps_path = _resolve_path(
                data_root, cfg.data.timestamps_file, "data.timestamps_file"
            )
            timestamps = load_timestamps_file(timestamps_path, data.shape[0])
            calendar_record = make_file_record(timestamps_path, role="timestamps")
            extra_files.append(calendar_record)

        if cfg.data.graph_file is None:
            if cfg.graph.topology != "identity":
                raise DataError("data.graph_file is required (identity is diagnostic only)")
            graph_path = None
            edges = np.zeros((0, 2), dtype=np.int64)
            graph_source = "identity diagnostic: no road graph supplied"
        else:
            graph_path = _resolve_path(data_root, cfg.data.graph_file, "data.graph_file")
            edges = load_edge_list(graph_path, node_ids=None)
            # Edge files may use original node ids; map through node_ids if needed.
            max_edge = int(edges.max()) if edges.size else 0
            if max_edge >= data.shape[1]:
                edges = load_edge_list(graph_path, node_ids=node_ids)
            elif node_ids.tolist() != list(range(data.shape[1])):
                edges = load_edge_list(graph_path, node_ids=node_ids)
            graph_source = str(graph_path)
        source = str(data_path)
        license_note = "see dataset manifest / original publisher"

    if cfg.data.timestamps_available is False:
        timestamps = None

    if data.shape[0] < cfg.data.input_steps + cfg.data.output_steps + 10:
        raise DataError(
            f"dataset has only {data.shape[0]} steps, too few for L={cfg.data.input_steps}, "
            f"T={cfg.data.output_steps}"
        )

    finite = np.isfinite(data)
    if source_mask is not None:
        valid = source_mask.astype(bool) & finite
    else:
        valid = finite
    if cfg.data.missing_policy == "official_compat_nonzero":
        valid = valid & (data != 0)
    elif not cfg.data.zeros_are_valid:
        valid = valid & (data != 0)

    _, _, calendar_available = build_calendar(
        timestamps, data.shape[0], cfg.data.steps_per_day
    )
    if cfg.data.day_of_week and not calendar_available:
        raise DataError(
            "data.day_of_week=True requires reliable timestamps; refusing to guess weekdays"
        )

    scheme = build_window_scheme(
        num_steps=data.shape[0],
        input_steps=cfg.data.input_steps,
        output_steps=cfg.data.output_steps,
        ratios=cfg.data.split_ratios,
    )
    scaler = fit_scaler(
        data,
        valid,
        train_end=scheme.boundaries.train_end,
        quantile=cfg.data.envelope_quantile,
        envelope_min_count=cfg.data.envelope_min_count,
    )
    node_hash = node_order_hash(node_ids)
    graph = build_graph_context(
        num_nodes=data.shape[1],
        edges=edges,
        node_ids=node_ids,
        topology=cfg.graph.topology,
        node_order_hash_value=node_hash,
    )

    audit = audit_array(data, valid)
    files = [make_file_record(data_path, role="data")]
    if graph_path is not None:
        files.append(make_file_record(graph_path, role="graph"))
    files.extend(extra_files)
    if calendar_record is not None:
        timestamps_source = (
            f"external calendar file {calendar_record.path} "
            f"sha256={calendar_record.sha256[:16]}"
        )
    elif timestamps is not None:
        timestamps_source = f"embedded in {data_path}"
    else:
        timestamps_source = "not available; relative time-of-day only"
    manifest = DatasetManifest(
        dataset=cfg.data.dataset + protocol_suffix,
        source=source,
        license=license_note,
        files=files,
        shape=(int(data.shape[0]), int(data.shape[1])),
        flow_channel=cfg.data.flow_channel,
        unit=cfg.data.unit,
        interval_minutes=max(1, int(round(24 * 60 / cfg.data.steps_per_day))),
        start_index=0,
        end_index=int(data.shape[0]) - 1,
        timestamps_source=timestamps_source,
        node_ids=node_ids.tolist(),
        node_order_hash=node_hash,
        adjacency_source=graph_source,
        missing_definition=cfg.data.missing_policy,
        audit=audit,
        calendar_available=calendar_available,
        protocol_id=cfg.protocol_id,
        extra={
            "synthetic": use_synthetic,
            "data_sha256": sha256_array(data),
            "valid_sha256": sha256_array(valid.astype(np.uint8)),
            "envelope_fallback_nodes": int(scaler.envelope_fallback.sum().item()),
            "train_end": scheme.boundaries.train_end,
            "val_end": scheme.boundaries.val_end,
            "prepare_signature": signature
            or compute_prepare_signature(cfg, data_root, synthetic=use_synthetic),
        },
    )

    prepared = PreparedData(
        dataset=manifest.dataset,
        data=data,
        valid=valid,
        timestamps=timestamps,
        node_ids=node_ids,
        scaler=scaler,
        scheme=scheme,
        graph=graph,
        manifest=manifest,
        calendar_available=calendar_available,
        steps_per_day=cfg.data.steps_per_day,
        topology=cfg.graph.topology,
        protocol_id=cfg.protocol_id,
    )
    out_dir = output_root / "prepared" / manifest.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = save_prepared(prepared, out_dir / "prepared.npz")
    manifest.save(out_dir / "manifest.json")
    return prepared, prepared_path


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_prepared(prepared: PreparedData, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = (
        np.asarray(prepared.timestamps)
        if prepared.timestamps is not None
        else np.zeros(0, dtype=np.float64)
    )
    edges = prepared.graph.edge_index.detach().cpu().numpy().astype(np.int64)
    np.savez_compressed(
        path,
        schema_version=np.int64(PREPARED_SCHEMA_VERSION),
        data=prepared.data,
        valid=prepared.valid,
        timestamps=timestamps,
        has_timestamps=np.bool_(prepared.timestamps is not None),
        node_ids=prepared.node_ids,
        edge_index=edges,
        mu=np.float64(prepared.scaler.mu),
        sigma=np.float64(prepared.scaler.sigma),
        envelope=prepared.scaler.envelope.detach().cpu().numpy(),
        envelope_fallback=prepared.scaler.envelope_fallback.detach().cpu().numpy(),
        envelope_quantile=np.float64(prepared.scaler.envelope_quantile),
        envelope_min_count=np.int64(prepared.scaler.envelope_min_count),
        windows_json=np.asarray(json.dumps(prepared.scheme.to_dict())),
        graph_hash=np.asarray(prepared.graph.graph_hash),
        node_order_hash=np.asarray(prepared.graph.node_order_hash),
        topology=np.asarray(prepared.topology),
        steps_per_day=np.int64(prepared.steps_per_day),
        calendar_available=np.bool_(prepared.calendar_available),
        protocol_id=np.asarray(prepared.protocol_id),
        dataset=np.asarray(prepared.dataset),
    )
    return path


def load_prepared(path: str | Path) -> PreparedData:
    path = Path(path)
    if not path.is_file():
        raise DataError(f"prepared dataset not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        version = int(archive["schema_version"])
        if version != PREPARED_SCHEMA_VERSION:
            raise DataError(f"unsupported prepared schema {version}")
        data = np.asarray(archive["data"], dtype=np.float32)
        valid = np.asarray(archive["valid"], dtype=bool)
        timestamps = np.asarray(archive["timestamps"]) if bool(archive["has_timestamps"]) else None
        node_ids = np.asarray(archive["node_ids"], dtype=np.int64)
        edges = np.asarray(archive["edge_index"], dtype=np.int64)
        scaler = ScalerStats(
            mu=float(archive["mu"]),
            sigma=float(archive["sigma"]),
            envelope=torch.from_numpy(np.asarray(archive["envelope"], dtype=np.float32)),
            envelope_fallback=torch.from_numpy(np.asarray(archive["envelope_fallback"], dtype=bool)),
            envelope_quantile=float(archive["envelope_quantile"]),
            envelope_min_count=int(archive["envelope_min_count"]),
        )
        scheme = WindowScheme.from_dict(json.loads(str(archive["windows_json"])))
        topology = str(archive["topology"]) if "topology" in archive else "published_binary_undirected"
        graph = build_graph_context(
            num_nodes=data.shape[1],
            edges=edges.T if edges.ndim == 2 and edges.shape[0] == 2 else edges,
            node_ids=node_ids,
            topology=topology,
            node_order_hash_value=str(archive["node_order_hash"]),
        )
        calendar_available = bool(archive["calendar_available"])
        steps_per_day = int(archive["steps_per_day"]) if "steps_per_day" in archive else 288
        protocol_id = str(archive["protocol_id"])
        dataset = str(archive["dataset"])
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise DataError(f"manifest missing next to prepared data: {manifest_path}")
    manifest = DatasetManifest.load(manifest_path)
    return PreparedData(
        dataset=dataset,
        data=data,
        valid=valid,
        timestamps=timestamps,
        node_ids=node_ids,
        scaler=scaler,
        scheme=scheme,
        graph=graph,
        manifest=manifest,
        calendar_available=calendar_available,
        steps_per_day=steps_per_day,
        topology=topology,
        protocol_id=protocol_id,
    )


def prepared_path_for(cfg: Config, output_root: str | Path, synthetic: bool | None = None) -> Path:
    suffix = "_synthetic" if (cfg.data.synthetic if synthetic is None else synthetic) else ""
    return Path(output_root) / "prepared" / (cfg.data.dataset + suffix) / "prepared.npz"


def load_or_prepare(
    cfg: Config,
    data_root: str | Path,
    output_root: str | Path,
    synthetic: bool | None = None,
) -> PreparedData:
    path = prepared_path_for(cfg, output_root, synthetic)
    expected = compute_prepare_signature(cfg, data_root, synthetic)
    if path.is_file():
        # A schema mismatch is an error; only an input/config change triggers a
        # rebuild.  Old artifacts without the signature are rebuilt once.
        prepared = load_prepared(path)
        if prepared.manifest.extra.get("prepare_signature") == expected:
            return prepared
    prepared, _ = prepare_dataset(
        cfg, data_root, output_root, synthetic=synthetic, signature=expected
    )
    return prepared


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------


def build_calendar(
    timestamps: np.ndarray | None,
    num_steps: int,
    steps_per_day: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return ``tod [M]``, ``dow [M]`` and whether weekdays are trustworthy.

    ``timestamps`` may be ``[M]`` datetime64/integer, or an explicit
    ``[M, 2]`` table of ``(time_of_day, day_of_week)``.  An absent calendar
    falls back to a relative time-of-day phase and reports weekdays unavailable.
    """
    if timestamps is None:
        index = np.arange(num_steps, dtype=np.float64)
        tod = (index % steps_per_day) / float(steps_per_day)
        dow = np.zeros(num_steps, dtype=np.int64)
        return tod.astype(np.float32), dow, False

    ts = np.asarray(timestamps)
    if ts.ndim == 2:
        # Explicit (time-of-day, day-of-week) table; day-of-week is 0..6 with
        # Monday=0.  Fractional encodings such as day/7 are rejected because
        # they are ambiguous at 0 and 1.
        if ts.shape[1] != 2:
            raise DataError(f"calendar table must be [M, 2], got {ts.shape}")
        tod = ts[:, 0]
        dow_raw = ts[:, 1]
        if not np.isfinite(tod).all() or float(tod.min()) < 0.0 or float(tod.max()) >= 1.0:
            raise DataError("calendar time_of_day column must be finite and in [0, 1)")
        dow = np.rint(dow_raw).astype(np.int64)
        if not np.isfinite(dow_raw).all() or not np.allclose(dow_raw, dow, atol=1e-6):
            raise DataError("calendar day_of_week column must contain integer days 0..6")
        if int(dow.min()) < 0 or int(dow.max()) > 6:
            raise DataError("calendar day_of_week column must be in 0..6")
        return tod.astype(np.float32), dow, True

    if np.issubdtype(ts.dtype, np.datetime64):
        seconds = ts.astype("datetime64[s]").astype(np.int64)
        day_seconds = 24 * 3600
        tod = (seconds % day_seconds) / float(day_seconds)
        # 1970-01-01 was a Thursday; use a calendar-relative day index but keep
        # the 0..6 convention Monday=0.
        days = seconds // day_seconds
        dow = ((days + 3) % 7).astype(np.int64)
        return tod.astype(np.float32), dow, True

    if np.issubdtype(ts.dtype, np.integer):
        # Large integers are treated as minutes since epoch and can identify a
        # weekday.  Small integers are just sampling-step indices: they give a
        # time-of-day phase but *cannot* reveal the day of week, so weekday
        # features are reported as unavailable instead of being guessed.
        if float(np.max(np.abs(ts))) > 1e5:
            minutes = ts.astype(np.int64)
            tod = (minutes % 1440) / 1440.0
            days = minutes // 1440
            dow = ((days + 3) % 7).astype(np.int64)
            return tod.astype(np.float32), dow, True
        steps = ts.astype(np.int64)
        tod = (steps % steps_per_day) / float(steps_per_day)
        dow = np.zeros(num_steps, dtype=np.int64)
        return tod.astype(np.float32), dow, False

    raise DataError(
        "timestamps must be datetime64 or integer; strings must be converted before use"
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TrafficWindowDataset(Dataset):
    """One split of a prepared dataset.

    ``__getitem__`` returns :class:`WindowSample`; batch metadata stays outside
    the model API.
    """

    def __init__(
        self,
        prepared: PreparedData,
        split: str,
        allow_empty: bool = False,
    ) -> None:
        if split not in prepared.scheme.starts:
            raise DataError(f"unknown split {split!r}")
        self.prepared = prepared
        self.split = split
        self.starts = prepared.scheme.starts[split]
        if self.starts.size == 0 and not allow_empty:
            raise DataError(f"split {split!r} has no windows")
        self.L = prepared.scheme.input_steps
        self.T = prepared.scheme.output_steps
        self.mu = prepared.scaler.mu
        self.sigma = prepared.scaler.sigma
        self.filled = fill_invalid_with_mean(prepared.data, prepared.valid, self.mu)
        self.standardized = standardize_array(self.filled, prepared.scaler)
        self.tod, self.dow, self.calendar_available = build_calendar(
            prepared.timestamps, prepared.num_steps, prepared.steps_per_day
        )
        self.sample_ids = np.asarray(self.starts, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, index: int) -> WindowSample:
        start = int(self.starts[index])
        hist_start = start - self.L
        history = self.standardized[hist_start:start]  # [L, N]
        observed = self.prepared.valid[hist_start:start]  # [L, N]
        if history.shape[0] != self.L:
            raise DataError(f"history slice length {history.shape[0]} != {self.L}")
        target_raw = self.prepared.data[start : start + self.T].copy()
        target_valid = self.prepared.valid[start : start + self.T].copy()
        # Safe replacement before any arithmetic; the mask keeps these out.
        target_raw[~target_valid] = 0.0
        target_raw[~np.isfinite(target_raw)] = 0.0

        model_input = ModelInput(
            history=torch.from_numpy(history[..., None])[None],
            observed=torch.from_numpy(observed[..., None])[None],
            time_of_day=torch.from_numpy(self.tod[hist_start:start].copy())[None],
            day_of_week=torch.from_numpy(self.dow[hist_start:start].copy())[None],
        )
        target = TargetBatch(
            values_raw=torch.from_numpy(target_raw[..., None])[None],
            valid=torch.from_numpy(target_valid[..., None])[None],
            sample_ids=torch.tensor([start], dtype=torch.int64),
        )
        return WindowSample(model_input=model_input, target=target)

    def metadata(self) -> dict[str, object]:
        return {
            "split": self.split,
            "num_windows": len(self),
            "sample_ids": self.sample_ids.tolist(),
            "input_steps": self.L,
            "output_steps": self.T,
            "calendar_available": bool(self.calendar_available),
        }


@dataclass
class WindowSample:
    model_input: ModelInput
    target: TargetBatch


def collate_samples(samples: list[WindowSample]) -> WindowSample:
    """Concatenate per-sample batches along the batch axis.

    Each dataset item is already a batch of one (the ``ModelInput`` contract is
    batched), so collation concatenates rather than stacks.
    """
    if not samples:
        raise DataError("cannot collate an empty batch")
    histories = torch.cat([s.model_input.history for s in samples], dim=0)
    observed = torch.cat([s.model_input.observed for s in samples], dim=0)
    tod = torch.cat([s.model_input.time_of_day for s in samples], dim=0)
    dow = torch.cat([s.model_input.day_of_week for s in samples], dim=0)
    values = torch.cat([s.target.values_raw for s in samples], dim=0)
    valid = torch.cat([s.target.valid for s in samples], dim=0)
    ids = torch.cat([s.target.sample_ids for s in samples], dim=0)
    return WindowSample(
        model_input=ModelInput(
            history=histories,
            observed=observed,
            time_of_day=tod,
            day_of_week=dow,
        ),
        target=TargetBatch(values_raw=values, valid=valid, sample_ids=ids),
    )


def make_loader(
    prepared: PreparedData,
    split: str,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 11,
    allow_empty: bool = False,
) -> DataLoader:
    dataset = TrafficWindowDataset(prepared, split, allow_empty=allow_empty)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_samples,
        generator=generator,
        drop_last=False,
        persistent_workers=False,
    )


def split_sample_ids(prepared: PreparedData) -> dict[str, np.ndarray]:
    return {k: v.copy() for k, v in prepared.scheme.starts.items()}


__all__ = [
    "PREPARED_SCHEMA_VERSION",
    "PreparedData",
    "WindowSample",
    "generate_synthetic",
    "compute_prepare_signature",
    "prepare_dataset",
    "save_prepared",
    "load_prepared",
    "prepared_path_for",
    "load_or_prepare",
    "build_calendar",
    "TrafficWindowDataset",
    "collate_samples",
    "make_loader",
    "split_sample_ids",
]
