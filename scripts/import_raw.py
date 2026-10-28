"""Convert upstream raw dataset files into the layout ``prepare-data`` expects.

The importer never invents values: it copies a series file verbatim when that
file is already canonical, and otherwise performs a documented, loss-free
conversion.  Every import writes ``RAW_SOURCE.json`` with the upstream file
hashes and the exact transform, so a prepared artifact can be traced back to
the published files.

Supported inputs
----------------
* series: ``.npz`` (key ``data`` or ``values``; optional ``mask``/``node_ids``),
  ``.npy``, ``.h5`` (requires ``h5py``), ``.csv`` with an optional datetime
  column;
* graph: ``.pkl`` with ``(ids, id_to_ind, adj)``, ``(ids, adj)``, a dense
  ``ndarray`` or a scipy sparse matrix (restricted unpickler, no arbitrary
  code); ``.npz``/``.npy`` matrices or edge lists; ``.csv``/``.txt`` edges;
* calendar: ``tod``/``dow`` channels inside the series, or an explicit start
  date for uniform sampling.

Usage::

    python scripts/import_raw.py --all --raw-dir raw --data-root data
    python scripts/import_raw.py --dataset PEMS08 --raw-dir raw --data-root data
    python scripts/import_raw.py --dataset METRLA --start-weekday 3 --force

Then::

    python -m sbona.cli prepare-data --config configs/experiments/pems08.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_calendar  # noqa: E402  (scripts/ helper, not a package)

DEFAULT_RAW = REPO_ROOT / "raw"
DEFAULT_DATA = REPO_ROOT / "data"


class ImportFailure(RuntimeError):
    """Raised when a raw file is missing, malformed or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Dataset presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Where the published files live and what shape they must have."""

    name: str
    series: tuple[str, ...]
    edges: tuple[str, ...]
    node_ids: str | None
    start: str
    start_weekday: int
    expected_steps: int | None
    expected_nodes: int
    expected_edges: int | None
    source_note: str


SPECS: dict[str, DatasetSpec] = {
    "PEMS03": DatasetSpec(
        name="PEMS03",
        series=("PEMS03.npz", "data.npz"),
        edges=("PEMS03.csv",),
        node_ids="PEMS03.txt",
        start="2018-09-01",
        start_weekday=5,  # Saturday
        expected_steps=26208,
        expected_nodes=358,
        expected_edges=546,
        source_note="Caltrans PeMS via the ASTGNN / STAEformer mirrors",
    ),
    "PEMS04": DatasetSpec(
        name="PEMS04",
        series=("PEMS04.npz", "data.npz"),
        edges=("PEMS04.csv",),
        node_ids=None,
        start="2018-01-01",
        start_weekday=0,
        expected_steps=16992,
        expected_nodes=307,
        expected_edges=340,
        source_note="Caltrans PeMS via the ASTGNN mirror",
    ),
    "PEMS07": DatasetSpec(
        name="PEMS07",
        series=("PEMS07.npz", "data.npz"),
        edges=("PEMS07.csv",),
        node_ids=None,
        start="2017-05-01",
        start_weekday=0,
        expected_steps=28224,
        expected_nodes=883,
        expected_edges=866,
        source_note="Caltrans PeMS via the ASTGNN mirror",
    ),
    "PEMS08": DatasetSpec(
        name="PEMS08",
        series=("PEMS08.npz", "data.npz"),
        edges=("PEMS08.csv",),
        node_ids=None,
        start="2016-07-01",
        start_weekday=4,
        expected_steps=17856,
        expected_nodes=170,
        expected_edges=274,
        source_note="Caltrans PeMS via the ASTGNN / STAEformer mirrors",
    ),
    "METRLA": DatasetSpec(
        name="METRLA",
        series=("data.npz", "METRLA.npz", "metr-la.h5", "METR-LA.csv"),
        edges=("adj_mx_METR-LA.pkl", "adj_mx.pkl", "adj_edges.npz"),
        node_ids=None,
        start="2012-03-01",
        start_weekday=3,  # Thursday
        expected_steps=34272,
        expected_nodes=207,
        expected_edges=1313,
        source_note="DCRNN / Zenodo 10.5281/zenodo.5724362 (CC-BY-4.0)",
    ),
    "PEMSBAY": DatasetSpec(
        name="PEMSBAY",
        series=("data.npz", "PEMSBAY.npz", "pems-bay.h5", "PEMS-BAY.csv"),
        edges=("adj_mx_PEMS-BAY.pkl", "adj_mx_bay.pkl", "adj_edges.npz"),
        node_ids=None,
        start="2017-01-01",
        start_weekday=6,  # Sunday
        expected_steps=52116,
        expected_nodes=325,
        expected_edges=2079,
        source_note="DCRNN / Zenodo 10.5281/zenodo.5724362 (CC-BY-4.0)",
    ),
}


# ---------------------------------------------------------------------------
# Series readers
# ---------------------------------------------------------------------------


def _as_series(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim not in (2, 3):
        raise ImportFailure(f"{name}: expected [M, N] or [M, N, C], got {array.shape}")
    if array.dtype.kind not in "fiu":
        raise ImportFailure(f"{name}: series dtype must be numeric, got {array.dtype}")
    return array


def read_series(path: Path) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Return ``(series, timestamps or None, format_note)``."""
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = "data" if "data" in archive else ("values" if "values" in archive else None)
            if key is None:
                raise ImportFailure(f"{path} must contain 'data' or 'values'; keys={archive.files}")
            series = _as_series(archive[key], path.name)
            timestamps = np.asarray(archive["timestamps"]) if "timestamps" in archive else None
        return series, timestamps, "npz"
    if suffix == ".npy":
        return _as_series(np.load(path, allow_pickle=False), path.name), None, "npy"
    if suffix in (".h5", ".hdf5"):
        try:
            import h5py  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportFailure(
                f"{path.name}: reading HDF5 needs the optional 'h5py' package; "
                "install it or pass the CSV mirror of the same dataset"
            ) from exc
        with h5py.File(path, "r") as handle:
            keys = list(handle.keys())
            key = "df" if "df" in keys else keys[0]
            series = _as_series(handle[key][()], path.name)
            timestamps = None
            axis0 = handle[key].dims[0].get("axis0") if handle[key].dims else None
            if axis0 is not None:
                try:
                    timestamps = np.asarray(axis0[()]).astype("datetime64[s]")
                except (TypeError, ValueError):
                    timestamps = None
        return series, timestamps, "h5"
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip()
        columns = [part.strip() for part in header.split(",")]
        has_time = bool(columns) and _looks_like_timestamp(columns[0])
        skip = 1 if not has_time and not _looks_numeric(columns[0]) else 0
        if has_time:
            stamps: list[str] = []
            rows: list[list[float]] = []
            with path.open("r", encoding="utf-8") as handle:
                next(handle)
                for line in handle:
                    if not line.strip():
                        continue
                    parts = line.rstrip("\n").split(",")
                    stamps.append(parts[0].strip())
                    rows.append([float(x) for x in parts[1:]])
            series = _as_series(np.asarray(rows, dtype=np.float64), path.name)
            timestamps = np.asarray(stamps).astype("datetime64[s]")
            return series, timestamps, "csv_with_time"
        series = _as_series(np.loadtxt(path, delimiter=",", skiprows=skip, dtype=np.float64), path.name)
        return series, None, "csv"
    raise ImportFailure(f"unsupported series format: {path.suffix}")


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _looks_like_timestamp(token: str) -> bool:
    if _looks_numeric(token):
        return False
    lowered = token.lower()
    return any(sep in lowered for sep in ("-", ":", "/")) or lowered.startswith("20")


# ---------------------------------------------------------------------------
# Adjacency readers
# ---------------------------------------------------------------------------


class _RestrictedUnpickler(pickle.Unpickler):
    """Only numpy/scipy containers are allowed; never load arbitrary objects."""

    ALLOWED_ROOTS = ("numpy", "scipy")

    def find_class(self, module: str, name: str) -> Any:  # noqa: D102
        if module.split(".")[0] in self.ALLOWED_ROOTS:
            return super().find_class(module, name)
        raise ImportFailure(f"refusing to unpickle {module}.{name} from a dataset graph file")


def _adjacency_matrix(obj: Any, path: Path) -> tuple[np.ndarray, list[str] | None]:
    """Unpack the known DCRNN/BasicTS pickle layouts into ``(adj, ids)``."""
    if isinstance(obj, (list, tuple)):
        if len(obj) == 3:
            ids, _id_to_ind, adj = obj
        elif len(obj) == 2:
            ids, adj = obj
        else:
            raise ImportFailure(f"{path.name}: unsupported pickle tuple of length {len(obj)}")
        ids = [str(item) for item in ids] if ids is not None else None
        return _as_dense(adj, path), ids
    return _as_dense(obj, path), None


def _as_dense(obj: Any, path: Path) -> np.ndarray:
    if hasattr(obj, "toarray"):
        obj = obj.toarray()
    array = np.asarray(obj)
    if array.ndim == 1:
        side = int(round(np.sqrt(array.size)))
        if side * side != array.size:
            raise ImportFailure(f"{path.name}: flat adjacency is not square")
        array = array.reshape(side, side)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ImportFailure(f"{path.name}: adjacency must be square, got {array.shape}")
    return array.astype(np.float64, copy=False)


def read_adjacency(path: Path) -> tuple[np.ndarray, list[str] | None]:
    """Return ``(edges [E, 2] int64, ids or None)`` in column-index space."""
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        with path.open("rb") as handle:
            obj = _RestrictedUnpickler(handle, encoding="latin1").load()
        adj, ids = _adjacency_matrix(obj, path)
        return binarize_undirected(adj), ids
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "edge_index" in archive:
                edges = np.asarray(archive["edge_index"])
                if edges.ndim == 2 and edges.shape[0] == 2:
                    edges = edges.T
                return edges.astype(np.int64), None
            if "edges" in archive:
                return np.asarray(archive["edges"], dtype=np.int64), None
            key = "adj" if "adj" in archive else None
            if key is None:
                raise ImportFailure(f"{path.name}: npz has no edge_index/edges/adj; keys={archive.files}")
            return binarize_undirected(np.asarray(archive[key])), None
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.ndim == 2 and array.shape[1] == 2:
            return array.astype(np.int64), None
        return binarize_undirected(array), None
    if suffix in (".csv", ".txt"):
        rows: list[tuple[int, int]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line in handle:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                try:
                    i, j = int(parts[0]), int(parts[1])
                except ValueError:
                    continue  # header/comments
                rows.append((i, j))
        if not rows:
            raise ImportFailure(f"{path.name}: no edges parsed")
        return np.asarray(rows, dtype=np.int64), None
    raise ImportFailure(f"unsupported graph format: {path.suffix}")


def binarize_undirected(adj: np.ndarray) -> np.ndarray:
    """``adj > 0`` -> drop diagonal -> undirected union -> unique ``i < j`` pairs."""
    matrix = np.asarray(adj)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ImportFailure(f"adjacency must be square, got {matrix.shape}")
    positive = np.isfinite(matrix) & (matrix > 0)
    np.fill_diagonal(positive, False)
    rows, cols = np.nonzero(positive)
    if rows.size == 0:
        raise ImportFailure("adjacency has no positive off-diagonal entries")
    pairs = np.stack([np.minimum(rows, cols), np.maximum(rows, cols)], axis=1)
    return np.unique(pairs, axis=0).astype(np.int64)


def unique_undirected_pairs(edges: np.ndarray) -> np.ndarray:
    """Drop self-loops, sort each pair and deduplicate; returns ``[E, 2]``."""
    array = np.asarray(edges, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ImportFailure(f"edge list must be [E, 2], got {array.shape}")
    array = array[array[:, 0] != array[:, 1]]
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.sort(array, axis=1), axis=0).astype(np.int64)


def undirected_edge_count_consistent(edges: np.ndarray, num_nodes: int) -> None:
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ImportFailure(
            f"edge endpoint out of range for {num_nodes} nodes: [{edges.min()}, {edges.max()}]"
        )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **arrays)
    tmp.replace(path)


def _copy_verbatim(source: Path, dest: Path) -> bool:
    """Copy when the file is already canonical; return True if bytes are reused."""
    if dest.resolve() == source.resolve():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return True


def write_series(
    source: Path,
    series: np.ndarray,
    dest: Path,
    *,
    canonical: bool,
) -> str:
    """Persist the series as ``data.npz``; copy verbatim when possible."""
    if canonical and source.suffix.lower() == ".npz":
        _copy_verbatim(source, dest)
        return "copied"
    _write_npz_atomic(dest, data=series)
    return "rewritten"


def write_edge_csv(edges: np.ndarray, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        handle.write("from,to,cost\n")
        for i, j in edges.tolist():
            handle.write(f"{i},{j},1.0\n")
    tmp.replace(dest)


def write_edge_npz(edges: np.ndarray, dest: Path, *, node_ids: np.ndarray, rule: str) -> None:
    _write_npz_atomic(
        dest,
        edge_index=edges.T.astype(np.int64),
        node_ids=node_ids.astype(np.int64),
        topology=np.asarray("derived_binary_undirected"),
        rule=np.asarray(rule),
    )


def write_node_ids(ids: list[str] | np.ndarray, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text("".join(f"{item}\n" for item in ids), encoding="utf-8")
    tmp.replace(dest)


# ---------------------------------------------------------------------------
# Import driver
# ---------------------------------------------------------------------------


def _pick_existing(raw_dir: Path, candidates: tuple[str, ...], role: str) -> Path:
    for name in candidates:
        candidate = raw_dir / name
        if candidate.is_file():
            return candidate
    raise ImportFailure(f"no {role} file found in {raw_dir}; looked for {list(candidates)}")


def import_dataset(
    spec: DatasetSpec,
    raw_root: Path,
    data_root: Path,
    *,
    force: bool,
    start_weekday: int | None,
    start: str | None,
    calendar_mode: str = "auto",
) -> dict[str, Any]:
    raw_dir = raw_root / spec.name
    if not raw_dir.is_dir():
        raise ImportFailure(f"raw directory not found: {raw_dir}")
    dest_dir = data_root / spec.name

    series_path = _pick_existing(raw_dir, spec.series, "series")
    edges_path = _pick_existing(raw_dir, spec.edges, "graph")

    series, timestamps, series_format = read_series(series_path)
    steps, nodes = int(series.shape[0]), int(series.shape[1])
    if spec.expected_steps is not None and steps != spec.expected_steps:
        raise ImportFailure(f"{spec.name}: expected {spec.expected_steps} steps, got {steps}")
    if nodes != spec.expected_nodes:
        raise ImportFailure(f"{spec.name}: expected {spec.expected_nodes} nodes, got {nodes}")

    edges, edge_ids = read_adjacency(edges_path)

    node_ids_path: Path | None = None
    node_id_values: np.ndarray | None = None
    if spec.node_ids is not None:
        node_ids_path = raw_dir / spec.node_ids
        if not node_ids_path.is_file():
            raise ImportFailure(f"{spec.name}: missing node id file {node_ids_path}")
        node_id_values = _read_node_ids(node_ids_path, nodes)
    edges_in_index_space = unique_undirected_pairs(_map_edges(edges, node_id_values, nodes))
    undirected_edge_count_consistent(edges_in_index_space, nodes)
    if spec.expected_edges is not None and edges_in_index_space.shape[0] != spec.expected_edges:
        raise ImportFailure(
            f"{spec.name}: expected {spec.expected_edges} undirected edges, "
            f"got {edges_in_index_space.shape[0]}"
        )

    # --- series ---
    canonical = (
        series_path.suffix.lower() == ".npz"
        and series.ndim == 3
        and series.shape[2] >= 2
        and str(series.dtype) in ("float64", "float32")
    )
    data_dest = dest_dir / "data.npz"
    _guard_existing(data_dest, force)
    series_action = write_series(series_path, series, data_dest, canonical=canonical)

    # --- graph ---
    graph_file: str
    if edges_path.suffix.lower() in (".csv", ".txt") and edge_ids is None:
        graph_dest = dest_dir / f"{spec.name}.csv"
        _guard_existing(graph_dest, force)
        _copy_verbatim(edges_path, graph_dest)
        graph_file = graph_dest.name
        graph_action = "copied_edge_list"
    else:
        graph_dest = dest_dir / "adj_edges.npz"
        _guard_existing(graph_dest, force)
        write_edge_npz(
            edges_in_index_space,
            graph_dest,
            node_ids=np.arange(nodes, dtype=np.int64),
            rule="adj>0; drop diagonal; undirected union; keep i<j",
        )
        graph_file = graph_dest.name
        graph_action = "converted_adjacency"
        if edge_ids is not None and len(edge_ids) != nodes:
            raise ImportFailure(
                f"{spec.name}: adjacency ids ({len(edge_ids)}) do not match series nodes ({nodes})"
            )

    if node_ids_path is not None:
        ids_dest = dest_dir / spec.node_ids
        _guard_existing(ids_dest, force)
        _copy_verbatim(node_ids_path, ids_dest)

    # --- calendar ---
    calendar_dest = dest_dir / "calendar.npz"
    _guard_existing(calendar_dest, force)
    weekday = spec.start_weekday if start_weekday is None else int(start_weekday)
    calendar_info = _build_calendar(
        series=series,
        timestamps=timestamps,
        dest=calendar_dest,
        start=spec.start if start is None else start,
        start_weekday=weekday,
        steps_per_day=288,
        mode=calendar_mode,
    )

    record = {
        "dataset": spec.name,
        "spec_source": spec.source_note,
        "raw_dir": str(raw_dir),
        "series": {
            "file": series_path.name,
            "sha256": sha256_file(series_path),
            "format": series_format,
            "shape": [steps, nodes] + list(series.shape[2:]),
            "action": series_action,
        },
        "graph": {
            "file": edges_path.name,
            "sha256": sha256_file(edges_path),
            "raw_rows": int(edges.shape[0]),
            "undirected_edges": int(edges_in_index_space.shape[0]),
            "action": graph_action,
            "output": graph_file,
        },
        "node_ids_file": spec.node_ids,
        "calendar": calendar_info,
        "outputs": {
            "data.npz": sha256_file(data_dest),
            "graph": sha256_file(dest_dir / graph_file),
            "calendar.npz": sha256_file(calendar_dest),
        },
    }
    if node_ids_path is not None:
        record["node_ids"] = {
            "file": node_ids_path.name,
            "sha256": sha256_file(node_ids_path),
        }
    (dest_dir / "RAW_SOURCE.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def _guard_existing(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise ImportFailure(
            f"{path} already exists; pass --force to overwrite (its hash is recorded in "
            "RAW_SOURCE.json / SOURCE.json)"
        )


def _calendar_channels_look_valid(series: np.ndarray, steps_per_day: int) -> bool:
    """True only when channels 1/2 really are (tod, dow) and not occupancy/speed.

    The day channel must be an integer 0..6 that advances by exactly one day at
    every day boundary.  This tolerates gaps in the time axis (PEMS-BAY loses
    one hour) while rejecting occupancy/speed channels, which never march
    through the week.
    """
    if series.ndim != 3 or series.shape[2] < 3 or series.shape[0] < 2 * steps_per_day:
        return False
    tod = np.asarray(series[:, 0, 1], dtype=np.float64)
    dow = np.asarray(series[:, 0, 2], dtype=np.float64)
    if not np.isfinite(tod).all() or tod.min() < 0.0 or tod.max() >= 1.0:
        return False
    if not np.isfinite(dow).all():
        return False
    if not np.allclose(dow, np.rint(dow), atol=1e-6) or dow.min() < 0 or dow.max() > 6:
        return False
    days = np.rint(dow).astype(np.int64)
    deltas = np.diff(days)
    boundaries = deltas != 0
    expected = series.shape[0] / float(steps_per_day) - 1.0
    if not (0.85 * expected <= boundaries.sum() <= 1.15 * expected + 1.0):
        return False
    if not np.all(np.mod(deltas[boundaries], 7) == 1):
        return False
    if not tod.max() - tod.min() > 0.9:
        return False
    # A real tod channel resets at each midnight; occupancy/speed never do.
    wraps = int(np.sum(np.diff(tod) < -1e-3))
    return bool(0.85 * expected <= wraps <= 1.15 * expected + 1.0)


def _build_calendar(
    *,
    series: np.ndarray,
    timestamps: np.ndarray | None,
    dest: Path,
    start: str,
    start_weekday: int,
    steps_per_day: int,
    mode: str = "auto",
) -> dict[str, Any]:
    steps = int(series.shape[0])
    channels_ok = _calendar_channels_look_valid(series, steps_per_day)
    if mode == "channels" and not channels_ok:
        raise ImportFailure(
            "calendar mode 'channels' requested but channels 1/2 are not a valid (tod, dow) table"
        )
    if mode == "uniform" and timestamps is not None:
        raise ImportFailure("calendar mode 'uniform' conflicts with embedded timestamps")

    use_timestamps = timestamps is not None and mode in ("auto", "timestamps")
    use_channels = channels_ok and mode in ("auto", "channels") and not use_timestamps

    if use_timestamps:
        if timestamps.shape[0] != steps:
            raise ImportFailure("embedded timestamps length does not match the series")
        if timestamps.dtype.kind != "M":
            timestamps = timestamps.astype("datetime64[s]")
        minutes = (
            (timestamps - timestamps.astype("datetime64[D]")).astype("timedelta64[m]").astype(np.int64)
        )
        tod = minutes.astype(np.float64) / (24 * 60)
        weekday = (timestamps.astype("datetime64[D]").astype(np.int64) + 3) % 7
        table = np.stack([tod.astype(np.float32), weekday.astype(np.float32)], axis=1)
        info = {
            "mode": "embedded_timestamps",
            "start": str(timestamps[0]),
            "end": str(timestamps[-1]),
        }
    elif use_channels:
        table, info = build_calendar.build_table_from_array(
            series,
            start_weekday=start_weekday,
            steps_per_day=steps_per_day,
            tod_channel=1,
            dow_channel=2,
        )
        info["mode"] = "data_channels"
        info["assumption"] = (
            "tod/dow channels come from the published npz; the weekday offset is inferred "
            f"from the documented start date {start}"
        )
    else:
        stamps = build_calendar.build_uniform(start, steps, interval_minutes=60 * 24 // steps_per_day)
        minutes = (stamps - stamps.astype("datetime64[D]")).astype("timedelta64[m]").astype(np.int64)
        tod = minutes.astype(np.float64) / (24 * 60)
        weekday = (stamps.astype("datetime64[D]").astype(np.int64) + 3) % 7
        table = np.stack([tod.astype(np.float32), weekday.astype(np.float32)], axis=1)
        info = {
            "mode": "uniform",
            "start": start,
            "steps": steps,
            "interval_minutes": 60 * 24 // steps_per_day,
            "assumption": "uniform sampling from the documented start date; no gaps assumed",
        }

    np.savez_compressed(
        dest,
        timestamps=table,
        note=np.asarray(f"imported by scripts/import_raw.py ({info['mode']})"),
        mode=np.asarray(str(info["mode"])),
    )
    return info


def _read_node_ids(path: Path, num_nodes: int) -> np.ndarray:
    values: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip().split(",")[0].strip()
            if not token:
                continue
            try:
                values.append(int(float(token)))
            except ValueError:
                continue
    ids = np.asarray(values, dtype=np.int64)
    if ids.shape != (num_nodes,):
        raise ImportFailure(f"{path.name}: expected {num_nodes} ids, got {ids.shape[0]}")
    return ids


def _map_edges(edges: np.ndarray, node_ids: np.ndarray | None, num_nodes: int) -> np.ndarray:
    if node_ids is None:
        return edges.astype(np.int64)
    mapping = {int(node_id): index for index, node_id in enumerate(node_ids)}
    try:
        return np.vectorize(mapping.__getitem__, otypes=[np.int64])(edges)
    except KeyError as exc:
        raise ImportFailure(f"edge endpoint {exc.args[0]} is not in the node id file") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dataset", nargs="+", action="extend", default=None, metavar="NAME")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite existing canonical files")
    parser.add_argument("--start-weekday", type=int, default=None, help="override Monday=0")
    parser.add_argument("--start", default=None, help="override the documented ISO start date")
    parser.add_argument(
        "--calendar-mode",
        choices=("auto", "channels", "uniform", "timestamps"),
        default="auto",
        help="how to build calendar.npz (default: embedded timestamps > tod/dow channels > uniform)",
    )
    args = parser.parse_args(argv)

    selected = list(SPECS) if args.all else (args.dataset or [])
    if not selected:
        parser.error("pass --all or --dataset NAME (available: " + ", ".join(SPECS) + ")")
    unknown = [name for name in selected if name not in SPECS]
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; available: {sorted(SPECS)}")

    failures = 0
    for name in selected:
        try:
            record = import_dataset(
                SPECS[name],
                args.raw_dir,
                args.data_root,
                force=args.force,
                start_weekday=args.start_weekday,
                start=args.start,
                calendar_mode=args.calendar_mode,
            )
        except ImportFailure as exc:
            failures += 1
            print(f"[{name}] FAILED: {exc}", file=sys.stderr)
            continue
        print(
            f"[{name}] {record['series']['action']:<9} "
            f"series={record['series']['file']} "
            f"graph={record['graph']['file']} -> {record['graph']['output']} "
            f"edges={record['graph']['undirected_edges']} "
            f"calendar={record['calendar']['mode']}"
        )
        print(f"        wrote {args.data_root / name / 'RAW_SOURCE.json'}")
    if failures:
        print(f"{failures} dataset(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
