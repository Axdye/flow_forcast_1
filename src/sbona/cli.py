"""Command line entry points.

The CLI contains path/argument plumbing only.  Data, model, objective and
evaluation formulas live in their own modules.  Every training command writes a
self-describing run directory; a failure is recorded in ``failure.json``
instead of being silently swallowed.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import traceback
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sbona.backbones import build_backbone
from sbona.baselines import build_adapter
from sbona.config import Config, ConfigError, load_config, resolve_roots
from sbona.data.dataset import (
    PreparedData,
    TrafficWindowDataset,
    collate_samples,
    load_or_prepare,
    prepare_dataset,
)
from sbona.data.manifest import sha256_file
from sbona.evaluation.artifacts import (
    METRICS_SCHEMA_VERSION,
    RunArtifacts,
    collect_environment,
    create_run_id,
    read_json,
    write_json,
)
from sbona.evaluation.profiler import profile_callable, profile_model
from sbona.evaluation.runner import (
    evaluate_adapter_split,
    evaluate_base_split,
    metric_units,
)
from sbona.training.checkpoint import (
    load_checkpoint,
    load_state_dict_strict,
    validate_resume,
)
from sbona.training.probe import run_route_probe
from sbona.training.trainer import (
    BaseTrainer,
    CandidateTrainer,
    RouterTrainer,
    adapter_identity_hash,
    backbone_identity_hash,
    make_hashes,
    move_input,
    resolve_device,
)

EXPERIMENT_COLUMNS = (
    "run_id",
    "method",
    "dataset",
    "backbone",
    "split",
    "seed",
    "protocol_id",
    "data_hash",
    "metric",
    "value",
    "units",
    "deployable",
    "artifact",
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_cfg(args: argparse.Namespace) -> Config:
    if getattr(args, "config", None) is None:
        raise ConfigError("--config is required for this command")
    # The seed comes from the config and nowhere else: one fixed seed per
    # experiment, so a CLI flag cannot relabel a run after the fact.
    return load_config(args.config)


def _roots(
    args: argparse.Namespace, cfg: Config
) -> tuple[Path, Path]:
    return resolve_roots(
        cfg,
        data_root=getattr(args, "data_root", None),
        output_root=getattr(args, "output_root", None),
    )


def _synthetic_flag(args: argparse.Namespace) -> bool | None:
    value = getattr(args, "synthetic", None)
    return None if value is None else bool(value)


def _device(args: argparse.Namespace, cfg: Config) -> torch.device:
    name = getattr(args, "device", None) or cfg.training.device
    return resolve_device(name)


def _resolve_run_dir(run: str | Path, output_root: Path) -> Path:
    path = Path(run).expanduser()
    if not path.is_absolute():
        # ``--run outputs/RUN_ID`` must not become
        # ``outputs/outputs/RUN_ID`` when the default root is also ``outputs``.
        direct = Path.cwd() / path
        if direct.is_dir():
            path = direct
        else:
            path = output_root / path
    if not path.is_dir():
        raise FileNotFoundError(f"run directory not found: {path}")
    return path


def _resolve_checkpoint(
    value: str | Path | None, run_dir: Path | None, default_name: str = "checkpoint_best.pt"
) -> Path:
    if value is not None:
        path = Path(value)
        if not path.is_absolute() and run_dir is not None and not path.exists():
            path = run_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path
    if run_dir is None:
        raise FileNotFoundError("no run directory was supplied")
    path = run_dir / default_name
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def _manifest_base(
    cfg: Config,
    prepared: PreparedData,
    *,
    stage: str,
    hashes: dict[str, Any],
    device: torch.device,
    num_candidates: int,
    parent_checkpoint: Path | None,
) -> dict[str, Any]:
    parent_hash = None
    if parent_checkpoint is not None and parent_checkpoint.is_file():
        # Imported lazily to keep the manifest helper usable without a checkpoint.
        from sbona.training.checkpoint import checkpoint_sha256

        parent_hash = checkpoint_sha256(parent_checkpoint)
    return {
        "schema_version": 1,
        "status": "running",
        "stage": stage,
        "dataset": prepared.dataset,
        "protocol_id": cfg.protocol_id,
        "method_id": cfg.method_id,
        "backbone": cfg.backbone.name,
        "backbone_mode": cfg.backbone.mode,
        "adapter_kind": cfg.method_id,
        "num_candidates": int(num_candidates),
        "seed": cfg.seed,
        "device": str(device),
        "created_at": _utc_now(),
        "command": sys.argv,
        "config_hash": cfg.hash,
        "hashes": hashes,
        "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
        "parent_checkpoint_hash": parent_hash,
        "data_manifest": prepared.manifest.to_dict(),
        "environment": collect_environment(_repo_root()),
        "precision": cfg.training.precision,
    }


def _make_run(
    args: argparse.Namespace,
    cfg: Config,
    prepared: PreparedData,
    output_root: Path,
    *,
    stage: str,
    num_candidates: int,
    device: torch.device,
    parent_checkpoint: Path | None,
) -> tuple[RunArtifacts, dict[str, Any], dict[str, Any]]:
    run_id = getattr(args, "run_id", None) or create_run_id(
        cfg.hash, prefix=f"{stage}-{cfg.data.dataset}-{cfg.method_id}"
    )
    run = RunArtifacts(output_root / run_id)
    run.write_resolved_config(cfg.to_yaml())
    hashes = make_hashes(prepared, cfg, num_candidates)
    manifest = _manifest_base(
        cfg,
        prepared,
        stage=stage,
        hashes=hashes,
        device=device,
        num_candidates=num_candidates,
        parent_checkpoint=parent_checkpoint,
    )
    run.write_manifest(manifest)
    args._run = run
    return run, hashes, manifest


def _update_manifest(
    run: RunArtifacts, manifest: dict[str, Any], **updates: Any
) -> dict[str, Any]:
    manifest.update(updates)
    run.write_manifest(manifest)
    return manifest


def _write_failure(args: argparse.Namespace, stage: str, exc: BaseException) -> None:
    run = getattr(args, "_run", None)
    if run is None:
        return
    latest = run.path("checkpoint_latest.pt")
    run.write_failure(
        {
            "schema_version": 1,
            "stage": stage,
            "timestamp": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "last_checkpoint": str(latest) if latest.is_file() else None,
        }
    )


#: Hashes a run must reproduce before its checkpoint may be reused.
#:
#: ``backbone_identity_hash`` / ``adapter_identity_hash`` are the important ones:
#: they cover seed, method/adapter kind, architecture and K.  Without them two
#: runs of the same stage/data/split (for example seed 11 and seed 22) are
#: indistinguishable and the stale one is picked silently.
DATA_SELECTION_KEYS = (
    "data_hash",
    "valid_hash",
    "graph_hash",
    "split_hash",
    "scaler_hash",
    "num_nodes",
    "num_candidates",
)
BACKBONE_SELECTION_KEYS = ("backbone_identity_hash",) + DATA_SELECTION_KEYS
ADAPTER_SELECTION_KEYS = ("adapter_identity_hash",) + BACKBONE_SELECTION_KEYS


def _selection_keys(stage: str) -> tuple[str, ...]:
    """Stage A holds only a backbone, so its adapter identity must not be pinned."""
    return BACKBONE_SELECTION_KEYS if stage == "A" else ADAPTER_SELECTION_KEYS


def _iter_run_manifests(output_root: Path):
    if not output_root.is_dir():
        return
    children = sorted(
        (path for path in output_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for child in children:
        manifest_path = child / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        if isinstance(manifest, dict) and "stage" in manifest and "hashes" in manifest:
            yield child, manifest


def _find_run(
    output_root: Path,
    *,
    stage: str,
    prepared: PreparedData,
    cfg: Config,
    num_candidates: int,
    explicit_run: str | Path | None = None,
    explicit_checkpoint: str | Path | None = None,
    label: str,
) -> Path:
    if explicit_checkpoint is not None:
        return _resolve_checkpoint(explicit_checkpoint, None)
    if explicit_run is not None:
        run_dir = _resolve_run_dir(explicit_run, output_root)
        return _resolve_checkpoint(None, run_dir)

    expected = make_hashes(prepared, cfg, num_candidates)
    keys = _selection_keys(stage)
    matches: list[Path] = []
    for run_dir, manifest in _iter_run_manifests(output_root):
        if manifest.get("stage") != stage:
            continue
        stored = manifest.get("hashes", {})
        if any(stored.get(key) != expected.get(key) for key in keys):
            continue
        candidate = run_dir / "checkpoint_best.pt"
        if not candidate.is_file():
            candidate = run_dir / "checkpoint_latest.pt"
        if candidate.is_file():
            matches.append(candidate)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # One seed produces exactly one run per stage.  More than one match means
        # the output root holds interchangeable-looking runs (same seed/method/
        # data but re-trained, or a stale copy); guessing by mtime silently picks
        # the wrong weights.
        names = ", ".join(sorted(path.parent.name for path in matches))
        raise ValueError(
            f"{len(matches)} stage {stage} {label} runs match the current seed/method/"
            f"data/split/scaler/graph/K: {names}; expected exactly one. Pass an explicit "
            f"--{label}-run or --{label}-checkpoint, or remove the stale run(s)."
        )
    raise FileNotFoundError(
        f"no stage {stage} {label} checkpoint matching the current seed/method/"
        f"data/split/scaler/graph/K was found under {output_root}; pass an explicit "
        f"--{label}-checkpoint or --{label}-run"
    )


# ---------------------------------------------------------------------------
# Experiment rows
# ---------------------------------------------------------------------------


def _units_for(metric: str, data_unit: str) -> str:
    units = metric_units(data_unit)
    for key, label in units.items():
        if key in metric:
            return label
    return data_unit


def _read_experiment_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        return [dict(row) for row in reader]


def _write_experiment_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in EXPERIMENT_COLUMNS})


def _append_experiment_rows(
    run: RunArtifacts,
    output_root: Path,
    cfg: Config,
    prepared: PreparedData,
    rows: list[dict[str, Any]],
) -> None:
    artifact = run.path("metrics.json")
    if not artifact.is_file():
        raise FileNotFoundError("metrics.json must exist before experiments.csv is written")
    data_hash = prepared.manifest.extra.get("data_sha256")
    prepared_rows: list[dict[str, str]] = []
    for row in rows:
        value = row.get("value")
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric):
            continue
        split = str(row["split"])
        if split not in ("train", "val", "test"):
            raise ValueError(f"experiment row has unknown split {split!r}")
        prepared_rows.append(
            {
                "run_id": run.run_dir.name,
                "method": str(row["method"]),
                "dataset": prepared.dataset,
                "backbone": cfg.backbone.name,
                "split": split,
                "seed": str(cfg.seed),
                "protocol_id": cfg.protocol_id,
                "data_hash": str(data_hash),
                "metric": str(row["metric"]),
                "value": repr(numeric),
                "units": str(
                    row.get("units") or _units_for(str(row["metric"]), cfg.data.unit)
                ),
                "deployable": "true" if bool(row.get("deployable", False)) else "false",
                "artifact": str(artifact),
            }
        )
    _write_experiment_rows(run.path("experiments.csv"), prepared_rows)

    root_path = output_root / "experiments.csv"
    if root_path.is_file():
        with root_path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), None)
        if header is not None and "split" not in header:
            # Silently appending would leave old rows with an empty split and
            # make val/test indistinguishable, so refuse instead.
            raise ValueError(
                f"{root_path} was written by an older schema without a 'split' column; "
                "move it aside (or re-run the earlier evaluations) before appending"
            )
    existing = [
        row
        for row in _read_experiment_rows(root_path)
        if row.get("run_id") != run.run_dir.name
    ]
    existing.extend(prepared_rows)
    _write_experiment_rows(root_path, existing)


def _metric_document(
    cfg: Config,
    *,
    stage: str,
    methods: dict[str, Any],
    counts: dict[str, Any],
    best_metric: float | None,
    epochs_run: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "protocol_id": cfg.protocol_id,
        "method_id": cfg.method_id,
        "stage": stage,
        "metric_units": metric_units(cfg.data.unit),
        "counts": counts,
        "best_metric": best_metric,
        "epochs_run": epochs_run,
        "methods": methods,
    }
    for name in ("base", "single", "candidate", "router", "oracle_mae"):
        if name in methods:
            document[name] = methods[name]
    if extra:
        document.update(extra)
    return document


# ---------------------------------------------------------------------------
# prepare-data
# ---------------------------------------------------------------------------


def cmd_prepare_data(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    data_root, output_root = _roots(args, cfg)
    _set_determinism(cfg.seed)
    prepared, path = prepare_dataset(
        cfg,
        data_root,
        output_root,
        synthetic=_synthetic_flag(args),
    )
    print(f"prepared: {path}")
    print(f"dataset: {prepared.dataset} shape={prepared.data.shape} nodes={prepared.num_nodes}")
    print(f"graph edges: {prepared.graph.num_edges} hash={prepared.graph.graph_hash[:12]}")
    print(
        "windows: "
        + ", ".join(f"{name}={prepared.scheme.starts[name].size}" for name in ("train", "val", "test"))
    )
    return 0


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------


def cmd_train_base(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    data_root, output_root = _roots(args, cfg)
    _set_determinism(cfg.seed)
    prepared = load_or_prepare(cfg, data_root, output_root, synthetic=_synthetic_flag(args))
    device = _device(args, cfg)
    run, _, manifest = _make_run(
        args,
        cfg,
        prepared,
        output_root,
        stage="A",
        num_candidates=1,
        device=device,
        parent_checkpoint=None,
    )
    trainer = BaseTrainer(cfg, prepared, run, device, synthetic=bool(
        prepared.manifest.extra.get("synthetic", False)
    ))
    result = trainer.fit(
        resume=getattr(args, "resume", None),
        max_epochs=getattr(args, "max_epochs", None),
        patience_override=getattr(args, "patience", None),
    )
    if result.best_checkpoint is None:
        raise RuntimeError("base training produced no best checkpoint")
    payload = load_checkpoint(result.best_checkpoint, map_location=device)
    load_state_dict_strict(trainer.model, payload["state_dict"], "base backbone")
    trainer.model.set_mode("full")
    metrics = trainer._validate()
    document = _metric_document(
        cfg,
        stage="A",
        methods={"base": metrics},
        counts={
            "split": "val",
            "samples": metrics["counts"]["samples"],
            "valid_elements": metrics["counts"]["valid_elements"],
        },
        best_metric=result.best_metric,
        epochs_run=result.epochs_run,
        extra={"selection_metric": "raw_mae_all"},
    )
    run.write_metrics(document)
    write_json(run.path("metrics_val.json"), document)
    _update_manifest(
        run,
        manifest,
        status="completed",
        finished_at=_utc_now(),
        best_checkpoint=str(result.best_checkpoint),
        latest_checkpoint=str(result.latest_checkpoint) if result.latest_checkpoint else None,
        best_metric=result.best_metric,
        epochs_run=result.epochs_run,
        metrics_file="metrics.json",
    )
    _append_experiment_rows(
        run,
        output_root,
        cfg,
        prepared,
        [
            {
                "method": "sbona_base",
                "metric": "raw_mae_all",
                "value": metrics.get("raw_mae_all"),
                "split": "val",
                "deployable": True,
            }
        ],
    )
    print(f"run: {run.run_dir}")
    print(f"best checkpoint: {result.best_checkpoint}")
    print(f"val raw MAE: {metrics.get('raw_mae_all')}")
    return 0


def _base_checkpoint(
    args: argparse.Namespace,
    output_root: Path,
    prepared: PreparedData,
    cfg: Config,
) -> Path:
    return _find_run(
        output_root,
        stage="A",
        prepared=prepared,
        cfg=cfg,
        num_candidates=1,
        explicit_run=getattr(args, "base_run", None),
        explicit_checkpoint=getattr(args, "base_checkpoint", None),
        label="base",
    )


def _load_backbone_from_checkpoint(
    model, checkpoint: Path, device: torch.device, cfg: Config
) -> None:
    payload = load_checkpoint(checkpoint, map_location=device)
    if payload.get("stage") != "A":
        raise ValueError(f"expected a stage A base checkpoint, got {payload.get('stage')}")
    # An explicit --base-checkpoint must still come from this seed/protocol/
    # architecture: a foreign backbone would make the whole adapter stage
    # silently incomparable to its Base.
    stored = payload.get("hashes", {}).get("backbone_identity_hash")
    if stored != backbone_identity_hash(cfg):
        raise ValueError(
            f"base checkpoint {checkpoint} does not match the current seed/protocol/"
            "backbone architecture; pass the Base checkpoint trained for this config"
        )
    load_state_dict_strict(model.backbone, payload["state_dict"], "base backbone")
    model.backbone.set_mode(model.config.backbone.mode)


def _finish_candidate_run(
    args: argparse.Namespace,
    cfg: Config,
    prepared: PreparedData,
    output_root: Path,
    run: RunArtifacts,
    manifest: dict[str, Any],
    trainer: CandidateTrainer,
    result,
    *,
    stage: str,
    method: str,
    deployable: bool,
) -> None:
    if result.best_checkpoint is None:
        raise RuntimeError(f"stage {stage} produced no best checkpoint")
    payload = load_checkpoint(result.best_checkpoint, map_location=trainer.device)
    load_state_dict_strict(trainer.model, payload["state_dict"], f"stage {stage}")
    trainer.model.set_stage(stage)
    metrics = trainer._validate()
    selection = metrics.get("selection_metric")
    methods = {"candidate": metrics}
    if stage == "B":
        # K=1 Stage B is the matched single baseline; expose the same metric
        # block under an explicit name so it cannot be read as a Router result.
        methods["single"] = metrics
    document = _metric_document(
        cfg,
        stage=stage,
        methods=methods,
        counts={
            "split": "val",
            "samples": metrics["counts"]["samples"],
            "valid_elements": metrics["counts"]["valid_elements"],
            "num_candidates": int(trainer.model.num_candidates),
        },
        best_metric=result.best_metric,
        epochs_run=result.epochs_run,
        extra={
            "selection_metric": "candidate_oracle_mae_all" if cfg.oracle.alpha >= 1.0 else "candidate_best_cost_mean",
            "selection_value": selection,
        },
    )
    run.write_metrics(document)
    write_json(run.path("metrics_val.json"), document)
    _update_manifest(
        run,
        manifest,
        status="completed",
        finished_at=_utc_now(),
        best_checkpoint=str(result.best_checkpoint),
        latest_checkpoint=str(result.latest_checkpoint) if result.latest_checkpoint else None,
        best_metric=result.best_metric,
        epochs_run=result.epochs_run,
        metrics_file="metrics.json",
    )
    metric_key = metrics.get("candidate_oracle_mae_all", metrics.get("raw_mae_all"))
    _append_experiment_rows(
        run,
        output_root,
        cfg,
        prepared,
        [
            {
                "method": method,
                "metric": "raw_mae_all" if stage == "B" else "candidate_oracle_mae_all",
                "value": metric_key,
                "split": "val",
                "deployable": bool(deployable),
            }
        ],
    )
    print(f"run: {run.run_dir}")
    print(f"best checkpoint: {result.best_checkpoint}")
    print(f"val selection metric: {selection}")


def cmd_train_single(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    data_root, output_root = _roots(args, cfg)
    _set_determinism(cfg.seed)
    prepared = load_or_prepare(cfg, data_root, output_root, synthetic=_synthetic_flag(args))
    device = _device(args, cfg)
    base_checkpoint = _base_checkpoint(args, output_root, prepared, cfg)
    active_cfg = replace(
        cfg,
        adapter=replace(cfg.adapter, num_candidates=1),
        method_id=f"{cfg.method_id}_single",
    )
    active_cfg.validate()
    run, _, manifest = _make_run(
        args,
        active_cfg,
        prepared,
        output_root,
        stage="B",
        num_candidates=1,
        device=device,
        parent_checkpoint=base_checkpoint,
    )
    model = build_adapter(
        "single",
        active_cfg,
        num_nodes=prepared.num_nodes,
        steps_per_day=prepared.steps_per_day,
    )
    _load_backbone_from_checkpoint(model, base_checkpoint, device, cfg)
    trainer = CandidateTrainer(
        active_cfg, prepared, model, run, device, stage="B"
    )
    result = trainer.fit(
        resume=getattr(args, "resume", None),
        max_epochs=getattr(args, "max_epochs", None),
        patience_override=getattr(args, "patience", None),
        parent_checkpoint=base_checkpoint,
    )
    _finish_candidate_run(
        args,
        active_cfg,
        prepared,
        output_root,
        run,
        manifest,
        trainer,
        result,
        stage="B",
        method=active_cfg.method_id,
        deployable=True,
    )
    return 0


def cmd_train_candidates(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    data_root, output_root = _roots(args, cfg)
    _set_determinism(cfg.seed)
    prepared = load_or_prepare(cfg, data_root, output_root, synthetic=_synthetic_flag(args))
    device = _device(args, cfg)
    base_checkpoint = _base_checkpoint(args, output_root, prepared, cfg)
    kind = getattr(args, "kind", "structured")
    if kind == "single":
        raise ValueError("use the train-single command for the K=1 Stage-B baseline")
    # An ablation adapter must not share the main method's identity: otherwise a
    # run trained with ``--kind independent_heads`` and one trained with the
    # default kind are indistinguishable to ``_find_run`` and either could be
    # handed to the next stage.
    active_cfg = cfg
    if kind != "structured":
        active_cfg = replace(cfg, method_id=f"{cfg.method_id}_{kind}")
        active_cfg.validate()
    run, _, manifest = _make_run(
        args,
        active_cfg,
        prepared,
        output_root,
        stage="C",
        num_candidates=active_cfg.adapter.num_candidates,
        device=device,
        parent_checkpoint=base_checkpoint,
    )
    model = build_adapter(
        kind,
        active_cfg,
        num_nodes=prepared.num_nodes,
        steps_per_day=prepared.steps_per_day,
    )
    _load_backbone_from_checkpoint(model, base_checkpoint, device, cfg)
    trainer = CandidateTrainer(active_cfg, prepared, model, run, device, stage="C")
    result = trainer.fit(
        resume=getattr(args, "resume", None),
        max_epochs=getattr(args, "max_epochs", None),
        patience_override=getattr(args, "patience", None),
        parent_checkpoint=base_checkpoint,
    )
    _finish_candidate_run(
        args,
        active_cfg,
        prepared,
        output_root,
        run,
        manifest,
        trainer,
        result,
        stage="C",
        method=active_cfg.method_id,
        deployable=False,
    )
    return 0


def _candidate_checkpoint(
    args: argparse.Namespace,
    output_root: Path,
    prepared: PreparedData,
    cfg: Config,
) -> Path:
    return _find_run(
        output_root,
        stage="C",
        prepared=prepared,
        cfg=cfg,
        num_candidates=cfg.adapter.num_candidates,
        explicit_run=getattr(args, "candidate_run", None),
        explicit_checkpoint=getattr(args, "candidate_checkpoint", None),
        label="candidate",
    )


def cmd_train_router(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    data_root, output_root = _roots(args, cfg)
    _set_determinism(cfg.seed)
    prepared = load_or_prepare(cfg, data_root, output_root, synthetic=_synthetic_flag(args))
    device = _device(args, cfg)
    candidate_checkpoint = _candidate_checkpoint(args, output_root, prepared, cfg)
    payload = load_checkpoint(candidate_checkpoint, map_location=device)
    if payload.get("hashes", {}).get("num_candidates") != cfg.adapter.num_candidates:
        raise ValueError(
            "candidate checkpoint K does not match adapter.num_candidates: "
            f"{payload.get('hashes', {}).get('num_candidates')} != {cfg.adapter.num_candidates}"
        )
    # Same guard as the Base checkpoint: an explicit --candidate-checkpoint from
    # another seed/method/K would train a Router on weights that the deployed
    # pipeline can never produce.
    if (
        payload.get("hashes", {}).get("adapter_identity_hash")
        != adapter_identity_hash(cfg, cfg.adapter.num_candidates)
    ):
        raise ValueError(
            f"candidate checkpoint {candidate_checkpoint} does not match the current "
            "seed/protocol/method/K; pass the candidate run trained by this config"
        )
    if int(cfg.adapter.num_candidates) < 2:
        raise ValueError("Router training requires K >= 2")
    kind = payload.get("model_kind", "structured")
    model = build_adapter(
        kind,
        cfg,
        num_nodes=prepared.num_nodes,
        steps_per_day=prepared.steps_per_day,
    )
    load_state_dict_strict(model, payload["state_dict"], "candidate model")
    run, _, manifest = _make_run(
        args,
        cfg,
        prepared,
        output_root,
        stage="D",
        num_candidates=cfg.adapter.num_candidates,
        device=device,
        parent_checkpoint=candidate_checkpoint,
    )
    trainer = RouterTrainer(cfg, prepared, model, run, device)
    result = trainer.fit(
        resume=getattr(args, "resume", None),
        max_epochs=getattr(args, "max_epochs", None),
        patience_override=getattr(args, "patience", None),
    )
    if result.best_checkpoint is None:
        raise RuntimeError("router training produced no best checkpoint")
    best_payload = load_checkpoint(result.best_checkpoint, map_location=device)
    load_state_dict_strict(model, best_payload["state_dict"], "router model")
    model.set_stage("D")
    metrics = trainer._validate()
    document = _metric_document(
        cfg,
        stage="D",
        methods={
            "router": metrics["router"],
            "oracle_mae": metrics["oracle_mae"],
        },
        counts={
            "split": "val",
            "samples": metrics["router"]["counts"]["samples"],
            "valid_elements": metrics["router"]["counts"]["valid_elements"],
            "num_candidates": cfg.adapter.num_candidates,
        },
        best_metric=result.best_metric,
        epochs_run=result.epochs_run,
        extra={
            "selection_metric": "router_raw_mae_all",
            "router_regret_mean": metrics.get("router_regret_mean"),
        },
    )
    run.write_metrics(document)
    write_json(run.path("metrics_val.json"), document)
    _update_manifest(
        run,
        manifest,
        status="completed",
        finished_at=_utc_now(),
        best_checkpoint=str(result.best_checkpoint),
        latest_checkpoint=str(result.latest_checkpoint) if result.latest_checkpoint else None,
        best_metric=result.best_metric,
        epochs_run=result.epochs_run,
        metrics_file="metrics.json",
    )
    _append_experiment_rows(
        run,
        output_root,
        cfg,
        prepared,
        [
            {
                "method": f"{cfg.method_id}_router",
                "metric": "router_raw_mae_all",
                "value": metrics.get("router_raw_mae_all"),
                "split": "val",
                "deployable": True,
            },
            {
                "method": "oracle_mae_offline",
                "metric": "raw_mae_all",
                "value": metrics.get("oracle_mae_all"),
                "split": "val",
                "deployable": False,
            },
        ],
    )
    print(f"run: {run.run_dir}")
    print(f"best checkpoint: {result.best_checkpoint}")
    print(f"val Router@1 MAE: {metrics.get('router_raw_mae_all')}")
    print(f"val Oracle MAE: {metrics.get('oracle_mae_all')}")
    return 0


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def cmd_probe_route(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    data_root, output_root = _roots(args, cfg)
    _set_determinism(cfg.seed)
    prepared = load_or_prepare(cfg, data_root, output_root, synthetic=_synthetic_flag(args))
    device = _device(args, cfg)
    run, _, manifest = _make_run(
        args,
        cfg,
        prepared,
        output_root,
        stage="P",
        num_candidates=cfg.adapter.num_candidates,
        device=device,
        parent_checkpoint=None,
    )
    candidate_run = getattr(args, "candidate_run", None) or ""
    probe = run_route_probe(cfg, prepared, run, device, candidate_run=candidate_run)
    document = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "protocol_id": cfg.protocol_id,
        "method_id": cfg.method_id,
        "stage": "P",
        "metric_units": metric_units(cfg.data.unit),
        "counts": {
            "windows": probe.get("windows", {}),
            "num_candidates": cfg.adapter.num_candidates,
        },
        "probe": probe,
    }
    run.write_metrics(document)
    _update_manifest(
        run,
        manifest,
        status="completed",
        finished_at=_utc_now(),
        metrics_file="metrics.json",
        probe_results="probe_results.json",
    )
    print(f"run: {run.run_dir}")
    print(f"probe results: {run.path('probe_results.json')}")
    return 0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _checkpoint_config(cfg: Config | None, payload: dict[str, Any]) -> Config | None:
    stored = payload.get("resolved_config")
    if not isinstance(stored, dict):
        return cfg
    try:
        return Config.from_dict(stored)
    except ConfigError:
        return cfg


def _run_output_root(args: argparse.Namespace) -> Path:
    """Resolve the run root before a config has been loaded."""
    explicit = getattr(args, "output_root", None)
    if explicit is not None:
        return Path(explicit).expanduser()
    env = os.environ.get("SBONA_OUTPUT_ROOT")
    if env:
        return Path(env).expanduser()
    config_path = getattr(args, "config", None)
    if config_path is not None:
        cfg = load_config(config_path)
        return resolve_roots(cfg)[1]
    return Path("outputs")


def _check_hashes_against_data(
    payload: dict[str, Any], prepared: PreparedData, cfg: Config, num_candidates: int
) -> None:
    expected = make_hashes(prepared, cfg, num_candidates)
    validate_resume(payload, expected, keys=_selection_keys(str(payload.get("stage"))))


def _router_checkpoint_for(
    args: argparse.Namespace,
    output_root: Path,
    prepared: PreparedData,
    cfg: Config,
) -> Path | None:
    explicit = getattr(args, "router_checkpoint", None)
    if explicit is not None:
        return _resolve_checkpoint(explicit, None)
    expected = make_hashes(prepared, cfg, cfg.adapter.num_candidates)
    matches = [
        run_dir / "checkpoint_best.pt"
        for run_dir, manifest in _iter_run_manifests(output_root)
        if manifest.get("stage") == "D"
        and all(
            manifest.get("hashes", {}).get(key) == expected.get(key)
            for key in ADAPTER_SELECTION_KEYS
        )
        and (run_dir / "checkpoint_best.pt").is_file()
    ]
    if len(matches) > 1:
        names = ", ".join(sorted(path.parent.name for path in matches))
        raise ValueError(
            f"{len(matches)} stage D Router runs match the current seed/method/K: {names}; "
            "expected exactly one. Pass an explicit --router-checkpoint or remove the stale run(s)."
        )
    return matches[0] if matches else None


def cmd_evaluate(args: argparse.Namespace) -> int:
    # A config is optional here; the run's resolved config is preferred.
    run_dir = _resolve_run_dir(args.run, _run_output_root(args))
    checkpoint = _resolve_checkpoint(getattr(args, "checkpoint", None), run_dir)
    payload = load_checkpoint(checkpoint, map_location="cpu")
    if getattr(args, "config", None):
        cfg = _load_cfg(args)
    elif (run_dir / "resolved_config.yaml").is_file():
        cfg = load_config(run_dir / "resolved_config.yaml")
    else:
        cfg = _checkpoint_config(None, payload)
        if cfg is None:
            raise ConfigError("run has no resolved_config.yaml; pass --config")
    data_root, output_root = _roots(args, cfg)
    _set_determinism(cfg.seed)
    prepared = load_or_prepare(cfg, data_root, output_root, synthetic=_synthetic_flag(args))
    device = _device(args, cfg)
    split = getattr(args, "split", "test")
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}")

    stage = str(payload.get("stage"))
    num_candidates = int(payload.get("hashes", {}).get("num_candidates", cfg.adapter.num_candidates))
    _check_hashes_against_data(payload, prepared, cfg, num_candidates)

    # Evaluation never overwrites the training run's metrics.  It writes a
    # sibling ``<source-run>-eval-<split>`` directory so both artifacts survive.
    eval_run_id = getattr(args, "eval_run_id", None) or f"{run_dir.name}-eval-{split}"
    eval_run = RunArtifacts(output_root / eval_run_id)
    eval_hashes = make_hashes(prepared, cfg, num_candidates)
    eval_manifest = _manifest_base(
        cfg,
        prepared,
        stage=f"eval-{stage}",
        hashes=eval_hashes,
        device=device,
        num_candidates=num_candidates,
        parent_checkpoint=checkpoint,
    )
    eval_manifest["evaluation"] = {
        "source_run": str(run_dir),
        "source_stage": stage,
        "split": split,
        "checkpoint": str(checkpoint),
    }
    eval_run.write_resolved_config(cfg.to_yaml())
    eval_run.write_manifest(eval_manifest)
    args._run = eval_run

    protocol_lock = getattr(args, "protocol_lock", None)
    protocol_lock_info = None
    if protocol_lock:
        lock_path = Path(protocol_lock)
        if not lock_path.is_file():
            raise FileNotFoundError(f"protocol lock not found: {lock_path}")
        protocol_lock_info = {
            "path": str(lock_path),
            "sha256": sha256_file(lock_path),
        }

    if stage == "A":
        model = build_backbone(
            cfg.backbone,
            num_nodes=prepared.num_nodes,
            in_steps=cfg.data.input_steps,
            out_steps=cfg.data.output_steps,
            steps_per_day=prepared.steps_per_day,
        )
        load_state_dict_strict(model, payload["state_dict"], "base backbone")
        model.set_mode("full")
        model.to(device)
        result = evaluate_base_split(
            model,
            cfg,
            prepared,
            split,
            device,
            batch_size=getattr(args, "batch_size", None),
            run=eval_run,
            save_predictions=True,
        )
        result.document["protocol_lock"] = protocol_lock_info
        result.document["source_run"] = str(run_dir)
        result.document["checkpoint"] = str(checkpoint)
        result.document["checkpoint_stage"] = stage
        eval_run.write_metrics(result.document)
        write_json(eval_run.path(f"metrics_{split}.json"), result.document)
        _update_manifest(
            eval_run,
            eval_manifest,
            status="completed",
            finished_at=_utc_now(),
            metrics_file="metrics.json",
        )
        _append_experiment_rows(
            eval_run,
            output_root,
            cfg,
            prepared,
            [
                {
                    "method": "sbona_base",
                    "metric": "raw_mae_all",
                    "value": result.document["base"].get("raw_mae_all"),
                    "split": split,
                    "deployable": True,
                }
            ],
        )
        print(f"eval run: {eval_run.run_dir}")
        print(f"split: {split}")
        print(f"base raw MAE: {result.document['base'].get('raw_mae_all')}")
        print(f"metrics: {eval_run.path('metrics.json')}")
        return 0

    if stage not in ("B", "C", "D"):
        raise ValueError(f"cannot evaluate unsupported checkpoint stage {stage!r}")
    kind = str(payload.get("model_kind", "structured"))
    model = build_adapter(
        kind,
        cfg,
        num_nodes=prepared.num_nodes,
        steps_per_day=prepared.steps_per_day,
    )
    load_state_dict_strict(model, payload["state_dict"], f"stage {stage} model")

    router_available = stage == "D"
    if not router_available and int(model.num_candidates) > 1:
        router_checkpoint = _router_checkpoint_for(args, output_root, prepared, cfg)
        if router_checkpoint is None:
            raise FileNotFoundError(
                "this candidate checkpoint has K > 1 but no trained Router checkpoint was "
                "found; run train-router first or pass --router-checkpoint"
            )
        router_payload = load_checkpoint(router_checkpoint, map_location=device)
        if router_payload.get("stage") != "D":
            raise ValueError(
                f"router checkpoint must be stage D, got {router_payload.get('stage')}"
            )
        _check_hashes_against_data(
            router_payload,
            prepared,
            cfg,
            int(router_payload.get("hashes", {}).get("num_candidates", cfg.adapter.num_candidates)),
        )
        load_state_dict_strict(model, router_payload["state_dict"], "router model")
        router_available = True

    if stage == "B" and int(model.num_candidates) == 1:
        model.set_stage("B")
    else:
        model.set_stage("D" if router_available else "C")
    model.to(device)
    model.eval()
    result = evaluate_adapter_split(
        model,
        cfg,
        prepared,
        split,
        device,
        batch_size=getattr(args, "batch_size", None),
        run=eval_run,
        save_predictions=True,
        router_available=router_available,
        include_bootstrap=not bool(getattr(args, "no_bootstrap", False)),
    )
    if router_available and result.router_index is not None:
        eval_run.save_top1(
            split,
            result.router_index,
            result.router_logits if result.router_logits is not None else np.zeros((result.router_index.size, 1)),
            result.router_actions
            if result.router_actions is not None
            else np.zeros((result.router_index.size, 5)),
        )
    result.document["protocol_lock"] = protocol_lock_info
    result.document["checkpoint"] = str(checkpoint)
    result.document["checkpoint_stage"] = stage
    result.document["source_run"] = str(run_dir)
    eval_run.write_metrics(result.document)
    write_json(eval_run.path(f"metrics_{split}.json"), result.document)
    _update_manifest(
        eval_run,
        eval_manifest,
        status="completed",
        finished_at=_utc_now(),
        metrics_file="metrics.json",
    )

    rows: list[dict[str, Any]] = []
    if router_available:
        rows.append(
            {
                "method": f"{cfg.method_id}_router",
                "metric": "router_raw_mae_all",
                "value": result.document["router"].get("raw_mae_all"),
                "split": split,
                "deployable": True,
            }
        )
    if "single" in result.document:
        rows.append(
            {
                "method": f"{cfg.method_id}_single",
                "metric": "raw_mae_all",
                "value": result.document["single"].get("raw_mae_all"),
                "split": split,
                "deployable": True,
            }
        )
    rows.extend(
        [
            {
                "method": "sbona_base",
                "metric": "raw_mae_all",
                "value": result.document["base"].get("raw_mae_all"),
                "split": split,
                "deployable": True,
            },
            {
                "method": "oracle_mae_offline",
                "metric": "raw_mae_all",
                "value": result.document["oracle_mae"].get("raw_mae_all"),
                "split": split,
                "deployable": False,
            },
            {
                "method": "oracle_mse_offline",
                "metric": "raw_rmse_all",
                "value": result.document["oracle_mse"].get("raw_rmse_all"),
                "split": split,
                "deployable": False,
            },
            {
                "method": "oracle_cost_offline",
                "metric": "raw_mae_all",
                "value": result.document["oracle_cost"].get("raw_mae_all"),
                "split": split,
                "deployable": False,
            },
        ]
    )
    _append_experiment_rows(eval_run, output_root, cfg, prepared, rows)
    print(f"eval run: {eval_run.run_dir}")
    print(f"split: {split}")
    print(f"base raw MAE: {result.document['base'].get('raw_mae_all')}")
    if router_available:
        print(f"Router@1 raw MAE: {result.document['router'].get('raw_mae_all')}")
    if "single" in result.document:
        print(f"single raw MAE: {result.document['single'].get('raw_mae_all')}")
    print(f"Oracle MAE raw MAE: {result.document['oracle_mae'].get('raw_mae_all')}")
    retention_key = "single_vs_base" if "single" in result.document else "router_vs_base"
    retention_doc = result.document.get("retention", {}).get(retention_key, {})
    print(f"retention: {retention_doc.get('value')} ({retention_doc.get('reason')})")
    print(f"metrics: {eval_run.path('metrics.json')}")
    return 0


# ---------------------------------------------------------------------------
# Profiling and aggregation
# ---------------------------------------------------------------------------


def cmd_profile(args: argparse.Namespace) -> int:
    run_dir = _resolve_run_dir(args.run, _run_output_root(args))
    checkpoint = _resolve_checkpoint(getattr(args, "checkpoint", None), run_dir)
    payload = load_checkpoint(checkpoint, map_location="cpu")
    if getattr(args, "config", None):
        cfg = _load_cfg(args)
    elif (run_dir / "resolved_config.yaml").is_file():
        cfg = load_config(run_dir / "resolved_config.yaml")
    else:
        cfg = _checkpoint_config(None, payload)
        if cfg is None:
            raise ConfigError("run has no resolved_config.yaml; pass --config")
    data_root, output_root = _roots(args, cfg)
    prepared = load_or_prepare(cfg, data_root, output_root, synthetic=_synthetic_flag(args))
    device = _device(args, cfg)
    split = getattr(args, "split", "test")
    dataset = TrafficWindowDataset(prepared, split)
    if len(dataset) == 0:
        raise ValueError(f"split {split!r} has no windows to profile")
    graph = prepared.graph.to(device)

    def make_input(batch_size: int):
        indices = [index % len(dataset) for index in range(batch_size)]
        batch = collate_samples([dataset[index] for index in indices])
        return move_input(batch.model_input, device), graph

    stage = str(payload.get("stage"))
    if stage == "A":
        model = build_backbone(
            cfg.backbone,
            num_nodes=prepared.num_nodes,
            in_steps=cfg.data.input_steps,
            out_steps=cfg.data.output_steps,
            steps_per_day=prepared.steps_per_day,
        )
        load_state_dict_strict(model, payload["state_dict"], "base backbone")
        model.set_mode("full")
        model.to(device)
        model.eval()

        @torch.inference_mode()
        def run_base(batch_size: int):
            return model.encode_and_predict(*make_input(batch_size))

        measurements = profile_callable(
            run_base,
            batch_sizes=tuple(args.batch_sizes),
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
    else:
        kind = str(payload.get("model_kind", "structured"))
        model = build_adapter(
            kind,
            cfg,
            num_nodes=prepared.num_nodes,
            steps_per_day=prepared.steps_per_day,
        )
        load_state_dict_strict(model, payload["state_dict"], f"stage {stage} model")
        router_available = stage == "D"
        if stage in ("B", "C") and int(model.num_candidates) > 1:
            # Top-1 latency without a trained Router would profile an
            # untrained policy and cannot be reported as deployment latency.
            router_checkpoint = _router_checkpoint_for(args, output_root, prepared, cfg)
            if router_checkpoint is None:
                raise FileNotFoundError(
                    "profiling K > 1 Top-1 requires a trained Router checkpoint; "
                    "run train-router first or pass --router-checkpoint"
                )
            router_payload = load_checkpoint(router_checkpoint, map_location=device)
            if router_payload.get("stage") != "D":
                raise ValueError(
                    f"router checkpoint must be stage D, got {router_payload.get('stage')}"
                )
            _check_hashes_against_data(
                router_payload,
                prepared,
                cfg,
                int(router_payload.get("hashes", {}).get("num_candidates", cfg.adapter.num_candidates)),
            )
            load_state_dict_strict(model, router_payload["state_dict"], "router model")
            router_available = True
        if int(model.num_candidates) == 1:
            model.set_stage("B")
        else:
            model.set_stage("D" if router_available else "C")
        model.to(device)
        model.eval()
        measurements = profile_model(
            model,
            make_input,
            batch_sizes=tuple(args.batch_sizes),
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )

    document = {
        "schema_version": 1,
        "protocol_id": cfg.protocol_id,
        "run_id": run_dir.name,
        "checkpoint": str(checkpoint),
        "stage": stage,
        "dataset": prepared.dataset,
        "split": split,
        "precision": cfg.training.precision,
        "device": str(device),
        "acceptance_note": (
            "CPU timing is not a GPU acceptance result"
            if device.type != "cuda"
            else "measure on a dedicated GPU with no co-running jobs"
        ),
        "measurements": measurements,
    }
    RunArtifacts(run_dir).write_profile(document)
    print(f"profile: {run_dir / 'profile.json'}")
    print(f"device: {device}")
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    from sbona.evaluation.artifacts import aggregate_experiments

    manifest = Path(args.manifest)
    output = Path(args.output) if args.output else manifest.with_suffix(".aggregate.json")
    output_csv = (
        Path(args.csv)
        if args.csv
        else manifest.with_suffix(".aggregate.csv")
    )
    document = aggregate_experiments(manifest, output, output_csv, split=args.split)
    print(f"aggregate: {output}")
    print(
        f"split: {document['split']} seed: {document['seed']} "
        f"rows: {document['rows']} selected: {document['selected_rows']} "
        f"groups: {len(document['groups'])}"
    )
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser, *, config_required: bool = True) -> None:
    parser.add_argument("--config", type=Path, required=config_required)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--synthetic",
        action="store_const",
        const=True,
        default=None,
        help="force the labelled synthetic fixture (config may already enable it)",
    )
    parser.add_argument("--device", type=str, default=None)


def _add_training(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sbona", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-data", help="audit raw data and build prepared artifacts")
    _add_common(prepare)
    prepare.set_defaults(func=cmd_prepare_data)

    base = sub.add_parser("train-base", help="Stage A: train the backbone")
    _add_common(base)
    _add_training(base)
    base.set_defaults(func=cmd_train_base)

    single = sub.add_parser("train-single", help="Stage B: train the K=1 structured baseline")
    _add_common(single)
    _add_training(single)
    single.add_argument("--base-run", type=str, default=None)
    single.add_argument("--base-checkpoint", type=Path, default=None)
    single.set_defaults(func=cmd_train_single)

    candidates = sub.add_parser("train-candidates", help="Stage C: train K candidates")
    _add_common(candidates)
    _add_training(candidates)
    candidates.add_argument("--base-run", type=str, default=None)
    candidates.add_argument("--base-checkpoint", type=Path, default=None)
    candidates.add_argument(
        "--kind",
        choices=("structured", "unstructured", "independent_heads"),
        default="structured",
    )
    candidates.set_defaults(func=cmd_train_candidates)

    router = sub.add_parser("train-router", help="Stage D: train the Router on frozen candidates")
    _add_common(router)
    _add_training(router)
    router.add_argument("--candidate-run", type=str, default=None)
    router.add_argument("--candidate-checkpoint", type=Path, default=None)
    router.set_defaults(func=cmd_train_router)

    probe = sub.add_parser("probe-route", help="F/R/Q temporal route probe")
    _add_common(probe)
    _add_training(probe)
    probe.add_argument("--candidate-run", type=str, default=None)
    probe.set_defaults(func=cmd_probe_route)

    evaluate = sub.add_parser("evaluate", help="evaluate a frozen run checkpoint")
    _add_common(evaluate, config_required=False)
    evaluate.add_argument("--run", type=str, required=True)
    evaluate.add_argument("--split", choices=("train", "val", "test"), default="test")
    evaluate.add_argument("--checkpoint", type=Path, default=None)
    evaluate.add_argument("--router-checkpoint", type=Path, default=None)
    evaluate.add_argument(
        "--eval-run-id",
        type=str,
        default=None,
        help="sibling evaluation run id (default: <source-run>-eval-<split>)",
    )
    evaluate.add_argument("--protocol-lock", type=Path, default=None)
    evaluate.add_argument("--batch-size", type=int, default=None)
    evaluate.add_argument("--no-bootstrap", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    profile = sub.add_parser("profile", help="profile latency and memory for a run")
    _add_common(profile, config_required=False)
    profile.add_argument("--run", type=str, required=True)
    profile.add_argument("--split", choices=("train", "val", "test"), default="test")
    profile.add_argument("--checkpoint", type=Path, default=None)
    profile.add_argument("--router-checkpoint", type=Path, default=None)
    profile.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64])
    profile.add_argument("--warmup", type=int, default=100)
    profile.add_argument("--iterations", type=int, default=1000)
    profile.set_defaults(func=cmd_profile)

    aggregate = sub.add_parser("aggregate", help="build the single-seed comparison table")
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--split", choices=("train", "val", "test"), default="test")
    aggregate.add_argument("--output", type=Path, default=None)
    aggregate.add_argument("--csv", type=Path, default=None)
    aggregate.set_defaults(func=cmd_aggregate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except Exception as exc:  # noqa: BLE001 - CLI must turn failures into artifacts
        _write_failure(args, getattr(args, "command", "unknown"), exc)
        print(f"error: {exc}", file=sys.stderr)
        if os.environ.get("SBONA_DEBUG"):
            traceback.print_exc()
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
