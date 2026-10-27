"""Run artifacts, environment manifests and experiment aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sbona.evaluation.statistics import aggregate_seeds

METRICS_SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record), allow_nan=False) + "\n")


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo_root, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    dirty = status is not None and status != ""
    diff = run("diff") or ""
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "git_dirty_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if dirty else None,
    }


def _gpu_driver_version() -> str | list[str] | None:
    """Read the NVIDIA driver version without making CUDA a hard requirement."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    versions = sorted({line.strip() for line in output.splitlines() if line.strip()})
    if not versions:
        return None
    return versions[0] if len(versions) == 1 else versions


def collect_environment(repo_root: str | Path | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root or Path.cwd())
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if cuda_available
        else [],
        "gpu_driver": _gpu_driver_version(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if cuda_available else None,
        **_git_state(repo_root),
    }


def create_run_id(cfg_hash: str, prefix: str = "run") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{cfg_hash[:8]}"


@dataclass
class RunArtifacts:
    run_dir: Path
    schema_version: int = METRICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def write_resolved_config(self, text: str) -> None:
        self.path("resolved_config.yaml").write_text(text, encoding="utf-8")

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        write_json(self.path("manifest.json"), manifest)

    def log_train(self, record: dict[str, Any]) -> None:
        append_jsonl(self.path("train.jsonl"), record)

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        write_json(self.path("metrics.json"), metrics)

    def write_failure(self, failure: dict[str, Any]) -> None:
        write_json(self.path("failure.json"), failure)

    def write_profile(self, profile: dict[str, Any]) -> None:
        write_json(self.path("profile.json"), profile)

    def save_predictions(
        self,
        split: str,
        pred_raw: np.ndarray,
        valid: np.ndarray,
        sample_ids: np.ndarray,
        shard: int = 0,
    ) -> Path:
        if pred_raw.ndim != 4:
            raise ValueError("pred_raw must be [S, T, N, 1]")
        if valid.shape != pred_raw.shape:
            raise ValueError("valid shape must match predictions")
        path = self.path(f"predictions_{split}_{shard:04d}.npz")
        np.savez_compressed(
            path,
            pred_raw=pred_raw.astype(np.float32),
            valid=valid.astype(bool),
            sample_ids=sample_ids.astype(np.int64),
        )
        return path

    def save_top1(self, split: str, index: np.ndarray, logits: np.ndarray, actions: np.ndarray) -> None:
        np.savez_compressed(
            self.path(f"top1_{split}.npz"),
            index=index.astype(np.int64),
            logits=logits.astype(np.float32),
            selected_action=actions.astype(np.float32),
        )


def validate_metrics_document(document: dict[str, Any]) -> None:
    required = {"schema_version", "protocol_id", "metric_units", "counts"}
    missing = required - set(document)
    if missing:
        raise ValueError(f"metrics document missing keys: {sorted(missing)}")
    if document["schema_version"] != METRICS_SCHEMA_VERSION:
        raise ValueError(f"unsupported metrics schema {document['schema_version']}")
    counts = document["counts"]
    if not isinstance(counts, dict):
        raise ValueError("metrics.counts must be an object")


def aggregate_experiments(
    manifest_csv: str | Path,
    output_json: str | Path,
    output_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Read experiments.csv, validate provenance and aggregate by seed.

    Required columns: ``run_id, method, dataset, backbone, seed, protocol_id,
    data_hash, metric, value, units, deployable, artifact``.  Oracle rows that
    claim ``deployable=true`` are rejected.
    """
    manifest_csv = Path(manifest_csv)
    required = {
        "run_id",
        "method",
        "dataset",
        "backbone",
        "seed",
        "protocol_id",
        "data_hash",
        "metric",
        "value",
        "units",
        "deployable",
        "artifact",
    }
    rows: list[dict[str, Any]] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("experiments.csv has no header")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"experiments.csv missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            method = row["method"].strip()
            deployable = str(row["deployable"]).strip().lower() in ("1", "true", "yes")
            if "oracle" in method.lower() and deployable:
                raise ValueError(
                    f"line {line_number}: oracle method {method!r} cannot be marked deployable"
                )
            artifact = Path(row["artifact"])
            if not artifact.exists():
                raise ValueError(f"line {line_number}: missing artifact {artifact}")
            rows.append(
                {
                    **row,
                    "seed": int(row["seed"]),
                    "value": float(row["value"]),
                    "deployable": deployable,
                    "artifact": str(artifact),
                }
            )

    protocol_ids = {row["protocol_id"] for row in rows}
    if len(protocol_ids) > 1:
        raise ValueError(f"mixed protocol ids: {sorted(protocol_ids)}")
    metric_units: dict[str, set[str]] = {}
    for row in rows:
        metric_units.setdefault(row["metric"], set()).add(row["units"])
    mixed_units = {metric: sorted(units) for metric, units in metric_units.items() if len(units) > 1}
    if mixed_units:
        raise ValueError(f"metric(s) reported with inconsistent units: {mixed_units}")
    dataset_hashes: dict[str, set[str]] = {}
    for row in rows:
        dataset_hashes.setdefault(row["dataset"], set()).add(row["data_hash"])
    inconsistent = {k: v for k, v in dataset_hashes.items() if len(v) > 1}
    if inconsistent:
        raise ValueError(f"inconsistent data hashes: {inconsistent}")

    groups: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        key = (row["dataset"], row["backbone"], row["method"], row["metric"])
        groups.setdefault(key, []).append(row["value"])
    aggregated = []
    for (dataset, backbone, method, metric), values in sorted(groups.items()):
        stats = aggregate_seeds(values)
        aggregated.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "method": method,
                "metric": metric,
                "seeds": stats["n"],
                "mean": stats["mean"],
                "std": stats["std"],
            }
        )

    document = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "protocol_id": next(iter(protocol_ids)) if protocol_ids else None,
        "rows": len(rows),
        "groups": aggregated,
        "warnings": [
            "Only deployable rows belong in the main comparison table.",
            "Oracle rows are offline diagnostics and are not deployable.",
        ],
    }
    write_json(output_json, document)
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["dataset", "backbone", "method", "metric", "seeds", "mean", "std"],
            )
            writer.writeheader()
            writer.writerows(aggregated)
    return document


__all__ = [
    "METRICS_SCHEMA_VERSION",
    "write_json",
    "read_json",
    "append_jsonl",
    "collect_environment",
    "create_run_id",
    "RunArtifacts",
    "validate_metrics_document",
    "aggregate_experiments",
]
