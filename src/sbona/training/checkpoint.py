"""Atomic checkpoints and strict resume validation."""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sbona.contracts import SCHEMA_VERSION


class CheckpointError(RuntimeError):
    """Raised for incompatible or corrupted checkpoints."""


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write to a temporary file in the same directory, then atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "rng" not in payload:
        payload = {**payload, "rng": _rng_state()}
    payload.setdefault("schema_version", SCHEMA_VERSION)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=path.parent, prefix=path.name + ".tmp-"
    )
    temporary = Path(handle.name)
    try:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, path)
    finally:
        if not handle.closed:
            handle.close()
        if temporary.exists():
            temporary.unlink()
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint payload must be a dict")
    version = int(payload.get("schema_version", -1))
    if version != SCHEMA_VERSION:
        raise CheckpointError(f"unsupported checkpoint schema {version}")
    required = {"stage", "state_dict", "epoch", "resolved_config", "hashes"}
    missing = required - set(payload)
    if missing:
        raise CheckpointError(f"checkpoint missing keys: {sorted(missing)}")
    return payload


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def upstream_lock_hash(repo_root: str | Path | None = None) -> str | None:
    """Hash the local third-party source lock/registry.

    The contract requires every checkpoint to record an upstream lock hash, so
    a run can be tied back to the reviewed source manifest.  Returns ``None``
    only when the repository has no lock files at all (for example when the
    package is installed without its source tree); it never guesses a version.
    """
    root = Path(repo_root).expanduser() if repo_root is not None else _repo_root()
    candidates = (
        root / "third_party" / "source_manifest.json",
        root / "third_party" / "LOCK.md",
    )
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return None
    digest = hashlib.sha256()
    for path in existing:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_resume(
    payload: dict[str, Any],
    expected: dict[str, Any],
    keys: tuple[str, ...] = (
        "config_hash",
        "data_hash",
        "valid_hash",
        "graph_hash",
        "split_hash",
        "scaler_hash",
        "num_nodes",
        "num_candidates",
    ),
) -> None:
    """Reject resume when K/N/graph/split/scaler/architecture changed."""
    stored = payload.get("hashes", {})
    for key in keys:
        if key not in expected:
            continue
        if key not in stored:
            raise CheckpointError(f"checkpoint has no {key}; refusing resume")
        if stored[key] != expected[key]:
            raise CheckpointError(
                f"resume mismatch for {key}: checkpoint={stored[key]!r}, current={expected[key]!r}"
            )


def load_state_dict_strict(module: torch.nn.Module, state_dict: dict[str, Any], name: str) -> None:
    try:
        module.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise CheckpointError(f"failed to load {name} state_dict: {exc}") from exc


__all__ = [
    "CheckpointError",
    "save_checkpoint",
    "load_checkpoint",
    "checkpoint_sha256",
    "upstream_lock_hash",
    "validate_resume",
    "restore_rng_state",
    "load_state_dict_strict",
]
