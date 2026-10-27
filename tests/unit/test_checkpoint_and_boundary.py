from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from sbona.models.adapter import SBoN
from sbona.training.checkpoint import (
    CheckpointError,
    checkpoint_sha256,
    load_checkpoint,
    save_checkpoint,
    upstream_lock_hash,
    validate_resume,
)


def test_checkpoint_roundtrip_and_atomic_hash(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    payload = {
        "stage": "A",
        "state_dict": model.state_dict(),
        "optimizer": {},
        "epoch": 3,
        "global_step": 42,
        "resolved_config": {"schema_version": 1},
        "hashes": {"data_hash": "abc", "num_candidates": 1},
        "upstream_lock_hash": upstream_lock_hash(),
    }
    path = save_checkpoint(tmp_path / "checkpoint.pt", payload)
    loaded = load_checkpoint(path)
    assert loaded["stage"] == "A"
    assert loaded["global_step"] == 42
    assert "rng" in loaded
    assert loaded["schema_version"] == 1
    assert checkpoint_sha256(path) == checkpoint_sha256(path)
    assert len(checkpoint_sha256(path)) == 64


def test_resume_hash_mismatch_is_rejected() -> None:
    payload = {"hashes": {"data_hash": "a", "num_nodes": 5}}
    validate_resume(payload, {"data_hash": "a", "num_nodes": 5})
    with pytest.raises(CheckpointError, match="resume mismatch"):
        validate_resume(payload, {"data_hash": "b", "num_nodes": 5})
    with pytest.raises(CheckpointError, match="no graph_hash"):
        validate_resume(payload, {"graph_hash": "x"})


def test_upstream_lock_hash_is_available_in_repo() -> None:
    digest = upstream_lock_hash()
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_predict_top1_has_no_target_or_kwargs() -> None:
    signature = inspect.signature(SBoN.predict_top1)
    assert list(signature.parameters) == ["self", "inp", "graph"]
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    signature = inspect.signature(SBoN.forward_candidates)
    assert list(signature.parameters) == ["self", "inp", "graph"]
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
