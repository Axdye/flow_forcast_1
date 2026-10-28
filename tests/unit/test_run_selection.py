"""Run discovery must never hand out another seed's / another method's weights."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from sbona.cli import _append_experiment_rows, _find_run, _router_checkpoint_for
from sbona.config import Config
from sbona.data.dataset import prepare_dataset
from sbona.evaluation.artifacts import RunArtifacts
from sbona.training.trainer import (
    adapter_identity_hash,
    backbone_identity_hash,
    make_hashes,
)

BASE = {
    "schema_version": 1,
    "protocol_id": "selection_test_v1",
    "method_id": "sbona",
    "seed": 11,
    "data": {
        "dataset": "synthetic",
        "input_steps": 12,
        "output_steps": 12,
        "split_ratios": [0.6, 0.2, 0.2],
        "steps_per_day": 24,
        "synthetic": True,
        "synthetic_num_nodes": 5,
        "synthetic_num_steps": 240,
        "synthetic_missing_fraction": 0.0,
        "day_of_week": False,
    },
    "graph": {"topology": "published_binary_undirected"},
    "backbone": {
        "name": "staeformer",
        "mode": "frozen",
        "input_embedding_dim": 8,
        "tod_embedding_dim": 8,
        "adaptive_embedding_dim": 8,
        "feed_forward_dim": 16,
        "num_heads": 4,
        "num_layers": 1,
        "dropout": 0.0,
    },
    "adapter": {"num_candidates": 8, "hidden_dim": 8, "residual_dim": 4, "graph_hops": 1},
    "training": {
        "precision": "fp32",
        "initial_batch_size": 4,
        "eval_batch_size": 4,
        "max_epochs": {"base": 1, "single": 1, "candidates": 1, "router": 1, "probe": 1},
        "patience": {"base": 1, "single": 1, "candidates": 1, "router": 1, "probe": 1},
        "num_workers": 0,
        "device": "cpu",
    },
    "evaluation": {"horizons": [3, 6, 12], "bootstrap_replicates": 20, "block_origins": 12},
}


def _cfg(**overrides: object) -> Config:
    raw = json.loads(json.dumps(BASE))
    raw.update(overrides)
    return Config.from_dict(raw)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("selection")
    prepared, _ = prepare_dataset(_cfg(), root / "data", root / "outputs")
    return prepared


@pytest.fixture()
def output_root(tmp_path: Path) -> Path:
    """Fresh output root per test so runs from other tests cannot leak in."""
    path = tmp_path / "outputs"
    path.mkdir()
    return path


def _write_run(
    output_root: Path,
    name: str,
    prepared,
    *,
    seed: int,
    method_id: str,
    stage: str,
    num_candidates: int,
) -> Path:
    run = output_root / name
    run.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(seed=seed, method_id=method_id)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "stage": stage,
        "method_id": method_id,
        "adapter_kind": method_id,
        "seed": seed,
        "config_hash": cfg.hash,
        "hashes": make_hashes(prepared, cfg, num_candidates),
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run / "checkpoint_best.pt").write_bytes(f"weights-of-{name}".encode())
    return run


def _age(output_root: Path, run: Path, seconds: int) -> None:
    """Make ``run`` look either newer or older than its siblings."""
    stamp = time.time() + seconds
    os.utime(run / "manifest.json", (stamp, stamp))


def test_find_run_matches_only_the_requested_seed(prepared, output_root) -> None:
    mine = _write_run(
        output_root, "A-seed11", prepared, seed=11, method_id="sbona_base", stage="A", num_candidates=1
    )
    other = _write_run(
        output_root, "A-seed22", prepared, seed=22, method_id="sbona_base", stage="A", num_candidates=1
    )
    _age(output_root, other, 100)  # newer than mine: the old mtime-first rule would win

    found = _find_run(
        output_root, stage="A", prepared=prepared, cfg=_cfg(seed=11, method_id="sbona_base"), num_candidates=1, label="base"
    )
    assert found == mine / "checkpoint_best.pt"

    found = _find_run(
        output_root, stage="A", prepared=prepared, cfg=_cfg(seed=22, method_id="sbona_base"), num_candidates=1, label="base"
    )
    assert found == other / "checkpoint_best.pt"


def test_find_run_fails_when_only_another_seed_exists(prepared, output_root) -> None:
    _write_run(
        output_root, "A-only-seed22", prepared, seed=22, method_id="sbona_base", stage="A", num_candidates=1
    )
    with pytest.raises(FileNotFoundError, match="matching the current seed/method"):
        _find_run(
            output_root,
            stage="A",
            prepared=prepared,
            cfg=_cfg(seed=33, method_id="sbona_base"),
            num_candidates=1,
            label="base",
        )


def test_find_run_distinguishes_ablation_methods(prepared, output_root) -> None:
    # ``train-candidates --kind independent_heads`` suffixes the method_id, so an
    # ablation run and the main structured run never share an adapter identity.
    structured = _write_run(
        output_root,
        "C-structured",
        prepared,
        seed=11,
        method_id="sbona_structured",
        stage="C",
        num_candidates=8,
    )
    independent = _write_run(
        output_root,
        "C-independent",
        prepared,
        seed=11,
        method_id="sbona_structured_independent_heads",
        stage="C",
        num_candidates=8,
    )
    _age(output_root, independent, 100)

    found = _find_run(
        output_root,
        stage="C",
        prepared=prepared,
        cfg=_cfg(seed=11, method_id="sbona_structured"),
        num_candidates=8,
        label="candidate",
    )
    assert found == structured / "checkpoint_best.pt"


def test_find_run_refuses_to_guess_between_two_matching_runs(prepared, output_root) -> None:
    for name in ("C-dup-a", "C-dup-b"):
        _write_run(
            output_root,
            name,
            prepared,
            seed=11,
            method_id="sbona_structured",
            stage="C",
            num_candidates=8,
        )
    with pytest.raises(ValueError, match="expected exactly one"):
        _find_run(
            output_root,
            stage="C",
            prepared=prepared,
            cfg=_cfg(seed=11, method_id="sbona_structured"),
            num_candidates=8,
            label="candidate",
        )


def test_router_checkpoint_is_seed_specific(prepared, output_root) -> None:
    # ``train-router`` is driven by the candidate config (method_id
    # "sbona_structured" for configs/experiments/pems08_staeformer_k8.yaml), so
    # the stage D run inherits that method_id rather than inventing "sbona_router".
    mine = _write_run(
        output_root,
        "D-seed11",
        prepared,
        seed=11,
        method_id="sbona_structured",
        stage="D",
        num_candidates=8,
    )
    other = _write_run(
        output_root,
        "D-seed22",
        prepared,
        seed=22,
        method_id="sbona_structured",
        stage="D",
        num_candidates=8,
    )
    _age(output_root, other, 100)

    class Args:
        router_checkpoint = None

    assert _router_checkpoint_for(
        Args(), output_root, prepared, _cfg(seed=11, method_id="sbona_structured")
    ) == (mine / "checkpoint_best.pt")
    assert _router_checkpoint_for(
        Args(), output_root, prepared, _cfg(seed=22, method_id="sbona_structured")
    ) == (other / "checkpoint_best.pt")
    assert (
        _router_checkpoint_for(
            Args(), output_root, prepared, _cfg(seed=33, method_id="sbona_structured")
        )
        is None
    )


def test_identity_hashes_ignore_tuning_knobs_but_not_seed_method_or_k() -> None:
    backbone = backbone_identity_hash(_cfg())
    adapter = adapter_identity_hash(_cfg(), 8)
    assert backbone_identity_hash(_cfg(seed=22)) != backbone
    assert adapter_identity_hash(_cfg(seed=22), 8) != adapter
    assert adapter_identity_hash(_cfg(), 4) != adapter
    assert adapter_identity_hash(_cfg(method_id="sbona_probe"), 8) != adapter
    # K and method must not matter for backbone reuse (Stage A is method-agnostic).
    assert backbone_identity_hash(_cfg(method_id="sbona_probe")) == backbone

    retuned = json.loads(json.dumps(BASE))
    retuned["training"]["initial_batch_size"] = 256
    retuned["training"]["max_epochs"]["base"] = 100
    retuned["training"]["device"] = "cuda"
    retuned_cfg = Config.from_dict(retuned)
    assert backbone_identity_hash(retuned_cfg) == backbone
    assert adapter_identity_hash(retuned_cfg, 8) == adapter


def test_appending_to_a_legacy_experiments_csv_is_refused(prepared, tmp_path: Path) -> None:
    """An old root table has no ``split`` column; appending would blank it out."""
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    legacy = output_root / "experiments.csv"
    legacy.write_text(
        "run_id,method,dataset,backbone,seed,protocol_id,data_hash,metric,value,units,"
        "deployable,artifact\n"
        "old-run,sbona_base,PEMS08,staeformer,11,flow_12x12_v1,abc,raw_mae_all,1.0,u,true,x\n",
        encoding="utf-8",
    )
    run = RunArtifacts(output_root / "new-run")
    run.path("metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="older schema without a 'split' column"):
        _append_experiment_rows(
            run,
            output_root,
            _cfg(),
            prepared,
            [
                {
                    "method": "sbona_base",
                    "metric": "raw_mae_all",
                    "value": 1.0,
                    "split": "val",
                    "deployable": True,
                }
            ],
        )


def test_explicit_foreign_base_checkpoint_is_rejected(prepared, tmp_path: Path) -> None:
    """Explicit paths must not bypass the seed/architecture check."""
    import torch

    from sbona.baselines import build_adapter
    from sbona.cli import _load_backbone_from_checkpoint
    from sbona.contracts import SCHEMA_VERSION

    cfg = _cfg(seed=11, method_id="sbona_base")
    foreign = _cfg(seed=22, method_id="sbona_base")
    model = build_adapter(
        "single", cfg, num_nodes=prepared.num_nodes, steps_per_day=prepared.steps_per_day
    )
    path = tmp_path / "foreign-base.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "A",
            "state_dict": model.backbone.state_dict(),
            "epoch": 0,
            "resolved_config": foreign.to_dict(),
            "hashes": make_hashes(prepared, foreign, 1),
        },
        path,
    )
    with pytest.raises(ValueError, match="does not match the current seed"):
        _load_backbone_from_checkpoint(model, path, torch.device("cpu"), cfg)
