from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sbona.baselines import build_adapter
from sbona.config import Config
from sbona.data.dataset import (
    TrafficWindowDataset,
    collate_samples,
    load_or_prepare,
    prepare_dataset,
)
from sbona.training.trainer import make_hashes, move_input


def _cfg() -> Config:
    return Config.from_dict(
        {
            "schema_version": 1,
            "protocol_id": "synthetic_test_v1",
            "method_id": "sbona_test",
            "seed": 3,
            "data": {
                "dataset": "synthetic",
                "input_steps": 12,
                "output_steps": 12,
                "split_ratios": [0.6, 0.2, 0.2],
                "steps_per_day": 24,
                "synthetic": True,
                "synthetic_num_nodes": 6,
                "synthetic_num_steps": 120,
                "synthetic_missing_fraction": 0.02,
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
            "adapter": {
                "num_candidates": 2,
                "hidden_dim": 8,
                "residual_dim": 4,
                "graph_hops": 1,
            },
            "evaluation": {"horizons": [3, 6, 12]},
        }
    )


@pytest.mark.slow
def test_synthetic_prepare_and_forward(tmp_path: Path) -> None:
    cfg = _cfg()
    prepared, artifact = prepare_dataset(
        cfg,
        tmp_path / "data",
        tmp_path / "outputs",
        synthetic=True,
    )
    assert artifact.is_file()
    assert prepared.num_nodes == 6
    assert prepared.data.shape == (120, 6)
    assert prepared.manifest.extra["synthetic"] is True
    assert "prepare_signature" in prepared.manifest.extra

    dataset = TrafficWindowDataset(prepared, "train")
    assert len(dataset) > 0
    batch = collate_samples([dataset[0], dataset[1]])
    assert batch.model_input.history.shape == (2, 12, 6, 1)
    assert batch.target.values_raw.shape == (2, 12, 6, 1)
    assert batch.model_input.day_of_week.shape == (2, 12)
    assert not bool(batch.model_input.day_of_week.any())

    model = build_adapter(
        "structured",
        cfg,
        num_nodes=prepared.num_nodes,
        steps_per_day=prepared.steps_per_day,
    )
    model.eval()
    graph = prepared.graph
    inp = move_input(batch.model_input, torch.device("cpu"))
    with torch.no_grad():
        candidates = model.forward_candidates(inp, graph)
        top1 = model.predict_top1(inp, graph)
    assert candidates.pred_z.shape == (2, 2, 12, 6, 1)
    assert top1.pred_z.shape == (2, 12, 6, 1)
    expected = candidates.pred_z[
        torch.arange(2), top1.index
    ]
    assert torch.allclose(top1.pred_z, expected, atol=1e-6)

    hashes = make_hashes(prepared, cfg, cfg.adapter.num_candidates)
    assert hashes["num_candidates"] == 2
    assert hashes["num_nodes"] == 6
    assert hashes["graph_hash"] == prepared.graph.graph_hash

    again = load_or_prepare(cfg, tmp_path / "data", tmp_path / "outputs", synthetic=True)
    assert again.manifest.extra["data_sha256"] == prepared.manifest.extra["data_sha256"]
