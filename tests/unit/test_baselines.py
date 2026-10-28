"""MCL-WTA baseline tests (TimeMCL-style hard WTA transplant)."""

from __future__ import annotations

import pytest
import torch

from sbona.baselines import ADAPTER_KINDS, build_adapter
from sbona.baselines.mcl_wta import MCLWTA, ScoreHead
from sbona.config import Config
from sbona.contracts import GraphContext, ModelInput
from sbona.objectives.losses import wta_loss, wta_score_loss


def _cfg(num_candidates: int = 3) -> Config:
    return Config.from_dict(
        {
            "schema_version": 1,
            "protocol_id": "flow_12x12_v1",
            "method_id": "sbona_mcl_wta",
            "seed": 11,
            "data": {
                "dataset": "PEMS08",
                "data_file": "PEMS08/data.npz",
                "graph_file": "PEMS08/PEMS08.csv",
                "synthetic_num_nodes": 5,
                "synthetic_num_steps": 200,
            },
            "graph": {"topology": "published_binary_undirected"},
            "backbone": {"name": "staeformer", "mode": "frozen", "input_dim": 3},
            "adapter": {"num_candidates": num_candidates, "hidden_dim": 16, "residual_dim": 8},
        }
    )


def _model(num_candidates: int = 3) -> MCLWTA:
    torch.manual_seed(0)
    model = build_adapter("mcl_wta", _cfg(num_candidates), num_nodes=5, steps_per_day=288)
    assert isinstance(model, MCLWTA)
    return model


@pytest.fixture
def inp() -> ModelInput:
    torch.manual_seed(3)
    return ModelInput(
        history=torch.randn(2, 12, 5, 1),
        observed=torch.ones(2, 12, 5, 1, dtype=torch.bool),
        time_of_day=torch.rand(2, 12),
        day_of_week=torch.zeros(2, 12, dtype=torch.int64),
    )


def test_mcl_wta_is_a_registered_adapter_kind() -> None:
    assert "mcl_wta" in ADAPTER_KINDS


def test_mcl_wta_contract(inp: ModelInput, tiny_graph: GraphContext) -> None:
    model = _model(3)
    model.eval()
    out = model.forward_candidates(inp, tiny_graph)
    assert out.pred_z.shape == (2, 3, 12, 5, 1)
    assert out.candidate_logits is not None
    assert out.candidate_logits.shape == (2, 3)
    assert torch.isfinite(out.candidate_logits).all()
    assert model.loss_kind == "wta"
    assert model.uses_score_selector is True
    assert model.supports_separation is False
    # There is deliberately no Stage-D Router on this baseline.
    assert not hasattr(model, "router")


def test_mcl_wta_top1_follows_the_score_head(inp: ModelInput, tiny_graph: GraphContext) -> None:
    model = _model(4)
    model.eval()
    with torch.no_grad():
        out = model.forward_candidates(inp, tiny_graph)
        top1 = model.predict_top1(inp, tiny_graph)
    expected = out.candidate_logits.argmax(dim=-1)
    assert torch.equal(top1.index, expected)
    assert torch.allclose(top1.logits, out.candidate_logits)
    picked = out.pred_z.gather(
        1, top1.index[:, None, None, None, None].expand(-1, 1, 12, 5, 1)
    ).squeeze(1)
    assert torch.allclose(top1.pred_z, picked)


def test_mcl_wta_rejects_stage_d() -> None:
    model = _model(3)
    with pytest.raises(ValueError, match="stages B/C"):
        model.set_stage("D")


def test_mcl_wta_needs_at_least_two_heads() -> None:
    with pytest.raises(ValueError, match="K >= 2"):
        _model(1)


def test_hard_wta_trains_only_the_winning_head(inp: ModelInput, tiny_graph: GraphContext) -> None:
    model = _model(3)
    model.train()
    single = ModelInput(
        history=inp.history[:1],
        observed=inp.observed[:1],
        time_of_day=inp.time_of_day[:1],
        day_of_week=inp.day_of_week[:1],
    )
    out = model.forward_candidates(single, tiny_graph)
    error = out.pred_z.pow(2).mean(dim=(2, 3, 4))  # [1, K] z-space MSE to a zero target
    loss, count = wta_loss(error, torch.ones(1, dtype=torch.bool))
    assert count == 1
    loss.backward()

    winner = int(error.detach().argmin(dim=-1))
    for index, head in enumerate(model.heads):
        grads = [p.grad for p in head.parameters()]
        if index == winner:
            assert any(g is not None and g.abs().sum() > 0 for g in grads), (
                "the winning head must receive the WTA gradient"
            )
        else:
            # ``gather`` sends an all-zero gradient through the other heads, so
            # "hard" means zero (or absent) signal, never a real update.
            assert all(
                g is None or float(g.abs().sum()) == 0.0 for g in grads
            ), f"losing head {index} received a WTA gradient; routing is not hard"
    # The score head is trained by its own term, never by the WTA head loss.
    assert all(p.grad is None for p in model.score_head.parameters())


def test_score_head_is_trained_by_its_own_cross_entropy(inp: ModelInput, tiny_graph: GraphContext) -> None:
    model = _model(3)
    model.train()
    out = model.forward_candidates(inp, tiny_graph)
    error = out.pred_z.detach().pow(2).mean(dim=(2, 3, 4))
    assert out.candidate_logits is not None
    loss, count = wta_score_loss(out.candidate_logits, error, torch.ones(2, dtype=torch.bool))
    assert count == 2
    loss.backward()
    assert all(p.grad is not None for p in model.score_head.parameters())


def test_score_head_reads_global_state_and_residual_magnitude() -> None:
    head = ScoreHead(hidden_dim=8, width=16)
    h_global = torch.zeros(2, 8)
    small = torch.full((2, 3, 12, 5, 1), 0.01)
    large = torch.full((2, 3, 12, 5, 1), 10.0)
    with torch.no_grad():
        a = head(h_global, small)
        b = head(h_global, large)
    assert a.shape == b.shape == (2, 3)
    # The output layer is initialised near zero, so the score is not constant in
    # the residual summary: the summary reaches the output.
    assert not torch.allclose(a, b)
