from __future__ import annotations

import torch

from sbona.config import ActionRangesConfig
from sbona.contracts import GraphContext
from sbona.models.actions import ActionGenerator
from sbona.models.decoder import StructuredDecoder


def test_actions_respect_physical_ranges() -> None:
    generator = ActionGenerator(
        hidden_dim=8,
        num_candidates=4,
        output_steps=12,
        ranges=ActionRangesConfig(),
    )
    out = generator(torch.randn(3, 8))
    assert out.physical.shape == (3, 4, 5)
    assert out.normalized.shape == (3, 4, 5)
    assert ((out.physical[..., 0] >= 0) & (out.physical[..., 0] <= 11)).all()
    assert ((out.physical[..., 1] >= 0.25) & (out.physical[..., 1] <= 3.0)).all()
    assert ((out.physical[..., 2] >= -3.0) & (out.physical[..., 2] <= 3.0)).all()
    assert ((out.physical[..., 3] >= 1.0) & (out.physical[..., 3] <= 12.0)).all()
    assert ((out.physical[..., 4] >= 0.01) & (out.physical[..., 4] <= 1.0)).all()
    assert ((out.normalized >= -1e-6) & (out.normalized <= 1 + 1e-6)).all()


def test_t1_action_o_is_zero_and_anchors_are_not_all_zero() -> None:
    generator = ActionGenerator(hidden_dim=8, num_candidates=1, output_steps=1)
    out = generator(torch.randn(2, 8))
    assert torch.allclose(out.physical[..., 0], torch.zeros(2, 1))
    assert torch.allclose(out.normalized[..., 0], torch.zeros(2, 1))
    assert not torch.allclose(generator.anchors, torch.zeros_like(generator.anchors))


def test_m_zero_gives_exactly_zero_residual(tiny_graph: GraphContext) -> None:
    decoder = StructuredDecoder(hidden_dim=8, residual_dim=4, graph_hops=2, output_steps=6)
    h_node = torch.randn(2, tiny_graph.num_nodes, 8)
    actions = torch.tensor(
        [
            [1.0, 1.0, 0.0, 2.0, 0.5],
            [2.0, 0.5, 0.0, 1.0, 0.2],
        ]
    )[:, None, :]
    residual = decoder.forward_all(h_node, actions, tiny_graph)
    assert residual.shape == (2, 1, 6, tiny_graph.num_nodes, 1)
    assert torch.equal(residual, torch.zeros_like(residual))


def test_selected_decode_matches_all_candidate_decode(tiny_graph: GraphContext) -> None:
    torch.manual_seed(0)
    decoder = StructuredDecoder(hidden_dim=8, residual_dim=4, graph_hops=2, output_steps=6)
    h_node = torch.randn(2, tiny_graph.num_nodes, 8)
    actions = torch.randn(2, 3, 5)
    actions[..., 0] = actions[..., 0].sigmoid() * 5
    actions[..., 1] = 0.25 + 2.75 * actions[..., 1].sigmoid()
    actions[..., 2] = 3.0 * actions[..., 2].tanh()
    actions[..., 3] = 1.0 + 5.0 * actions[..., 3].sigmoid()
    actions[..., 4] = 0.01 + 0.99 * actions[..., 4].sigmoid()

    all_residual = decoder.forward_all(h_node, actions, tiny_graph)
    selected = actions[torch.arange(2), torch.tensor([0, 2])]
    selected_residual = decoder.forward_selected(h_node, selected, tiny_graph)

    assert torch.allclose(
        selected_residual,
        all_residual[torch.arange(2), torch.tensor([0, 2])],
    )


def test_selected_decode_never_calls_forward_all(tiny_graph: GraphContext, monkeypatch) -> None:
    decoder = StructuredDecoder(hidden_dim=8, residual_dim=4, graph_hops=2, output_steps=6)
    h_node = torch.randn(1, tiny_graph.num_nodes, 8)
    selected = torch.tensor([[1.0, 1.0, 0.5, 2.0, 0.5]])

    def fail(*args, **kwargs):
        raise AssertionError("forward_all must not be used by the single-decode path")

    monkeypatch.setattr(decoder, "forward_all", fail)
    residual = decoder.forward_selected(h_node, selected, tiny_graph)
    assert residual.shape == (1, 6, tiny_graph.num_nodes, 1)
