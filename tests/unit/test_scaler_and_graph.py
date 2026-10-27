from __future__ import annotations

import numpy as np
import pytest
import torch

from sbona.contracts import GraphContext
from sbona.data.graph import build_graph_context, multi_hop_basis, propagate
from sbona.data.scaler import fill_invalid_with_mean, fit_scaler


def test_scaler_uses_training_interval_only() -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0], [1000.0, 2000.0]], dtype=np.float32)
    valid = np.ones_like(data, dtype=bool)
    stats = fit_scaler(data, valid, train_end=2, envelope_min_count=10)
    assert stats.mu == pytest.approx(2.5)
    assert stats.sigma == pytest.approx(np.std([1, 2, 3, 4], ddof=0))
    # Too few per-node observations -> global fallback is recorded.
    assert bool(stats.envelope_fallback.all())
    assert stats.envelope.tolist() == pytest.approx([stats.envelope[0].item()] * 2)


def test_scaler_ignores_invalid_entries_and_keeps_fill_at_zero() -> None:
    data = np.array([[1.0], [3.0]], dtype=np.float32)
    valid = np.array([[True], [False]])
    stats = fit_scaler(data, valid, train_end=2, envelope_min_count=1)
    assert stats.mu == pytest.approx(1.0)
    filled = fill_invalid_with_mean(data, valid, stats.mu)
    assert filled[1, 0] == pytest.approx(stats.mu)
    assert float(stats.standardize(torch.tensor(filled[1, 0]))) == pytest.approx(0.0)


def test_scaler_rejects_empty_training_interval() -> None:
    data = np.array([[1.0], [2.0]], dtype=np.float32)
    valid = np.zeros_like(data, dtype=bool)
    with pytest.raises(ValueError):
        fit_scaler(data, valid, train_end=1)


def _dense(graph: GraphContext) -> torch.Tensor:
    return graph.diffusion.to_dense()


def test_graph_is_undirected_deduplicated_and_row_normalized() -> None:
    edges = np.array([[0, 1], [1, 0], [1, 2], [2, 1], [1, 1]], dtype=np.int64)
    graph = build_graph_context(4, edges)
    # (0,1) and (1,2) once each; self-loop input is dropped from constraints.
    assert graph.edge_index.tolist() == [[0, 1], [1, 2]]
    dense = _dense(graph)
    assert torch.allclose(dense.sum(dim=1), torch.ones(4), atol=1e-6)
    # Undirected topology: both directions exist before row normalization.
    assert dense[0, 1] > 0 and dense[1, 0] > 0
    # Isolated node 3 keeps its self-loop.
    assert dense[3, 3] == pytest.approx(1.0)


def test_identity_graph_is_not_a_published_graph() -> None:
    graph = build_graph_context(3, np.zeros((0, 2), dtype=np.int64), topology="identity")
    assert torch.allclose(_dense(graph), torch.eye(3))


def test_sparse_propagate_matches_dense_matmul(tiny_graph: GraphContext) -> None:
    features = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3)
    sparse_out = propagate(tiny_graph.diffusion, features)
    dense_out = tiny_graph.diffusion.to_dense() @ features
    assert torch.allclose(sparse_out, dense_out, atol=1e-6)
    basis = multi_hop_basis(tiny_graph.diffusion, features, hops=2)
    assert basis.shape == (3, 5, 3)
    assert torch.allclose(basis[0], features)
    assert torch.allclose(basis[1], dense_out)
