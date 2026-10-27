"""Shared fixtures for contract-level tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sbona.contracts import GraphContext, ModelInput, TargetBatch
from sbona.data.graph import build_graph_context


@pytest.fixture
def tiny_graph() -> GraphContext:
    edges = np.array([[0, 1], [1, 0], [1, 2], [2, 3]], dtype=np.int64)
    return build_graph_context(5, edges)


@pytest.fixture
def model_input() -> ModelInput:
    torch.manual_seed(0)
    batch, length, nodes = 2, 12, 5
    return ModelInput(
        history=torch.randn(batch, length, nodes, 1),
        observed=torch.ones(batch, length, nodes, 1, dtype=torch.bool),
        time_of_day=torch.rand(batch, length),
        day_of_week=torch.zeros(batch, length, dtype=torch.int64),
    )


@pytest.fixture
def target_batch() -> TargetBatch:
    torch.manual_seed(1)
    batch, horizon, nodes = 2, 12, 5
    return TargetBatch(
        values_raw=torch.rand(batch, horizon, nodes, 1) * 100.0,
        valid=torch.ones(batch, horizon, nodes, 1, dtype=torch.bool),
        sample_ids=torch.arange(batch, dtype=torch.int64),
    )
