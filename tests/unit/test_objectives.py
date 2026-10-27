from __future__ import annotations

import math

import torch

from sbona.contracts import OracleOutput, ScalerStats, TargetBatch
from sbona.evaluation.oracle import candidate_errors, gather_candidates, select_whole_sample
from sbona.objectives.constraints import target_mse_z
from sbona.objectives.losses import (
    candidate_loss,
    mean_over_valid,
    separation_loss,
)
from sbona.objectives.oracle import group_zscore, soft_weights


def test_target_mse_z_keeps_candidate_axis() -> None:
    stats = ScalerStats(
        mu=0.0,
        sigma=1.0,
        envelope=torch.zeros(2),
        envelope_fallback=torch.zeros(2, dtype=torch.bool),
    )
    target = TargetBatch(
        values_raw=torch.ones(2, 1, 2, 1),
        valid=torch.ones(2, 1, 2, 1, dtype=torch.bool),
        sample_ids=torch.tensor([0, 1], dtype=torch.int64),
    )
    for candidates in (1, 3):
        pred = torch.zeros(2, candidates, 1, 2, 1)
        error = target_mse_z(pred, target, stats)
        assert error.shape == (2, candidates)
        assert torch.allclose(error, torch.ones(2, candidates))


def test_group_zscore_is_population_and_k1_is_zero() -> None:
    values = torch.tensor([[1.0, 3.0], [2.0, 2.0]])
    z = group_zscore(values)
    assert torch.allclose(z[0], torch.tensor([-1.0, 1.0]))
    assert torch.allclose(z[1], torch.zeros(2))
    assert torch.allclose(group_zscore(torch.tensor([[5.0]])), torch.zeros(1, 1))


def test_soft_weights_are_uniform_for_invalid_samples() -> None:
    cost = torch.tensor([[0.0, 1.0], [100.0, -100.0]])
    valid = torch.tensor([True, False])
    weights = soft_weights(cost, 0.5, valid)
    assert torch.allclose(weights[1], torch.tensor([0.5, 0.5]))
    assert math.isclose(float(weights.sum(dim=-1)[0]), 1.0, abs_tol=1e-6)


def test_mean_over_valid_skips_empty_batches() -> None:
    value, count = mean_over_valid(torch.tensor([4.0, 8.0]), torch.tensor([False, False]))
    assert count == 0
    assert float(value) == 0.0
    value, count = mean_over_valid(torch.tensor([4.0, 8.0]), torch.tensor([True, False]))
    assert count == 1
    assert float(value) == 4.0


def _oracle_output() -> OracleOutput:
    error = torch.tensor([[1.0, 3.0]])
    consistency = torch.zeros_like(error)
    cost = torch.tensor([[0.0, 1.0]])
    soft = torch.tensor([[1.0, 0.0]])
    valid = torch.tensor([True])
    best = torch.tensor([0], dtype=torch.int64)
    return OracleOutput(error, consistency, cost, soft, best, valid)


def test_candidate_loss_matches_hand_computation() -> None:
    oracle = _oracle_output()
    loss, count = candidate_loss(oracle, epsilon=0.0, lambda_consistency=0.0)
    assert count == 1
    assert math.isclose(float(loss), 1.0)
    # epsilon=0.5, K=2: w_tilde=[0.75,0.25] -> 0.75*1 + 0.25*3 = 1.5
    loss, _ = candidate_loss(oracle, epsilon=0.5, lambda_consistency=0.0)
    assert math.isclose(float(loss), 1.5)


def test_separation_loss_is_zero_for_one_candidate() -> None:
    actions = torch.zeros(2, 1, 5)
    loss, count = separation_loss(actions, torch.tensor([True, True]), margin=0.25)
    assert count == 2
    assert float(loss) == 0.0


def test_whole_sample_oracle_does_not_mix_per_node_or_horizon() -> None:
    # Candidate 0 is best on the first half, candidate 1 on the second half.
    pred = torch.tensor([[[[[0.0]], [[0.0]]], [[[0.0]], [[10.0]]]]])
    target = torch.zeros(1, 2, 1, 1)
    from sbona.contracts import TargetBatch

    target_batch = TargetBatch(
        values_raw=target,
        valid=torch.ones_like(target, dtype=torch.bool),
        sample_ids=torch.zeros(1, dtype=torch.int64),
    )
    mae, mse, valid = candidate_errors(pred, target_batch)
    # candidate 0: mean(0,0)=0; candidate 1: mean(0,10)=5
    assert torch.allclose(mae, torch.tensor([[0.0, 5.0]]))
    index = select_whole_sample(mae, "oracle_mae", valid)
    assert index.tolist() == [0]
    selected = gather_candidates(pred, index)
    assert selected.shape == (1, 2, 1, 1)
    assert torch.equal(selected, pred[:, 0])
