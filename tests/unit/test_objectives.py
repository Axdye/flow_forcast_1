from __future__ import annotations

import math

import pytest
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


def test_wta_loss_routes_gradient_to_the_winner_only() -> None:
    from sbona.objectives.losses import wta_loss

    # Sample 0: candidate 1 is best (1.0); sample 1: candidate 2 is best (0.5).
    error = torch.tensor([[3.0, 1.0, 2.0], [4.0, 5.0, 0.5]], requires_grad=True)
    loss, count = wta_loss(error, torch.tensor([True, True]))
    assert count == 2
    assert math.isclose(float(loss), (1.0 + 0.5) / 2)
    loss.backward()
    grad = error.grad
    assert grad is not None
    # Only the winner coordinate per sample carries gradient (1 / batch).
    assert torch.allclose(grad, torch.tensor([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]))


def test_wta_loss_ignores_invalid_samples() -> None:
    from sbona.objectives.losses import wta_loss

    error = torch.tensor([[1.0, 9.0], [0.0, 0.0]])
    loss, count = wta_loss(error, torch.tensor([True, False]))
    assert count == 1
    assert math.isclose(float(loss), 1.0)
    empty, count = wta_loss(error, torch.tensor([False, False]))
    assert count == 0
    assert float(empty) == 0.0


def test_wta_loss_rejects_bad_shapes() -> None:
    from sbona.objectives.losses import LossError, wta_loss

    with pytest.raises(LossError, match=r"\[B, K\]"):
        wta_loss(torch.zeros(2, 3, 1), torch.tensor([True, True]))


def test_wta_score_loss_is_cross_entropy_against_the_detached_winner() -> None:
    from sbona.objectives.losses import wta_score_loss

    logits = torch.zeros(2, 3, requires_grad=True)
    error = torch.tensor([[3.0, 1.0, 2.0], [4.0, 5.0, 0.5]])
    loss, count = wta_score_loss(logits, error, torch.tensor([True, True]))
    assert count == 2
    assert math.isclose(float(loss), math.log(3.0), rel_tol=1e-6)
    loss.backward()
    grad = logits.grad
    assert grad is not None
    # Uniform logits -> ``(softmax - onehot) / batch``; the sign proves the target
    # is the *winner* (index 1 for sample 0, index 2 for sample 1).
    expected = torch.tensor([[1 / 6, -1 / 3, 1 / 6], [1 / 6, 1 / 6, -1 / 3]])
    assert torch.allclose(grad, expected, atol=1e-6)


def test_wta_score_loss_shape_mismatch_is_rejected() -> None:
    from sbona.objectives.losses import LossError, wta_score_loss

    with pytest.raises(LossError, match="share"):
        wta_score_loss(torch.zeros(2, 2), torch.zeros(2, 3), torch.tensor([True, True]))


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
