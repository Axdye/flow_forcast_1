from __future__ import annotations

import math

import torch

from sbona.evaluation.metrics import MetricAccumulator


def _accumulator() -> MetricAccumulator:
    return MetricAccumulator(mape_min_abs_target=1.0, horizons=(1, 2))


def test_masked_mae_and_rmse_are_not_batch_averages() -> None:
    # errors = [0, 2, 4, 2] -> MAE = 2, MSE = 6, RMSE = sqrt(6)
    pred = torch.tensor([[[[10.0]], [[12.0]], [[14.0]], [[12.0]]]])
    target = torch.full_like(pred, 10.0)
    valid = torch.tensor([[[[True]], [[True]], [[True]], [[False]]]])
    # Last element invalid: active errors are [0,2,4] -> MAE 2, MSE 20/3.
    acc = _accumulator()
    acc.update(pred, target, valid)
    result = acc.result()
    assert result["counts"]["valid_elements"] == 3
    assert math.isclose(result["raw_mae_all"], 2.0)
    assert math.isclose(result["raw_rmse_all"], math.sqrt(20.0 / 3.0))

    valid_all = torch.ones_like(valid)
    acc = _accumulator()
    acc.update(pred, target, valid_all)
    result = acc.result()
    assert math.isclose(result["raw_mae_all"], 2.0)
    assert math.isclose(result["raw_rmse_all"], math.sqrt(6.0))


def test_mape_uses_raw_target_threshold_and_percent_units() -> None:
    pred = torch.tensor([[[[6.0]], [[8.0]]]])
    target = torch.tensor([[[[4.0]], [[4.0]]]])
    valid = torch.ones_like(pred, dtype=torch.bool)
    acc = _accumulator()
    acc.update(pred, target, valid)
    result = acc.result()
    # |6-4|/4=50%, |8-4|/4=100% -> mean 75%.
    assert math.isclose(result["raw_mape_all"], 75.0)
    assert result["counts"]["mape_elements"] == 2


def test_undefined_metric_is_none_not_zero() -> None:
    pred = torch.ones(1, 2, 1, 1)
    target = torch.ones(1, 2, 1, 1)
    valid = torch.zeros(1, 2, 1, 1, dtype=torch.bool)
    acc = _accumulator()
    acc.update(pred, target, valid)
    result = acc.result()
    assert result["raw_mae_all"] is None
    assert result["raw_rmse_all"] is None
    assert result["raw_mape_all"] is None


def test_nvr_and_euvr_magnitudes() -> None:
    pred = torch.tensor([[[[-2.0]], [[3.0]]]])
    target = torch.tensor([[[[0.0]], [[0.0]]]])
    valid = torch.ones_like(pred, dtype=torch.bool)
    envelope = torch.tensor([2.5])
    acc = _accumulator()
    acc.update(pred, target, valid, envelope=envelope)
    result = acc.result()
    assert result["raw_nvr"] == 0.5  # one of two valid elements is negative
    assert result["raw_euvr"] == 0.5  # 3.0 > 2.5
    assert result["raw_nvr_magnitude_mean"] == 1.0  # /2 valid elements
    assert result["raw_euvr_magnitude_mean"] == 0.25  # 0.5 / 2 valid elements


def test_horizon_metrics_are_single_steps() -> None:
    pred = torch.tensor([[[[1.0]], [[9.0]]]])
    target = torch.zeros_like(pred)
    valid = torch.ones_like(pred, dtype=torch.bool)
    acc = MetricAccumulator(1.0, (2,))
    acc.update(pred, target, valid)
    result = acc.result()
    assert math.isclose(result["per_horizon"]["2"]["raw_mae"], 9.0)
    assert result["per_horizon"]["2"]["count"] == 1
