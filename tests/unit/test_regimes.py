from __future__ import annotations

from types import SimpleNamespace

import torch

from sbona.contracts import TargetBatch
from sbona.evaluation.regimes import RegimeAccumulator, regime_metrics


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(mape_min_abs_target=1.0),
        evaluation=SimpleNamespace(horizons=(1,)),
    )


def _target(values: torch.Tensor, valid: torch.Tensor) -> TargetBatch:
    return TargetBatch(
        values_raw=values,
        valid=valid,
        sample_ids=torch.arange(values.shape[0], dtype=torch.int64),
    )


def test_invalid_sample_is_not_forced_into_low_regime() -> None:
    values = torch.tensor([[[[1.0]]], [[[0.0]]]])
    valid = torch.tensor([[[[True]]], [[[False]]]])
    target = _target(values, valid)
    accumulator = RegimeAccumulator(_cfg())
    accumulator.set_thresholds((5.0, 10.0))
    accumulator.update(torch.ones_like(values), target)
    result = accumulator.result()
    assert result["groups"]["low"]["samples"] == 1
    assert result["groups"]["middle"]["samples"] == 0
    assert result["groups"]["high"]["samples"] == 0
    assert result["groups"]["middle"]["defined"] is False


def test_regime_metrics_excludes_all_invalid_samples() -> None:
    values = torch.tensor([[[[1.0]]], [[[100.0]]], [[[0.0]]]])
    valid = torch.tensor([[[[True]]], [[[True]]], [[[False]]]])
    result = regime_metrics(
        torch.ones_like(values),
        values,
        valid,
        thresholds=(5.0, 10.0),
        mape_min_abs_target=1.0,
        horizons=(1,),
    )
    assert result["groups"]["low"]["samples"] == 1
    assert result["groups"]["high"]["samples"] == 1
    assert result["groups"]["middle"]["samples"] == 0
