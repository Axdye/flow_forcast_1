"""Flow-regime grouping.

There is no speed/density label in v1, so these groups describe *flow level*,
not congestion states.  Thresholds must come from training data.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from sbona.config import Config
from sbona.evaluation.metrics import MetricAccumulator


def flow_regime_thresholds(
    train_values: np.ndarray,
    quantiles: tuple[float, float] = (1 / 3, 2 / 3),
) -> tuple[float, float]:
    """Return the low/high cut points from valid training flow only."""
    values = np.asarray(train_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite training values for regime thresholds")
    return tuple(float(x) for x in np.quantile(values, quantiles, method="linear"))  # type: ignore[return-value]


def assign_flow_regime(values: Tensor, thresholds: tuple[float, float]) -> Tensor:
    """0 = low, 1 = middle, 2 = high according to training cut points."""
    low, high = thresholds
    return torch.where(values < low, 0, torch.where(values < high, 1, 2)).to(torch.int64)


class RegimeAccumulator:
    """Streaming per-flow-regime metric accumulators for one method."""

    def __init__(self, cfg: Config) -> None:
        self.thresholds: tuple[float, float] | None = None
        self._new = lambda: MetricAccumulator(
            cfg.data.mape_min_abs_target,
            cfg.evaluation.horizons,
        )
        self.groups = {0: self._new(), 1: self._new(), 2: self._new()}
        self.samples = {0: 0, 1: 0, 2: 0}

    def set_thresholds(self, thresholds: tuple[float, float]) -> None:
        self.thresholds = thresholds

    def update(self, pred_raw: Tensor, target) -> None:
        if self.thresholds is None:
            return
        mask = target.valid.bool()
        valid_counts = mask.sum(dim=(1, 2, 3))
        # A sample with no valid target element has no defined mean flow.  It
        # must not be silently grouped with the "low" regime.
        has_valid = valid_counts > 0
        count = valid_counts.clamp_min(1)
        target_mean = (target.values_raw * mask).sum(dim=(1, 2, 3)) / count
        regimes = assign_flow_regime(target_mean, self.thresholds)
        for group, accumulator in self.groups.items():
            select = (regimes == group) & has_valid
            if bool(select.any()):
                self.samples[group] += int(select.sum())
                accumulator.update(
                    pred_raw[select],
                    target.values_raw[select],
                    target.valid[select],
                )

    def result(self) -> dict[str, object]:
        if self.thresholds is None:
            return {
                "thresholds": None,
                "groups": {},
                "reason": "regime thresholds unavailable",
            }
        groups: dict[str, object] = {}
        for group, name in ((0, "low"), (1, "middle"), (2, "high")):
            if self.samples[group] == 0:
                groups[name] = {
                    "samples": 0,
                    "defined": False,
                    "reason": "no samples in this flow regime",
                }
                continue
            metrics = self.groups[group].result()
            metrics["samples"] = self.samples[group]
            metrics["defined"] = True
            groups[name] = metrics
        return {
            "thresholds": {"low": self.thresholds[0], "high": self.thresholds[1]},
            "regime_definition": "mean raw flow over valid target elements",
            "groups": groups,
        }


def regime_metrics(
    pred_raw: Tensor,
    target_raw: Tensor,
    valid: Tensor,
    thresholds: tuple[float, float],
    mape_min_abs_target: float = 1.0,
    horizons: tuple[int, ...] = (3, 6, 12),
) -> dict[str, object]:
    """Metrics grouped by the mean target flow of the whole prediction window.

    One sample belongs to exactly one regime to keep the comparison honest.
    """
    if pred_raw.dim() != 4:
        raise ValueError("pred_raw must be [B, T, N, 1]")
    mask = valid.bool()
    valid_counts = mask.sum(dim=(1, 2, 3))
    has_valid = valid_counts > 0
    target_mean = (target_raw * mask).sum(dim=(1, 2, 3)) / valid_counts.clamp_min(1)
    regimes = assign_flow_regime(target_mean, thresholds)
    result: dict[str, object] = {
        "thresholds": {"low": thresholds[0], "high": thresholds[1]},
        "regime_definition": "mean raw flow over valid target elements; low/mid/high",
        "groups": {},
    }
    for group, name in ((0, "low"), (1, "middle"), (2, "high")):
        # Samples without any valid target element have no regime and are
        # excluded instead of being counted as "low".
        select = (regimes == group) & has_valid
        if not bool(select.any()):
            result["groups"][name] = {
                "samples": 0,
                "defined": False,
                "reason": "no samples in this flow regime",
            }
            continue
        accumulator = MetricAccumulator(mape_min_abs_target, horizons)
        accumulator.update(
            pred_raw[select],
            target_raw[select],
            valid[select],
        )
        metrics = accumulator.result()
        metrics["samples"] = int(select.sum())
        result["groups"][name] = metrics
    return result


__all__ = [
    "flow_regime_thresholds",
    "assign_flow_regime",
    "RegimeAccumulator",
    "regime_metrics",
]
