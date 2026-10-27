"""Streaming metric aggregation in raw units.

MAE/RMSE are computed from total sums and total valid counts, never as an
average of batch RMSEs.  Undefined metrics are returned as ``None`` with a
reason string; they are never replaced by 0 or NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class MetricSums:
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    count: int = 0
    mape_sum: float = 0.0
    mape_count: int = 0
    negative_count: int = 0
    negative_magnitude: float = 0.0
    envelope_count: int = 0
    envelope_magnitude: float = 0.0
    samples: int = 0
    total_elements: int = 0
    per_horizon: dict[int, dict[str, float]] = field(default_factory=dict)
    per_sample: list[dict[str, float]] = field(default_factory=list)

    def merge(self, other: MetricSums) -> MetricSums:
        self.abs_sum += other.abs_sum
        self.sq_sum += other.sq_sum
        self.count += other.count
        self.mape_sum += other.mape_sum
        self.mape_count += other.mape_count
        self.negative_count += other.negative_count
        self.negative_magnitude += other.negative_magnitude
        self.envelope_count += other.envelope_count
        self.envelope_magnitude += other.envelope_magnitude
        self.samples += other.samples
        self.total_elements += other.total_elements
        for horizon, values in other.per_horizon.items():
            target = self.per_horizon.setdefault(
                horizon, {"abs_sum": 0.0, "sq_sum": 0.0, "count": 0, "mape_sum": 0.0, "mape_count": 0}
            )
            for key, value in values.items():
                target[key] += value
        self.per_sample.extend(other.per_sample)
        return self


class MetricAccumulator:
    def __init__(
        self,
        mape_min_abs_target: float = 1.0,
        horizons: tuple[int, ...] = (3, 6, 12),
        collect_per_sample: bool = False,
    ) -> None:
        self.mape_min_abs_target = mape_min_abs_target
        self.horizons = tuple(horizons)
        self.collect_per_sample = collect_per_sample
        self.sums = MetricSums()

    def update(
        self,
        pred_raw: Tensor,
        target_raw: Tensor,
        valid: Tensor,
        envelope: Tensor | None = None,
        sample_ids: Tensor | None = None,
    ) -> None:
        """Accumulate one batch.

        ``pred_raw``/``target_raw``: ``[B, T, N, 1]``; ``valid`` bool.
        ``envelope``: optional ``[N]`` raw per-node cap for EUVR.
        """
        if pred_raw.shape != target_raw.shape or pred_raw.shape != valid.shape:
            raise ValueError("pred/target/valid shapes must match")
        if pred_raw.dim() != 4:
            raise ValueError("pred/target must be [B, T, N, 1]")
        pred = pred_raw.detach().float()
        target = target_raw.detach().float()
        mask = valid.detach().bool()
        error = (pred - target).abs()
        sq = (pred - target) ** 2
        per_sample_abs = (error * mask).sum(dim=(1, 2, 3))
        per_sample_sq = (sq * mask).sum(dim=(1, 2, 3))
        per_sample_count = mask.sum(dim=(1, 2, 3))
        self.sums.abs_sum += float(per_sample_abs.sum())
        self.sums.sq_sum += float(per_sample_sq.sum())
        self.sums.count += int(per_sample_count.sum())
        self.sums.samples += int(pred.shape[0])
        self.sums.total_elements += int(mask.numel())

        mape_ok = mask & (target.abs() >= self.mape_min_abs_target)
        mape_values = (error / target.abs().clamp_min(1e-12)) * 100.0
        mape_sum = (mape_values * mape_ok).sum()
        mape_count = mape_ok.sum()
        self.sums.mape_sum += float(mape_sum)
        self.sums.mape_count += int(mape_count)

        negative = mask & (pred < 0)
        self.sums.negative_count += int(negative.sum())
        self.sums.negative_magnitude += float((torch.relu(-pred) * mask).sum())

        if envelope is not None:
            cap = envelope.detach().float().to(pred.device)[None, None, :, None]
            violation = mask & (pred > cap)
            self.sums.envelope_count += int(violation.sum())
            self.sums.envelope_magnitude += float(
                (torch.relu(pred - cap) * mask).sum()
            )

        for horizon in self.horizons:
            if horizon > pred.shape[1]:
                continue
            m = mask[:, horizon - 1]
            e = error[:, horizon - 1]
            s = sq[:, horizon - 1]
            t = target[:, horizon - 1]
            entry = self.sums.per_horizon.setdefault(
                horizon, {"abs_sum": 0.0, "sq_sum": 0.0, "count": 0, "mape_sum": 0.0, "mape_count": 0}
            )
            entry["abs_sum"] += float((e * m).sum())
            entry["sq_sum"] += float((s * m).sum())
            entry["count"] += int(m.sum())
            ok = m & (t.abs() >= self.mape_min_abs_target)
            entry["mape_sum"] += float(((e / t.abs().clamp_min(1e-12)) * 100.0 * ok).sum())
            entry["mape_count"] += int(ok.sum())

        if self.collect_per_sample and sample_ids is not None:
            if sample_ids.shape != (pred.shape[0],):
                raise ValueError("sample_ids must be [B]")
            for index in range(pred.shape[0]):
                self.sums.per_sample.append(
                    {
                        "sample_id": float(sample_ids[index].item()),
                        "abs_sum": float(per_sample_abs[index].item()),
                        "sq_sum": float(per_sample_sq[index].item()),
                        "count": float(per_sample_count[index].item()),
                    }
                )

    def merge(self, other: MetricAccumulator) -> MetricAccumulator:
        if self.mape_min_abs_target != other.mape_min_abs_target:
            raise ValueError("cannot merge accumulators with different MAPE thresholds")
        self.sums.merge(other.sums)
        return self

    def result(self, prefix: str = "raw") -> dict[str, object]:
        return metric_result(self.sums, prefix=prefix, horizons=self.horizons)

    def reset(self) -> None:
        self.sums = MetricSums()


def _metric(value_sum: float, count: int, reason: str) -> float | None:
    if count <= 0:
        return None
    return value_sum / count


def metric_result(
    sums: MetricSums,
    prefix: str = "raw",
    horizons: tuple[int, ...] = (3, 6, 12),
) -> dict[str, object]:
    mae = _metric(sums.abs_sum, sums.count, "no valid target elements")
    mse = _metric(sums.sq_sum, sums.count, "no valid target elements")
    rmse = None if mse is None else float(mse**0.5)
    mape = _metric(sums.mape_sum, sums.mape_count, "no valid target with |y| >= threshold")
    nvr = _metric(float(sums.negative_count), sums.count, "no valid target elements")
    nvr_mag = _metric(sums.negative_magnitude, sums.count, "no valid target elements")
    nvr_cond = _metric(
        sums.negative_magnitude, sums.negative_count, "no negative predictions"
    )
    envr = _metric(float(sums.envelope_count), sums.count, "no valid target elements")
    envr_mag = _metric(sums.envelope_magnitude, sums.count, "no valid target elements")
    envr_cond = _metric(
        sums.envelope_magnitude, sums.envelope_count, "no envelope violations"
    )

    undefined: dict[str, str] = {}
    values = {
        f"{prefix}_mae_all": mae,
        f"{prefix}_rmse_all": rmse,
        f"{prefix}_mape_all": mape,
        f"{prefix}_nvr": nvr,
        f"{prefix}_nvr_magnitude_mean": nvr_mag,
        f"{prefix}_nvr_magnitude_conditional": nvr_cond,
        f"{prefix}_euvr": envr,
        f"{prefix}_euvr_magnitude_mean": envr_mag,
        f"{prefix}_euvr_magnitude_conditional": envr_cond,
    }
    for key, value in values.items():
        if value is None:
            undefined[key] = "undefined metric: empty support"

    per_horizon: dict[str, dict[str, object]] = {}
    for horizon in horizons:
        entry = sums.per_horizon.get(horizon)
        if entry is None or entry["count"] == 0:
            per_horizon[str(horizon)] = {
                f"{prefix}_mae": None,
                f"{prefix}_rmse": None,
                f"{prefix}_mape": None,
                "count": 0,
            }
            continue
        h_mae = entry["abs_sum"] / entry["count"]
        h_rmse = (entry["sq_sum"] / entry["count"]) ** 0.5
        h_mape = (
            entry["mape_sum"] / entry["mape_count"] if entry["mape_count"] > 0 else None
        )
        per_horizon[str(horizon)] = {
            f"{prefix}_mae": h_mae,
            f"{prefix}_rmse": h_rmse,
            f"{prefix}_mape": h_mape,
            "count": entry["count"],
        }

    return {
        **values,
        "per_horizon": per_horizon,
        "counts": {
            "samples": sums.samples,
            "total_elements": sums.total_elements,
            "valid_elements": sums.count,
            "mape_elements": sums.mape_count,
            "negative_predictions": sums.negative_count,
            "envelope_violations": sums.envelope_count,
            "coverage": (sums.count / sums.total_elements) if sums.total_elements else None,
        },
        "undefined": undefined,
    }


def per_sample_abs_error(pred_raw: Tensor, target_raw: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(sum_abs [B], count [B])`` over valid elements."""
    if pred_raw.shape != target_raw.shape or pred_raw.shape != valid.shape:
        raise ValueError("pred/target/valid shapes must match")
    error = (pred_raw.detach().float() - target_raw.detach().float()).abs()
    mask = valid.bool()
    return (error * mask).sum(dim=(1, 2, 3)), mask.sum(dim=(1, 2, 3))


def per_sample_sq_error(pred_raw: Tensor, target_raw: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    error = (pred_raw.detach().float() - target_raw.detach().float()) ** 2
    mask = valid.bool()
    return (error * mask).sum(dim=(1, 2, 3)), mask.sum(dim=(1, 2, 3))


__all__ = [
    "MetricSums",
    "MetricAccumulator",
    "metric_result",
    "per_sample_abs_error",
    "per_sample_sq_error",
]
