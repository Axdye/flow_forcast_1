"""Offline metric-specific oracle selection.

Selection is always one whole ``T x N`` trajectory per sample.  Different
criteria produce different oracle names and must not be mixed on one curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from sbona.evaluation.metrics import MetricAccumulator


@dataclass
class OracleSelection:
    index: Tensor  # [B]
    pred_raw: Tensor  # [B, T, N, 1]
    sample_mae: Tensor  # [B]
    sample_mse: Tensor  # [B]
    valid_sample: Tensor  # [B]
    criterion: str


def candidate_errors(
    pred_raw: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(mae [B,K], mse [B,K], count [B])`` over valid target elements.

    The target's own valid mask is used once and broadcast across candidates.
    """
    if pred_raw.dim() != 5:
        raise ValueError("pred_raw must be [B, K, T, N, 1]")
    if target.values_raw.shape != pred_raw.shape[:1] + pred_raw.shape[2:]:
        raise ValueError("target shape must equal pred without the K axis")
    batch_size, num_candidates = pred_raw.shape[0], pred_raw.shape[1]
    target_expanded = target.values_raw[:, None, :, :, :].expand_as(pred_raw)
    valid = target.valid[:, None, :, :, :].expand_as(pred_raw)
    error = (pred_raw.detach().float() - target_expanded.detach().float()).abs()
    sq = error**2
    count = valid.sum(dim=(2, 3, 4))
    mae = (error * valid).sum(dim=(2, 3, 4)) / count.clamp_min(1)
    mse = (sq * valid).sum(dim=(2, 3, 4)) / count.clamp_min(1)
    valid_sample = count > 0
    if batch_size != mae.shape[0] or num_candidates != mae.shape[1]:
        raise RuntimeError("internal candidate error shape mismatch")
    return mae, mse, valid_sample.any(dim=-1)


def select_whole_sample(errors: Tensor, criterion: str, valid_sample: Tensor | None = None) -> Tensor:
    """Argmin over K for every sample; invalid samples select index 0."""
    if errors.dim() != 2:
        raise ValueError("errors must be [B, K]")
    index = torch.argmin(errors, dim=-1)
    if valid_sample is not None:
        index = torch.where(valid_sample, index, torch.zeros_like(index))
    return index


def gather_candidates(pred_raw: Tensor, index: Tensor) -> Tensor:
    """Select one complete candidate per sample: ``[B,K,T,N,1] -> [B,T,N,1]``."""
    if pred_raw.dim() != 5:
        raise ValueError("pred_raw must be [B, K, T, N, 1]")
    if index.shape != (pred_raw.shape[0],):
        raise ValueError("index must be [B]")
    return pred_raw[torch.arange(pred_raw.shape[0], device=pred_raw.device), index]


def oracle_mae(
    pred_raw: Tensor,
    target,
    mape_min_abs_target: float = 1.0,
    horizons: tuple[int, ...] = (3, 6, 12),
) -> OracleSelection:
    mae, mse, valid_sample = candidate_errors(pred_raw, target)
    index = select_whole_sample(mae, "oracle_mae", valid_sample)
    selected = gather_candidates(pred_raw, index)
    return OracleSelection(
        index=index,
        pred_raw=selected,
        sample_mae=mae.gather(1, index[:, None]).squeeze(1),
        sample_mse=mse.gather(1, index[:, None]).squeeze(1),
        valid_sample=valid_sample,
        criterion="oracle_mae",
    )


def oracle_mse(
    pred_raw: Tensor,
    target,
) -> OracleSelection:
    mae, mse, valid_sample = candidate_errors(pred_raw, target)
    index = select_whole_sample(mse, "oracle_mse", valid_sample)
    selected = gather_candidates(pred_raw, index)
    return OracleSelection(
        index=index,
        pred_raw=selected,
        sample_mae=mae.gather(1, index[:, None]).squeeze(1),
        sample_mse=mse.gather(1, index[:, None]).squeeze(1),
        valid_sample=valid_sample,
        criterion="oracle_mse",
    )


def oracle_cost(pred_raw: Tensor, cost: Tensor, valid_sample: Tensor, target) -> OracleSelection:
    if cost.dim() != 2 or cost.shape != pred_raw.shape[:2]:
        raise ValueError("cost must be [B, K] matching pred_raw")
    index = select_whole_sample(cost.detach(), "oracle_cost", valid_sample)
    selected = gather_candidates(pred_raw, index)
    mae, mse, _ = candidate_errors(pred_raw, target)
    return OracleSelection(
        index=index,
        pred_raw=selected,
        sample_mae=mae.gather(1, index[:, None]).squeeze(1),
        sample_mse=mse.gather(1, index[:, None]).squeeze(1),
        valid_sample=valid_sample,
        criterion="oracle_cost",
    )


def oracle_metrics(
    selection: OracleSelection,
    target,
    mape_min_abs_target: float = 1.0,
    horizons: tuple[int, ...] = (3, 6, 12),
) -> dict[str, object]:
    accumulator = MetricAccumulator(mape_min_abs_target, horizons, collect_per_sample=True)
    accumulator.update(
        selection.pred_raw,
        target.values_raw,
        target.valid,
        sample_ids=target.sample_ids,
    )
    return accumulator.result()


def retention(base_mae: float | None, router_mae: float | None, oracle_mae_value: float | None) -> dict[str, object]:
    """``(base-router)/(base-oracle)`` with an explicit undefined branch."""
    if base_mae is None or router_mae is None or oracle_mae_value is None:
        return {
            "value": None,
            "numerator": None,
            "denominator": None,
            "reason": "missing metric",
        }
    numerator = base_mae - router_mae
    denominator = base_mae - oracle_mae_value
    if denominator <= 1e-6:
        return {
            "value": None,
            "numerator": numerator,
            "denominator": denominator,
            "reason": "oracle gain denominator <= 1e-6 raw units",
        }
    return {
        "value": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
        "reason": None,
    }


__all__ = [
    "OracleSelection",
    "candidate_errors",
    "select_whole_sample",
    "gather_candidates",
    "oracle_mae",
    "oracle_mse",
    "oracle_cost",
    "oracle_metrics",
    "retention",
]
