"""Supervised candidate costs, detached group normalization and soft targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from sbona.config import Config
from sbona.contracts import (
    CandidateOutput,
    GraphContext,
    OracleOutput,
    ScalerStats,
    TargetBatch,
)
from sbona.objectives.constraints import (
    consistency_cost,
    target_mse_z,
    valid_sample_mask,
)


@dataclass(frozen=True)
class OracleSettings:
    alpha: float = 1.0
    temperature: float = 0.5
    std_floor: float = 0.001
    penalty_negative: float = 1.0
    penalty_envelope: float = 1.0
    penalty_spatial: float = 0.1

    @classmethod
    def from_config(cls, cfg: Config) -> OracleSettings:
        return cls(
            alpha=cfg.oracle.alpha,
            temperature=cfg.oracle.temperature,
            std_floor=cfg.oracle.std_floor,
            penalty_negative=cfg.loss.penalty_negative,
            penalty_envelope=cfg.loss.penalty_envelope,
            penalty_spatial=cfg.loss.penalty_spatial,
        )


def group_zscore(values: Tensor, std_floor: float = 0.001) -> Tensor:
    """Population z-score across the candidate axis, computed after detach.

    ``values``: ``[B, K]``.  ``K=1`` or all-equal groups return zeros.  The
    standard deviation uses ``correction=0``; this is an ordinary z-score, not
    a robust z-score.
    """
    if values.dim() != 2:
        raise ValueError(f"values must be [B, K], got {tuple(values.shape)}")
    detached = values.detach().float()
    if detached.shape[1] == 1:
        return torch.zeros_like(detached)
    mean = detached.mean(dim=-1, keepdim=True)
    std = detached.std(dim=-1, unbiased=False, keepdim=True)
    return (detached - mean) / std.clamp_min(std_floor)


def soft_weights(cost: Tensor, temperature: float, valid: Tensor) -> Tensor:
    """``softmax(-C/tau)`` in fp32; invalid samples get a uniform placeholder."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    weights = torch.softmax(-cost.float() / temperature, dim=-1)
    if not bool(valid.all()):
        uniform = torch.full_like(weights, 1.0 / weights.shape[1])
        weights = torch.where(valid[:, None], weights, uniform)
    return weights.detach()


def build_oracle(
    out: CandidateOutput,
    target: TargetBatch,
    stats: ScalerStats,
    graph: GraphContext,
    settings: OracleSettings,
) -> OracleOutput:
    """Build E, P, detached cost/soft targets for ``[B,K]`` candidates.

    ``error`` and ``consistency`` retain candidate gradients; ``cost`` and
    ``soft_target`` are detached.  Whole invalid samples keep ``C=0``,
    ``w=uniform`` and ``valid_sample=False``.
    """
    if out.pred_z.shape[0] != target.values_raw.shape[0]:
        raise ValueError("candidate/target batch mismatch")
    if out.pred_z.shape[2] != target.values_raw.shape[1]:
        raise ValueError("candidate/target horizon mismatch")
    if out.pred_z.shape[3] != target.values_raw.shape[2]:
        raise ValueError("candidate/target node count mismatch")

    error = target_mse_z(out.pred_z, target, stats)
    pred_raw = stats.inverse(out.pred_z)
    consistency, _ = consistency_cost(
        pred_raw,
        target,
        stats,
        graph,
        penalty_negative=settings.penalty_negative,
        penalty_envelope=settings.penalty_envelope,
        penalty_spatial=settings.penalty_spatial,
    )
    valid = valid_sample_mask(target)
    z_error = group_zscore(error, settings.std_floor)
    z_consistency = group_zscore(consistency, settings.std_floor)
    cost = settings.alpha * z_error + (1.0 - settings.alpha) * z_consistency
    if not bool(valid.all()):
        cost = torch.where(valid[:, None], cost, torch.zeros_like(cost))
    weights = soft_weights(cost, settings.temperature, valid)
    best_index = torch.argmin(cost, dim=-1)
    if not bool(valid.all()):
        best_index = torch.where(valid, best_index, torch.zeros_like(best_index))
    return OracleOutput(
        error=error,
        consistency=consistency,
        cost=cost.detach(),
        soft_target=weights,
        best_index=best_index,
        valid_sample=valid,
    )


__all__ = ["OracleSettings", "group_zscore", "soft_weights", "build_oracle"]
