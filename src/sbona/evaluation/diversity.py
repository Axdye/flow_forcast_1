"""Candidate health, usage and trajectory diversity diagnostics."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def entropy_metrics(weights: Tensor, eps: float = 1e-12) -> dict[str, float]:
    """Entropy diagnostics for soft probabilities ``[B, K]``."""
    if weights.dim() != 2:
        raise ValueError("weights must be [B, K]")
    probs = weights.detach().float().clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    entropy = -(probs * probs.log()).sum(dim=-1)
    mean_entropy = float(entropy.mean())
    num_candidates = probs.shape[1]
    return {
        "mean_entropy": mean_entropy,
        "normalized_entropy": mean_entropy / math.log(num_candidates) if num_candidates > 1 else 0.0,
        "effective_k": float(math.exp(mean_entropy)),
        "mean_probabilities": probs.mean(dim=0).tolist(),
        "max_share": float(probs.mean(dim=0).max()),
    }


def hard_usage(index: Tensor, num_candidates: int) -> dict[str, object]:
    counts = torch.bincount(index.detach().cpu(), minlength=num_candidates).float()
    total = counts.sum().clamp_min(1)
    return {
        "counts": counts.long().tolist(),
        "fractions": (counts / total).tolist(),
        "num_candidates": int(num_candidates),
    }


def binary_usage(selected: Tensor, num_candidates: int, name: str) -> dict[str, object]:
    return {"name": name, **hard_usage(selected, num_candidates)}


def trajectory_distances(pred_raw: Tensor, valid: Tensor, sigma: float) -> dict[str, float]:
    """Mean pairwise raw/standardized L1 over valid elements.

    ``pred_raw``: ``[B, K, T, N, 1]``.  Only pairs ``i<j`` are counted.
    """
    if pred_raw.dim() != 5:
        raise ValueError("pred_raw must be [B, K, T, N, 1]")
    batch_size, num_candidates = pred_raw.shape[:2]
    if num_candidates < 2:
        return {"raw_l1": 0.0, "standardized_l1": 0.0, "pairs": 0}
    mask = valid[:, None, :, :, :].to(pred_raw.dtype)
    total_raw = 0.0
    total_count = 0.0
    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            diff = (pred_raw[:, i] - pred_raw[:, j]).abs() * mask[:, 0]
            total_raw += float(diff.sum())
            total_count += float(mask[:, 0].sum())
    pairs = batch_size * num_candidates * (num_candidates - 1) // 2
    raw = total_raw / max(total_count, 1.0)
    return {
        "raw_l1": raw,
        "standardized_l1": raw / max(sigma, 1e-12),
        "pairs": pairs,
    }


def action_distances(actions_normalized: Tensor) -> dict[str, float]:
    """Mean normalized L2 distance over candidate pairs."""
    if actions_normalized.dim() != 3:
        raise ValueError("actions must be [B, K, 5]")
    num_candidates = actions_normalized.shape[1]
    if num_candidates < 2:
        return {"mean_l2": 0.0}
    total = 0.0
    count = 0
    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            total += float(
                torch.linalg.norm(
                    actions_normalized[:, i] - actions_normalized[:, j], dim=-1
                ).sum()
            )
            count += actions_normalized.shape[0]
    return {"mean_l2": total / max(count, 1)}


def branch_errors(pred_raw: Tensor, target_raw: Tensor, valid: Tensor) -> dict[str, object]:
    """Per-branch masked raw MAE, useful to detect identical low-error branches."""
    if pred_raw.dim() != 5:
        raise ValueError("pred_raw must be [B, K, T, N, 1]")
    target = target_raw[:, None, :, :, :]
    mask = valid[:, None, :, :, :]
    error = (pred_raw.detach() - target.detach()).abs() * mask
    count = mask.expand_as(error).sum(dim=(0, 2, 3, 4)).clamp_min(1)
    totals = error.sum(dim=(0, 2, 3, 4))
    return {"per_branch_mae": (totals / count).tolist()}


def candidate_health(
    pred_raw: Tensor,
    target_raw: Tensor,
    valid: Tensor,
    actions_normalized: Tensor,
    sigma: float,
    oracle_index: Tensor,
    oracle_soft: Tensor,
    router_index: Tensor | None = None,
    saturation_ratio: float | None = None,
) -> dict[str, object]:
    """One JSON-serialisable health document."""
    num_candidates = pred_raw.shape[1]
    out: dict[str, object] = {
        "oracle_hard_usage": binary_usage(oracle_index, num_candidates, "oracle"),
        "soft_oracle": entropy_metrics(oracle_soft),
        "trajectory": trajectory_distances(pred_raw, valid, sigma),
        "action": action_distances(actions_normalized),
        "branch": branch_errors(pred_raw, target_raw, valid),
    }
    if router_index is not None:
        out["router_hard_usage"] = binary_usage(router_index, num_candidates, "router")
    if saturation_ratio is not None:
        out["residual_saturation_ratio"] = float(saturation_ratio)
    return out


__all__ = [
    "entropy_metrics",
    "hard_usage",
    "binary_usage",
    "trajectory_distances",
    "action_distances",
    "branch_errors",
    "candidate_health",
]
