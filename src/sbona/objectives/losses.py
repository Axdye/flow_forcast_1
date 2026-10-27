"""Candidate, separation and router losses (fp32 reductions)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from sbona.contracts import OracleOutput


class LossError(ValueError):
    """Raised for malformed loss inputs."""


def mean_over_valid(values: Tensor, valid: Tensor) -> tuple[Tensor, int]:
    """Mean over valid samples; returns ``(loss, valid_count)``.

    The caller must skip the optimizer step when ``valid_count == 0``.  We do
    not return a fake zero-error "good" sample.
    """
    if values.dim() != 1:
        raise LossError(f"values must be [B], got {tuple(values.shape)}")
    if valid.shape != values.shape:
        raise LossError("valid mask shape must match values")
    values = values.float()
    valid = valid.bool()
    count = int(valid.sum().item())
    if count == 0:
        return values.new_zeros(()), 0
    return values[valid].mean(), count


def candidate_loss(
    oracle: OracleOutput,
    epsilon: float = 0.05,
    lambda_consistency: float = 0.0,
) -> tuple[Tensor, int]:
    """``mean_valid(sum_k w_tilde * (E + lambda*P))``.

    ``w_tilde = (1-epsilon)*w + epsilon/K`` keeps a non-negative floor for
    losing candidates.  This is a candidate-dimension **sum** and a
    sample-dimension mean; do not multiply or divide by K again.
    """
    if not (0.0 <= epsilon <= 1.0):
        raise LossError("epsilon must be in [0, 1]")
    if lambda_consistency < 0:
        raise LossError("lambda_consistency must be >= 0")
    weights = oracle.soft_target.detach().float()
    if weights.dim() != 2:
        raise LossError("soft_target must be [B, K]")
    num_candidates = weights.shape[1]
    floor = epsilon / float(num_candidates)
    w_tilde = (1.0 - epsilon) * weights + floor
    per_candidate = oracle.error.float() + lambda_consistency * oracle.consistency.float()
    per_sample = (w_tilde * per_candidate).sum(dim=-1)
    return mean_over_valid(per_sample, oracle.valid_sample)


def separation_loss(
    actions_normalized: Tensor,
    valid: Tensor,
    margin: float = 0.25,
) -> tuple[Tensor, int]:
    """``mean_valid(mean_{i<j} relu(margin - ||a_i-a_j||)^2)``."""
    if actions_normalized.dim() != 3 or actions_normalized.shape[-1] != 5:
        raise LossError("actions_normalized must be [B, K, 5]")
    actions_normalized = actions_normalized.float()
    num_candidates = actions_normalized.shape[1]
    if num_candidates < 2:
        return actions_normalized.new_zeros(()), int(valid.sum().item())
    diff = actions_normalized[:, :, None, :] - actions_normalized[:, None, :, :]
    distances = torch.linalg.norm(diff, dim=-1)  # [B,K,K]
    upper = torch.triu(
        torch.ones(
            (num_candidates, num_candidates),
            dtype=torch.bool,
            device=actions_normalized.device,
        ),
        diagonal=1,
    )
    pair_distances = distances[:, upper]  # [B, K*(K-1)/2]
    pair_loss = torch.relu(margin - pair_distances) ** 2
    per_sample = pair_loss.mean(dim=-1)
    return mean_over_valid(per_sample, valid)


@dataclass
class RouterLossBreakdown:
    total: Tensor
    kl: Tensor
    regret: Tensor
    balance: Tensor
    count: int


def router_loss(
    logits: Tensor,
    oracle: OracleOutput,
    beta: float,
    kl_weight: float = 1.0,
    regret_weight: float = 0.2,
) -> RouterLossBreakdown:
    """KL(w || p) + regret_weight * E_p[C-min C] + beta * balance."""
    if logits.dim() != 2:
        raise LossError("logits must be [B, K]")
    if logits.shape != oracle.soft_target.shape:
        raise LossError("logits and oracle soft targets must have the same [B,K] shape")
    if beta < 0 or kl_weight < 0 or regret_weight < 0:
        raise LossError("router loss weights must be non-negative")
    logits = logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    weights = oracle.soft_target.detach().float()
    cost = oracle.cost.detach().float()
    valid = oracle.valid_sample.bool()
    num_candidates = weights.shape[1]

    kl_per_sample = (
        weights * (torch.log(weights.clamp_min(1e-12)) - logp)
    ).sum(dim=-1)
    regret_per_sample = (
        logp.exp() * (cost - cost.min(dim=-1, keepdim=True).values)
    ).sum(dim=-1)
    kl, count = mean_over_valid(kl_per_sample, valid)
    regret, _ = mean_over_valid(regret_per_sample, valid)
    if count == 0:
        zero = logits.new_zeros(())
        return RouterLossBreakdown(zero, zero, zero, zero, 0)

    probabilities = logp[valid].exp()
    usage = probabilities.mean(dim=0)
    balance = ((usage - 1.0 / num_candidates) ** 2).sum()
    total = kl_weight * kl + regret_weight * regret + beta * balance
    return RouterLossBreakdown(
        total=total,
        kl=kl,
        regret=regret,
        balance=balance,
        count=count,
    )


def beta_schedule(
    epoch: int,
    max_epochs: int,
    max_beta: float = 0.01,
    final_beta: float = 0.002,
) -> float:
    """Linear ramp 0 -> max (first 10%), hold (10%-50%), decay -> final."""
    if max_epochs <= 0:
        raise LossError("max_epochs must be > 0")
    if epoch < 0:
        raise LossError("epoch must be >= 0")
    progress = min(max(epoch / float(max_epochs), 0.0), 1.0)
    if progress < 0.1:
        return max_beta * (progress / 0.1)
    if progress < 0.5:
        return max_beta
    decay = (progress - 0.5) / 0.5
    return max_beta + (final_beta - max_beta) * decay


def base_mse_loss(
    base_pred_z: Tensor,
    target_z: Tensor,
    valid: Tensor,
) -> tuple[Tensor, int]:
    """Masked MSE for Stage A; returns ``(loss, valid_count)``, reduced in fp32."""
    error = (base_pred_z.float() - target_z.float()) ** 2
    expanded = valid.expand_as(error)
    per_sample = (error * expanded).sum(dim=tuple(range(1, error.dim()))) / expanded.sum(
        dim=tuple(range(1, error.dim()))
    ).clamp_min(1)
    valid_sample = valid.reshape(valid.shape[0], -1).any(dim=1)
    return mean_over_valid(per_sample, valid_sample)


__all__ = [
    "LossError",
    "mean_over_valid",
    "candidate_loss",
    "separation_loss",
    "RouterLossBreakdown",
    "router_loss",
    "beta_schedule",
    "base_mse_loss",
]
