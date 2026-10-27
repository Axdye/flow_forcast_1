"""Dimensionless continuous consistency costs.

All reductions are per *sample* so that the candidate loss can sum over K and
average over valid samples.  Invalid targets never contribute.  ``P_spa`` uses
the undirected constraint edges once (``i < j``) and uses the same valid-target
support as the non-negativity and envelope terms.
"""

from __future__ import annotations

import torch
from torch import Tensor

from sbona.contracts import GraphContext, ScalerStats, TargetBatch


def _masked_mean_over_nonbatch(values: Tensor, mask: Tensor) -> Tensor:
    """Per-sample mean over all dimensions after the first, in fp32.

    ``values``: ``[B, ...]``; ``mask``: broadcastable to ``values`` with a
    leading batch dimension.  Returns ``[B]`` in float32; empty support returns
    exactly 0 because the caller separately counts validity.
    """
    values = values.float()
    expanded = mask.expand_as(values)
    # Dim 0 is the batch axis and dim 1 is the candidate axis; both must stay.
    dims = tuple(range(2, values.dim()))
    denom = expanded.sum(dim=dims).clamp_min(1)
    return (values * expanded).sum(dim=dims) / denom


def target_mse_z(pred_z: Tensor, target: TargetBatch, stats: ScalerStats) -> Tensor:
    """Masked MSE in z space, shaped ``[B, K]``, reduced in fp32.

    ``pred_z`` is ``[B, K, T, N, 1]``.  ``(yhat_raw-y_raw)/s`` is identical to
    ``yhat_z - y_z`` because the global scalar transform is affine.
    """
    target_z = stats.standardize(target.values_raw.float())
    error = (pred_z.float() - target_z[:, None, :, :, :]) ** 2
    mask = target.valid[:, None, :, :, :]
    return _masked_mean_over_nonbatch(error, mask)


def negative_penalty(pred_raw: Tensor, target: TargetBatch, stats: ScalerStats) -> Tensor:
    """``mean((relu(-yhat)/s)^2)`` over valid target elements, ``[B, K]``."""
    values = (torch.relu(-pred_raw.float()) / float(stats.sigma)) ** 2
    return _masked_mean_over_nonbatch(values, target.valid[:, None, :, :, :])


def envelope_penalty(
    pred_raw: Tensor,
    target: TargetBatch,
    stats: ScalerStats,
    envelope: Tensor | None = None,
) -> Tensor:
    """``mean((relu(yhat-C_node)/s)^2)`` over valid target elements, ``[B, K]``."""
    cap = stats.envelope.float() if envelope is None else envelope.float()
    if cap.dim() != 1:
        raise ValueError("envelope must be [N]")
    values = (
        torch.relu(pred_raw.float() - cap[None, None, None, :, None]) / float(stats.sigma)
    ) ** 2
    return _masked_mean_over_nonbatch(values, target.valid[:, None, :, :, :])


def _huber_unit(x: Tensor) -> Tensor:
    abs_x = x.abs()
    return torch.where(abs_x <= 1.0, 0.5 * x * x, abs_x - 0.5)


def spatial_penalty(
    pred_raw: Tensor,
    target: TargetBatch,
    stats: ScalerStats,
    graph: GraphContext,
) -> tuple[Tensor, Tensor]:
    """Weighted Huber total variation over valid undirected edges.

    Returns ``(penalty [B, K], edge_weight_sum [B])``.  If a sample has no
    valid edge, its penalty is exactly zero and the reported weight sum is 0.
    """
    if graph.num_edges == 0:
        zeros = pred_raw.new_zeros((pred_raw.shape[0], pred_raw.shape[1]))
        counts = pred_raw.new_zeros((pred_raw.shape[0],))
        return zeros, counts
    pred = pred_raw.float()
    u = graph.edge_index[0]
    v = graph.edge_index[1]
    pred_u = pred[:, :, :, u, :]
    pred_v = pred[:, :, :, v, :]
    diff = (pred_u - pred_v) / float(stats.sigma)
    huber = _huber_unit(diff)  # [B,K,T,E,1]

    valid_u = target.valid[:, :, u, :]  # [B,T,E,1]
    valid_v = target.valid[:, :, v, :]
    edge_valid = (valid_u & valid_v).to(huber.dtype)  # [B,T,E,1]
    weights = graph.edge_weight.float()[None, None, None, :, None] * edge_valid[:, None, :, :, :]
    weights = weights.expand_as(huber)  # [B,K,T,E,1]
    numerator = (huber * weights).sum(dim=(2, 3, 4))
    denominator = weights.sum(dim=(2, 3, 4))  # [B,K]
    # Denominator is identical across K; use candidate 0 for reporting.
    edge_weight_sum = denominator[:, 0]
    penalty = torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1e-12),
        torch.zeros_like(numerator),
    )
    return penalty, edge_weight_sum


def consistency_cost(
    pred_raw: Tensor,
    target: TargetBatch,
    stats: ScalerStats,
    graph: GraphContext,
    penalty_negative: float = 1.0,
    penalty_envelope: float = 1.0,
    penalty_spatial: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Return ``(P [B,K], spatial_edge_weight_sum [B])``."""
    p_neg = negative_penalty(pred_raw, target, stats)
    p_env = envelope_penalty(pred_raw, target, stats)
    p_spa, spa_support = spatial_penalty(pred_raw, target, stats, graph)
    total = penalty_negative * p_neg + penalty_envelope * p_env + penalty_spatial * p_spa
    return total, spa_support


def valid_sample_mask(target: TargetBatch) -> Tensor:
    """``[B]`` bool: sample has at least one valid target element."""
    return target.valid.reshape(target.valid.shape[0], -1).any(dim=1)


__all__ = [
    "target_mse_z",
    "negative_penalty",
    "envelope_penalty",
    "spatial_penalty",
    "consistency_cost",
    "valid_sample_mask",
]
