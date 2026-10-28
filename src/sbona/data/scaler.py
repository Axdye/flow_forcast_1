"""Train-only global scaler and per-node empirical envelope."""

from __future__ import annotations

import numpy as np
import torch

from sbona.contracts import ScalerStats


class ScalerError(ValueError):
    """Raised when the training interval has no valid observations."""


def fit_scaler(
    data: np.ndarray,
    valid: np.ndarray,
    train_end: int,
    quantile: float = 0.995,
    envelope_min_count: int = 1000,
) -> ScalerStats:
    """Fit ``mu``/``sigma`` and the envelope using only ``[0, train_end)``.

    ``data`` is raw ``[M, N]`` and may contain non-finite values; ``valid`` is
    the raw observation mask.  Invalid values are excluded from every statistic.
    Population std is used (``ddof=0``), matching the design contract.
    """
    if data.shape != valid.shape:
        raise ScalerError("data and valid must have the same shape")
    if not (0 < train_end <= data.shape[0]):
        raise ScalerError(f"train_end {train_end} outside [1, {data.shape[0]}]")
    train_data = data[:train_end]
    train_valid = valid[:train_end] & np.isfinite(train_data)
    values = train_data[train_valid]
    if values.size == 0:
        raise ScalerError("training interval has no valid flow values")
    mu = float(values.mean())
    sigma = float(values.std(ddof=0))
    sigma = max(sigma, 1e-6)

    num_nodes = data.shape[1]
    envelope = np.empty(num_nodes, dtype=np.float32)
    fallback = np.zeros(num_nodes, dtype=bool)
    global_q = float(np.quantile(values, quantile, method="linear"))
    for node in range(num_nodes):
        node_values = train_data[:, node][train_valid[:, node]]
        if node_values.size >= envelope_min_count:
            envelope[node] = float(np.quantile(node_values, quantile, method="linear"))
        else:
            envelope[node] = global_q
            fallback[node] = True
    return ScalerStats(
        mu=mu,
        sigma=sigma,
        envelope=torch.from_numpy(envelope),
        envelope_fallback=torch.from_numpy(fallback),
        envelope_quantile=quantile,
        envelope_min_count=envelope_min_count,
    )


def fill_invalid_with_mean(
    data: np.ndarray,
    valid: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Replace invalid raw values with the training global mean.

    The standardized value of the fill is exactly zero.  No future value or
    interpolation is used.
    """
    if data.shape != valid.shape:
        raise ScalerError("data and valid must have the same shape")
    filled = np.array(data, dtype=np.float32, copy=True)
    filled[~valid] = np.float32(mu)
    return filled


def standardize_array(data: np.ndarray, stats: ScalerStats) -> np.ndarray:
    return ((data - stats.mu) / stats.sigma).astype(np.float32, copy=False)


__all__ = ["ScalerError", "fit_scaler", "fill_invalid_with_mean", "standardize_array"]
