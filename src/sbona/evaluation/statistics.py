"""Paired moving-block bootstrap and multiple-comparison helpers."""

from __future__ import annotations

import numpy as np


def _as_1d(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def paired_moving_block_bootstrap(
    errors_a: np.ndarray,
    counts_a: np.ndarray,
    errors_b: np.ndarray,
    counts_b: np.ndarray,
    block_origins: int = 288,
    replicates: int = 10000,
    seed: int = 20261028,
) -> dict[str, object]:
    """Bootstrap a paired MAE difference using contiguous origin blocks.

    Inputs are per-origin error sums and valid counts (not already-averaged
    batch MAE).  The same resampled origins are applied to both methods.
    """
    errors_a = _as_1d("errors_a", errors_a)
    counts_a = _as_1d("counts_a", counts_a)
    errors_b = _as_1d("errors_b", errors_b)
    counts_b = _as_1d("counts_b", counts_b)
    if not (len(errors_a) == len(counts_a) == len(errors_b) == len(counts_b)):
        raise ValueError("all per-origin arrays must have the same length")
    if block_origins < 1:
        raise ValueError("block_origins must be >= 1")
    if replicates < 1:
        raise ValueError("replicates must be >= 1")
    n = len(errors_a)
    if n == 0:
        return {
            "mae_a": None,
            "mae_b": None,
            "difference": None,
            "ci95": None,
            "replicates": 0,
            "reason": "no origins",
        }

    block_ids = np.arange(n) // block_origins
    unique_blocks = np.unique(block_ids)
    members = [np.where(block_ids == b)[0] for b in unique_blocks]
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    observed_a = errors_a.sum() / max(counts_a.sum(), 1.0)
    observed_b = errors_b.sum() / max(counts_b.sum(), 1.0)
    for replicate in range(replicates):
        chosen = rng.integers(0, len(unique_blocks), size=len(unique_blocks))
        indices = np.concatenate([members[i] for i in chosen])
        mae_a = errors_a[indices].sum() / max(counts_a[indices].sum(), 1.0)
        mae_b = errors_b[indices].sum() / max(counts_b[indices].sum(), 1.0)
        differences[replicate] = mae_a - mae_b

    ci_low, ci_high = np.quantile(differences, [0.025, 0.975], method="linear")
    return {
        "mae_a": float(observed_a),
        "mae_b": float(observed_b),
        "difference": float(observed_a - observed_b),
        "ci95": [float(ci_low), float(ci_high)],
        "replicates": int(replicates),
        "blocks": int(len(unique_blocks)),
        "origins": int(n),
        "note": "bootstrap CI; this is not a p-value",
    }


def block_sign_permutation(
    errors_a: np.ndarray,
    counts_a: np.ndarray,
    errors_b: np.ndarray,
    counts_b: np.ndarray,
    block_origins: int = 288,
    replicates: int = 10000,
    seed: int = 20261028,
) -> dict[str, object]:
    """Two-sided paired block sign-permutation p-value.

    The per-block statistic is the sum of error differences; counts must be
    paired (same origins for both methods).
    """
    errors_a = _as_1d("errors_a", errors_a)
    counts_a = _as_1d("counts_a", counts_a)
    errors_b = _as_1d("errors_b", errors_b)
    counts_b = _as_1d("counts_b", counts_b)
    if not (len(errors_a) == len(errors_b) == len(counts_a) == len(counts_b)):
        raise ValueError("all per-origin arrays must have the same length")
    block_ids = np.arange(len(errors_a)) // block_origins
    statistic = np.zeros(int(block_ids.max()) + 1 if len(block_ids) else 0, dtype=np.float64)
    for index, block in enumerate(block_ids):
        if counts_a[index] > 0 and counts_b[index] > 0:
            statistic[block] += errors_a[index] - errors_b[index]
    statistic = statistic[statistic != 0.0]
    if statistic.size == 0:
        return {"p_value": None, "reason": "no non-zero block differences"}
    observed = abs(statistic.sum())
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(replicates):
        signs = rng.choice([-1.0, 1.0], size=statistic.size)
        if abs((statistic * signs).sum()) >= observed:
            exceed += 1
    return {
        "p_value": (exceed + 1.0) / (replicates + 1.0),
        "blocks": int(statistic.size),
        "replicates": int(replicates),
        "observed": float(observed),
    }


def holm_correction(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, preserving input order."""
    m = len(pvalues)
    if m == 0:
        return []
    order = np.argsort(np.asarray(pvalues, dtype=np.float64))
    adjusted = np.ones(m, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        value = (m - rank) * float(pvalues[index])
        running = max(running, value)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def per_origin_records(records: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort per-sample records by sample id and return ``(ids, errors, counts)``."""
    if not records:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    ordered = sorted(records, key=lambda item: item["sample_id"])
    ids = np.asarray([item["sample_id"] for item in ordered], dtype=np.int64)
    errors = np.asarray([item["abs_sum"] for item in ordered], dtype=np.float64)
    counts = np.asarray([item["count"] for item in ordered], dtype=np.float64)
    return ids, errors, counts


__all__ = [
    "paired_moving_block_bootstrap",
    "block_sign_permutation",
    "holm_correction",
    "per_origin_records",
]
