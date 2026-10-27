"""Gaussian perturbation control.

Only a coverage/capacity diagnostic; it is never a deployable strong baseline.
The noise scale must be selected on validation data, never on test oracle
performance.
"""

from __future__ import annotations

import torch
from torch import Tensor


def gaussian_candidates(
    base_pred_z: Tensor,
    sigma: float,
    num_candidates: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return ``[B, K, T, N, 1]`` base prediction plus fixed-scale Gaussian noise."""
    if base_pred_z.dim() != 4:
        raise ValueError("base_pred_z must be [B, T, N, 1]")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if num_candidates < 1:
        raise ValueError("num_candidates must be >= 1")
    noise = torch.randn(
        (base_pred_z.shape[0], num_candidates) + base_pred_z.shape[1:],
        device=base_pred_z.device,
        dtype=base_pred_z.dtype,
        generator=generator,
    )
    return base_pred_z[:, None] + sigma * noise


__all__ = ["gaussian_candidates"]
