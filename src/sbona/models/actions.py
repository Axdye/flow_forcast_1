"""Bounded five-dimensional control actions and Sobol anchors.

Action dimensions (all per candidate ``k``):

====  ==================  ==========================================  =============
dim   symbol              mapping                                     unit
====  ==================  ==========================================  =============
0     ``o``               ``(T-1) * sigmoid(z0)``                     forecast step
1     ``delta``           ``lo + (hi-lo) * sigmoid(z1)``              steps / hop
2     ``m``               ``hi * tanh(z2)`` (symmetric range)         z residual
3     ``d``               ``lo + (hi-lo) * sigmoid(z3)``              steps
4     ``rho``             ``lo + (hi-lo) * sigmoid(z4)``              1 / step
====  ==================  ==========================================  =============

The physical action is an inductive bias, not an identified traffic
parameter.  ``m=0`` must produce exactly zero residual.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch import Tensor, nn

from sbona.config import ActionRangesConfig
from sbona.contracts import ActionOutput


def _sobol_anchors(num_candidates: int, seed: int) -> np.ndarray:
    try:
        from scipy.stats import qmc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sampler = qmc.Sobol(d=5, scramble=True, seed=seed)
            points = sampler.random(num_candidates)
        anchors = points * 2.0 - 1.0
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        rng = np.random.default_rng(seed)
        anchors = rng.uniform(-1.0, 1.0, size=(num_candidates, 5))
    anchors = anchors.astype(np.float32)
    if np.allclose(anchors, 0.0):
        # Sobol(0) is exactly zero; scramble=True avoids this, but never ship
        # an all-zero anchor set even under a degenerate fallback.
        anchors[0, 0] = 0.5
    return anchors


class ActionGenerator(nn.Module):
    """Two-layer MLP ``D -> 64 -> 10`` (first 5 = mu, last 5 = sigma)."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_candidates: int = 8,
        output_steps: int = 12,
        ranges: ActionRangesConfig | None = None,
        sigma_min: float = 0.05,
        anchor_seed: int = 20261028,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_candidates = num_candidates
        self.output_steps = output_steps
        self.ranges = ranges or ActionRangesConfig()
        self.sigma_min = sigma_min
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 10),
        )
        anchors = torch.from_numpy(_sobol_anchors(num_candidates, anchor_seed))
        self.anchors = nn.Parameter(anchors)

    def forward(self, h_global: Tensor) -> ActionOutput:
        if h_global.dim() != 2 or h_global.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"h_global must be [B, {self.hidden_dim}], got {tuple(h_global.shape)}"
            )
        out = self.mlp(h_global)
        mu = out[:, :5]
        sigma = self.sigma_min + torch.nn.functional.softplus(out[:, 5:])
        # [B, 1, 5] + [1, K, 5] -> [B, K, 5]
        z = mu[:, None, :] + sigma[:, None, :] * self.anchors[None, :, :]
        physical = self._to_physical(z).float()
        normalized = self._normalize(physical).float()
        return ActionOutput(
            physical=physical,
            normalized=normalized,
            mu=mu.float(),
            sigma=sigma.float(),
        )

    def _duration_bounds(self) -> tuple[float, float]:
        """Contract: ``d`` lies in ``[1, T]``; never exceed the forecast horizon."""
        horizon = float(self.output_steps)
        high = min(float(self.ranges.duration[1]), horizon)
        low = min(float(self.ranges.duration[0]), high)
        return low, high

    def _to_physical(self, z: Tensor) -> Tensor:
        horizon = float(self.output_steps)
        r = self.ranges
        if self.output_steps == 1:
            o = torch.zeros_like(z[..., 0])
        else:
            o = (horizon - 1.0) * torch.sigmoid(z[..., 0])
        delay_lo, delay_hi = r.delay
        delta = delay_lo + (delay_hi - delay_lo) * torch.sigmoid(z[..., 1])
        mag_lo, mag_hi = r.magnitude
        if abs(mag_lo + mag_hi) < 1e-9:
            magnitude = mag_hi * torch.tanh(z[..., 2])
        else:
            center = 0.5 * (mag_lo + mag_hi)
            half = 0.5 * (mag_hi - mag_lo)
            magnitude = center + half * torch.tanh(z[..., 2])
        duration_lo, duration_hi = self._duration_bounds()
        duration = duration_lo + (duration_hi - duration_lo) * torch.sigmoid(z[..., 3])
        rho_lo, rho_hi = r.recovery
        rho = rho_lo + (rho_hi - rho_lo) * torch.sigmoid(z[..., 4])
        return torch.stack([o, delta, magnitude, duration, rho], dim=-1)

    def _bounds(self) -> tuple[Tensor, Tensor]:
        device = self.anchors.device
        dtype = self.anchors.dtype
        duration_lo, duration_hi = self._duration_bounds()
        lows = [
            0.0,
            self.ranges.delay[0],
            self.ranges.magnitude[0],
            duration_lo,
            self.ranges.recovery[0],
        ]
        highs = [
            max(float(self.output_steps - 1), 0.0),
            self.ranges.delay[1],
            self.ranges.magnitude[1],
            duration_hi,
            self.ranges.recovery[1],
        ]
        return (
            torch.tensor(lows, device=device, dtype=dtype),
            torch.tensor(highs, device=device, dtype=dtype),
        )

    def _normalize(self, physical: Tensor) -> Tensor:
        low, high = self._bounds()
        span = (high - low).clamp_min(1e-12)
        normalized = (physical - low) / span
        if self.output_steps == 1:
            normalized = normalized.clone()
            normalized[..., 0] = 0.0
        return normalized


__all__ = ["ActionGenerator", "_sobol_anchors"]
