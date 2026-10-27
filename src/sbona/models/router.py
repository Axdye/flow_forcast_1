"""Pre-decoding router: history representation -> candidate logits."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Router(nn.Module):
    """``LayerNorm -> Linear(64,128) -> GELU -> Dropout(.1) -> Linear(128,K)``."""

    def __init__(self, hidden_dim: int = 64, num_candidates: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")
        self.hidden_dim = hidden_dim
        self.num_candidates = num_candidates
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_candidates),
        )

    def forward(self, h_global: Tensor) -> Tensor:
        """Return logits ``[B, K]`` in fp32."""
        if h_global.dim() != 2 or h_global.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"h_global must be [B, {self.hidden_dim}], got {tuple(h_global.shape)}"
            )
        return self.net(h_global).float()

    def select(self, h_global: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(index [B], logits [B,K])``.  Ties pick the smallest index."""
        logits = self.forward(h_global)
        if self.num_candidates == 1:
            index = torch.zeros(logits.shape[0], dtype=torch.int64, device=logits.device)
        else:
            index = torch.argmax(logits, dim=-1)
        return index, logits


__all__ = ["Router"]
