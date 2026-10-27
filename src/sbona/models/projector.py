"""Feature projector: maps any backbone hidden into the shared adapter space."""

from __future__ import annotations

from torch import Tensor, nn


class FeatureProjector(nn.Module):
    """``h_node = Linear(D, D)(GELU(Linear(D_b, D)(LayerNorm(hidden))))``.

    No residual bypass, no dropout, global mean pooling.
    """

    def __init__(self, backbone_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if backbone_dim < 1 or hidden_dim < 1:
            raise ValueError("backbone_dim and hidden_dim must be >= 1")
        self.backbone_dim = backbone_dim
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(backbone_dim)
        self.fc1 = nn.Linear(backbone_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """``hidden [B, N, D_b] -> (h_node [B, N, D], h_global [B, D])``."""
        if hidden.dim() != 3:
            raise ValueError(f"hidden must be [B, N, D_b], got {tuple(hidden.shape)}")
        if hidden.shape[-1] != self.backbone_dim:
            raise ValueError(
                f"expected backbone dim {self.backbone_dim}, got {hidden.shape[-1]}"
            )
        h_node = self.fc2(self.act(self.fc1(self.norm(hidden)))).float()
        h_global = h_node.mean(dim=1).float()
        return h_node, h_global


__all__ = ["FeatureProjector"]
