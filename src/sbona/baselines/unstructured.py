"""Unstructured five-dimensional latent baseline.

It keeps the same K, graph/input availability, router and shared projector, but
replaces the structured ``o/delta/m/d/rho`` envelope with a shared low-rank
time-node decoder.  This is the structure ablation, not a second method.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sbona.contracts import GraphContext


class UnstructuredDecoder(nn.Module):
    """``residual[t,n] = r_scale * tanh(sum_r coeff_r(a) * E_time[t,r] * E_node[n,r] / r_scale)``."""

    def __init__(
        self,
        latent_dim: int = 5,
        rank: int = 32,
        output_steps: int = 12,
        num_nodes: int = 1,
        residual_scale: float = 3.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.rank = rank
        self.output_steps = output_steps
        self.num_nodes = num_nodes
        self.residual_scale = residual_scale
        self.coeff = nn.Linear(latent_dim, rank)
        self.time_embedding = nn.Parameter(torch.empty(output_steps, rank))
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, rank))
        nn.init.normal_(self.time_embedding, std=0.02)
        nn.init.normal_(self.node_embedding, std=0.02)
        self.last_saturation: Tensor | None = None

    def _decode(self, latent: Tensor) -> Tensor:
        # latent [B,K,5], coeff [B,K,R], time [T,R], node [N,R]
        coeff = self.coeff(latent)
        mixed = torch.einsum("bkr,tr,nr->bktn", coeff, self.time_embedding, self.node_embedding)
        residual = self.residual_scale * torch.tanh(mixed / self.residual_scale)
        self.last_saturation = (
            (mixed.detach().abs() > 3.0 * self.residual_scale).float().mean()
        )
        return residual.unsqueeze(-1).float()

    def forward_all(self, h_node: Tensor, actions: Tensor, graph: GraphContext) -> Tensor:
        if actions.dim() != 3 or actions.shape[-1] != self.latent_dim:
            raise ValueError("actions must be [B, K, 5]")
        if h_node.shape[1] != self.num_nodes:
            raise ValueError("node count mismatch in unstructured decoder")
        return self._decode(actions)

    def forward_selected(self, h_node: Tensor, selected: Tensor, graph: GraphContext) -> Tensor:
        if selected.dim() != 2 or selected.shape[-1] != self.latent_dim:
            raise ValueError("selected must be [B, 5]")
        return self._decode(selected[:, None, :])[:, 0]


__all__ = ["UnstructuredDecoder"]
