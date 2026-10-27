"""Structured shared-residual decoder (v1: no free bypass).

For a candidate action ``(o, delta, m, d, rho)`` and graph hops ``r=0..R``::

    q[b,k,t,r] = -((t-o)-r*delta)^2 / (2*s_p^2)
    pi         = softmax(q, dim=r)
    e[b,k,t]   = sigmoid((t-o)/s_o) * sigmoid((o+d-t)/s_d)
                 * exp(-rho * relu(t-o-d))
    u[b,k,t,n] = m[b,k] * e[b,k,t] * sum_r pi * (W_out S_r)[b,r,n]
    residual_z = r_scale * tanh(u / r_scale)

``W_out`` has no bias.  Therefore ``m=0`` produces exactly zero residual.
The linear map is applied to the graph basis *before* the hop mixing because
``W_out`` is linear; this is mathematically identical and avoids a large
``[B, K, T, N, D_r]`` activation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sbona.contracts import GraphContext


def _propagate_node_features(diffusion: Tensor, features: Tensor) -> Tensor:
    """Sparse ``A @ X`` for ``X`` shaped ``[B, N, D]`` in fp32.

    Internally reshapes to ``[N, B*D]`` so one ``torch.sparse.mm`` call runs
    and no ``[B, K, T, N, N]`` tensor is ever created.  Sparse ops are kept in
    fp32 regardless of autocast, per the design contract.
    """
    features = features.float()
    batch_size, nodes, dim = features.shape
    flat = features.permute(1, 0, 2).reshape(nodes, batch_size * dim)
    out = torch.sparse.mm(diffusion.float(), flat)
    return out.reshape(nodes, batch_size, dim).permute(1, 0, 2)


class StructuredDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        residual_dim: int = 32,
        graph_hops: int = 3,
        output_steps: int = 12,
        s_p: float = 1.0,
        s_o: float = 0.5,
        s_d: float = 0.5,
        residual_scale: float = 3.0,
        output_init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual_dim = residual_dim
        self.graph_hops = graph_hops
        self.output_steps = output_steps
        self.s_p = s_p
        self.s_o = s_o
        self.s_d = s_d
        self.residual_scale = residual_scale

        self.in_proj1 = nn.Linear(hidden_dim, residual_dim)
        self.act = nn.GELU()
        self.in_proj2 = nn.Linear(residual_dim, residual_dim)
        self.out_proj = nn.Linear(residual_dim, 1, bias=False)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=output_init_std)

        self.last_saturation: Tensor | None = None
        self._diffusion_cache: dict[tuple[int, str], Tensor] = {}

    def _diffusion(self, graph: GraphContext, device: torch.device) -> Tensor:
        key = (graph.graph_hash, str(device))
        cached = self._diffusion_cache.get(key)
        if cached is None:
            cached = graph.diffusion
            if cached.device != device:
                cached = cached.to(device)
            if cached.dtype != torch.float32:
                cached = cached.to(torch.float32)
            self._diffusion_cache[key] = cached
        return cached

    def _basis(self, h_node: Tensor, graph: GraphContext) -> Tensor:
        """Return hop-mapped scalars ``[B, R+1, N]``."""
        if h_node.dim() != 3 or h_node.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"h_node must be [B, N, {self.hidden_dim}], got {tuple(h_node.shape)}"
            )
        if h_node.shape[1] != graph.num_nodes:
            raise ValueError(
                f"h_node has {h_node.shape[1]} nodes but graph has {graph.num_nodes}"
            )
        diffusion = self._diffusion(graph, h_node.device)
        features = self.in_proj2(self.act(self.in_proj1(h_node))).float()  # [B,N,D_r]
        basis = [features]
        current = features
        for _ in range(self.graph_hops):
            current = _propagate_node_features(diffusion, current)
            basis.append(current)
        stacked = torch.stack(basis, dim=1)  # [B, R+1, N, D_r]
        scalars = self.out_proj(stacked).squeeze(-1).float()  # [B, R+1, N]
        return scalars

    def _kernel(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(pi [B,K,T,R+1], e [B,K,T], m [B,K])``."""
        if actions.dim() != 3 or actions.shape[-1] != 5:
            raise ValueError(f"actions must be [B, K, 5], got {tuple(actions.shape)}")
        actions = actions.float()
        batch_size, num_candidates, _ = actions.shape
        device = actions.device
        horizon = self.output_steps
        steps = torch.arange(horizon, device=device, dtype=actions.dtype)
        hops = torch.arange(self.graph_hops + 1, device=device, dtype=actions.dtype)

        o = actions[..., 0]
        delta = actions[..., 1]
        magnitude = actions[..., 2]
        duration = actions[..., 3]
        rho = actions[..., 4]

        centered = steps[None, None, :, None] - o[:, :, None, None]
        q = -((centered - hops[None, None, None, :] * delta[:, :, None, None]) ** 2) / (
            2.0 * self.s_p**2
        )
        pi = torch.softmax(q, dim=-1)

        onset = torch.sigmoid((steps[None, None, :] - o[:, :, None]) / self.s_o)
        finish = torch.sigmoid((o[:, :, None] + duration[:, :, None] - steps[None, None, :]) / self.s_d)
        tail = torch.exp(
            -rho[:, :, None] * torch.relu(steps[None, None, :] - o[:, :, None] - duration[:, :, None])
        )
        envelope = onset * finish * tail
        if pi.shape[:2] != (batch_size, num_candidates):
            raise RuntimeError("internal candidate-axis mismatch")
        return pi, envelope, magnitude

    def _decode(self, h_node: Tensor, actions: Tensor, graph: GraphContext) -> Tensor:
        scalars = self._basis(h_node, graph)  # [B, R+1, N]
        pi, envelope, magnitude = self._kernel(actions)
        mixed = torch.einsum("bktr,brn->bktn", pi, scalars)
        u = magnitude[:, :, None, None] * envelope[:, :, :, None] * mixed
        residual = self.residual_scale * torch.tanh(u / self.residual_scale)
        self.last_saturation = (
            (u.detach().abs() > 3.0 * self.residual_scale).float().mean()
        )
        return residual.unsqueeze(-1).float()

    def forward_all(self, h_node: Tensor, actions: Tensor, graph: GraphContext) -> Tensor:
        """Decode all ``K`` candidates: returns residual ``[B, K, T, N, 1]``."""
        return self._decode(h_node, actions, graph)

    def forward_selected(self, h_node: Tensor, selected: Tensor, graph: GraphContext) -> Tensor:
        """Decode exactly one candidate per sample: returns ``[B, T, N, 1]``.

        The candidate axis is length 1 and is immediately squeezed; this method
        never calls :meth:`forward_all` and never creates a ``K``-sized
        trajectory tensor.
        """
        if selected.dim() != 2 or selected.shape[-1] != 5:
            raise ValueError(f"selected must be [B, 5], got {tuple(selected.shape)}")
        residual = self._decode(h_node, selected[:, None, :], graph)
        return residual[:, 0]


__all__ = ["StructuredDecoder", "_propagate_node_features"]
