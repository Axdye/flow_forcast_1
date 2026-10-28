"""AGCRN backbone adapter.

Upstream: https://github.com/LeiBAI/AGCRN, commit
``7fbbf2aeb099242098a3cf482b55cd45d7295c28`` (MIT licence).

AGCRN learns node-specific ("adaptive") graph convolutions from node embeddings
instead of consuming the road graph, so it is the natural *adaptive-graph*
contrast in the backbone matrix: the graph prior reaches the prediction only
through SBoN's residual decoder.

Faithfulness notes
------------------
* ``AVWGCN`` recomputes ``softmax(relu(E E^T))`` and the Chebyshev support set
  at every call, exactly like upstream.  This is O(N^2 D) per cell per step and
  is intentionally not cached: caching it would change the autograd graph.
* The published PEMS08 configuration uses ``input_dim=1`` (flow only),
  ``embed_dim=2``, ``rnn_units=64``, ``num_layers=2`` and ``cheb_order=2``.
  Those are the defaults here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sbona.backbones.common import TorchBackboneAdapter, stack_features
from sbona.config import BackboneConfig
from sbona.contracts import GraphContext, ModelInput


class AVWGCN(nn.Module):
    """Node-adaptive Chebyshev graph convolution with generated weights."""

    def __init__(self, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int) -> None:
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, dim_out))
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)

    def forward(self, x: Tensor, node_embeddings: Tensor) -> Tensor:
        nodes = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.t())), dim=1)
        support_set = [torch.eye(nodes, device=supports.device), supports]
        for _ in range(2, self.cheb_k):
            support_set.append(
                torch.matmul(2 * supports, support_set[-1]) - support_set[-2]
            )
        stacked = torch.stack(support_set, dim=0)
        weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", stacked, x)
        x_g = x_g.permute(0, 2, 1, 3)
        return torch.einsum("bnki,nkio->bno", x_g, weights) + bias


class AGCRNCell(nn.Module):
    def __init__(
        self, node_num: int, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int
    ) -> None:
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in + dim_out, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + dim_out, dim_out, cheb_k, embed_dim)

    def forward(self, x: Tensor, state: Tensor, node_embeddings: Tensor) -> Tensor:
        state = state.to(x.device)
        z_r = torch.sigmoid(self.gate(torch.cat((x, state), dim=-1), node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        return r * state + (1 - r) * hc

    def init_hidden_state(self, batch_size: int) -> Tensor:
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(
        self,
        node_num: int,
        dim_in: int,
        dim_out: int,
        cheb_k: int,
        embed_dim: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("AGCRN needs at least one recurrent layer")
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList(
            [AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim)]
            + [
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim)
                for _ in range(1, num_layers)
            ]
        )

    def forward(
        self, x: Tensor, init_state: Tensor, node_embeddings: Tensor
    ) -> tuple[Tensor, list[Tensor]]:
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden: list[Tensor] = []
        for layer in range(self.num_layers):
            state = init_state[layer]
            inner_states = []
            for step in range(seq_length):
                state = self.dcrnn_cells[layer](
                    current_inputs[:, step, :, :], state, node_embeddings
                )
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size: int) -> Tensor:
        return torch.stack(
            [cell.init_hidden_state(batch_size) for cell in self.dcrnn_cells], dim=0
        )


class AGCRNArch(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int = 1,
        rnn_units: int = 64,
        output_dim: int = 1,
        horizon: int = 12,
        num_layers: int = 2,
        cheb_k: int = 2,
        embed_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.encoder = AVWDCRNN(
            num_nodes, input_dim, rnn_units, cheb_k, embed_dim, num_layers
        )
        self.end_conv = nn.Conv2d(
            1, horizon * output_dim, kernel_size=(1, rnn_units), bias=True
        )
        self.node_feature_dim = rnn_units

    def forward(self, source: Tensor) -> tuple[Tensor, Tensor]:
        """``source [B, L, N, C]`` -> ``(hidden [B, N, rnn_units], pred [B, T, N, 1])``."""
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)
        hidden = output[:, -1, :, :]
        last = output[:, -1:, :, :]
        prediction = self.end_conv(last)  # [B, T*C, N, 1]
        prediction = prediction.squeeze(-1).reshape(
            -1, self.horizon, self.output_dim, self.num_nodes
        )
        prediction = prediction.permute(0, 1, 3, 2).contiguous()
        return hidden, prediction


class AGCRNAdapter(TorchBackboneAdapter):
    def __init__(
        self,
        cfg: BackboneConfig,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        steps_per_day: int = 288,
    ) -> None:
        params = cfg.params
        model = AGCRNArch(
            num_nodes=num_nodes,
            input_dim=1,
            rnn_units=int(params["rnn_units"]),
            output_dim=1,
            horizon=out_steps,
            num_layers=int(params["num_layers"]),
            cheb_k=int(params["cheb_k"]),
            embed_dim=int(params["embed_dim"]),
        )
        super().__init__(
            model=model,
            mode=cfg.mode,
            partial_patterns=("end_conv",),
            node_feature_dim=model.node_feature_dim,
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
        )

    def forward_features(self, inp: ModelInput, graph: GraphContext) -> tuple[Tensor, Tensor]:
        x = stack_features(inp, ("flow",))
        return self.model(x)


__all__ = ["AGCRNArch", "AGCRNAdapter", "AVWGCN", "AGCRNCell", "AVWDCRNN"]
