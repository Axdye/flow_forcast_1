"""HimNet backbone adapter.

Upstream: https://github.com/XDZhelheim/HimNet (KDD 2024), commit
``2a9aa44d1864f6fb60bdf9d14bafd6c1681e81a5``.  The repository ships **no
licence file**; the vendoring decision stays ``pending`` in
``third_party/LOCK.md`` and this module is an attributed re-implementation.

HimNet ("Heterogeneous Meta-parameters") runs two GCRU encoders whose graph
convolution weights are *generated* from node embeddings (``meta_axis="S"``)
and from the timestamp embedding (``meta_axis="T"``), adds both summaries,
builds a sample-specific adaptive graph from the resulting spatial-temporal
embedding, and decodes autoregressively.

Faithfulness notes (all recorded in ``reviews/EXTRA_WORK.md``)
-------------------------------------------------------------
* ``nn.init.xavier_normal_(nn.Parameter(...))`` returns ``None``, so the
  published code silently sets ``weights_pool`` / ``bias_pool`` /
  ``node_embedding`` to ``None``.  We register the parameters and initialise
  them in place, which is the behaviour the paper describes.
* The official decoder feed-forward input is ``[y_prev, y_cov]`` where
  ``y_cov`` are *future* time-of-day / weekday covariates.  Those are
  deterministic calendar values, so the adapter extrapolates them
  (:func:`sbona.backbones.common.future_calendar`) instead of reading targets.
* Teacher forcing is disabled: feeding ``labels[:, t]`` would leak the target
  into a forward pass, which the contract forbids.
* The road graph is **not** consumed; the supports are learned from node
  embeddings.  HimNet is therefore an adaptive-graph backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from sbona.backbones.common import TorchBackboneAdapter, future_calendar, stack_features
from sbona.config import BackboneConfig
from sbona.contracts import GraphContext, ModelInput


class HimGCN(nn.Module):
    """Chebyshev graph convolution whose weights come from meta-parameters."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cheb_k: int,
        embed_dim: int,
        meta_axis: str | None = None,
    ) -> None:
        super().__init__()
        self.cheb_k = cheb_k
        self.meta_axis = meta_axis.upper() if meta_axis else None
        if self.meta_axis:
            self.weights_pool = nn.Parameter(torch.empty(embed_dim, cheb_k * input_dim, output_dim))
            self.bias_pool = nn.Parameter(torch.empty(embed_dim, output_dim))
            nn.init.xavier_normal_(self.weights_pool)
            nn.init.zeros_(self.bias_pool)
        else:
            self.weights = nn.Parameter(torch.empty(cheb_k * input_dim, output_dim))
            self.bias = nn.Parameter(torch.zeros(output_dim))
            nn.init.xavier_normal_(self.weights)

    def forward(self, x: Tensor, support: Tensor, embeddings: Tensor) -> Tensor:
        if support.dim() == 2:
            graphs = [torch.eye(support.shape[0], device=support.device), support]
            for _ in range(2, self.cheb_k):
                graphs.append(torch.matmul(2 * support, graphs[-1]) - graphs[-2])
            gathered = [torch.einsum("nm,bmc->bnc", graph, x) for graph in graphs]
        elif support.dim() == 3:
            graphs = [
                torch.eye(support.shape[1], device=support.device).repeat(support.shape[0], 1, 1),
                support,
            ]
            for _ in range(2, self.cheb_k):
                graphs.append(torch.matmul(2 * support, graphs[-1]) - graphs[-2])
            gathered = [torch.einsum("bnm,bmc->bnc", graph, x) for graph in graphs]
        else:
            raise ValueError(f"support must be [N,N] or [B,N,N], got {tuple(support.shape)}")
        x_g = torch.cat(gathered, dim=-1)

        if self.meta_axis == "T":
            weights = torch.einsum("bd,dio->bio", embeddings, self.weights_pool)
            bias = torch.matmul(embeddings, self.bias_pool)
            return torch.einsum("bni,bio->bno", x_g, weights) + bias[:, None, :]
        if self.meta_axis == "S":
            weights = torch.einsum("nd,dio->nio", embeddings, self.weights_pool)
            bias = torch.matmul(embeddings, self.bias_pool)
            return torch.einsum("bni,nio->bno", x_g, weights) + bias
        if self.meta_axis == "ST":
            weights = torch.einsum("bnd,dio->bnio", embeddings, self.weights_pool)
            bias = torch.einsum("bnd,do->bno", embeddings, self.bias_pool)
            return torch.einsum("bni,bnio->bno", x_g, weights) + bias
        return torch.einsum("bni,io->bno", x_g, self.weights) + self.bias


class HimGCRU(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        cheb_k: int,
        embed_dim: int,
        meta_axis: str | None = "S",
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = output_dim
        self.gate = HimGCN(input_dim + output_dim, 2 * output_dim, cheb_k, embed_dim, meta_axis)
        self.update = HimGCN(input_dim + output_dim, output_dim, cheb_k, embed_dim, meta_axis)

    def forward(self, x: Tensor, state: Tensor, support: Tensor, embeddings: Tensor) -> Tensor:
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, support, embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, support, embeddings))
        return r * state + (1 - r) * hc

    def init_hidden_state(self, batch_size: int) -> Tensor:
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim)


class HimEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        cheb_k: int,
        num_layers: int,
        embed_dim: int,
        meta_axis: str = "S",
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [HimGCRU(num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis)]
            + [
                HimGCRU(num_nodes, output_dim, output_dim, cheb_k, embed_dim, meta_axis)
                for _ in range(1, num_layers)
            ]
        )

    def forward(self, x: Tensor, support: Tensor, embeddings: Tensor) -> Tensor:
        batch_size, in_steps = x.shape[0], x.shape[1]
        current = x
        for cell in self.cells:
            state = cell.init_hidden_state(batch_size).to(x.device)
            inner = []
            for step in range(in_steps):
                state = cell(current[:, step, :, :], state, support, embeddings)
                inner.append(state)
            current = torch.stack(inner, dim=1)
        return current


class HimDecoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        cheb_k: int,
        num_layers: int,
        embed_dim: int,
        meta_axis: str | None = "ST",
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [HimGCRU(num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis)]
            + [
                HimGCRU(num_nodes, output_dim, output_dim, cheb_k, embed_dim, meta_axis)
                for _ in range(1, num_layers)
            ]
        )

    def forward(
        self, x: Tensor, init_state: list[Tensor], support: Tensor, embeddings: Tensor
    ) -> tuple[Tensor, list[Tensor]]:
        current = x
        hidden = []
        for index in range(self.num_layers):
            state = self.cells[index](current, init_state[index], support, embeddings)
            hidden.append(state)
            current = state
        return current, hidden


class HimNetArch(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        input_dim: int = 3,
        output_dim: int = 1,
        hidden_dim: int = 96,
        num_layers: int = 1,
        cheb_k: int = 2,
        ycov_dim: int = 2,
        tod_embedding_dim: int = 10,
        dow_embedding_dim: int = 2,
        node_embedding_dim: int = 14,
        st_embedding_dim: int = 10,
        steps_per_day: int = 288,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.ycov_dim = ycov_dim
        self.steps_per_day = steps_per_day

        self.encoder_s = HimEncoder(
            num_nodes, input_dim, hidden_dim, cheb_k, num_layers, node_embedding_dim, meta_axis="S"
        )
        self.encoder_t = HimEncoder(
            num_nodes,
            input_dim,
            hidden_dim,
            cheb_k,
            num_layers,
            tod_embedding_dim + dow_embedding_dim,
            meta_axis="T",
        )
        self.decoder = HimDecoder(
            num_nodes, output_dim + ycov_dim, hidden_dim, cheb_k, num_layers, st_embedding_dim
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, node_embedding_dim))
        nn.init.xavier_normal_(self.node_embedding)
        self.st_proj = nn.Linear(hidden_dim, st_embedding_dim)
        self.node_feature_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, y_cov: Tensor
    ) -> tuple[Tensor, Tensor]:
        """``x [B, L, N, 3]``, ``y_cov [B, T, N, 2]`` -> ``(hidden, pred)``."""
        tod = x[:, -1, 0, 1]
        dow = x[:, -1, 0, 2].long().clamp(0, 6)
        tod_emb = self.tod_embedding(
            (tod.clamp(0.0, 1.0) * self.steps_per_day).long().clamp_max(self.steps_per_day - 1)
        )
        dow_emb = self.dow_embedding(dow)
        time_embedding = torch.cat([tod_emb, dow_emb], dim=-1)

        support = torch.softmax(torch.relu(self.node_embedding @ self.node_embedding.t()), dim=-1)
        h_s = self.encoder_s(x, support, self.node_embedding)
        h_t = self.encoder_t(x, support, time_embedding)
        h_last = (h_s + h_t)[:, -1, :, :]

        st_embedding = self.st_proj(h_last)
        support = torch.softmax(
            torch.relu(torch.einsum("bnc,bmc->bnm", st_embedding, st_embedding)), dim=-1
        )

        states = [h_last] * self.decoder.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        outputs = []
        for step in range(self.out_steps):
            h_de, states = self.decoder(
                torch.cat([go, y_cov[:, step, ...]], dim=-1), states, support, st_embedding
            )
            go = self.out_proj(h_de)
            outputs.append(go)
        prediction = torch.stack(outputs, dim=1)
        return h_last, prediction


class HimNetAdapter(TorchBackboneAdapter):
    def __init__(
        self,
        cfg: BackboneConfig,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        steps_per_day: int = 288,
    ) -> None:
        params = cfg.params
        model = HimNetArch(
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
            input_dim=3,
            output_dim=1,
            hidden_dim=int(params["hidden_dim"]),
            num_layers=int(params["num_layers"]),
            cheb_k=int(params["cheb_k"]),
            ycov_dim=2,
            tod_embedding_dim=int(params["tod_embedding_dim"]),
            dow_embedding_dim=int(params["dow_embedding_dim"]),
            node_embedding_dim=int(params["node_embedding_dim"]),
            st_embedding_dim=int(params["st_embedding_dim"]),
            steps_per_day=steps_per_day,
            dropout=float(params["dropout"]),
        )
        super().__init__(
            model=model,
            mode=cfg.mode,
            partial_patterns=("decoder", "out_proj"),
            node_feature_dim=model.node_feature_dim,
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
        )
        self.steps_per_day = steps_per_day

    def forward_features(self, inp: ModelInput, graph: GraphContext) -> tuple[Tensor, Tensor]:
        x = stack_features(inp, ("flow", "tod", "dow_index"))
        tod, dow = future_calendar(inp, self.steps_per_day, self.out_steps)
        batch, nodes = inp.batch_size, inp.num_nodes
        y_cov = torch.stack([tod, dow.float() / 7.0], dim=-1)
        y_cov = y_cov[:, :, None, :].expand(batch, self.out_steps, nodes, 2)
        hidden, prediction = self.model(x, y_cov)
        # ``HimNetArch`` already returns [B, T, N, output_dim=1].
        return hidden, prediction


__all__ = ["HimNetArch", "HimNetAdapter", "HimGCN", "HimGCRU", "HimEncoder", "HimDecoder"]
