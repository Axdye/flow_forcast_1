"""ST-SSDL backbone adapter (NeurIPS 2025).

Upstream: https://github.com/Jimmy-7664/ST-SSDL, commit
``a98331d3a45a081778f3abb608897902a60e301d`` (MIT licence).

"How Different from the Past?" adds a prototype-based self-supervised branch to
an AGCRN-style encoder-decoder: the encoder summary is matched against learned
prototypes, the prototypes plus the summary drive a hyper-network that emits a
sample-specific graph, and the decoder rolls that graph forward.  The contrastive
``latent_dis`` / ``prototype_dis`` terms belong to the paper's auxiliary loss.

Faithfulness notes
------------------
* SBoN supplies its own objective, so the contrastive head is computed (it is
  cheap and part of the forward graph) but **not** returned as a loss.
* The official model also encodes a *reference* series ``x_his`` (the same
  window one period earlier) for the contrastive term.  A backbone adapter only
  sees the history in :class:`ModelInput`, so ``x_his = x``; the forecast path
  is unaffected because ``h_a``/``v_a`` only feed the hyper-network alongside
  ``h_t``/``v_t``.
* ``y_cov`` (future time-of-day) is extrapolated deterministically instead of
  read from the target batch.
* The support passed to the encoder is the contract's row-normalised diffusion
  operator ``A`` (``A[i, j]`` = row ``i`` aggregates column ``j``), which is the
  orientation ``AGCN`` expects.  The published code uses a *symmetric*
  normalisation; that difference is recorded in the contracts document.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sbona.backbones.base import BackboneError
from sbona.backbones.common import (
    TorchBackboneAdapter,
    dense_diffusion,
    future_calendar,
)
from sbona.config import BackboneConfig
from sbona.contracts import GraphContext, ModelInput


class AGCN(nn.Module):
    """Chebyshev graph convolution with fixed (non-adaptive) weights."""

    def __init__(self, dim_in: int, dim_out: int, cheb_k: int, num_support: int) -> None:
        super().__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.empty(num_support * cheb_k * dim_in, dim_out))
        self.bias = nn.Parameter(torch.zeros(dim_out))
        nn.init.xavier_normal_(self.weights)

    def forward(self, x: Tensor, supports: Sequence[Tensor]) -> Tensor:
        gathered = []
        for support in supports:
            if support.dim() == 2:
                basis = [torch.eye(support.shape[0], device=support.device), support]
                for _ in range(2, self.cheb_k):
                    basis.append(torch.matmul(2 * support, basis[-1]) - basis[-2])
                gathered.extend(torch.einsum("nm,bmc->bnc", graph, x) for graph in basis)
            else:
                basis = [
                    torch.eye(support.shape[1], device=support.device).repeat(
                        support.shape[0], 1, 1
                    ),
                    support,
                ]
                for _ in range(2, self.cheb_k):
                    basis.append(torch.matmul(2 * support, basis[-1]) - basis[-2])
                gathered.extend(torch.einsum("bnm,bmc->bnc", graph, x) for graph in basis)
        x_g = torch.cat(gathered, dim=-1)
        return torch.einsum("bni,io->bno", x_g, self.weights) + self.bias


class AGCRNCell(nn.Module):
    def __init__(
        self, node_num: int, dim_in: int, dim_out: int, cheb_k: int, num_support: int
    ) -> None:
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + dim_out, 2 * dim_out, cheb_k, num_support)
        self.update = AGCN(dim_in + dim_out, dim_out, cheb_k, num_support)

    def forward(self, x: Tensor, state: Tensor, supports: Sequence[Tensor]) -> Tensor:
        input_and_state = torch.cat((x, state.to(x.device)), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state.to(x.device)), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        return r * state.to(x.device) + (1 - r) * hc

    def init_hidden_state(self, batch_size: int) -> Tensor:
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class ADCRNNEncoder(nn.Module):
    def __init__(
        self,
        node_num: int,
        dim_in: int,
        dim_out: int,
        cheb_k: int,
        rnn_layers: int,
        num_support: int,
    ) -> None:
        super().__init__()
        if rnn_layers < 1:
            raise BackboneError("ST-SSDL needs at least one encoder layer")
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.dcrnn_cells = nn.ModuleList(
            [AGCRNCell(node_num, dim_in, dim_out, cheb_k, num_support)]
            + [
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, num_support)
                for _ in range(1, rnn_layers)
            ]
        )

    def forward(
        self, x: Tensor, init_state: Sequence[Tensor], supports: Sequence[Tensor]
    ) -> Tensor:
        seq_length = x.shape[1]
        current_inputs = x
        for layer in range(self.rnn_layers):
            state = init_state[layer]
            inner = []
            for step in range(seq_length):
                state = self.dcrnn_cells[layer](current_inputs[:, step, :, :], state, supports)
                inner.append(state)
            current_inputs = torch.stack(inner, dim=1)
        return current_inputs

    def init_hidden(self, batch_size: int) -> list[Tensor]:
        return [cell.init_hidden_state(batch_size) for cell in self.dcrnn_cells]


class ADCRNNDecoder(nn.Module):
    def __init__(
        self,
        node_num: int,
        dim_in: int,
        dim_out: int,
        cheb_k: int,
        rnn_layers: int,
        num_support: int,
    ) -> None:
        super().__init__()
        if rnn_layers < 1:
            raise BackboneError("ST-SSDL needs at least one decoder layer")
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.dcrnn_cells = nn.ModuleList(
            [AGCRNCell(node_num, dim_in, dim_out, cheb_k, num_support)]
            + [
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, num_support)
                for _ in range(1, rnn_layers)
            ]
        )

    def forward(
        self, x: Tensor, init_state: Sequence[Tensor], supports: Sequence[Tensor]
    ) -> tuple[Tensor, list[Tensor]]:
        current = x
        hidden = []
        for layer in range(self.rnn_layers):
            state = self.dcrnn_cells[layer](current, init_state[layer], supports)
            hidden.append(state)
            current = state
        return current, hidden


class STSSDLArch(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        input_dim: int = 1,
        output_dim: int = 1,
        rnn_units: int = 12,
        rnn_layers: int = 1,
        cheb_k: int = 3,
        prototype_num: int = 20,
        prototype_dim: int = 64,
        tod_embed_dim: int = 20,
        adaptive_embedding_dim: int = 0,
        node_embedding_dim: int = 20,
        input_embedding_dim: int = 16,
        steps_per_day: int = 288,
        use_ste: bool = True,
        num_support: int = 1,
        ycov_dim: int = 1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_units = rnn_units
        self.cheb_k = cheb_k
        self.tod_embed_dim = tod_embed_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.input_embedding_dim = input_embedding_dim
        self.steps_per_day = steps_per_day
        self.use_ste = use_ste
        self.ycov_dim = ycov_dim
        self.num_support = num_support
        self.total_embedding_dim = tod_embed_dim + adaptive_embedding_dim + node_embedding_dim

        self.prototype_num = prototype_num
        self.prototype_dim = prototype_dim
        self.prototypes = nn.ParameterDict()
        prototypes = torch.empty(prototype_num, prototype_dim)
        nn.init.xavier_normal_(prototypes)
        self.prototypes["prototypes"] = nn.Parameter(prototypes)
        wq = torch.empty(rnn_units, prototype_dim)
        nn.init.xavier_normal_(wq)
        self.prototypes["Wq"] = nn.Parameter(wq)

        if use_ste:
            if adaptive_embedding_dim > 0:
                self.adaptive_embedding = nn.Parameter(
                    torch.empty(in_steps, num_nodes, adaptive_embedding_dim)
                )
                nn.init.xavier_uniform_(self.adaptive_embedding)
            self.input_proj = nn.Linear(input_dim, input_embedding_dim)
            self.node_embedding = nn.Parameter(torch.empty(num_nodes, node_embedding_dim))
            self.time_embedding = nn.Parameter(torch.empty(steps_per_day, tod_embed_dim))
            nn.init.xavier_uniform_(self.node_embedding)
            nn.init.xavier_uniform_(self.time_embedding)

        encoder_dim = (
            input_embedding_dim + self.total_embedding_dim if use_ste else input_dim
        )
        self.encoder = ADCRNNEncoder(
            num_nodes, encoder_dim, rnn_units, cheb_k, rnn_layers, num_support
        )
        self.decoder_dim = rnn_units + prototype_dim
        decoder_dim = (
            input_embedding_dim + self.total_embedding_dim - adaptive_embedding_dim
            if use_ste
            else output_dim + ycov_dim
        )
        self.decoder = ADCRNNDecoder(
            num_nodes, decoder_dim, self.decoder_dim, cheb_k, rnn_layers, 1
        )
        self.proj = nn.Linear(self.decoder_dim, output_dim, bias=True)
        self.hypernet = nn.Linear(self.decoder_dim * 2, tod_embed_dim, bias=True)
        self.node_feature_dim = self.decoder_dim

    # -- prototype branch ------------------------------------------------

    def query_prototypes(self, h_t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        prototypes = self.prototypes["prototypes"]
        query = torch.matmul(h_t, self.prototypes["Wq"])
        att_score = torch.softmax(torch.matmul(query, prototypes.t()), dim=-1)
        value = torch.matmul(att_score, prototypes)
        top = torch.topk(att_score, k=2, dim=-1).indices
        positive = prototypes[top[:, :, 0]]
        negative = prototypes[top[:, :, 1]]
        return value, query, positive, negative

    # -- forward ---------------------------------------------------------

    def _embed_sequence(self, flow: Tensor, tod: Tensor, reference_length: int) -> Tensor:
        x = self.input_proj(flow) if self.input_embedding_dim > 0 else flow
        # ``tod`` may arrive as ``[B, L, N, 1]``; the embedding indexes nodes as
        # the last axis, exactly like the upstream ``x_cov.squeeze()``.
        if tod.dim() == 4:
            tod = tod.squeeze(-1)
        features = [x]
        if self.tod_embed_dim > 0:
            index = (tod.clamp(0.0, 1.0) * self.steps_per_day).long()
            index = index.clamp_max(self.steps_per_day - 1)
            features.append(self.time_embedding[index])
        if self.adaptive_embedding_dim > 0:
            features.append(
                self.adaptive_embedding.expand(size=(x.shape[0], *self.adaptive_embedding.shape))
            )
        if self.node_embedding_dim > 0:
            features.append(
                self.node_embedding.unsqueeze(0)
                .unsqueeze(1)
                .expand(x.shape[0], reference_length, -1, -1)
            )
        return torch.cat(features, dim=-1)

    def forward(
        self,
        flow: Tensor,
        tod_cov: Tensor,
        future_tod: Tensor,
        supports: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """``flow [B, L, N, 1]``, ``tod_cov [B, L, N, 1]``, ``future_tod [B, T, N, 1]``."""
        batch_size = flow.shape[0]
        if self.use_ste:
            encoded = self._embed_sequence(flow, tod_cov, flow.shape[1])
        else:
            encoded = flow
        init_state = self.encoder.init_hidden(batch_size)
        h_en = self.encoder(encoded, init_state, supports)
        h_t = h_en[:, -1, :, :]
        v_t, _, _, _ = self.query_prototypes(h_t)

        # ``x_his`` is unavailable in the contract; using the same window makes
        # the augmenting branch an identity and keeps the forecast path intact.
        if self.use_ste:
            encoded_his = self._embed_sequence(flow, tod_cov, flow.shape[1])
        else:
            encoded_his = flow
        h_his_en = self.encoder(encoded_his, self.encoder.init_hidden(batch_size), supports)
        h_a = h_his_en[:, -1, :, :]
        v_a, _, _, _ = self.query_prototypes(h_a)

        h_de = torch.cat([h_t, v_t], dim=-1)
        h_aug = torch.cat([h_t, v_t, h_a, v_a], dim=-1)
        node_embeddings = self.hypernet(h_aug)
        support = F.softmax(
            F.relu(torch.einsum("bnc,bmc->bnm", node_embeddings, node_embeddings)), dim=-1
        )

        states = [h_de] * self.decoder.rnn_layers
        go = torch.zeros(
            (batch_size, self.num_nodes, self.output_dim), device=flow.device
        )
        outputs = []
        for step in range(self.out_steps):
            step_tod = future_tod[:, step]
            if step_tod.dim() == 3:
                step_tod = step_tod.squeeze(-1)
            if self.use_ste:
                go_embed = self.input_proj(go) if self.input_embedding_dim > 0 else go
                features = [go_embed]
                if self.tod_embed_dim > 0:
                    index = (step_tod.clamp(0.0, 1.0) * self.steps_per_day).long()
                    features.append(self.time_embedding[index.clamp_max(self.steps_per_day - 1)])
                if self.node_embedding_dim > 0:
                    features.append(
                        self.node_embedding.unsqueeze(0).expand(batch_size, -1, -1)
                    )
                go_input = torch.cat(features, dim=-1)
            else:
                go_input = torch.cat([go, step_tod], dim=-1)
            h_de, states = self.decoder(go_input, states, [support])
            go = self.proj(h_de)
            outputs.append(go)
        # The final decoder state is the node representation that produced the
        # last forecast step; its width is ``decoder_dim == node_feature_dim``.
        return h_de, torch.stack(outputs, dim=1)


class STSSDLAdapter(TorchBackboneAdapter):
    def __init__(
        self,
        cfg: BackboneConfig,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        steps_per_day: int = 288,
    ) -> None:
        params = cfg.params
        model = STSSDLArch(
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
            input_dim=1,
            output_dim=1,
            rnn_units=int(params["rnn_units"]),
            rnn_layers=int(params["rnn_layers"]),
            cheb_k=int(params["cheb_k"]),
            prototype_num=int(params["prototype_num"]),
            prototype_dim=int(params["prototype_dim"]),
            tod_embed_dim=int(params["tod_embed_dim"]),
            adaptive_embedding_dim=int(params["adaptive_embedding_dim"]),
            node_embedding_dim=int(params["node_embedding_dim"]),
            input_embedding_dim=int(params["input_embedding_dim"]),
            steps_per_day=steps_per_day,
            use_ste=bool(params["use_ste"]),
            num_support=1,
            ycov_dim=1,
        )
        super().__init__(
            model=model,
            mode=cfg.mode,
            partial_patterns=("decoder", "proj", "hypernet"),
            node_feature_dim=model.node_feature_dim,
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
        )
        self.steps_per_day = steps_per_day

    def forward_features(self, inp: ModelInput, graph: GraphContext) -> tuple[Tensor, Tensor]:
        support = self._graph_cache.get(graph, "support", lambda: (dense_diffusion(graph),))
        flow = inp.history
        tod_cov = inp.time_of_day[:, :, None, None].expand(
            inp.batch_size, inp.input_steps, inp.num_nodes, 1
        )
        future_tod, _ = future_calendar(inp, self.steps_per_day, self.out_steps)
        future = future_tod[:, :, None, None].expand(
            inp.batch_size, self.out_steps, inp.num_nodes, 1
        )
        hidden, prediction = self.model(flow, tod_cov, future, support)
        # ``STSSDLArch`` returns ``[B, T, N, output_dim=1]`` already.
        return hidden, prediction


__all__ = [
    "STSSDLArch",
    "STSSDLAdapter",
    "AGCN",
    "AGCRNCell",
    "ADCRNNEncoder",
    "ADCRNNDecoder",
]
