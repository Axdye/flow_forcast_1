"""D2STGNN backbone adapter.

Upstream: https://github.com/zezhishao/D2STGNN, commit
``82c2d38ba098a6aa87e8099c1ee8da58480692ec``.  The repository ships **no
licence file**; the vendoring decision is therefore ``pending`` in
``third_party/LOCK.md`` and this module is an attributed re-implementation for
the research prototype, not a redistribution of upstream source.

Structure (all of ``models/``): decoupling layers that split the signal into a
diffusion component (dynamic + static hidden graphs) and an inherent component
(GRU + multi-head self-attention), each with its own forecast branch.  The
forecast branches emit ``out_steps / gap`` hidden states and expand every state
into ``gap`` consecutive steps.

Graph
-----
``STLocalizedConv`` expects the upstream orientation ``out[v] = sum_w A[v, w]
x[w]`` and a ``doubletransition`` pair, exactly what ``util.load_adj`` returns
for PEMS08.  The adapter therefore feeds ``(A.T, A)`` from
``GraphContext.diffusion``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sbona.backbones.base import BackboneError
from sbona.backbones.common import TorchBackboneAdapter, graph_supports, stack_features
from sbona.config import BackboneConfig
from sbona.contracts import GraphContext, ModelInput

# ---------------------------------------------------------------------------
# Decoupling helpers
# ---------------------------------------------------------------------------


class ResidualDecomp(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.ac = nn.ReLU()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.ln(x - self.ac(y))


class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim: int, time_emb_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2 * node_emb_dim + time_emb_dim * 2, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_embedding_u: Tensor,
        node_embedding_d: Tensor,
        time_in_day_feat: Tensor,
        day_in_week_feat: Tensor,
        history_data: Tensor,
    ) -> Tensor:
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        feat = torch.cat(
            [
                time_in_day_feat,
                day_in_week_feat,
                node_embedding_u.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1, -1),
                node_embedding_d.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1, -1),
            ],
            dim=-1,
        )
        hidden = self.activation(self.fc1(feat))
        gate = torch.sigmoid(self.fc2(hidden))[:, -history_data.shape[1] :, :, :]
        return history_data * gate


# ---------------------------------------------------------------------------
# Dynamic graph construction
# ---------------------------------------------------------------------------


class DistanceFunction(nn.Module):
    def __init__(
        self, hidden_dim: int, node_dim: int, seq_length: int, time_emb_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.time_slot_emb_dim = hidden_dim
        self.input_seq_len = seq_length
        self.dropout = nn.Dropout(dropout)
        self.fc_ts_emb1 = nn.Linear(seq_length, hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ts_feat_dim = hidden_dim
        self.time_slot_embedding = nn.Linear(time_emb_dim, hidden_dim)
        self.all_feat_dim = self.ts_feat_dim + node_dim + time_emb_dim * 2
        self.WQ = nn.Linear(self.all_feat_dim, hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(hidden_dim * 2)

    def forward(
        self, x: Tensor, e_d: Tensor, e_u: Tensor, t_d: Tensor, d_w: Tensor
    ) -> list[Tensor]:
        t_d = t_d[:, -1, :, :]
        d_w = d_w[:, -1, :, :]
        series = x[:, :, :, 0].transpose(1, 2).contiguous()  # [B, N, L]
        batch_size, num_nodes, seq_len = series.shape
        flat = series.view(batch_size * num_nodes, seq_len)
        dy_feat = self.fc_ts_emb2(self.dropout(self.bn(F.relu(self.fc_ts_emb1(flat)))))
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)
        emb1 = e_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = e_u.unsqueeze(0).expand(batch_size, -1, -1)
        left = torch.cat([dy_feat, t_d, d_w, emb1], dim=-1)
        right = torch.cat([dy_feat, t_d, d_w, emb2], dim=-1)
        result: list[Tensor] = []
        for features in (left, right):
            q = self.WQ(features)
            k = self.WK(features)
            scores = torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
            result.append(torch.softmax(scores, dim=-1))
        return result


class GraphMask(nn.Module):
    """Multiplies candidate adjacencies by the predefined graphs."""

    def __init__(self, adjs: Sequence[Tensor]) -> None:
        super().__init__()
        self.register_buffer("mask", torch.stack([a.float() for a in adjs]), persistent=False)

    def forward(self, adjacency: list[Tensor]) -> list[Tensor]:
        result = []
        for index, matrix in enumerate(adjacency):
            base = self.mask[index] + torch.ones_like(self.mask[index]) * 1e-7
            result.append(base.to(matrix.device) * matrix)
        return result


class Normalizer(nn.Module):
    @staticmethod
    def _norm(graph: Tensor) -> Tensor:
        degree = torch.sum(graph, dim=2)
        inverse = 1.0 / degree
        inverse = torch.where(torch.isfinite(inverse), inverse, torch.zeros_like(inverse))
        return torch.bmm(torch.diag_embed(inverse), graph)

    def forward(self, adjacency: list[Tensor]) -> list[Tensor]:
        return [self._norm(matrix) for matrix in adjacency]


class MultiOrder(nn.Module):
    def __init__(self, order: int = 2) -> None:
        super().__init__()
        self.order = order

    def _multi_order(self, graph: Tensor) -> list[Tensor]:
        ordered = []
        current = graph
        mask = 1 - torch.eye(graph.shape[1], device=graph.device)
        ordered.append(current * mask)
        for _ in range(2, self.order + 1):
            current = torch.matmul(current, graph)
            ordered.append(current * mask)
        return ordered

    def forward(self, adjacency: list[Tensor]) -> list[list[Tensor]]:
        return [self._multi_order(matrix) for matrix in adjacency]


class DynamicGraphConstructor(nn.Module):
    def __init__(
        self,
        *,
        k_s: int,
        k_t: int,
        hidden_dim: int,
        node_dim: int,
        seq_length: int,
        time_emb_dim: int,
        dropout: float,
        adjs: Sequence[Tensor],
    ) -> None:
        super().__init__()
        self.k_s = k_s
        self.k_t = k_t
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.distance_function = DistanceFunction(
            hidden_dim, node_dim, seq_length, time_emb_dim, dropout
        )
        self.mask = GraphMask(adjs)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=k_s)

    def st_localization(self, graph_ordered: list[list[Tensor]]) -> list[Tensor]:
        local = []
        for modality in graph_ordered:
            for order_graph in modality:
                graph = order_graph.unsqueeze(-2).expand(-1, -1, self.k_t, -1)
                graph = graph.reshape(graph.shape[0], graph.shape[1], graph.shape[2] * graph.shape[3])
                local.append(graph)
        return local

    def forward(
        self,
        history_data: Tensor,
        node_embedding_d: Tensor,
        node_embedding_u: Tensor,
        time_in_day_feat: Tensor,
        day_in_week_feat: Tensor,
    ) -> list[Tensor]:
        distance = self.distance_function(
            history_data, node_embedding_d, node_embedding_u, time_in_day_feat, day_in_week_feat
        )
        distance = self.mask(distance)
        distance = self.normalizer(distance)
        return self.st_localization(self.multi_order(distance))


# ---------------------------------------------------------------------------
# Diffusion block
# ---------------------------------------------------------------------------


class STLocalizedConv(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        k_s: int,
        k_t: int,
        dropout: float,
        predefined_graphs: Sequence[Tensor],
    ) -> None:
        super().__init__()
        self.k_s = k_s
        self.k_t = k_t
        self.hidden_dim = hidden_dim
        self.use_dynamic_hidden_graph = True
        self.use_static_hidden_graph = True
        # Upstream hard-codes ``use_pre=False`` for every published config: the
        # road graph enters only through the dynamic-graph *mask*, and the
        # support count is ``(len(predefined) + 1) * k_s + 1``.
        self.use_predefined_graph = False
        self.support_len = len(predefined_graphs) + 2
        self.num_matric = (len(predefined_graphs) + 1) * self.k_s + 1
        self.dropout = nn.Dropout(dropout)
        self.predefined = self.get_graph(predefined_graphs)
        self.fc_list_updt = nn.Linear(k_t * hidden_dim, k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(hidden_dim * self.num_matric, hidden_dim)
        self.activation = nn.ReLU()

    def set_predefined(self, supports: Sequence[Tensor]) -> None:
        """Replace the placeholder road graphs once the real graph is known."""
        self.predefined = self.get_graph(supports)

    def gconv(self, support: list[Tensor], x_k: Tensor, x_0: Tensor) -> Tensor:
        out = [x_0]
        for graph in support:
            if graph.dim() != 2:
                graph = graph.unsqueeze(1)
            out.append(torch.matmul(graph, x_k))
        concatenated = torch.cat(out, dim=-1)
        return self.dropout(self.gcn_updt(concatenated))

    def get_graph(self, support: Sequence[Tensor]) -> list[Tensor]:
        ordered: list[Tensor] = []
        mask = 1 - torch.eye(support[0].shape[0], device=support[0].device)
        for graph in support:
            first = graph * mask
            ordered.append(first)
            for _ in range(2, self.k_s + 1):
                first = torch.matmul(graph, first)
                ordered.append(first * mask)
        local: list[Tensor] = []
        for graph in ordered:
            graph = graph.unsqueeze(-2).expand(-1, self.k_t, -1)
            local.append(graph.reshape(graph.shape[0], graph.shape[1] * graph.shape[2]))
        return local

    def forward(self, x: Tensor, dynamic_graph: list[Tensor], static_graph: Tensor) -> Tensor:
        x = x.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        batch_size, seq_len, num_nodes, kernel_size, num_feat = x.shape
        support: list[Tensor] = []
        if self.use_predefined_graph:
            support += list(self.predefined)
        if self.use_dynamic_hidden_graph:
            support += list(dynamic_graph)
        if self.use_static_hidden_graph:
            support += self.get_graph([static_graph])
        x = x.reshape(batch_size, seq_len, num_nodes, kernel_size * num_feat)
        out = self.activation(self.fc_list_updt(x))
        out = out.view(batch_size, seq_len, num_nodes, kernel_size, num_feat)
        x_0 = torch.mean(out, dim=-2)
        x_k = out.transpose(-3, -2).reshape(
            batch_size, seq_len, kernel_size * num_nodes, num_feat
        )
        return self.gconv(support, x_k, x_0)


class DiffusionForecast(nn.Module):
    def __init__(self, hidden_dim: int, forecast_hidden_dim: int, k_t: int, out_steps: int, gap: int):
        super().__init__()
        self.k_t = k_t
        self.out_steps = out_steps
        self.gap = gap
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)

    def forward(
        self,
        gated_history: Tensor,
        hidden_states: Tensor,
        localized_conv: STLocalizedConv,
        dynamic_graph: list[Tensor],
        static_graph: Tensor,
    ) -> Tensor:
        predict = [hidden_states[:, -1, :, :].unsqueeze(1)]
        for _ in range(int(self.out_steps / self.gap) - 1):
            window = predict[-self.k_t :]
            if len(window) < self.k_t:
                shortage = self.k_t - len(window)
                window = [gated_history[:, -shortage:, :, :]] + window
            predict.append(localized_conv(torch.cat(window, dim=1), dynamic_graph, static_graph))
        return self.forecast_fc(torch.cat(predict, dim=1))


class DiffusionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        forecast_hidden_dim: int,
        *,
        k_s: int,
        k_t: int,
        out_steps: int,
        gap: int,
        dropout: float,
        predefined_graphs: Sequence[Tensor],
    ) -> None:
        super().__init__()
        self.localized_st_conv = STLocalizedConv(
            hidden_dim,
            k_s=k_s,
            k_t=k_t,
            dropout=dropout,
            predefined_graphs=predefined_graphs,
        )
        self.forecast_branch = DiffusionForecast(
            hidden_dim, forecast_hidden_dim, k_t, out_steps, gap
        )
        self.backcast_branch = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp(hidden_dim)

    def forward(
        self,
        history_data: Tensor,
        gated_history_data: Tensor,
        dynamic_graph: list[Tensor],
        static_graph: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden_states = self.localized_st_conv(gated_history_data, dynamic_graph, static_graph)
        forecast_hidden = self.forecast_branch(
            gated_history_data, hidden_states, self.localized_st_conv, dynamic_graph, static_graph
        )
        backcast = self.backcast_branch(hidden_states)
        history_data = history_data[:, -backcast.shape[1] :, :, :]
        return self.residual_decompose(history_data, backcast), forecast_hidden


# ---------------------------------------------------------------------------
# Inherent block
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float | None, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe[: x.size(0)])


class RNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float | None) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, num_nodes, hidden_dim = x.shape
        x = x.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)
        state = torch.zeros_like(x[:, 0, :])
        output = []
        for step in range(x.shape[1]):
            state = self.gru_cell(x[:, step, :], state)
            output.append(state)
        return self.dropout(torch.stack(output, dim=0))


class TransformerLayer(nn.Module):
    def __init__(
        self, hidden_dim: int, num_heads: int = 4, dropout: float | None = None, bias: bool = True
    ) -> None:
        super().__init__()
        self.multi_head_self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout or 0.0, bias=bias
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, k: Tensor, v: Tensor) -> Tensor:
        hidden, _ = self.multi_head_self_attention(x, k, v)
        return self.dropout(hidden)


class InherentForecast(nn.Module):
    def __init__(self, hidden_dim: int, fk_dim: int, out_steps: int, gap: int) -> None:
        super().__init__()
        self.out_steps = out_steps
        self.gap = gap
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)

    def forward(
        self,
        x: Tensor,
        rnn_h: Tensor,
        z: Tensor,
        transformer_layer: TransformerLayer,
        rnn_layer: RNNLayer,
        pe: PositionalEncoding,
    ) -> Tensor:
        batch_size, _, num_nodes, num_feat = x.shape
        predict = [z[-1, :, :].unsqueeze(0)]
        for _ in range(int(self.out_steps / self.gap) - 1):
            state = rnn_layer.gru_cell(predict[-1][0], rnn_h[-1]).unsqueeze(0)
            rnn_h = torch.cat([rnn_h, state], dim=0)
            rnn_h = pe(rnn_h)
            predict.append(transformer_layer(state, k=rnn_h, v=rnn_h))
        stacked = torch.cat(predict, dim=0)
        stacked = stacked.reshape(-1, batch_size, num_nodes, num_feat).transpose(0, 1)
        return self.forecast_fc(stacked)


class InherentBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        forecast_hidden_dim: int,
        out_steps: int,
        gap: int,
        dropout: float,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        self.rnn_layer = RNNLayer(hidden_dim, dropout)
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads, dropout, True)
        self.forecast_block = InherentForecast(hidden_dim, forecast_hidden_dim, out_steps, gap)
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp(hidden_dim)

    def forward(self, hidden_inherent_signal: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, seq_len, num_nodes, num_feat = hidden_inherent_signal.shape
        rnn_states = self.rnn_layer(hidden_inherent_signal)
        rnn_states = self.pos_encoder(rnn_states)
        transformer_states = self.transformer_layer(rnn_states, rnn_states, rnn_states)
        forecast_hidden = self.forecast_block(
            hidden_inherent_signal,
            rnn_states,
            transformer_states,
            self.transformer_layer,
            self.rnn_layer,
            self.pos_encoder,
        )
        backcast = transformer_states.reshape(seq_len, batch_size, num_nodes, num_feat)
        backcast = self.backcast_fc(backcast.transpose(0, 1))
        return self.residual_decompose(hidden_inherent_signal, backcast), forecast_hidden


class DecoupleLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        fk_dim: int,
        node_emb_dim: int,
        time_emb_dim: int,
        k_s: int,
        k_t: int,
        out_steps: int,
        gap: int,
        dropout: float,
        predefined_graphs: Sequence[Tensor],
        num_heads: int,
    ) -> None:
        super().__init__()
        self.estimation_gate = EstimationGate(node_emb_dim, time_emb_dim, hidden_dim=64)
        self.dif_layer = DiffusionBlock(
            hidden_dim,
            fk_dim,
            k_s=k_s,
            k_t=k_t,
            out_steps=out_steps,
            gap=gap,
            dropout=dropout,
            predefined_graphs=predefined_graphs,
        )
        self.inh_layer = InherentBlock(
            hidden_dim,
            forecast_hidden_dim=fk_dim,
            out_steps=out_steps,
            gap=gap,
            dropout=dropout,
            num_heads=num_heads,
        )

    def forward(
        self,
        history_data: Tensor,
        dynamic_graph: list[Tensor],
        static_graph: Tensor,
        node_embedding_u: Tensor,
        node_embedding_d: Tensor,
        time_in_day_feat: Tensor,
        day_in_week_feat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        gated = self.estimation_gate(
            node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data
        )
        backcast_res, dif_forecast = self.dif_layer(
            history_data, gated, dynamic_graph, static_graph
        )
        backcast_res, inh_forecast = self.inh_layer(backcast_res)
        return backcast_res, dif_forecast, inh_forecast


class D2STGNNArch(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        num_feat: int = 1,
        num_hidden: int = 32,
        node_hidden: int = 10,
        time_emb_dim: int = 10,
        dropout: float = 0.1,
        k_t: int = 3,
        k_s: int = 2,
        gap: int = 3,
        num_layers: int = 5,
        forecast_dim: int = 256,
        output_hidden: int = 512,
        steps_per_day: int = 288,
        predefined_graphs: Sequence[Tensor] = (),
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if out_steps % gap != 0:
            raise BackboneError(f"D2STGNN gap={gap} must divide out_steps={out_steps}")
        if in_steps < k_t:
            raise BackboneError(f"D2STGNN needs in_steps >= k_t ({in_steps} < {k_t})")
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.num_feat = num_feat
        self.out_steps = out_steps
        self.gap = gap
        self.hidden_dim = num_hidden
        self.forecast_dim = forecast_dim
        self.num_layers = num_layers
        self.k_s = k_s
        self.k_t = k_t
        self.steps_per_day = steps_per_day

        self.embedding = nn.Linear(num_feat, num_hidden)
        self.time_in_day_emb = nn.Parameter(torch.empty(steps_per_day, time_emb_dim))
        self.day_in_week_emb = nn.Parameter(torch.empty(7, time_emb_dim))
        self.layers = nn.ModuleList(
            [
                DecoupleLayer(
                    num_hidden,
                    fk_dim=forecast_dim,
                    node_emb_dim=node_hidden,
                    time_emb_dim=time_emb_dim,
                    k_s=k_s,
                    k_t=k_t,
                    out_steps=out_steps,
                    gap=gap,
                    dropout=dropout,
                    predefined_graphs=predefined_graphs,
                    num_heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )
        self.dynamic_graph_constructor = DynamicGraphConstructor(
            k_s=k_s,
            k_t=k_t,
            hidden_dim=num_hidden,
            node_dim=node_hidden,
            seq_length=in_steps,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
            adjs=predefined_graphs,
        )
        self.node_emb_u = nn.Parameter(torch.empty(num_nodes, node_hidden))
        self.node_emb_d = nn.Parameter(torch.empty(num_nodes, node_hidden))
        self.out_fc_1 = nn.Linear(forecast_dim, output_hidden)
        self.out_fc_2 = nn.Linear(output_hidden, gap)
        self.node_feature_dim = forecast_dim
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.time_in_day_emb)
        nn.init.xavier_uniform_(self.day_in_week_emb)

    def _prepare_inputs(self, history_data: Tensor):
        node_emb_u = self.node_emb_u
        node_emb_d = self.node_emb_d
        tod = history_data[:, :, :, self.num_feat]
        dow = history_data[:, :, :, self.num_feat + 1].long().clamp(0, 6)
        tod_index = (tod.clamp(0.0, 1.0) * self.steps_per_day).long()
        tod_index = tod_index.clamp_max(self.steps_per_day - 1)
        time_in_day_feat = self.time_in_day_emb[tod_index]
        day_in_week_feat = self.day_in_week_emb[dow]
        signal = history_data[:, :, :, : self.num_feat]
        return signal, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat

    def forward(self, history_data: Tensor) -> tuple[Tensor, Tensor]:
        """``history_data [B, L, N, 1+2]`` -> ``(hidden, pred [B, T, N, 1])``."""
        if history_data.shape[-1] < self.num_feat + 2:
            raise BackboneError("D2STGNN needs [flow, tod, dow] channels")
        signal, emb_u, emb_d, tod_feat, dow_feat = self._prepare_inputs(history_data)
        static_graph = F.softmax(F.relu(torch.mm(emb_u, emb_d.t())), dim=1)
        dynamic_graph = self.dynamic_graph_constructor(
            history_data=signal,
            node_embedding_d=emb_d,
            node_embedding_u=emb_u,
            time_in_day_feat=tod_feat,
            day_in_week_feat=dow_feat,
        )
        state = self.embedding(signal)
        dif_hidden: list[Tensor] = []
        inh_hidden: list[Tensor] = []
        for layer in self.layers:
            state, dif_forecast, inh_forecast = layer(
                state, dynamic_graph, static_graph, emb_u, emb_d, tod_feat, dow_feat
            )
            dif_hidden.append(dif_forecast)
            inh_hidden.append(inh_forecast)
        forecast_hidden = sum(dif_hidden) + sum(inh_hidden)
        node_hidden = forecast_hidden[:, -1].contiguous()
        forecast = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1
        )
        return node_hidden, forecast.permute(0, 2, 1).unsqueeze(-1).contiguous()


class D2STGNNAdapter(TorchBackboneAdapter):
    def __init__(
        self,
        cfg: BackboneConfig,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        steps_per_day: int = 288,
    ) -> None:
        params = cfg.params
        # ``predefined_graphs`` is filled on first forward: the graph only
        # exists at runtime, so the module is built with placeholder buffers of
        # the right shape and the real matrices are injected via ``set_graph``.
        model = D2STGNNArch(
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
            num_feat=1,
            num_hidden=int(params["num_hidden"]),
            node_hidden=int(params["node_hidden"]),
            time_emb_dim=int(params["time_emb_dim"]),
            dropout=float(params["dropout"]),
            k_t=int(params["k_t"]),
            k_s=int(params["k_s"]),
            gap=int(params["gap"]),
            num_layers=int(params["num_layers"]),
            forecast_dim=int(params["forecast_dim"]),
            output_hidden=int(params["output_hidden"]),
            steps_per_day=steps_per_day,
            predefined_graphs=(torch.zeros(num_nodes, num_nodes), torch.zeros(num_nodes, num_nodes)),
            num_heads=int(params["num_heads"]),
        )
        super().__init__(
            model=model,
            mode=cfg.mode,
            partial_patterns=("out_fc_1", "out_fc_2"),
            node_feature_dim=model.node_feature_dim,
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
        )

    def _bind_graph(self, graph: GraphContext) -> None:
        supports = self._graph_cache.get(
            graph, "supports", lambda: graph_supports(graph, symmetric=True)
        )
        device = self.model.node_emb_u.device
        moved = [support.to(device) for support in supports]
        for layer in self.model.layers:
            layer.dif_layer.localized_st_conv.set_predefined(moved)
        for index, support in enumerate(moved):
            self.model.dynamic_graph_constructor.mask.mask[index] = support

    def forward_features(self, inp: ModelInput, graph: GraphContext) -> tuple[Tensor, Tensor]:
        self._bind_graph(graph)
        x = stack_features(inp, ("flow", "tod", "dow_index"))
        return self.model(x)


__all__ = ["D2STGNNArch", "D2STGNNAdapter"]
