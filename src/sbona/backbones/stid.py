"""STID backbone adapter.

Upstream: https://github.com/zezhishao/STID, commit
``e8b313bc591bdd0101a1619962c9b503e75127c0`` (Apache-2.0), and its BasicTS port
(``basicts`` v0.5.8, ``baselines/STID/arch/stid_arch.py``).

STID is the "no spatial path" control: it concatenates the flattened history,
a node identity embedding, and the *last* history step's time-of-day and
weekday embeddings, then applies residual MLPs.  It never reads the graph, so
any spatial effect in SBoN's residual decoder is attributable to the decoder,
not to the base.

Calendar
--------
Upstream indexes ``time_in_day_emb[(tod * 288)]`` and
``day_in_week_emb[(dow * day_of_week_size)]`` where ``dow`` is the BasicTS
fraction ``dow/7``.  The contract stores the weekday index, so the adapter uses
the index directly — the embeddings remain 7 rows, only the lookup convention
changes (a fraction ``k/7`` times 7 truncates to the same ``k``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sbona.backbones.common import TorchBackboneAdapter, stack_features
from sbona.config import BackboneConfig
from sbona.contracts import GraphContext, ModelInput


class MultiLayerPerceptron(nn.Module):
    """Residual 1x1 conv MLP (upstream ``mlp.py``)."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))
        self.fc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.fc2(self.drop(self.act(self.fc1(x))))
        return hidden + x


class STIDArch(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_len: int = 12,
        input_dim: int = 3,
        embed_dim: int = 32,
        output_len: int = 12,
        num_layer: int = 3,
        if_node: bool = True,
        node_dim: int = 32,
        if_time_in_day: bool = True,
        if_day_in_week: bool = True,
        temp_dim_tid: int = 32,
        temp_dim_diw: int = 32,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.input_len = input_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.num_layer = num_layer
        self.temp_dim_tid = temp_dim_tid
        self.temp_dim_diw = temp_dim_diw
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.if_spatial = if_node

        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(time_of_day_size, temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(day_of_week_size, temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        self.time_series_emb_layer = nn.Conv2d(
            input_dim * input_len, embed_dim, kernel_size=(1, 1), bias=True
        )
        self.hidden_dim = (
            embed_dim
            + node_dim * int(self.if_spatial)
            + temp_dim_tid * int(self.if_time_in_day)
            + temp_dim_diw * int(self.if_day_in_week)
        )
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim, dropout) for _ in range(num_layer)]
        )
        self.regression_layer = nn.Conv2d(
            self.hidden_dim, output_len, kernel_size=(1, 1), bias=True
        )
        self.node_feature_dim = self.hidden_dim

    def forward(self, history: Tensor) -> tuple[Tensor, Tensor]:
        """``history [B, L, N, C]`` with ``C >= 3`` -> ``(hidden, pred)``."""
        if history.shape[1] != self.input_len:
            raise ValueError(f"STID built for L={self.input_len}, got {history.shape[1]}")
        if history.shape[-1] < 3:
            raise ValueError("STID needs [flow, tod, dow] channels")
        batch_size, _, num_nodes, _ = history.shape
        input_data = history[..., : self.input_dim]
        if self.if_time_in_day:
            tod = input_data[:, -1, :, 1]
            time_in_day_emb = self.time_in_day_emb[
                (tod.clamp(0.0, 1.0) * self.time_of_day_size).long().clamp_max(self.time_of_day_size - 1)
            ]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            dow = history[:, -1, :, 2].long().clamp(0, self.day_of_week_size - 1)
            day_in_week_emb = self.day_in_week_emb[dow]
        else:
            day_in_week_emb = None

        flat = input_data.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
        time_series_emb = self.time_series_emb_layer(flat.transpose(1, 2).unsqueeze(-1))

        features = [time_series_emb]
        if self.if_spatial:
            features.append(
                self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
            )
        if time_in_day_emb is not None:
            features.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            features.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        hidden = self.encoder(torch.cat(features, dim=1))
        prediction = self.regression_layer(hidden)
        # [B, D, N, 1] -> [B, N, D]
        node_hidden = hidden.squeeze(-1).permute(0, 2, 1).contiguous()
        # ``regression_layer`` already returns [B, T, N, 1].
        pred = prediction.contiguous()
        return node_hidden, pred


class STIDAdapter(TorchBackboneAdapter):
    def __init__(
        self,
        cfg: BackboneConfig,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        steps_per_day: int = 288,
    ) -> None:
        params = cfg.params
        model = STIDArch(
            num_nodes=num_nodes,
            input_len=in_steps,
            input_dim=3,
            embed_dim=int(params["embed_dim"]),
            output_len=out_steps,
            num_layer=int(params["num_layers"]),
            if_node=bool(params["if_node"]),
            node_dim=int(params["node_dim"]),
            if_time_in_day=bool(params["if_time_in_day"]),
            if_day_in_week=bool(params["if_day_in_week"]),
            temp_dim_tid=int(params["temp_dim_tid"]),
            temp_dim_diw=int(params["temp_dim_diw"]),
            time_of_day_size=steps_per_day,
            day_of_week_size=7,
            dropout=float(params["dropout"]),
        )
        super().__init__(
            model=model,
            mode=cfg.mode,
            partial_patterns=("encoder", "regression_layer"),
            node_feature_dim=model.node_feature_dim,
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
        )

    def forward_features(self, inp: ModelInput, graph: GraphContext) -> tuple[Tensor, Tensor]:
        x = stack_features(inp, ("flow", "tod", "dow_index"))
        return self.model(x)


__all__ = ["STIDArch", "STIDAdapter", "MultiLayerPerceptron"]
