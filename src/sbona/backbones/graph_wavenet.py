"""Graph WaveNet backbone adapter.

Upstream: https://github.com/nnzhan/Graph-WaveNet, commit
``6b162e80c59a1d494809252eca055cff93dc66b1`` (MIT licence).

The model is a faithful re-implementation of ``model.py`` with one *documented*
compatibility patch: the upstream ``gate_convs`` are declared as ``Conv1d``
while they receive a 4-D ``[B, C, N, L]`` tensor.  Modern PyTorch rejects that,
and the BasicTS port of the same architecture already uses ``Conv2d``; we do
the same and record it in ``third_party/BACKBONE_CONTRACTS.md``.

Graph orientation
-----------------
``nconv`` computes ``einsum('ncvl,vw->ncwl')``, i.e. ``out[v] = sum_w A[v, w]
x[w]``.  The contract's diffusion operator is the other orientation
(``A[i, j]`` = row ``i`` aggregates column ``j``), so the adapter feeds
``A.T`` as the first support and ``A`` as the second — the same
``doubletransition`` pair the upstream training script builds from
``util.load_adj``.

Calendar
--------
Upstream uses ``in_dim=2`` (flow, time-of-day) for METR-LA.  We keep that:
``[flow_z, tod]``.  The weekday channel is *not* part of the published
Graph WaveNet input and enabling it would deviate from the paper.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sbona.backbones.base import BackboneError
from sbona.backbones.common import TorchBackboneAdapter, graph_supports, stack_features
from sbona.config import BackboneConfig
from sbona.contracts import GraphContext, ModelInput


class NConv(nn.Module):
    """``out[v] = sum_w A[v, w] * x[w]`` applied per ``(batch, channel, step)``."""

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        return torch.einsum("ncvl,vw->ncwl", (x, adjacency)).contiguous()


class GCN(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        dropout: float,
        support_len: int = 3,
        order: int = 2,
    ) -> None:
        super().__init__()
        self.nconv = NConv()
        self.mlp = nn.Conv2d((order * support_len + 1) * c_in, c_out, kernel_size=(1, 1))
        self.dropout = dropout
        self.order = order
        self.support_len = support_len

    def forward(self, x: Tensor, supports: Sequence[Tensor]) -> Tensor:
        if len(supports) != self.support_len:
            raise BackboneError(
                f"Graph-WaveNet GCN expects {self.support_len} supports, got {len(supports)}"
            )
        out = [x]
        for adjacency in supports:
            x1 = self.nconv(x, adjacency.to(x.device))
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, adjacency.to(x.device))
                out.append(x2)
                x1 = x2
        hidden = torch.cat(out, dim=1)
        hidden = self.mlp(hidden)
        return F.dropout(hidden, self.dropout, training=self.training)


class GraphWaveNetArch(nn.Module):
    """Upstream ``gwnet``; returns ``(node_hidden, prediction)`` in one pass."""

    def __init__(
        self,
        num_nodes: int,
        in_dim: int = 2,
        out_dim: int = 12,
        dropout: float = 0.3,
        support_len: int = 2,
        gcn_order: int = 2,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
        add_adaptive_adj: bool = True,
        adaptive_rank: int = 10,
    ) -> None:
        super().__init__()
        if support_len < 1:
            raise BackboneError("Graph-WaveNet needs at least one support matrix")
        if gcn_order < 1:
            raise BackboneError("Graph-WaveNet gcn_order must be >= 1")
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.add_adaptive_adj = add_adaptive_adj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))
        self.supports_len = support_len + int(add_adaptive_adj)
        if add_adaptive_adj:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, adaptive_rank))
            self.nodevec2 = nn.Parameter(torch.randn(adaptive_rank, num_nodes))

        receptive_field = 1
        dilation = 1
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            dilation = 1
            for _ in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=dilation,
                    )
                )
                # Upstream declares Conv1d here; Conv2d is the documented patch.
                self.gate_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1))
                )
                self.skip_convs.append(
                    nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1))
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(
                    GCN(
                        dilation_channels,
                        residual_channels,
                        dropout,
                        support_len=self.supports_len,
                        order=gcn_order,
                    )
                )
        self.receptive_field = receptive_field
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1))
        self.node_feature_dim = end_channels

    def forward(self, x: Tensor, supports: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        """``x [B, C, N, L]`` -> ``(hidden [B, N, end_channels], pred [B, T, N, 1])``."""
        in_len = x.shape[-1]
        if in_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - in_len, 0, 0, 0))
        x = self.start_conv(x)
        skip: Tensor | None = None

        current_supports: list[Tensor] = list(supports)
        if self.add_adaptive_adj:
            adaptive = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            current_supports = current_supports + [adaptive]

        for index in range(self.blocks * self.layers):
            residual = x
            filt = torch.tanh(self.filter_convs[index](residual))
            gate = torch.sigmoid(self.gate_convs[index](residual))
            x = filt * gate

            s = self.skip_convs[index](x)
            if skip is None:
                skip = s
            else:
                skip = s + skip[:, :, :, -s.size(3) :]

            x = self.gconv[index](x, current_supports)
            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[index](x)

        if skip is None:  # pragma: no cover - blocks * layers >= 1 by validation
            raise BackboneError("Graph-WaveNet produced no skip connection")
        hidden = F.relu(self.end_conv_1(F.relu(skip)))
        prediction = self.end_conv_2(hidden)
        # [B, end_channels, N, 1] -> [B, N, end_channels]
        node_hidden = hidden.squeeze(-1).permute(0, 2, 1).contiguous()
        # [B, T, N, 1]; ``prediction`` already is [B, out_dim, N, 1].
        pred = prediction.squeeze(-1).unsqueeze(-1).contiguous()
        return node_hidden, pred


class GraphWaveNetAdapter(TorchBackboneAdapter):
    def __init__(
        self,
        cfg: BackboneConfig,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        steps_per_day: int = 288,
    ) -> None:
        params = cfg.params
        model = GraphWaveNetArch(
            num_nodes=num_nodes,
            in_dim=2,
            out_dim=out_steps,
            dropout=float(params["dropout"]),
            support_len=2,
            gcn_order=int(params["gcn_order"]),
            residual_channels=int(params["residual_channels"]),
            dilation_channels=int(params["dilation_channels"]),
            skip_channels=int(params["skip_channels"]),
            end_channels=int(params["end_channels"]),
            kernel_size=int(params["kernel_size"]),
            blocks=int(params["blocks"]),
            layers=int(params["layers"]),
            add_adaptive_adj=bool(params["add_adaptive_adj"]),
            adaptive_rank=int(params["adaptive_rank"]),
        )
        super().__init__(
            model=model,
            mode=cfg.mode,
            partial_patterns=("end_conv_1", "end_conv_2"),
            node_feature_dim=model.node_feature_dim,
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
        )
        self.steps_per_day = steps_per_day

    def forward_features(self, inp: ModelInput, graph: GraphContext) -> tuple[Tensor, Tensor]:
        supports = self._graph_cache.get(
            graph, "supports", lambda: graph_supports(graph, symmetric=True)
        )
        x = stack_features(inp, ("flow", "tod"))
        x = x.permute(0, 3, 2, 1).contiguous()  # [B, C, N, L]
        return self.model(x, supports)


__all__ = ["GraphWaveNetArch", "GraphWaveNetAdapter", "GCN", "NConv"]
