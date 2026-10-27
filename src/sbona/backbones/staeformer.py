"""STAEformer backbone adapter.

The architecture below follows the published STAEformer model
(https://github.com/XDZhelheim/STAEformer, commit
``fc49d39b2f1a8e3cf37b6289d7240680e1690f3f``) and is re-implemented here so
the tensors that the adapter needs are returned by a *single* forward pass.

The upstream repository did not contain a LICENSE file at the pinned commit;
the vendoring/licence decision remains ``pending`` in ``third_party/LOCK.md``.
This implementation is therefore an attributed re-implementation for the
research prototype, not a claimed redistribution of upstream source.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sbona.backbones.base import BackboneWrapper, check_input_against_graph
from sbona.config import BackboneConfig
from sbona.contracts import BackboneOutput, GraphContext, ModelInput


class AttentionLayer(nn.Module):
    """Attention across ``-2`` dim (``-1`` is the model dim)."""

    def __init__(self, model_dim: int, num_heads: int = 8, mask: bool = False) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)
        attn_score = (query @ key) / self.head_dim**0.5

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()
            attn_score = attn_score.masked_fill(~mask, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        return self.out_proj(out)


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        mask: bool = False,
    ) -> None:
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, dim: int = -2) -> Tensor:
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        return out.transpose(dim, -2)


class STAEformer(nn.Module):
    """Upstream architecture with an explicit ``forward_features`` hook."""

    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
        input_dim: int = 1,
        output_dim: int = 1,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        spatial_embedding_dim: int = 0,
        adaptive_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_mixed_proj: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(in_steps, num_nodes, adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)

        if use_mixed_proj:
            self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(node_hidden [B,N,D], base_prediction [B,T,N,1])``."""
        batch_size = x.shape[0]
        tod = None
        dow = None
        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)
        features = [x]
        if self.tod_embedding_dim > 0:
            features.append(self.tod_embedding((tod * self.steps_per_day).long()))
        if self.dow_embedding_dim > 0:
            features.append(self.dow_embedding(dow.long()))
        if self.spatial_embedding_dim > 0:
            features.append(
                self.node_emb.expand(batch_size, self.in_steps, *self.node_emb.shape)
            )
        if self.adaptive_embedding_dim > 0:
            features.append(
                self.adaptive_embedding.expand(
                    size=(batch_size, *self.adaptive_embedding.shape)
                )
            )
        x = torch.cat(features, dim=-1)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        # x: [B, L, N, model_dim]
        node_hidden = x[:, -1, :, :]

        if self.use_mixed_proj:
            out = x.transpose(1, 2)
            out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)
        else:
            out = x.transpose(1, 3)
            out = self.temporal_proj(out)
            out = self.output_proj(out.transpose(1, 3))
        return node_hidden, out

    def forward(self, x: Tensor) -> Tensor:
        _, prediction = self.forward_features(x)
        return prediction


class STAEformerAdapter(BackboneWrapper):
    """Adapter that feeds ``[flow_z, tod, dow]`` and extracts the last hidden step."""

    def __init__(self, cfg: BackboneConfig, num_nodes: int, in_steps: int, out_steps: int, steps_per_day: int = 288) -> None:
        model = STAEformer(
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
            steps_per_day=steps_per_day,
            input_dim=cfg.input_dim,
            output_dim=1,
            input_embedding_dim=cfg.input_embedding_dim,
            tod_embedding_dim=cfg.tod_embedding_dim,
            dow_embedding_dim=cfg.dow_embedding_dim,
            spatial_embedding_dim=cfg.spatial_embedding_dim,
            adaptive_embedding_dim=cfg.adaptive_embedding_dim,
            feed_forward_dim=cfg.feed_forward_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            use_mixed_proj=cfg.use_mixed_proj,
        )
        patterns: tuple[str, ...] = ("output_proj",)
        if cfg.mode == "partial":
            start = max(0, cfg.num_layers - cfg.partial_unfreeze_last_n_spatial)
            patterns = tuple(
                [f"attn_layers_s.{index}." for index in range(start, cfg.num_layers)]
                + ["output_proj"]
            )
        super().__init__(
            model=model,
            mode=cfg.mode,
            partial_patterns=patterns,
            node_feature_dim=model.model_dim,
        )
        self.in_steps = in_steps
        self.out_steps = out_steps

    def encode_and_predict(self, inp: ModelInput, graph: GraphContext) -> BackboneOutput:
        check_input_against_graph(inp, graph)
        if inp.num_nodes != self.model.num_nodes:
            raise ValueError(
                f"STAEformer built for {self.model.num_nodes} nodes, input has {inp.num_nodes}"
            )
        if inp.input_steps != self.in_steps:
            raise ValueError(
                f"STAEformer built for L={self.in_steps}, input has L={inp.input_steps}"
            )
        batch_size, length, nodes, _ = inp.history.shape
        tod = inp.time_of_day[:, :, None, None].expand(batch_size, length, nodes, 1)
        dow = inp.day_of_week.float()[:, :, None, None].expand(batch_size, length, nodes, 1)
        if self.model.dow_embedding_dim > 0:
            if bool((inp.day_of_week < 0).any()) or bool((inp.day_of_week > 6).any()):
                raise ValueError("day_of_week must be in 0..6 when the weekday embedding is enabled")
        x = torch.cat([inp.history, tod, dow], dim=-1)
        with self._context():
            hidden, prediction = self.model.forward_features(x)
        if prediction.shape[1] != self.out_steps:
            raise ValueError(
                f"STAEformer produced T={prediction.shape[1]}, expected {self.out_steps}"
            )
        return BackboneOutput(base_pred_z=prediction.float(), node_hidden=hidden.float())


__all__ = ["STAEformer", "STAEformerAdapter", "AttentionLayer", "SelfAttentionLayer"]
