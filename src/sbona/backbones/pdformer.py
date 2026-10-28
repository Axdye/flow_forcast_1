"""PDFormer backbone adapter.

Upstream: https://github.com/BUAABIGSCity/PDFormer (AAAI 2023), commit
``f8c8f6ad007a04fad3baee958b89504711852ce9`` (MIT licence).

PDFormer stacks "spatial-temporal transformer" blocks whose spatial attention
mixes three heads: a *geographic* head masked by shortest-path hops, a
*semantic* head masked by DTW similarity, and a *temporal* head.  Two of the
official inputs are dataset-preprocessing artefacts rather than model
parameters; both are handled explicitly here and recorded in
``reviews/EXTRA_WORK.md``:

``lap_mx``
    Laplacian positional encoding.  Computed exactly from the road graph
    (:func:`sbona.backbones.common.laplacian_pe`), so this part is faithful.
``sem_mask`` / ``pattern_keys``
    Dynamic-time-warping clusters and per-node prototypical patterns.  They
    require the full training series, which a backbone adapter never sees.  The
    semantic mask is therefore **disabled** (the semantic head attends densely,
    no invented prior) and ``pattern_keys`` is a **learnable** per-node
    parameter initialised sinusoidally instead of a frozen cluster centroid.
    Both choices are configurable-free and documented, never silent.

Calendar
--------
Upstream consumes ``[value, time_in_day, one_hot(weekday, 7)]`` which matches
``feature_dim=1`` plus ``ext_dim=8``.  ``time_in_day`` is converted to a minute
index (``1440`` slots) exactly like ``DataEmbedding``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sbona.backbones.base import BackboneError
from sbona.backbones.common import (
    TorchBackboneAdapter,
    dense_diffusion,
    hop_distance,
    laplacian_pe,
    stack_features,
)
from sbona.config import BackboneConfig
from sbona.contracts import GraphContext, ModelInput


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, norm_layer: type[nn.Module] | None = None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.token_embed(x))


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 100) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:, : x.size(1)].unsqueeze(2).expand_as(x).detach()


class LaplacianPE(nn.Module):
    def __init__(self, lape_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx: Tensor) -> Tensor:
        return self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)


class DataEmbedding(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        lape_dim: int,
        drop: float = 0.0,
        add_time_in_day: bool = True,
        add_day_in_week: bool = True,
    ) -> None:
        super().__init__()
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        if add_time_in_day:
            self.minute_size = 1440
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        if add_day_in_week:
            self.weekday_embedding = nn.Embedding(7, embed_dim)
        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: Tensor, lap_mx: Tensor) -> Tensor:
        origin_x = x
        encoded = self.value_embedding(origin_x[:, :, :, : self.feature_dim])
        encoded = encoded + self.position_encoding(encoded)
        if self.add_time_in_day:
            minute = (origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long()
            minute = minute.clamp(0, self.minute_size - 1)
            encoded = encoded + self.daytime_embedding(minute)
        if self.add_day_in_week:
            weekday = origin_x[
                :, :, :, self.feature_dim + 1 : self.feature_dim + 8
            ].argmax(dim=3)
            encoded = encoded + self.weekday_embedding(weekday)
        encoded = encoded + self.spatial_embedding(lap_mx)
        return self.dropout(encoded)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class STSelfAttention(nn.Module):
    """Geographic + semantic + temporal self-attention (upstream)."""

    def __init__(
        self,
        dim: int,
        geo_num_heads: int = 4,
        sem_num_heads: int = 2,
        t_num_heads: int = 2,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        total_heads = geo_num_heads + sem_num_heads + t_num_heads
        if dim % total_heads != 0:
            raise BackboneError(
                f"PDFormer embed_dim {dim} must be divisible by {total_heads} heads"
            )
        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.t_num_heads = t_num_heads
        self.head_dim = dim // total_heads
        self.scale = self.head_dim**-0.5
        self.geo_ratio = geo_num_heads / total_heads
        self.sem_ratio = sem_num_heads / total_heads
        self.t_ratio = 1.0 - self.geo_ratio - self.sem_ratio
        self.output_dim = output_dim

        self.pattern_q_linears = nn.ModuleList(
            [nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)]
        )
        self.pattern_k_linears = nn.ModuleList(
            [nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)]
        )
        self.pattern_v_linears = nn.ModuleList(
            [nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)]
        )

        self.geo_q_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_k_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_v_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_attn_drop = nn.Dropout(attn_drop)

        self.sem_q_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_k_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_v_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_attn_drop = nn.Dropout(attn_drop)

        self.t_q_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: Tensor,
        x_patterns: Tensor,
        pattern_keys: Tensor,
        geo_mask: Tensor | None = None,
        sem_mask: Tensor | None = None,
    ) -> Tensor:
        batch, steps, nodes, dim = x.shape
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(batch, nodes, steps, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(batch, nodes, steps, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(batch, nodes, steps, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_attn = ((t_q @ t_k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(
            batch, nodes, steps, int(dim * self.t_ratio)
        ).transpose(1, 2)

        geo_q = self.geo_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_k = self.geo_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        for index in range(self.output_dim):
            pattern_q = self.pattern_q_linears[index](x_patterns[..., index])
            pattern_k = self.pattern_k_linears[index](pattern_keys[..., index])
            pattern_v = self.pattern_v_linears[index](pattern_keys[..., index])
            pattern_attn = ((pattern_q @ pattern_k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            geo_k = geo_k + pattern_attn @ pattern_v
        geo_v = self.geo_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_q = geo_q.reshape(batch, steps, nodes, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_k = geo_k.reshape(batch, steps, nodes, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_v = geo_v.reshape(batch, steps, nodes, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale
        if geo_mask is not None:
            geo_attn = geo_attn.masked_fill(geo_mask, float("-inf"))
        geo_attn = geo_attn.softmax(dim=-1)
        geo_attn = self.geo_attn_drop(geo_attn)
        geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(
            batch, steps, nodes, int(dim * self.geo_ratio)
        )

        sem_q = self.sem_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_k = self.sem_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_v = self.sem_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_q = sem_q.reshape(batch, steps, nodes, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_k = sem_k.reshape(batch, steps, nodes, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_v = sem_v.reshape(batch, steps, nodes, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
        if sem_mask is not None:
            sem_attn = sem_attn.masked_fill(sem_mask, float("-inf"))
        sem_attn = sem_attn.softmax(dim=-1)
        sem_attn = self.sem_attn_drop(sem_attn)
        sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(
            batch, steps, nodes, int(dim * self.sem_ratio)
        )

        out = self.proj(torch.cat([t_x, geo_x, sem_x], dim=-1))
        return self.proj_drop(out)


class STEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        geo_num_heads: int = 4,
        sem_num_heads: int = 2,
        t_num_heads: int = 2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_prob: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        type_ln: str = "post",
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.st_attn = STSelfAttention(
            dim,
            geo_num_heads=geo_num_heads,
            sem_num_heads=sem_num_heads,
            t_num_heads=t_num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            output_dim=output_dim,
        )
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
        )

    def forward(
        self,
        x: Tensor,
        x_patterns: Tensor,
        pattern_keys: Tensor,
        geo_mask: Tensor | None = None,
        sem_mask: Tensor | None = None,
    ) -> Tensor:
        if self.type_ln == "pre":
            x = x + self.drop_path(
                self.st_attn(self.norm1(x), x_patterns, pattern_keys, geo_mask, sem_mask)
            )
            return x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm1(
            x + self.drop_path(self.st_attn(x, x_patterns, pattern_keys, geo_mask, sem_mask))
        )
        return self.norm2(x + self.drop_path(self.mlp(x)))


class PDFormerArch(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        feature_dim: int = 1,
        ext_dim: int = 8,
        embed_dim: int = 64,
        skip_dim: int = 256,
        lape_dim: int = 8,
        geo_num_heads: int = 4,
        sem_num_heads: int = 2,
        t_num_heads: int = 2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_prob: float = 0.3,
        s_attn_size: int = 3,
        enc_depth: int = 6,
        type_ln: str = "post",
        far_mask_delta: int = 5,
        output_dim: int = 1,
        add_time_in_day: bool = True,
        add_day_in_week: bool = True,
    ) -> None:
        super().__init__()
        if in_steps < s_attn_size:
            raise BackboneError(
                f"PDFormer needs in_steps >= s_attn_size ({in_steps} < {s_attn_size})"
            )
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.feature_dim = feature_dim
        self.ext_dim = ext_dim
        self.embed_dim = embed_dim
        self.skip_dim = skip_dim
        self.s_attn_size = s_attn_size
        self.output_dim = output_dim
        self.far_mask_delta = far_mask_delta

        self.pattern_embeddings = nn.ModuleList(
            [TokenEmbedding(s_attn_size, embed_dim) for _ in range(output_dim)]
        )
        # Upstream freezes DTW cluster centroids here.  We keep the same shape
        # and the same downstream wiring but learn the prior instead.
        init = torch.linspace(0.0, 1.0, s_attn_size).view(1, s_attn_size, 1)
        self.pattern_keys = nn.Parameter(
            init.repeat(num_nodes, 1, output_dim).contiguous()
        )

        self.enc_embed_layer = DataEmbedding(
            feature_dim,
            embed_dim,
            lape_dim,
            drop=drop,
            add_time_in_day=add_time_in_day,
            add_day_in_week=add_day_in_week,
        )
        depths = [float(x) for x in torch.linspace(0, drop_path_prob, enc_depth)]
        self.encoder_blocks = nn.ModuleList(
            [
                STEncoderBlock(
                    dim=embed_dim,
                    geo_num_heads=geo_num_heads,
                    sem_num_heads=sem_num_heads,
                    t_num_heads=t_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path_prob=depths[index],
                    norm_layer=lambda dim: nn.LayerNorm(dim, eps=1e-6),
                    type_ln=type_ln,
                    output_dim=output_dim,
                )
                for index in range(enc_depth)
            ]
        )
        self.skip_convs = nn.ModuleList(
            [nn.Conv2d(embed_dim, skip_dim, kernel_size=1) for _ in range(enc_depth)]
        )
        self.end_conv1 = nn.Conv2d(in_steps, out_steps, kernel_size=1, bias=True)
        self.end_conv2 = nn.Conv2d(skip_dim, output_dim, kernel_size=1, bias=True)
        self.node_feature_dim = embed_dim

    def forward(
        self, x: Tensor, lap_mx: Tensor, geo_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        steps = x.shape[1]
        x_pattern_list = []
        for index in range(self.s_attn_size):
            pattern = F.pad(
                x[:, : steps + index + 1 - self.s_attn_size, :, : self.output_dim],
                (0, 0, 0, 0, self.s_attn_size - 1 - index, 0),
                "constant",
                0,
            ).unsqueeze(-2)
            x_pattern_list.append(pattern)
        x_patterns = torch.cat(x_pattern_list, dim=-2)

        pattern_list = []
        key_list = []
        for index in range(self.output_dim):
            pattern_list.append(
                self.pattern_embeddings[index](x_patterns[..., index]).unsqueeze(-1)
            )
            key_list.append(
                self.pattern_embeddings[index](self.pattern_keys[..., index]).unsqueeze(-1)
            )
        x_patterns = torch.cat(pattern_list, dim=-1)
        pattern_keys = torch.cat(key_list, dim=-1)

        enc = self.enc_embed_layer(x, lap_mx)
        skip: Tensor | None = None
        for index, block in enumerate(self.encoder_blocks):
            enc = block(enc, x_patterns, pattern_keys, geo_mask, None)
            contribution = self.skip_convs[index](enc.permute(0, 3, 2, 1))
            skip = contribution if skip is None else skip + contribution
        if skip is None:  # pragma: no cover - enc_depth >= 1 by construction
            raise BackboneError("PDFormer produced no skip connection")
        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))
        prediction = skip.permute(0, 3, 2, 1)
        node_hidden = enc[:, -1].contiguous()
        return node_hidden, prediction


class PDFormerAdapter(TorchBackboneAdapter):
    def __init__(
        self,
        cfg: BackboneConfig,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        steps_per_day: int = 288,
    ) -> None:
        params = cfg.params
        self.lape_dim = int(params["lape_dim"])
        self.far_mask_delta = int(params["far_mask_delta"])
        model = PDFormerArch(
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
            feature_dim=1,
            ext_dim=8,
            embed_dim=int(params["embed_dim"]),
            skip_dim=int(params["skip_dim"]),
            lape_dim=self.lape_dim,
            geo_num_heads=int(params["geo_num_heads"]),
            sem_num_heads=int(params["sem_num_heads"]),
            t_num_heads=int(params["t_num_heads"]),
            mlp_ratio=float(params["mlp_ratio"]),
            qkv_bias=bool(params["qkv_bias"]),
            drop=float(params["drop"]),
            attn_drop=float(params["attn_drop"]),
            drop_path_prob=float(params["drop_path"]),
            s_attn_size=int(params["s_attn_size"]),
            enc_depth=int(params["enc_depth"]),
            type_ln=str(params["type_ln"]),
            far_mask_delta=self.far_mask_delta,
            output_dim=1,
            add_time_in_day=True,
            add_day_in_week=True,
        )
        super().__init__(
            model=model,
            mode=cfg.mode,
            partial_patterns=(
                f"encoder_blocks.{len(model.encoder_blocks) - 1}.",
                "end_conv2",
            ),
            node_feature_dim=model.node_feature_dim,
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
        )
        self.steps_per_day = steps_per_day

    def forward_features(self, inp: ModelInput, graph: GraphContext) -> tuple[Tensor, Tensor]:
        lap_mx = self._graph_cache.get(
            graph,
            "lap_mx",
            lambda: laplacian_pe(dense_diffusion(graph), self.lape_dim),
        )
        geo_mask = self._graph_cache.get(
            graph,
            "geo_mask",
            lambda: hop_distance(graph, self.far_mask_delta) >= self.far_mask_delta,
        )
        x = stack_features(inp, ("flow", "tod", "dow_onehot"))
        return self.model(x, lap_mx, geo_mask)


__all__ = ["PDFormerArch", "PDFormerAdapter", "STSelfAttention", "STEncoderBlock"]
