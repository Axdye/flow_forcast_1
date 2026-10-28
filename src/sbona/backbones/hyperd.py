"""HyperD backbone adapter (AAAI 2026).

Upstream: https://github.com/start2020/HyperD, commit
``5bd93a825955cbe31b2d8dac0041ecb0e1d223d2``.  The repository ships **no
licence file**; the vendoring decision stays ``pending`` in
``third_party/LOCK.md`` and this module is an attributed re-implementation for
the research prototype.

HyperD decomposes a series into a *hybrid periodic pattern* (a learnable daily
profile plus a learnable weekly profile, each diffused over the road graph and
axis-attended) and a residual that a frequency-domain ``stfe`` block models.
The two halves are recombined:

``y = stfe(x - S_in) + S_out``

Faithfulness notes (all recorded in ``reviews/EXTRA_WORK.md``)
-------------------------------------------------------------
* The published model loads ``daily_init.npy`` / ``weekly_init.npy``, i.e. the
  *mean daily and weekly profiles computed once from the training split*.  An
  adapter never sees the training split, so the periodic tables start at zero
  (``S_in = S_out = 0`` and HyperD degenerates to the ``stfe`` residual path)
  and a hook, :meth:`HybridDAdapter.set_periodic_prior`, injects the official
  prior when a caller has it.  This is a documented deviation, not a silent
  re-initialisation.
* Upstream also returns a frequency split ``dual_view_loss`` consumed by
  ``SimpleTimeSeriesForecastingRunner``.  SBoN supplies its own objective, and
  the loss is a function of the prediction only, so it is not returned here.
* The upstream code indexes the weekly table with
  ``tod * 288 + dow * weekly_len`` where ``dow`` is BasicTS's *fraction*
  ``weekday / 7``; that equals ``daily_index + weekday_index * 288``.  The
  contract stores the integer weekday, so the adapter uses the index directly.
* ``Hybrid_Periodic_Pattern`` diffuses with ``data @ adj``; the contract's
  ``A[i, j]`` means "row ``i`` aggregates column ``j``", so the adapter passes
  ``A.T`` to preserve the upstream orientation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sbona.backbones.base import BackboneError
from sbona.backbones.common import TorchBackboneAdapter, dense_diffusion
from sbona.config import BackboneConfig
from sbona.contracts import GraphContext, ModelInput


class SelfAttention(nn.Module):
    """Single-head self-attention over the last axis (upstream ``attention.py``)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, q: Tensor) -> Tensor:
        k, v = q, q
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v)


class HybridPeriodicPattern(nn.Module):
    """Learnable daily/weekly profile with graph diffusion and axis attention."""

    def __init__(self, period_len: int, num_nodes: int) -> None:
        super().__init__()
        if period_len < 1:
            raise BackboneError("HyperD period_len must be >= 1")
        self.period_len = int(period_len)
        self.num_nodes = int(num_nodes)
        self.linear1 = nn.Linear(num_nodes, num_nodes)
        self.linear2 = nn.Linear(2 * num_nodes, num_nodes)
        self.attention_t = SelfAttention(num_nodes)
        self.attention_s = SelfAttention(period_len)
        self.register_buffer("adjacency", torch.zeros(num_nodes, num_nodes), persistent=False)
        # ``None`` means "no prior available": the table is all zeros and the
        # model learns the profile from scratch (documented deviation).
        self.data = nn.Parameter(torch.zeros(period_len, num_nodes))
        self.prior_loaded = False

    def set_adjacency(self, adjacency: Tensor) -> None:
        """Store the diffusion operator used by ``einsum('ln,nv->lv')``."""
        expected = (self.num_nodes, self.num_nodes)
        if tuple(adjacency.shape) != expected:
            raise BackboneError(
                f"HyperD adjacency must be {expected}, got {tuple(adjacency.shape)}"
            )
        self.adjacency = adjacency.detach().to(
            device=self.linear1.weight.device, dtype=self.linear1.weight.dtype
        )

    def set_prior(self, prior: Tensor) -> None:
        """Load the official ``daily_init.npy`` / ``weekly_init.npy`` table."""
        expected = (self.period_len, self.num_nodes)
        if tuple(prior.shape) != expected:
            raise BackboneError(
                f"HyperD prior must be {expected}, got {tuple(prior.shape)}"
            )
        with torch.no_grad():
            self.data.copy_(prior.to(device=self.data.device, dtype=self.data.dtype))
        self.prior_loaded = True

    def forward(self, index: Tensor, length: int) -> Tensor:
        gather_index = (
            index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)
        ) % self.period_len

        data = torch.einsum("ln,nv->lv", (self.data, self.adjacency.to(self.data.device)))
        data = F.relu(self.linear1(data))

        data_t = self.attention_t(data)
        data_s = self.attention_s(data.transpose(0, 1)).transpose(0, 1)

        data = self.linear2(torch.cat([data_t, data_s], dim=-1))
        return data[gather_index.long()]


class STFE(nn.Module):
    """Spatial-temporal frequency enhancer (upstream ``STFE.py``)."""

    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        pred_len: int,
        embed_size: int,
        hidden_size: int,
        fc_hidden_size: int,
    ) -> None:
        super().__init__()
        self.scale = 0.02
        self.feature_size = num_nodes
        self.seq_length = seq_len
        self.sparsity_threshold = 0.01
        self.embed_size = embed_size

        self.embeddings = nn.Parameter(torch.randn(1, embed_size))
        self.spatial_r1 = nn.Parameter(self.scale * torch.randn(embed_size, hidden_size))
        self.spatial_i1 = nn.Parameter(self.scale * torch.randn(embed_size, hidden_size))
        self.spatial_rb1 = nn.Parameter(self.scale * torch.randn(hidden_size))
        self.spatial_ib1 = nn.Parameter(self.scale * torch.randn(hidden_size))
        self.spatial_r2 = nn.Parameter(self.scale * torch.randn(hidden_size, embed_size))
        self.spatial_i2 = nn.Parameter(self.scale * torch.randn(hidden_size, embed_size))
        self.spatial_rb2 = nn.Parameter(self.scale * torch.randn(embed_size))
        self.spatial_ib2 = nn.Parameter(self.scale * torch.randn(embed_size))

        self.temporal_r1 = nn.Parameter(self.scale * torch.randn(embed_size, hidden_size))
        self.temporal_i1 = nn.Parameter(self.scale * torch.randn(embed_size, hidden_size))
        self.temporal_rb1 = nn.Parameter(self.scale * torch.randn(hidden_size))
        self.temporal_ib1 = nn.Parameter(self.scale * torch.randn(hidden_size))
        self.temporal_r2 = nn.Parameter(self.scale * torch.randn(hidden_size, embed_size))
        self.temporal_i2 = nn.Parameter(self.scale * torch.randn(hidden_size, embed_size))
        self.temporal_rb2 = nn.Parameter(self.scale * torch.randn(embed_size))
        self.temporal_ib2 = nn.Parameter(self.scale * torch.randn(embed_size))

        self.fc = nn.Sequential(
            nn.Linear(seq_len * embed_size, fc_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_size, pred_len),
        )

    def _token_embedding(self, x: Tensor) -> Tensor:
        return x.unsqueeze(3) * self.embeddings

    def _complex_mlp(
        self,
        x: Tensor,
        r1: Tensor,
        i1: Tensor,
        r2: Tensor,
        i2: Tensor,
        rb1: Tensor,
        rb2: Tensor,
        ib1: Tensor,
        ib2: Tensor,
    ) -> Tensor:
        o1_real = F.relu(
            torch.einsum("bijd,df->bijf", x.real, r1)
            - torch.einsum("bijd,df->bijf", x.imag, i1)
            + rb1
        )
        o1_imag = F.relu(
            torch.einsum("bijd,df->bijf", x.imag, r1)
            + torch.einsum("bijd,df->bijf", x.real, i1)
            + ib1
        )
        o2_real = F.relu(
            torch.einsum("bijf,fd->bijd", o1_real, r2)
            - torch.einsum("bijf,fd->bijd", o1_imag, i2)
            + rb2
        )
        o2_imag = F.relu(
            torch.einsum("bijf,fd->bijd", o1_imag, r2)
            + torch.einsum("bijf,fd->bijd", o1_real, i2)
            + ib2
        )
        stacked = torch.stack([o2_real, o2_imag], dim=-1)
        return torch.view_as_complex(F.softshrink(stacked, lambd=self.sparsity_threshold))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """``x [B, T, N]`` -> ``(hidden [B, N, T*E], pred [B, T', N])``."""
        batch_size, steps, nodes = x.shape
        embedded = self._token_embedding(x)
        bias = embedded

        spatial = torch.fft.rfft(embedded, dim=2, norm="ortho")
        spatial = self._complex_mlp(
            spatial,
            self.spatial_r1,
            self.spatial_i1,
            self.spatial_r2,
            self.spatial_i2,
            self.spatial_rb1,
            self.spatial_rb2,
            self.spatial_ib1,
            self.spatial_ib2,
        )
        embedded = torch.fft.irfft(spatial, n=self.feature_size, dim=2, norm="ortho")

        temporal = embedded.transpose(1, 2)
        temporal = torch.fft.rfft(temporal, dim=2, norm="ortho")
        temporal = self._complex_mlp(
            temporal,
            self.temporal_r1,
            self.temporal_i1,
            self.temporal_r2,
            self.temporal_i2,
            self.temporal_rb1,
            self.temporal_rb2,
            self.temporal_ib1,
            self.temporal_ib2,
        )
        embedded = torch.fft.irfft(temporal, n=self.seq_length, dim=2, norm="ortho")
        embedded = embedded.transpose(1, 2) + bias

        hidden = embedded.transpose(1, 2).reshape(batch_size, nodes, -1).contiguous()
        prediction = self.fc(hidden).permute(0, 2, 1).contiguous()
        return hidden, prediction


class HyperDArch(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        steps_per_day: int = 288,
        embed_size: int = 64,
        hidden_size: int = 128,
        fc_hidden_size: int = 512,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = in_steps
        self.pred_len = out_steps
        self.daily_len = steps_per_day
        self.weekly_len = 7 * steps_per_day
        self.embed_size = embed_size

        self.daily_emb = HybridPeriodicPattern(self.daily_len, num_nodes)
        self.weekly_emb = HybridPeriodicPattern(self.weekly_len, num_nodes)
        self.stfe = STFE(
            num_nodes, in_steps, out_steps, embed_size, hidden_size, fc_hidden_size
        )
        self.node_feature_dim = in_steps * embed_size

    def set_adjacency(self, adjacency: Tensor) -> None:
        self.daily_emb.set_adjacency(adjacency)
        self.weekly_emb.set_adjacency(adjacency)

    def set_periodic_prior(self, daily: Tensor, weekly: Tensor | None = None) -> None:
        self.daily_emb.set_prior(daily)
        if weekly is not None:
            self.weekly_emb.set_prior(weekly)

    def forward(self, x: Tensor, daily_index: Tensor, weekly_index: Tensor) -> tuple[Tensor, Tensor]:
        """``x [B, L, N]``; indices are the *last* history step's positions."""
        pattern_in = self.daily_emb(daily_index, self.seq_len) + self.weekly_emb(
            weekly_index, self.seq_len
        )
        residual_in = x - pattern_in
        hidden, residual_out = self.stfe(residual_in)

        pattern_out = self.daily_emb(
            (daily_index + self.seq_len) % self.daily_len, self.pred_len
        ) + self.weekly_emb((weekly_index + self.seq_len) % self.weekly_len, self.pred_len)
        prediction = residual_out + pattern_out
        return hidden, prediction


class HyperDAdapter(TorchBackboneAdapter):
    def __init__(
        self,
        cfg: BackboneConfig,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        steps_per_day: int = 288,
    ) -> None:
        params = cfg.params
        model = HyperDArch(
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
            steps_per_day=steps_per_day,
            embed_size=int(params["embed_size"]),
            hidden_size=int(params["hidden_size"]),
            fc_hidden_size=int(params["fc_hidden_size"]),
        )
        super().__init__(
            model=model,
            mode=cfg.mode,
            partial_patterns=("stfe.fc", "daily_emb.linear", "weekly_emb.linear"),
            node_feature_dim=model.node_feature_dim,
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
        )
        self.steps_per_day = steps_per_day
        self._adjacency_hash: str | None = None

    # -- hooks ---------------------------------------------------------------

    def set_periodic_prior(self, daily: Tensor, weekly: Tensor | None = None) -> None:
        """Inject the official per-dataset daily/weekly mean profiles."""
        self.model.set_periodic_prior(daily, weekly)

    # -- forward -------------------------------------------------------------

    def _bind_adjacency(self, graph: GraphContext) -> None:
        if self._adjacency_hash == graph.graph_hash:
            return
        adjacency = dense_diffusion(graph).transpose(0, 1).contiguous()
        self.model.set_adjacency(adjacency)
        self._adjacency_hash = graph.graph_hash

    def forward_features(self, inp: ModelInput, graph: GraphContext) -> tuple[Tensor, Tensor]:
        self._bind_adjacency(graph)
        x = inp.history[..., 0]
        tod = inp.time_of_day[:, -1]
        dow = inp.day_of_week[:, -1].long().clamp(0, 6)
        daily_index = (tod.clamp(0.0, 1.0) * self.steps_per_day).long()
        daily_index = daily_index.clamp_max(self.steps_per_day - 1)
        weekly_index = daily_index + dow * self.steps_per_day
        hidden, prediction = self.model(x, daily_index, weekly_index)
        return hidden, prediction.unsqueeze(-1)


__all__ = ["HyperDArch", "HyperDAdapter", "HybridPeriodicPattern", "STFE", "SelfAttention"]
