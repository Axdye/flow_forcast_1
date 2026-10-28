"""Shared dataclass / Protocol contracts.

This module is the single source of truth for tensor shapes and information
boundaries.  Models and backbones must not import objectives, training or the
evaluation oracle.  Future ground truth lives only in :class:`TargetBatch`.

All tensor shapes are documented explicitly because most implementation bugs in
this project are axis/scale bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

SCHEMA_VERSION = 1
"""Bump whenever a persisted schema (checkpoint/config/manifest) changes."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_float_tensor(name: str, value: Tensor, ndim: int) -> None:
    _require(isinstance(value, Tensor), f"{name} must be a torch.Tensor")
    _require(value.dim() == ndim, f"{name} must have {ndim} dims, got {tuple(value.shape)}")
    _require(value.dtype == torch.float32, f"{name} must be float32, got {value.dtype}")


def _check_bool_tensor(name: str, value: Tensor, ndim: int) -> None:
    _require(isinstance(value, Tensor), f"{name} must be a torch.Tensor")
    _require(value.dim() == ndim, f"{name} must have {ndim} dims, got {tuple(value.shape)}")
    _require(value.dtype == torch.bool, f"{name} must be bool, got {value.dtype}")


def _check_same_device(*tensors: Tensor) -> None:
    devices = {t.device for t in tensors}
    _require(len(devices) <= 1, f"tensors must share one device, got {devices}")


# ---------------------------------------------------------------------------
# Inputs and targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInput:
    """Everything the model is allowed to see.

    history:      float32 ``[B, L, N, 1]`` standardized flow (z-space).
    observed:     bool    ``[B, L, N, 1]`` whether the *raw* observation was valid.
    time_of_day:  float32 ``[B, L]`` in ``[0, 1)``.
    day_of_week:  int64   ``[B, L]`` in ``0..6``; all zeros when calendar is
                  unavailable (the backbone must then disable the weekday
                  embedding in config, not silently interpret zeros as Monday).
    """

    history: Tensor
    observed: Tensor
    time_of_day: Tensor
    day_of_week: Tensor

    def __post_init__(self) -> None:
        _check_float_tensor("history", self.history, 4)
        _check_bool_tensor("observed", self.observed, 4)
        _check_float_tensor("time_of_day", self.time_of_day, 2)
        _require(self.day_of_week.dtype == torch.int64, "day_of_week must be int64")
        _require(self.day_of_week.dim() == 2, "day_of_week must have 2 dims")
        b, length, nodes, channels = self.history.shape
        _require(channels == 1, f"history channel must be 1, got {channels}")
        _require(self.observed.shape == self.history.shape, "observed shape must match history")
        _require(
            self.time_of_day.shape == (b, length),
            f"time_of_day must be {(b, length)}, got {tuple(self.time_of_day.shape)}",
        )
        _require(self.day_of_week.shape == (b, length), "day_of_week shape mismatch")
        _check_same_device(
            self.history, self.observed, self.time_of_day, self.day_of_week
        )

    @property
    def batch_size(self) -> int:
        return self.history.shape[0]

    @property
    def input_steps(self) -> int:
        return self.history.shape[1]

    @property
    def num_nodes(self) -> int:
        return self.history.shape[2]


@dataclass(frozen=True)
class TargetBatch:
    """Ground truth.  Never passed to ``forward_candidates`` / ``predict_top1``.

    values_raw: float32 ``[B, T, N, 1]`` in original traffic-flow units.
    valid:      bool    ``[B, T, N, 1]``.
    sample_ids: int64   ``[B]`` absolute index of the first future step.
    """

    values_raw: Tensor
    valid: Tensor
    sample_ids: Tensor

    def __post_init__(self) -> None:
        _check_float_tensor("values_raw", self.values_raw, 4)
        _check_bool_tensor("valid", self.valid, 4)
        _require(self.values_raw.shape == self.valid.shape, "valid shape must match values_raw")
        channels = self.values_raw.shape[-1]
        _require(channels == 1, f"target channel must be 1, got {channels}")
        _require(self.sample_ids.dim() == 1, "sample_ids must be 1-D")
        _require(self.sample_ids.dtype == torch.int64, "sample_ids must be int64")
        _require(
            self.sample_ids.shape[0] == self.values_raw.shape[0],
            "sample_ids batch mismatch",
        )
        _check_same_device(self.values_raw, self.valid, self.sample_ids)


@dataclass(frozen=True)
class GraphContext:
    """Road-graph information.

    diffusion:      sparse COO float32 ``[N, N]`` row-normalized with self loops.
    edge_index:     int64 ``[2, E]`` undirected constraint edges, ``i < j``, no
                    self loops, deduplicated.
    edge_weight:    float32 ``[E]`` non-negative.
    graph_hash:     stable hash of node count + topology + normalization rule.
    node_order_hash: hash of the node id ordering used by the data columns.
    """

    diffusion: Tensor
    edge_index: Tensor
    edge_weight: Tensor
    graph_hash: str
    node_order_hash: str

    def __post_init__(self) -> None:
        _require(self.diffusion.is_sparse, "diffusion must be a sparse COO tensor")
        _require(self.diffusion.layout == torch.sparse_coo, "diffusion must be sparse COO")
        _require(self.diffusion.dim() == 2, "diffusion must be 2-D")
        nodes = self.diffusion.shape[0]
        _require(self.diffusion.shape[1] == nodes, "diffusion must be square")
        _require(self.edge_index.dtype == torch.int64, "edge_index must be int64")
        _require(self.edge_index.dim() == 2, "edge_index must be [2, E]")
        _require(self.edge_index.shape[0] == 2, "edge_index must be [2, E]")
        _require(self.edge_weight.dim() == 1, "edge_weight must be 1-D")
        _require(
            self.edge_weight.shape[0] == self.edge_index.shape[1],
            "edge_weight length must match E",
        )
        _check_same_device(self.diffusion, self.edge_index, self.edge_weight)
        if self.edge_index.shape[1] > 0:
            _require(
                bool((self.edge_weight >= 0).all()),
                "edge weights must be non-negative",
            )
            _require(
                bool((self.edge_index[0] < self.edge_index[1]).all()),
                "constraint edges must satisfy i < j",
            )
            _require(
                int(self.edge_index.max()) < nodes,
                "edge index out of range for diffusion",
            )

    @property
    def num_nodes(self) -> int:
        return self.diffusion.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def to(self, device: torch.device | str) -> GraphContext:
        """Return a copy on ``device``.  Used once by the trainer, not per batch."""
        return GraphContext(
            diffusion=self.diffusion.to(device),
            edge_index=self.edge_index.to(device),
            edge_weight=self.edge_weight.to(device),
            graph_hash=self.graph_hash,
            node_order_hash=self.node_order_hash,
        )


@dataclass(frozen=True)
class ScalerStats:
    """Global scalar standardization and per-node empirical envelope.

    mu / sigma are fitted on valid training flow only.  The envelope is the
    (linear) ``q``-quantile of valid training flow per node with a global
    fallback for nodes with too few observations.
    """

    mu: float
    sigma: float
    envelope: Tensor  # [N], raw units
    envelope_fallback: Tensor  # [N] bool
    envelope_quantile: float = 0.995
    envelope_min_count: int = 1000

    def __post_init__(self) -> None:
        _require(self.sigma > 0, "sigma must be positive")
        _require(self.envelope.dim() == 1, "envelope must be [N]")
        _require(
            self.envelope.shape == self.envelope_fallback.shape,
            "envelope/fallback shape mismatch",
        )
        _require(self.envelope_fallback.dtype == torch.bool, "envelope_fallback must be bool")

    def standardize(self, raw: Tensor) -> Tensor:
        """``(raw - mu) / sigma``; keeps autograd and device."""
        return (raw - self.mu) / self.sigma

    def inverse(self, z: Tensor) -> Tensor:
        """``z * sigma + mu``; keeps autograd and device."""
        return z * self.sigma + self.mu

    def to(self, device: torch.device | str) -> ScalerStats:
        return ScalerStats(
            mu=self.mu,
            sigma=self.sigma,
            envelope=self.envelope.to(device),
            envelope_fallback=self.envelope_fallback.to(device),
            envelope_quantile=self.envelope_quantile,
            envelope_min_count=self.envelope_min_count,
        )


# ---------------------------------------------------------------------------
# Model outputs
# ---------------------------------------------------------------------------


@dataclass
class BackboneOutput:
    base_pred_z: Tensor  # [B, T, N, 1]
    node_hidden: Tensor  # [B, N, D_b]

    def __post_init__(self) -> None:
        _check_float_tensor("base_pred_z", self.base_pred_z, 4)
        _check_float_tensor("node_hidden", self.node_hidden, 3)
        _require(self.base_pred_z.shape[0] == self.node_hidden.shape[0], "batch mismatch")
        _require(
            self.base_pred_z.shape[2] == self.node_hidden.shape[1], "node count mismatch"
        )
        _require(self.base_pred_z.shape[3] == 1, "base_pred_z channel must be 1")
        _check_same_device(self.base_pred_z, self.node_hidden)


@dataclass
class ActionOutput:
    physical: Tensor  # [B, K, 5] (o, delta, m, d, rho) in physical units
    normalized: Tensor  # [B, K, 5] each dimension in [0, 1]
    mu: Tensor  # [B, 5]
    sigma: Tensor  # [B, 5]

    def __post_init__(self) -> None:
        _check_float_tensor("physical", self.physical, 3)
        _check_float_tensor("normalized", self.normalized, 3)
        _check_float_tensor("mu", self.mu, 2)
        _check_float_tensor("sigma", self.sigma, 2)
        _require(self.physical.shape[-1] == 5, "actions must have 5 dims")
        _require(self.normalized.shape == self.physical.shape, "normalized shape mismatch")
        _require(self.mu.shape == self.sigma.shape, "mu/sigma shape mismatch")
        _require(
            self.mu.shape == (self.physical.shape[0], 5),
            f"mu must be [B, 5], got {tuple(self.mu.shape)}",
        )
        _check_same_device(self.physical, self.normalized, self.mu, self.sigma)


@dataclass
class CandidateOutput:
    pred_z: Tensor  # [B, K, T, N, 1]
    actions: ActionOutput
    h_global: Tensor  # [B, D]
    base_pred_z: Tensor  # [B, T, N, 1]
    # Optional selector logits for adapters that learn their own scorer instead
    # of the Stage-D Router (currently only the MCL-WTA baseline).  ``None``
    # keeps every other adapter byte-for-byte unchanged.
    candidate_logits: Tensor | None = None  # [B, K]

    def __post_init__(self) -> None:
        _check_float_tensor("pred_z", self.pred_z, 5)
        _check_float_tensor("h_global", self.h_global, 2)
        _check_float_tensor("base_pred_z", self.base_pred_z, 4)
        b, k, t, n, c = self.pred_z.shape
        _require(c == 1, "pred_z channel must be 1")
        _require(self.actions.physical.shape[:2] == (b, k), "action K mismatch")
        _require(self.h_global.shape[0] == b, "h_global batch mismatch")
        _require(self.base_pred_z.shape == (b, t, n, 1), "base_pred_z shape mismatch")
        if self.candidate_logits is not None:
            _check_float_tensor("candidate_logits", self.candidate_logits, 2)
            _require(
                self.candidate_logits.shape == (b, k),
                f"candidate_logits must be [B, K] = {(b, k)}, got "
                f"{tuple(self.candidate_logits.shape)}",
            )
        _check_same_device(self.pred_z, self.h_global, self.base_pred_z)

    @property
    def num_candidates(self) -> int:
        return self.pred_z.shape[1]


@dataclass
class Top1Output:
    pred_z: Tensor  # [B, T, N, 1]
    index: Tensor  # int64 [B]
    logits: Tensor  # [B, K]
    selected_action: Tensor  # [B, 5]

    def __post_init__(self) -> None:
        _check_float_tensor("pred_z", self.pred_z, 4)
        _check_float_tensor("logits", self.logits, 2)
        _check_float_tensor("selected_action", self.selected_action, 2)
        _require(self.index.dtype == torch.int64, "index must be int64")
        _require(self.index.shape[0] == self.pred_z.shape[0], "index batch mismatch")
        _require(self.logits.shape[0] == self.pred_z.shape[0], "logits batch mismatch")
        _require(self.selected_action.shape == (self.pred_z.shape[0], 5), "action shape mismatch")
        _require(self.pred_z.shape[3] == 1, "pred_z channel must be 1")


@dataclass
class OracleOutput:
    error: Tensor  # [B, K], gradient retained (z-space masked MSE)
    consistency: Tensor  # [B, K], gradient retained (dimensionless P)
    cost: Tensor  # [B, K], detached
    soft_target: Tensor  # [B, K], detached
    best_index: Tensor  # int64 [B]
    valid_sample: Tensor  # bool [B]

    def __post_init__(self) -> None:
        _check_float_tensor("error", self.error, 2)
        _check_float_tensor("consistency", self.consistency, 2)
        _check_float_tensor("cost", self.cost, 2)
        _check_float_tensor("soft_target", self.soft_target, 2)
        _require(self.best_index.dtype == torch.int64, "best_index must be int64")
        _require(self.valid_sample.dtype == torch.bool, "valid_sample must be bool")
        _require(self.cost.shape == self.error.shape, "cost shape mismatch")
        _require(self.soft_target.shape == self.error.shape, "soft target shape mismatch")
        _require(self.best_index.shape == (self.error.shape[0],), "best_index shape mismatch")
        _require(self.valid_sample.shape == (self.error.shape[0],), "valid_sample shape mismatch")


# ---------------------------------------------------------------------------
# Backbone protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BackboneAdapter(Protocol):
    """Minimal interface every backbone wrapper must satisfy.

    ``encode_and_predict`` must run the backbone exactly once and return both
    the untouched base prediction and the hidden representation used by the
    adapter.  ``set_mode`` is idempotent and must be re-applied after resume.
    """

    node_feature_dim: int

    def encode_and_predict(self, inp: ModelInput, graph: GraphContext) -> BackboneOutput:
        ...

    def set_mode(self, mode: str) -> None:
        ...

    def trainable_parameter_names(self) -> list[str]:
        ...


@dataclass(frozen=True)
class SampleMeta:
    """Loader-side metadata.  Never passed into a model forward."""

    sample_ids: Tensor
    split: str
    target_start: Tensor
    calendar_available: bool


__all__ = [
    "SCHEMA_VERSION",
    "ModelInput",
    "TargetBatch",
    "GraphContext",
    "ScalerStats",
    "BackboneOutput",
    "ActionOutput",
    "CandidateOutput",
    "Top1Output",
    "OracleOutput",
    "BackboneAdapter",
    "SampleMeta",
]
