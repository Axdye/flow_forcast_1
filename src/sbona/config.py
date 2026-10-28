"""Strict configuration parsing.

Every section rejects unknown keys.  The design reference
``configs/design_defaults.yaml`` is *not* meant to be executed; experiment
configs live under ``configs/experiments`` and are fully resolved.

Local paths are deliberately separate from the logical configuration.  They can
be supplied on the CLI or through ``SBONA_DATA_ROOT`` / ``SBONA_OUTPUT_ROOT``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_BACKBONES = ("staeformer", "stid", "graph_wavenet", "agcrn")
SUPPORTED_MODES = ("frozen", "partial", "full")
SUPPORTED_PRECISIONS = ("fp32", "bf16")
SUPPORTED_SPLITS = ("train", "val", "test")

_DEFAULT_MAX_EPOCHS = {"base": 100, "single": 80, "candidates": 80, "router": 40, "probe": 40}
_DEFAULT_PATIENCE = {"base": 15, "single": 15, "candidates": 15, "router": 10, "probe": 10}


class ConfigError(ValueError):
    """Raised for unknown keys, invalid values or incompatible sections."""


def _strict_map(data: Mapping[str, Any], allowed: tuple[str, ...], where: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {where}: {unknown}; allowed: {sorted(allowed)}"
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number, got {value!r}") from exc


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    return value


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    input_steps: int = 12
    output_steps: int = 12
    split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
    steps_per_day: int = 288
    scaler: str = "global_standard_train_only"
    missing_policy: str = "source_mask_else_nonfinite"
    zeros_are_valid: bool = True
    envelope_quantile: float = 0.995
    envelope_min_count: int = 1000
    mape_min_abs_target: float = 1.0
    unit: str = "flow per sampling interval (raw publisher units)"
    data_file: str | None = None
    graph_file: str | None = None
    node_ids_file: str | None = None
    timestamps_file: str | None = None
    timestamps_available: bool | None = None
    time_of_day: bool = True
    day_of_week: bool = False
    flow_channel: int = 0
    missing_sentinel: float | None = None
    synthetic: bool = False
    synthetic_num_nodes: int = 8
    synthetic_num_steps: int = 432
    synthetic_missing_fraction: float = 0.01

    ALLOWED = (
        "dataset",
        "input_steps",
        "output_steps",
        "split_ratios",
        "steps_per_day",
        "scaler",
        "missing_policy",
        "zeros_are_valid",
        "envelope_quantile",
        "envelope_min_count",
        "mape_min_abs_target",
        "unit",
        "data_file",
        "graph_file",
        "node_ids_file",
        "timestamps_file",
        "timestamps_available",
        "time_of_day",
        "day_of_week",
        "flow_channel",
        "missing_sentinel",
        "synthetic",
        "synthetic_num_nodes",
        "synthetic_num_steps",
        "synthetic_missing_fraction",
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> DataConfig:
        data = dict(data or {})
        _strict_map(data, cls.ALLOWED, "data")
        if "dataset" not in data:
            raise ConfigError("data.dataset is required")
        ratios = data.get("split_ratios", (0.6, 0.2, 0.2))
        if not isinstance(ratios, (list, tuple)) or len(ratios) != 3:
            raise ConfigError("data.split_ratios must be a sequence of three numbers")
        cfg = cls(
            dataset=str(data["dataset"]),
            input_steps=_as_int(data.get("input_steps", 12), "data.input_steps"),
            output_steps=_as_int(data.get("output_steps", 12), "data.output_steps"),
            split_ratios=tuple(float(x) for x in ratios),  # type: ignore[arg-type]
            steps_per_day=_as_int(data.get("steps_per_day", 288), "data.steps_per_day"),
            scaler=str(data.get("scaler", "global_standard_train_only")),
            missing_policy=str(data.get("missing_policy", "source_mask_else_nonfinite")),
            zeros_are_valid=bool(data.get("zeros_are_valid", True)),
            envelope_quantile=_as_float(data.get("envelope_quantile", 0.995), "data.envelope_quantile"),
            envelope_min_count=_as_int(data.get("envelope_min_count", 1000), "data.envelope_min_count"),
            mape_min_abs_target=_as_float(
                data.get("mape_min_abs_target", 1.0), "data.mape_min_abs_target"
            ),
            unit=str(
                data.get(
                    "unit", "flow per sampling interval (raw publisher units)"
                )
            ),
            data_file=data.get("data_file"),
            graph_file=data.get("graph_file"),
            node_ids_file=data.get("node_ids_file"),
            timestamps_file=data.get("timestamps_file"),
            timestamps_available=data.get("timestamps_available"),
            time_of_day=bool(data.get("time_of_day", True)),
            day_of_week=bool(data.get("day_of_week", False)),
            flow_channel=_as_int(data.get("flow_channel", 0), "data.flow_channel"),
            missing_sentinel=(
                None
                if data.get("missing_sentinel") is None
                else _as_float(data["missing_sentinel"], "data.missing_sentinel")
            ),
            synthetic=bool(data.get("synthetic", False)),
            synthetic_num_nodes=_as_int(
                data.get("synthetic_num_nodes", 8), "data.synthetic_num_nodes"
            ),
            synthetic_num_steps=_as_int(
                data.get("synthetic_num_steps", 432), "data.synthetic_num_steps"
            ),
            synthetic_missing_fraction=_as_float(
                data.get("synthetic_missing_fraction", 0.01),
                "data.synthetic_missing_fraction",
            ),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        _require(self.dataset.strip() != "", "data.dataset must be non-empty")
        _require(self.input_steps >= 1, "data.input_steps must be >= 1")
        _require(self.output_steps >= 1, "data.output_steps must be >= 1")
        _require(self.steps_per_day >= 1, "data.steps_per_day must be >= 1")
        _require(
            all(0 < r < 1 for r in self.split_ratios),
            "data.split_ratios entries must be in (0, 1)",
        )
        _require(
            abs(sum(self.split_ratios) - 1.0) < 1e-6,
            f"data.split_ratios must sum to 1, got {sum(self.split_ratios)}",
        )
        _require(self.scaler == "global_standard_train_only", "only the train-only global scaler is supported in v1")
        _require(
            self.missing_policy in ("source_mask_else_nonfinite", "official_compat_nonzero"),
            f"unknown data.missing_policy {self.missing_policy!r}",
        )
        _require(0.5 < self.envelope_quantile < 1.0, "data.envelope_quantile must be in (0.5, 1)")
        _require(self.envelope_min_count >= 1, "data.envelope_min_count must be >= 1")
        _require(self.mape_min_abs_target >= 0, "data.mape_min_abs_target must be >= 0")
        _require(self.unit.strip() != "", "data.unit must be non-empty")
        _require(self.flow_channel >= 0, "data.flow_channel must be >= 0")
        _require(
            0.0 <= self.synthetic_missing_fraction < 1.0,
            "data.synthetic_missing_fraction must be in [0, 1)",
        )
        if self.synthetic:
            _require(self.synthetic_num_nodes >= 1, "synthetic_num_nodes must be >= 1")
            _require(self.synthetic_num_steps >= self.input_steps + self.output_steps + 10,
                     "synthetic_num_steps is too small")
        else:
            _require(self.data_file is not None, "data.data_file is required for real datasets")
        if self.timestamps_file is not None:
            _require(
                self.timestamps_available is not False,
                "data.timestamps_file contradicts data.timestamps_available=false",
            )
        if not self.time_of_day:
            _require(not self.day_of_week, "day_of_week requires time_of_day")


@dataclass(frozen=True)
class GraphConfig:
    topology: str = "published_binary_undirected"
    diffusion: str = "row_normalized_with_self_loops"
    representation: str = "sparse_coo"

    ALLOWED = ("topology", "diffusion", "representation")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> GraphConfig:
        data = dict(data or {})
        _strict_map(data, cls.ALLOWED, "graph")
        cfg = cls(
            topology=str(data.get("topology", "published_binary_undirected")),
            diffusion=str(data.get("diffusion", "row_normalized_with_self_loops")),
            representation=str(data.get("representation", "sparse_coo")),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        _require(
            self.topology
            in ("published_binary_undirected", "derived_binary_undirected", "identity"),
            f"unknown graph.topology {self.topology!r}",
        )
        _require(
            self.diffusion == "row_normalized_with_self_loops",
            "v1 supports only row-normalized diffusion with self loops",
        )
        _require(self.representation == "sparse_coo", "v1 supports sparse COO only")


@dataclass(frozen=True)
class BackboneConfig:
    name: str = "staeformer"
    mode: str = "frozen"
    input_dim: int = 1
    use_mixed_proj: bool = True
    input_embedding_dim: int = 24
    tod_embedding_dim: int = 24
    dow_embedding_dim: int = 0
    spatial_embedding_dim: int = 0
    adaptive_embedding_dim: int = 80
    feed_forward_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    partial_unfreeze_last_n_spatial: int = 1

    ALLOWED = (
        "name",
        "mode",
        "input_dim",
        "use_mixed_proj",
        "input_embedding_dim",
        "tod_embedding_dim",
        "dow_embedding_dim",
        "spatial_embedding_dim",
        "adaptive_embedding_dim",
        "feed_forward_dim",
        "num_heads",
        "num_layers",
        "dropout",
        "partial_unfreeze_last_n_spatial",
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> BackboneConfig:
        data = dict(data or {})
        _strict_map(data, cls.ALLOWED, "backbone")
        cfg = cls(
            name=str(data.get("name", "staeformer")),
            mode=str(data.get("mode", "frozen")),
            input_dim=_as_int(data.get("input_dim", 1), "backbone.input_dim"),
            use_mixed_proj=bool(data.get("use_mixed_proj", True)),
            input_embedding_dim=_as_int(data.get("input_embedding_dim", 24), "backbone.input_embedding_dim"),
            tod_embedding_dim=_as_int(data.get("tod_embedding_dim", 24), "backbone.tod_embedding_dim"),
            dow_embedding_dim=_as_int(data.get("dow_embedding_dim", 0), "backbone.dow_embedding_dim"),
            spatial_embedding_dim=_as_int(data.get("spatial_embedding_dim", 0), "backbone.spatial_embedding_dim"),
            adaptive_embedding_dim=_as_int(data.get("adaptive_embedding_dim", 80), "backbone.adaptive_embedding_dim"),
            feed_forward_dim=_as_int(data.get("feed_forward_dim", 256), "backbone.feed_forward_dim"),
            num_heads=_as_int(data.get("num_heads", 4), "backbone.num_heads"),
            num_layers=_as_int(data.get("num_layers", 3), "backbone.num_layers"),
            dropout=_as_float(data.get("dropout", 0.1), "backbone.dropout"),
            partial_unfreeze_last_n_spatial=_as_int(
                data.get("partial_unfreeze_last_n_spatial", 1),
                "backbone.partial_unfreeze_last_n_spatial",
            ),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        _require(self.name in SUPPORTED_BACKBONES, f"unknown backbone.name {self.name!r}")
        _require(self.mode in SUPPORTED_MODES, f"unknown backbone.mode {self.mode!r}")
        _require(self.input_dim >= 1, "backbone.input_dim must be >= 1")
        _require(self.input_embedding_dim >= 1, "backbone.input_embedding_dim must be >= 1")
        _require(self.num_heads >= 1, "backbone.num_heads must be >= 1")
        _require(self.num_layers >= 1, "backbone.num_layers must be >= 1")
        _require(0.0 <= self.dropout < 1.0, "backbone.dropout must be in [0, 1)")
        _require(self.partial_unfreeze_last_n_spatial >= 1, "partial_unfreeze_last_n_spatial must be >= 1")
        if self.name == "staeformer":
            _require(self.input_dim == 1, "STAEformer v1 must use input_dim=1 (flow-only projection)")
            _require(self.use_mixed_proj, "STAEformer v1 requires use_mixed_proj=True")
            dims = (
                self.input_embedding_dim
                + self.tod_embedding_dim
                + self.dow_embedding_dim
                + self.spatial_embedding_dim
                + self.adaptive_embedding_dim
            )
            _require(dims % self.num_heads == 0, "STAEformer model_dim must be divisible by num_heads")


@dataclass(frozen=True)
class ActionRangesConfig:
    delay: tuple[float, float] = (0.25, 3.0)
    magnitude: tuple[float, float] = (-3.0, 3.0)
    duration: tuple[float, float] = (1.0, 12.0)
    recovery: tuple[float, float] = (0.01, 1.0)

    ALLOWED = ("delay", "magnitude", "duration", "recovery")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ActionRangesConfig:
        data = dict(data or {})
        _strict_map(data, cls.ALLOWED, "adapter.action_ranges")
        out: dict[str, tuple[float, float]] = {}
        for key in cls.ALLOWED:
            raw = data.get(key)
            if raw is None:
                raw = {
                    "delay": (0.25, 3.0),
                    "magnitude": (-3.0, 3.0),
                    "duration": (1.0, 12.0),
                    "recovery": (0.01, 1.0),
                }[key]
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ConfigError(f"adapter.action_ranges.{key} must be [low, high]")
            lo, hi = float(raw[0]), float(raw[1])
            _require(lo < hi, f"adapter.action_ranges.{key} must satisfy low < high")
            out[key] = (lo, hi)
        return cls(**out)


@dataclass(frozen=True)
class AdapterConfig:
    num_candidates: int = 8
    hidden_dim: int = 64
    residual_dim: int = 32
    graph_hops: int = 3
    sigma_min: float = 0.05
    local_branch: bool = False
    film: bool = False
    s_p: float = 1.0
    s_o: float = 0.5
    s_d: float = 0.5
    residual_scale: float = 3.0
    output_init_std: float = 0.001
    anchor_seed: int = 20261028
    action_ranges: ActionRangesConfig = field(default_factory=ActionRangesConfig)

    ALLOWED = (
        "num_candidates",
        "hidden_dim",
        "residual_dim",
        "graph_hops",
        "sigma_min",
        "local_branch",
        "film",
        "s_p",
        "s_o",
        "s_d",
        "residual_scale",
        "output_init_std",
        "anchor_seed",
        "action_ranges",
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> AdapterConfig:
        data = dict(data or {})
        _strict_map(data, cls.ALLOWED, "adapter")
        cfg = cls(
            num_candidates=_as_int(data.get("num_candidates", 8), "adapter.num_candidates"),
            hidden_dim=_as_int(data.get("hidden_dim", 64), "adapter.hidden_dim"),
            residual_dim=_as_int(data.get("residual_dim", 32), "adapter.residual_dim"),
            graph_hops=_as_int(data.get("graph_hops", 3), "adapter.graph_hops"),
            sigma_min=_as_float(data.get("sigma_min", 0.05), "adapter.sigma_min"),
            local_branch=bool(data.get("local_branch", False)),
            film=bool(data.get("film", False)),
            s_p=_as_float(data.get("s_p", 1.0), "adapter.s_p"),
            s_o=_as_float(data.get("s_o", 0.5), "adapter.s_o"),
            s_d=_as_float(data.get("s_d", 0.5), "adapter.s_d"),
            residual_scale=_as_float(data.get("residual_scale", 3.0), "adapter.residual_scale"),
            output_init_std=_as_float(data.get("output_init_std", 0.001), "adapter.output_init_std"),
            anchor_seed=_as_int(data.get("anchor_seed", 20261028), "adapter.anchor_seed"),
            action_ranges=ActionRangesConfig.from_dict(data.get("action_ranges")),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        _require(self.num_candidates >= 1, "adapter.num_candidates must be >= 1")
        _require(self.hidden_dim >= 1, "adapter.hidden_dim must be >= 1")
        _require(self.residual_dim >= 1, "adapter.residual_dim must be >= 1")
        _require(self.graph_hops >= 0, "adapter.graph_hops must be >= 0")
        _require(self.sigma_min > 0, "adapter.sigma_min must be > 0")
        _require(self.local_branch is False, "v1 forbids a free local branch")
        _require(self.film is False, "v1 forbids FiLM")
        _require(self.s_p > 0 and self.s_o > 0 and self.s_d > 0, "smoothing scales must be positive")
        _require(self.residual_scale > 0, "adapter.residual_scale must be > 0")
        _require(self.output_init_std > 0, "adapter.output_init_std must be > 0")
        _require(
            self.action_ranges.duration[1] >= 1.0,
            "action_ranges.duration upper bound must be >= 1",
        )


@dataclass(frozen=True)
class OracleConfig:
    alpha: float = 1.0
    temperature: float = 0.5
    std_floor: float = 0.001
    std_correction: int = 0

    ALLOWED = ("alpha", "temperature", "std_floor", "std_correction")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> OracleConfig:
        data = dict(data or {})
        _strict_map(data, cls.ALLOWED, "oracle")
        cfg = cls(
            alpha=_as_float(data.get("alpha", 1.0), "oracle.alpha"),
            temperature=_as_float(data.get("temperature", 0.5), "oracle.temperature"),
            std_floor=_as_float(data.get("std_floor", 0.001), "oracle.std_floor"),
            std_correction=_as_int(data.get("std_correction", 0), "oracle.std_correction"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        _require(0.0 <= self.alpha <= 1.0, "oracle.alpha must be in [0, 1]")
        _require(self.temperature > 0, "oracle.temperature must be > 0")
        _require(self.std_floor > 0, "oracle.std_floor must be > 0")
        _require(self.std_correction == 0, "oracle.std_correction must be 0 (population std)")


@dataclass(frozen=True)
class LossConfig:
    consistency: float = 0.0
    penalty_negative: float = 1.0
    penalty_envelope: float = 1.0
    penalty_spatial: float = 0.1
    epsilon_floor: float = 0.05
    separation: float = 0.01
    separation_margin: float = 0.25
    router_kl: float = 1.0
    regret: float = 0.2
    balance_max: float = 0.01

    ALLOWED = (
        "consistency",
        "penalty_negative",
        "penalty_envelope",
        "penalty_spatial",
        "epsilon_floor",
        "separation",
        "separation_margin",
        "router_kl",
        "regret",
        "balance_max",
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> LossConfig:
        data = dict(data or {})
        _strict_map(data, cls.ALLOWED, "loss")
        cfg = cls(
            consistency=_as_float(data.get("consistency", 0.0), "loss.consistency"),
            penalty_negative=_as_float(data.get("penalty_negative", 1.0), "loss.penalty_negative"),
            penalty_envelope=_as_float(data.get("penalty_envelope", 1.0), "loss.penalty_envelope"),
            penalty_spatial=_as_float(data.get("penalty_spatial", 0.1), "loss.penalty_spatial"),
            epsilon_floor=_as_float(data.get("epsilon_floor", 0.05), "loss.epsilon_floor"),
            separation=_as_float(data.get("separation", 0.01), "loss.separation"),
            separation_margin=_as_float(data.get("separation_margin", 0.25), "loss.separation_margin"),
            router_kl=_as_float(data.get("router_kl", 1.0), "loss.router_kl"),
            regret=_as_float(data.get("regret", 0.2), "loss.regret"),
            balance_max=_as_float(data.get("balance_max", 0.01), "loss.balance_max"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        for name in (
            "consistency",
            "penalty_negative",
            "penalty_envelope",
            "penalty_spatial",
            "separation",
            "separation_margin",
            "router_kl",
            "regret",
            "balance_max",
        ):
            _require(getattr(self, name) >= 0, f"loss.{name} must be >= 0")
        _require(0.0 <= self.epsilon_floor <= 1.0, "loss.epsilon_floor must be in [0, 1]")


@dataclass(frozen=True)
class TrainingConfig:
    precision: str = "fp32"
    initial_batch_size: int = 32
    eval_batch_size: int = 64
    accumulation: int = 1
    base_lr: float = 1e-3
    candidate_lr: float = 1e-3
    router_lr: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    max_epochs: dict[str, int] = field(
        default_factory=lambda: {"base": 100, "single": 80, "candidates": 80, "router": 40, "probe": 40}
    )
    patience: dict[str, int] = field(
        default_factory=lambda: {"base": 15, "single": 15, "candidates": 15, "router": 10, "probe": 10}
    )
    scheduler: str = "constant"
    checkpoint_min_delta: float = 1e-5
    freeze_candidates_for_router: bool = True
    num_workers: int = 0
    device: str = "auto"

    ALLOWED = (
        "precision",
        "initial_batch_size",
        "eval_batch_size",
        "accumulation",
        "base_lr",
        "candidate_lr",
        "router_lr",
        "weight_decay",
        "gradient_clip",
        "max_epochs",
        "patience",
        "scheduler",
        "checkpoint_min_delta",
        "freeze_candidates_for_router",
        "num_workers",
        "device",
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> TrainingConfig:
        data = dict(data or {})
        _strict_map(data, cls.ALLOWED, "training")
        cfg = cls(
            precision=str(data.get("precision", "fp32")),
            initial_batch_size=_as_int(data.get("initial_batch_size", 32), "training.initial_batch_size"),
            eval_batch_size=_as_int(data.get("eval_batch_size", 64), "training.eval_batch_size"),
            accumulation=_as_int(data.get("accumulation", 1), "training.accumulation"),
            base_lr=_as_float(data.get("base_lr", 1e-3), "training.base_lr"),
            candidate_lr=_as_float(data.get("candidate_lr", 1e-3), "training.candidate_lr"),
            router_lr=_as_float(data.get("router_lr", 3e-4), "training.router_lr"),
            weight_decay=_as_float(data.get("weight_decay", 1e-4), "training.weight_decay"),
            gradient_clip=_as_float(data.get("gradient_clip", 5.0), "training.gradient_clip"),
            max_epochs={
                **_DEFAULT_MAX_EPOCHS,
                **{
                    str(k): _as_int(v, f"training.max_epochs.{k}")
                    for k, v in dict(data.get("max_epochs") or {}).items()
                },
            },
            patience={
                **_DEFAULT_PATIENCE,
                **{
                    str(k): _as_int(v, f"training.patience.{k}")
                    for k, v in dict(data.get("patience") or {}).items()
                },
            },
            scheduler=str(data.get("scheduler", "constant")),
            checkpoint_min_delta=_as_float(data.get("checkpoint_min_delta", 1e-5), "training.checkpoint_min_delta"),
            freeze_candidates_for_router=bool(data.get("freeze_candidates_for_router", True)),
            num_workers=_as_int(data.get("num_workers", 0), "training.num_workers"),
            device=str(data.get("device", "auto")),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        _require(self.precision in SUPPORTED_PRECISIONS, f"unknown training.precision {self.precision!r}")
        _require(self.initial_batch_size >= 1, "training.initial_batch_size must be >= 1")
        _require(self.eval_batch_size >= 1, "training.eval_batch_size must be >= 1")
        _require(
            self.accumulation == 1,
            "v1 implements accumulation=1 only; gradient accumulation is pending",
        )
        _require(self.base_lr > 0 and self.candidate_lr > 0 and self.router_lr > 0, "learning rates must be positive")
        _require(self.weight_decay >= 0, "training.weight_decay must be >= 0")
        _require(self.gradient_clip > 0, "training.gradient_clip must be > 0")
        _require(self.scheduler == "constant", "v1 supports only constant LR")
        _require(self.checkpoint_min_delta >= 0, "training.checkpoint_min_delta must be >= 0")
        _require(self.num_workers >= 0, "training.num_workers must be >= 0")
        _require(
            self.freeze_candidates_for_router is True,
            "v1 always freezes candidates for Router training; joint training is pending",
        )
        for stage, epochs in self.max_epochs.items():
            _require(stage in ("base", "single", "candidates", "router", "probe"), f"unknown max_epochs stage {stage!r}")
            _require(epochs >= 1, f"training.max_epochs.{stage} must be >= 1")
        for stage, patience in self.patience.items():
            _require(stage in ("base", "single", "candidates", "router", "probe"), f"unknown patience stage {stage!r}")
            _require(patience >= 1, f"training.patience.{stage} must be >= 1")


@dataclass(frozen=True)
class EvaluationConfig:
    primary_metric: str = "raw_mae_all"
    horizons: tuple[int, ...] = (3, 6, 12)
    clip_predictions: bool = False
    selection_granularity: str = "whole_sample_trajectory"
    bootstrap_replicates: int = 10000
    block_origins: int = 288
    bootstrap_seed: int = 20261028

    ALLOWED = (
        "primary_metric",
        "horizons",
        "clip_predictions",
        "selection_granularity",
        "bootstrap_replicates",
        "block_origins",
        "bootstrap_seed",
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, output_steps: int) -> EvaluationConfig:
        data = dict(data or {})
        _strict_map(data, cls.ALLOWED, "evaluation")
        horizons = data.get("horizons", (3, 6, 12))
        if not isinstance(horizons, (list, tuple)):
            raise ConfigError("evaluation.horizons must be a sequence")
        cfg = cls(
            primary_metric=str(data.get("primary_metric", "raw_mae_all")),
            horizons=tuple(_as_int(h, "evaluation.horizons") for h in horizons),
            clip_predictions=bool(data.get("clip_predictions", False)),
            selection_granularity=str(data.get("selection_granularity", "whole_sample_trajectory")),
            bootstrap_replicates=_as_int(data.get("bootstrap_replicates", 10000), "evaluation.bootstrap_replicates"),
            block_origins=_as_int(data.get("block_origins", 288), "evaluation.block_origins"),
            bootstrap_seed=_as_int(data.get("bootstrap_seed", 20261028), "evaluation.bootstrap_seed"),
        )
        cfg.validate(output_steps)
        return cfg

    def validate(self, output_steps: int) -> None:
        _require(self.horizons, "evaluation.horizons must not be empty")
        _require(all(1 <= h <= output_steps for h in self.horizons), "evaluation.horizons must be within 1..T")
        _require(self.primary_metric == "raw_mae_all", "v1 primary metric is raw_mae_all")
        _require(not self.clip_predictions, "v1 main tables use unclipped predictions")
        _require(
            self.selection_granularity == "whole_sample_trajectory",
            "oracle selection is one whole trajectory per sample",
        )
        _require(self.bootstrap_replicates >= 1, "bootstrap_replicates must be >= 1")
        _require(self.block_origins >= 1, "block_origins must be >= 1")


@dataclass(frozen=True)
class PathsConfig:
    data_root: str | None = None
    output_root: str | None = None

    ALLOWED = ("data_root", "output_root")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> PathsConfig:
        data = dict(data or {})
        _strict_map(data, cls.ALLOWED, "paths")
        return cls(data_root=data.get("data_root"), output_root=data.get("output_root"))


@dataclass(frozen=True)
class Config:
    data: DataConfig
    graph: GraphConfig = field(default_factory=GraphConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=lambda: EvaluationConfig())
    paths: PathsConfig = field(default_factory=PathsConfig)
    schema_version: int = 1
    protocol_id: str = "flow_12x12_v1"
    method_id: str = "sbona_structured"
    seed: int = 11

    ALLOWED = (
        "schema_version",
        "protocol_id",
        "method_id",
        "seed",
        "data",
        "graph",
        "backbone",
        "adapter",
        "oracle",
        "loss",
        "training",
        "evaluation",
        "paths",
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Config:
        if not isinstance(data, Mapping):
            raise ConfigError("config root must be a mapping")
        data = dict(data)
        _strict_map(data, cls.ALLOWED, "root")
        schema_version = _as_int(data.get("schema_version", 1), "schema_version")
        _require(schema_version == 1, f"unsupported schema_version {schema_version}")
        data_cfg = DataConfig.from_dict(data.get("data"))
        cfg = cls(
            data=data_cfg,
            graph=GraphConfig.from_dict(data.get("graph")),
            backbone=BackboneConfig.from_dict(data.get("backbone")),
            adapter=AdapterConfig.from_dict(data.get("adapter")),
            oracle=OracleConfig.from_dict(data.get("oracle")),
            loss=LossConfig.from_dict(data.get("loss")),
            training=TrainingConfig.from_dict(data.get("training")),
            evaluation=EvaluationConfig.from_dict(data.get("evaluation"), data_cfg.output_steps),
            paths=PathsConfig.from_dict(data.get("paths")),
            schema_version=schema_version,
            protocol_id=str(data.get("protocol_id", "flow_12x12_v1")),
            method_id=str(data.get("method_id", "sbona_structured")),
            seed=_as_int(data.get("seed", 11), "seed"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        _require(self.protocol_id.strip() != "", "protocol_id must be non-empty")
        _require(self.method_id.strip() != "", "method_id must be non-empty")
        if not self.data.synthetic and self.graph.topology != "identity":
            _require(
                self.data.graph_file is not None,
                "data.graph_file is required for a real published graph; "
                "graph.topology=identity is an explicit diagnostic only",
            )
        if not self.data.day_of_week:
            _require(
                self.backbone.dow_embedding_dim == 0,
                "backbone.dow_embedding_dim must be 0 when data.day_of_week is disabled",
            )
        if self.oracle.alpha < 1.0 and self.loss.consistency == 0.0:
            # Legal ablation, but make the caller state it explicitly.
            _require(
                self.method_id != "sbona_structured",
                "alpha<1 with consistency=0 must use an explicit ablation method_id",
            )

    @property
    def num_candidates(self) -> int:
        return self.adapter.num_candidates

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @property
    def hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_config(path: str | Path) -> Config:
    """Load and strictly validate a YAML configuration file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raise ConfigError(f"config file is empty: {path}")
    cfg = Config.from_dict(raw)
    return cfg


def resolve_roots(
    cfg: Config,
    data_root: str | Path | None = None,
    output_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    """Resolve machine-specific roots in CLI-override -> env -> config order."""
    import os

    env = os.environ if env is None else env
    data = data_root or env.get("SBONA_DATA_ROOT") or cfg.paths.data_root or "data"
    out = output_root or env.get("SBONA_OUTPUT_ROOT") or cfg.paths.output_root or "outputs"
    return Path(data).expanduser(), Path(out).expanduser()


__all__ = [
    "ConfigError",
    "Config",
    "DataConfig",
    "GraphConfig",
    "BackboneConfig",
    "AdapterConfig",
    "ActionRangesConfig",
    "OracleConfig",
    "LossConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "PathsConfig",
    "load_config",
    "resolve_roots",
    "SUPPORTED_BACKBONES",
]
