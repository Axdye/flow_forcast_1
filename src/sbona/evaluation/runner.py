"""Offline evaluation loop for Base and full SBoN checkpoints.

This module contains the business logic behind ``sbona.cli evaluate``.  It is
deliberately separate from ``cli.py``: the CLI only resolves paths, loads
checkpoints and reports failures.

Information boundary: ``TargetBatch`` is used here for metrics and for the
*offline* oracle.  It is never passed to ``SBoN.forward_candidates`` or
``SBoN.predict_top1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from sbona.config import Config
from sbona.contracts import GraphContext, ScalerStats, TargetBatch
from sbona.data.dataset import PreparedData, make_loader
from sbona.evaluation.artifacts import METRICS_SCHEMA_VERSION, RunArtifacts
from sbona.evaluation.metrics import (
    MetricAccumulator,
    per_sample_abs_error,
)
from sbona.evaluation.oracle import (
    candidate_errors,
    gather_candidates,
    retention,
    select_whole_sample,
)
from sbona.evaluation.regimes import (
    RegimeAccumulator,
    flow_regime_thresholds,
)
from sbona.evaluation.statistics import paired_moving_block_bootstrap
from sbona.objectives.oracle import OracleSettings, build_oracle
from sbona.training.trainer import move_input, move_target

METRIC_UNITS = {
    "mae": "flow per sampling interval (raw publisher units)",
    "rmse": "flow per sampling interval (raw publisher units)",
    "mape": "percent",
    "nvr": "fraction of valid elements",
    "euvr": "fraction of valid elements",
    "regret": "flow per sampling interval (raw publisher units)",
    "retention": "dimensionless",
    "distance": "raw flow units or z units as named",
}


@dataclass
class EvaluationResult:
    split: str
    document: dict[str, Any]
    sample_ids: np.ndarray
    predictions: list[Path] = field(default_factory=list)
    router_index: np.ndarray | None = None
    router_logits: np.ndarray | None = None
    router_actions: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Small streaming diagnostics
# ---------------------------------------------------------------------------


class CandidateHealthAccumulator:
    """Streaming version of :mod:`sbona.evaluation.diversity` diagnostics.

    Every quantity is stored as a sum plus a denominator so that batch sizes
    never bias the result.
    """

    def __init__(self, num_candidates: int) -> None:
        self.num_candidates = int(num_candidates)
        self.samples = 0
        self.soft_sum = torch.zeros(self.num_candidates, dtype=torch.float64)
        self.entropy_sum = 0.0
        self.usage_counts: dict[str, torch.Tensor] = {}
        self.trajectory_sum = 0.0
        self.trajectory_count = 0.0
        self.action_sum = 0.0
        self.action_count = 0
        self.branch_sum = torch.zeros(self.num_candidates, dtype=torch.float64)
        self.branch_count = 0.0
        self.saturation_sum = 0.0
        self.saturation_count = 0
        self.sigma = 0.0

    def add_usage(self, name: str, index: Tensor) -> None:
        counts = torch.bincount(
            index.detach().cpu(), minlength=self.num_candidates
        ).to(torch.float64)
        current = self.usage_counts.get(name)
        self.usage_counts[name] = counts if current is None else current + counts

    def update(
        self,
        pred_raw: Tensor,
        target: TargetBatch,
        actions_normalized: Tensor,
        sigma: float,
        soft_target: Tensor,
        oracle_mae_index: Tensor,
        oracle_cost_index: Tensor,
        router_index: Tensor | None,
        saturation: float | None,
    ) -> None:
        if pred_raw.dim() != 5:
            raise ValueError("pred_raw must be [B, K, T, N, 1]")
        self.sigma = float(sigma)
        batch_size = pred_raw.shape[0]
        self.samples += int(batch_size)
        self.add_usage("oracle_mae", oracle_mae_index)
        self.add_usage("oracle_cost", oracle_cost_index)
        if router_index is not None:
            self.add_usage("router", router_index)

        weights = soft_target.detach().float()
        probs = weights.clamp_min(1e-12)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        self.soft_sum += probs.sum(dim=0).to(torch.float64).cpu()
        entropy = -(probs * probs.log()).sum(dim=-1)
        self.entropy_sum += float(entropy.sum())

        mask = target.valid[:, None, :, :, :].to(pred_raw.dtype)
        num_candidates = pred_raw.shape[1]
        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                diff = (pred_raw[:, i] - pred_raw[:, j]).abs() * mask[:, 0]
                self.trajectory_sum += float(diff.sum())
                self.trajectory_count += float(mask[:, 0].sum())
                distance = torch.linalg.norm(
                    actions_normalized[:, i] - actions_normalized[:, j], dim=-1
                )
                self.action_sum += float(distance.sum())
                self.action_count += int(batch_size)

        target_expanded = target.values_raw[:, None, :, :, :].expand_as(pred_raw)
        error = (pred_raw.detach() - target_expanded.detach()).abs() * mask
        self.branch_sum += error.sum(dim=(0, 2, 3, 4)).to(torch.float64).cpu()
        self.branch_count += float(mask.sum())

        if saturation is not None:
            # ``last_saturation`` is a batch mean over all candidates/elements.
            self.saturation_sum += float(saturation) * batch_size
            self.saturation_count += int(batch_size)

    def result(self) -> dict[str, Any]:
        samples = max(self.samples, 1)
        mean_probs = self.soft_sum / samples
        mean_entropy = self.entropy_sum / samples
        normalized_entropy = (
            mean_entropy / math.log(self.num_candidates)
            if self.num_candidates > 1
            else 0.0
        )
        usage = {}
        for name, counts in self.usage_counts.items():
            total = float(counts.sum().item()) or 1.0
            usage[name] = {
                "counts": counts.to(torch.int64).tolist(),
                "fractions": (counts / total).tolist(),
                "num_candidates": self.num_candidates,
            }
        per_branch = (self.branch_sum / max(self.branch_count, 1.0)).tolist()
        return {
            "hard_usage": usage,
            "soft_oracle": {
                "mean_entropy": mean_entropy,
                "normalized_entropy": normalized_entropy,
                "effective_k": float(math.exp(mean_entropy)),
                "mean_probabilities": mean_probs.tolist(),
                "max_share": float(mean_probs.max()) if mean_probs.numel() else 0.0,
                "samples": self.samples,
            },
            "trajectory": {
                "raw_l1": self.trajectory_sum / max(self.trajectory_count, 1.0),
                "standardized_l1": (
                    self.trajectory_sum / max(self.trajectory_count, 1.0) / max(self.sigma, 1e-12)
                ),
            },
            "action": {"mean_l2": self.action_sum / max(self.action_count, 1)},
            "branch": {"per_branch_mae": per_branch},
            "residual_saturation_ratio": (
                self.saturation_sum / self.saturation_count
                if self.saturation_count
                else None
            ),
        }


def _regime_thresholds(prepared: PreparedData) -> tuple[float, float] | None:
    train_end = prepared.scheme.boundaries.train_end
    train_valid = prepared.valid[:train_end]
    values = prepared.data[:train_end][train_valid]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return flow_regime_thresholds(values)


def _new_accumulator(cfg: Config) -> MetricAccumulator:
    return MetricAccumulator(
        cfg.data.mape_min_abs_target,
        cfg.evaluation.horizons,
        collect_per_sample=True,
    )


def _metric_document(
    cfg: Config,
    *,
    methods: dict[str, dict[str, Any]],
    counts: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "protocol_id": cfg.protocol_id,
        "method_id": cfg.method_id,
        "metric_units": METRIC_UNITS,
        "counts": counts,
        "methods": methods,
    }
    # Convenience top-level aliases used by the review checklist and tables.
    for name in ("base", "single", "router", "oracle_mae", "oracle_mse", "oracle_cost"):
        if name in methods:
            document[name] = methods[name]
    if extra:
        document.update(extra)
    return document


def _sample_arrays(records: list[Tensor]) -> np.ndarray:
    if not records:
        return np.zeros(0, dtype=np.float64)
    return torch.cat(records, dim=0).detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Base-only evaluation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def evaluate_base_split(
    model,
    cfg: Config,
    prepared: PreparedData,
    split: str,
    device: torch.device,
    *,
    batch_size: int | None = None,
    run: RunArtifacts | None = None,
    save_predictions: bool = False,
) -> EvaluationResult:
    """Evaluate a Stage-A backbone (Base prediction only)."""
    stats: ScalerStats = prepared.scaler.to(device)
    graph: GraphContext = prepared.graph.to(device)
    loader = make_loader(
        prepared,
        split,
        batch_size or cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        seed=cfg.seed,
    )
    accumulator = _new_accumulator(cfg)
    thresholds = _regime_thresholds(prepared)
    regimes = RegimeAccumulator(cfg)
    if thresholds is not None:
        regimes.set_thresholds(thresholds)
    errors: list[Tensor] = []
    counts: list[Tensor] = []
    sample_ids: list[np.ndarray] = []
    predictions: list[Path] = []
    model.eval()
    for shard, batch in enumerate(loader):
        inp = move_input(batch.model_input, device)
        target = move_target(batch.target, device)
        out = model.encode_and_predict(inp, graph)
        pred_raw = stats.inverse(out.base_pred_z)
        accumulator.update(
            pred_raw,
            target.values_raw,
            target.valid,
            envelope=stats.envelope,
            sample_ids=target.sample_ids,
        )
        regimes.update(pred_raw, target)
        abs_sum, count = per_sample_abs_error(pred_raw, target.values_raw, target.valid)
        errors.append(abs_sum)
        counts.append(count)
        ids = target.sample_ids.detach().cpu().numpy().astype(np.int64)
        sample_ids.append(ids)
        if save_predictions and run is not None:
            predictions.append(
                run.save_predictions(
                    f"{split}_base",
                    pred_raw.detach().cpu().numpy(),
                    target.valid.detach().cpu().numpy(),
                    ids,
                    shard=shard,
                )
            )
    metrics = accumulator.result()
    ids_all = np.concatenate(sample_ids) if sample_ids else np.zeros(0, dtype=np.int64)
    document = _metric_document(
        cfg,
        methods={"base": metrics},
        counts={
            "split": split,
            "samples": int(ids_all.size),
            "valid_elements": metrics["counts"]["valid_elements"],
            "stage": "A",
        },
        extra={
            "split": split,
            "stage": "A",
            "regimes": {"base": regimes.result()},
            "bootstrap": {},
        },
    )
    return EvaluationResult(
        split=split,
        document=document,
        sample_ids=ids_all,
        predictions=predictions,
    )


# ---------------------------------------------------------------------------
# Full adapter evaluation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def evaluate_adapter_split(
    model,
    cfg: Config,
    prepared: PreparedData,
    split: str,
    device: torch.device,
    *,
    batch_size: int | None = None,
    run: RunArtifacts | None = None,
    save_predictions: bool = False,
    router_available: bool = True,
    include_bootstrap: bool = True,
) -> EvaluationResult:
    """Evaluate candidates, Router@1 and every metric-specific oracle.

    Selection is always one whole ``T x N`` candidate per sample.  The three
    oracle names are kept separate on purpose.
    """
    stats: ScalerStats = prepared.scaler.to(device)
    graph: GraphContext = prepared.graph.to(device)
    settings = OracleSettings.from_config(cfg)
    loader = make_loader(
        prepared,
        split,
        batch_size or cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        seed=cfg.seed,
    )
    num_candidates = int(model.num_candidates)
    if not router_available and num_candidates > 1:
        raise ValueError(
            "K > 1 candidate evaluation requires a trained Router checkpoint; "
            "refusing to report an untrained Top-1 policy as deployable"
        )
    if router_available and num_candidates < 2:
        raise ValueError("a Router cannot be evaluated with K=1")
    single = num_candidates == 1
    base_acc = _new_accumulator(cfg)
    router_acc = _new_accumulator(cfg)
    oracle_mae_acc = _new_accumulator(cfg)
    oracle_mse_acc = _new_accumulator(cfg)
    oracle_cost_acc = _new_accumulator(cfg)
    health = CandidateHealthAccumulator(num_candidates)

    thresholds = _regime_thresholds(prepared)
    regime_base = RegimeAccumulator(cfg)
    regime_router = RegimeAccumulator(cfg)
    regime_oracle_mae = RegimeAccumulator(cfg)
    if thresholds is not None:
        for item in (regime_base, regime_router, regime_oracle_mae):
            item.set_thresholds(thresholds)

    router_errors: list[Tensor] = []
    router_counts: list[Tensor] = []
    oracle_errors: list[Tensor] = []
    oracle_counts: list[Tensor] = []
    base_errors: list[Tensor] = []
    base_counts: list[Tensor] = []
    sample_ids: list[np.ndarray] = []
    router_indices: list[np.ndarray] = []
    router_logits_all: list[np.ndarray] = []
    router_actions_all: list[np.ndarray] = []
    predictions: list[Path] = []

    cost_regret_sum = 0.0
    cost_regret_count = 0
    tie_aware_sum = 0.0
    selected_cost_sum = 0.0
    oracle_cost_sum = 0.0

    model.eval()
    for shard, batch in enumerate(loader):
        inp = move_input(batch.model_input, device)
        target = move_target(batch.target, device)

        top1 = model.predict_top1(inp, graph)
        candidates = model.forward_candidates(inp, graph)

        pred_raw = stats.inverse(candidates.pred_z)
        base_raw = stats.inverse(candidates.base_pred_z)
        router_raw = stats.inverse(top1.pred_z)
        oracle_output = build_oracle(candidates, target, stats, graph, settings)

        mae, mse, valid_sample = candidate_errors(pred_raw, target)
        index_mae = select_whole_sample(mae, "oracle_mae", valid_sample)
        index_mse = select_whole_sample(mse, "oracle_mse", valid_sample)
        index_cost = select_whole_sample(
            oracle_output.cost, "oracle_cost", oracle_output.valid_sample
        )
        pred_oracle_mae = gather_candidates(pred_raw, index_mae)
        pred_oracle_mse = gather_candidates(pred_raw, index_mse)
        pred_oracle_cost = gather_candidates(pred_raw, index_cost)

        base_acc.update(
            base_raw,
            target.values_raw,
            target.valid,
            envelope=stats.envelope,
            sample_ids=target.sample_ids,
        )
        router_acc.update(
            router_raw,
            target.values_raw,
            target.valid,
            envelope=stats.envelope,
            sample_ids=target.sample_ids,
        )
        oracle_mae_acc.update(
            pred_oracle_mae,
            target.values_raw,
            target.valid,
            envelope=stats.envelope,
            sample_ids=target.sample_ids,
        )
        oracle_mse_acc.update(
            pred_oracle_mse,
            target.values_raw,
            target.valid,
            envelope=stats.envelope,
            sample_ids=target.sample_ids,
        )
        oracle_cost_acc.update(
            pred_oracle_cost,
            target.values_raw,
            target.valid,
            envelope=stats.envelope,
            sample_ids=target.sample_ids,
        )

        regime_base.update(base_raw, target)
        regime_router.update(router_raw, target)
        regime_oracle_mae.update(pred_oracle_mae, target)

        abs_sum, count = per_sample_abs_error(router_raw, target.values_raw, target.valid)
        router_errors.append(abs_sum)
        router_counts.append(count)
        abs_sum, count = per_sample_abs_error(
            pred_oracle_mae, target.values_raw, target.valid
        )
        oracle_errors.append(abs_sum)
        oracle_counts.append(count)
        abs_sum, count = per_sample_abs_error(base_raw, target.values_raw, target.valid)
        base_errors.append(abs_sum)
        base_counts.append(count)

        # Router cost regret uses the exact same whole-sample cost used by the
        # offline oracle; it is not an MAE lower bound.
        selected_cost = oracle_output.cost.gather(1, top1.index[:, None]).squeeze(1)
        min_cost = oracle_output.cost.min(dim=-1).values
        valid = oracle_output.valid_sample.bool()
        cost_regret_sum += float((selected_cost - min_cost)[valid].sum())
        cost_regret_count += int(valid.sum())
        selected_cost_sum += float(selected_cost[valid].sum())
        oracle_cost_sum += float(min_cost[valid].sum())
        tie = (oracle_output.cost - min_cost[:, None]).abs() <= 1e-6
        selected_tie = tie.gather(1, top1.index[:, None]).squeeze(1)
        tie_aware_sum += float(selected_tie[valid].sum())

        saturation = None
        decoder = getattr(model, "decoder", None)
        if decoder is not None and getattr(decoder, "last_saturation", None) is not None:
            saturation = float(decoder.last_saturation.detach().cpu())
        health.update(
            pred_raw,
            target,
            candidates.actions.normalized,
            stats.sigma,
            oracle_output.soft_target,
            index_mae,
            index_cost,
            top1.index if router_available else None,
            saturation,
        )

        ids = target.sample_ids.detach().cpu().numpy().astype(np.int64)
        sample_ids.append(ids)
        router_indices.append(top1.index.detach().cpu().numpy().astype(np.int64))
        router_logits_all.append(top1.logits.detach().float().cpu().numpy())
        router_actions_all.append(top1.selected_action.detach().float().cpu().numpy())
        if save_predictions and run is not None:
            predictions.append(
                run.save_predictions(
                    split,
                    router_raw.detach().cpu().numpy(),
                    target.valid.detach().cpu().numpy(),
                    ids,
                    shard=shard,
                )
            )

    base_metrics = base_acc.result()
    router_metrics = router_acc.result()
    oracle_mae_metrics = oracle_mae_acc.result()
    oracle_mse_metrics = oracle_mse_acc.result()
    oracle_cost_doc = oracle_cost_acc.result()

    selected_key = "single" if single else "router"
    methods = {
        "base": base_metrics,
        selected_key: router_metrics,
        "oracle_mae": oracle_mae_metrics,
        "oracle_mse": oracle_mse_metrics,
        "oracle_cost": oracle_cost_doc,
    }
    ids_all = np.concatenate(sample_ids) if sample_ids else np.zeros(0, dtype=np.int64)
    # ``predict_top1`` with K=1 returns index 0 and zero logits by contract;
    # saving those as ``top1_*`` would misleadingly look like a Router trace.
    router_index = None if single else (np.concatenate(router_indices) if router_indices else None)
    router_logits = None if single else (np.concatenate(router_logits_all) if router_logits_all else None)
    router_actions = None if single else (np.concatenate(router_actions_all) if router_actions_all else None)

    base_mae = base_metrics.get("raw_mae_all")
    router_mae = router_metrics.get("raw_mae_all")
    oracle_mae_value = oracle_mae_metrics.get("raw_mae_all")

    regret_doc = {
        "accuracy_regret_raw_mae": (
            None
            if router_mae is None or oracle_mae_value is None
            else router_mae - oracle_mae_value
        ),
        "cost_regret_mean": (
            cost_regret_sum / cost_regret_count if cost_regret_count else None
        ),
        "selected_cost_mean": (
            selected_cost_sum / cost_regret_count if cost_regret_count else None
        ),
        "oracle_cost_mean": (
            oracle_cost_sum / cost_regret_count if cost_regret_count else None
        ),
        "valid_samples": cost_regret_count,
        "tie_aware_top1_fraction": (
            tie_aware_sum / cost_regret_count if cost_regret_count else None
        ),
    }

    bootstrap_doc: dict[str, Any] = {}
    if include_bootstrap and ids_all.size > 0:
        r_sum = _sample_arrays(router_errors)
        r_cnt = _sample_arrays(router_counts)
        o_sum = _sample_arrays(oracle_errors)
        o_cnt = _sample_arrays(oracle_counts)
        b_sum = _sample_arrays(base_errors)
        b_cnt = _sample_arrays(base_counts)
        bootstrap_doc["router_vs_oracle_mae"] = paired_moving_block_bootstrap(
            r_sum,
            r_cnt,
            o_sum,
            o_cnt,
            block_origins=cfg.evaluation.block_origins,
            replicates=cfg.evaluation.bootstrap_replicates,
            seed=cfg.evaluation.bootstrap_seed,
        )
        bootstrap_doc["base_vs_router"] = paired_moving_block_bootstrap(
            b_sum,
            b_cnt,
            r_sum,
            r_cnt,
            block_origins=cfg.evaluation.block_origins,
            replicates=cfg.evaluation.bootstrap_replicates,
            seed=cfg.evaluation.bootstrap_seed,
        )

    document = _metric_document(
        cfg,
        methods=methods,
        counts={
            "split": split,
            "samples": int(ids_all.size),
            "valid_elements": router_metrics["counts"]["valid_elements"],
            "num_candidates": num_candidates,
            "router_available": bool(router_available),
        },
        extra={
            "split": split,
            "stage": "B" if single else ("D" if router_available else "C"),
            "adapter_kind": getattr(model, "adapter_kind", "structured"),
            "router_available": bool(router_available),
            "retention": {
                # K=1 is the Stage-B single baseline, not a Router policy.
                ("single_vs_base" if single else "router_vs_base"): retention(
                    base_mae, router_mae, oracle_mae_value
                ),
            },
            "regret": regret_doc,
            "candidate_health": health.result(),
            "regimes": {
                "base": regime_base.result(),
                "router": regime_router.result(),
                "oracle_mae": regime_oracle_mae.result(),
            },
            "bootstrap": bootstrap_doc,
        },
    )
    return EvaluationResult(
        split=split,
        document=document,
        sample_ids=ids_all,
        predictions=predictions,
        router_index=router_index,
        router_logits=router_logits,
        router_actions=router_actions,
    )


__all__ = [
    "EvaluationResult",
    "CandidateHealthAccumulator",
    "RegimeAccumulator",
    "evaluate_base_split",
    "evaluate_adapter_split",
    "METRIC_UNITS",
]
