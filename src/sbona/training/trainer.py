"""Training loops for Stage A (base), B/C (candidates) and D (router).

All metrics used for checkpoint selection are computed from validation data
only.  Optimizer steps are skipped for batches with no valid target elements.
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sbona.backbones import build_backbone
from sbona.config import Config
from sbona.contracts import ModelInput, ScalerStats, TargetBatch
from sbona.data.dataset import PreparedData, make_loader
from sbona.data.manifest import sha256_array
from sbona.evaluation.artifacts import RunArtifacts
from sbona.evaluation.metrics import MetricAccumulator
from sbona.evaluation.oracle import (
    candidate_errors,
    gather_candidates,
    oracle_mae,
    select_whole_sample,
)
from sbona.models.adapter import AdapterModel
from sbona.objectives.losses import (
    base_mse_loss,
    beta_schedule,
    candidate_loss,
    router_loss,
    separation_loss,
    wta_loss,
    wta_score_loss,
)
from sbona.objectives.oracle import OracleSettings, build_oracle
from sbona.training.checkpoint import (
    CheckpointError,
    checkpoint_sha256,
    load_checkpoint,
    load_state_dict_strict,
    restore_rng_state,
    save_checkpoint,
    upstream_lock_hash,
    validate_resume,
)
from sbona.training.stages import (
    build_optimizer,
    parameter_report,
    stage_budget,
    stage_lr,
)


@dataclass
class TrainResult:
    run_dir: Path
    stage: str
    best_checkpoint: Path | None
    latest_checkpoint: Path | None
    best_metric: float | None
    epochs_run: int


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def autocast_context(cfg: Config, device: torch.device):
    if cfg.training.precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def backbone_identity_payload(cfg: Config) -> dict[str, object]:
    """What makes two backbones interchangeable: seed, protocol and architecture.

    The adapter method is *not* part of it, so a Base run trained once can serve
    every adapter stage of the same seed (that is exactly the reuse the experiment
    plan requires).
    """
    return {
        "role": "backbone",
        "seed": int(cfg.seed),
        "protocol_id": str(cfg.protocol_id),
        "backbone": asdict(cfg.backbone),
    }


def adapter_identity_payload(cfg: Config, num_candidates: int) -> dict[str, object]:
    """What makes two adapters interchangeable: seed, method/kind, params and K."""
    return {
        "role": "adapter",
        "seed": int(cfg.seed),
        "protocol_id": str(cfg.protocol_id),
        "method_id": str(cfg.method_id),
        "num_candidates": int(num_candidates),
        "adapter": asdict(cfg.adapter),
    }


def _sha256_payload(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def backbone_identity_hash(cfg: Config) -> str:
    return _sha256_payload(backbone_identity_payload(cfg))


def adapter_identity_hash(cfg: Config, num_candidates: int) -> str:
    return _sha256_payload(adapter_identity_payload(cfg, num_candidates))


def make_hashes(prepared: PreparedData, cfg: Config, num_candidates: int) -> dict[str, object]:
    """Hash every split (probe schemes may only have ``train``/``val``)."""
    split_items = sorted(prepared.scheme.starts.items())
    split_payload = b"".join(
        name.encode("utf-8")
        + np.ascontiguousarray(np.asarray(starts, dtype=np.int64)).tobytes()
        for name, starts in split_items
    )
    scaler_payload = b"".join(
        [
            np.asarray([prepared.scaler.mu, prepared.scaler.sigma], dtype=np.float64).tobytes(),
            prepared.scaler.envelope.detach().cpu().numpy().astype(np.float32).tobytes(),
            prepared.scaler.envelope_fallback.detach().cpu().numpy().astype(np.uint8).tobytes(),
        ]
    )
    return {
        # ``config_hash`` covers schedule/batch/device too, so it cannot be used
        # to decide whether a checkpoint may be reused.  These two can.
        "config_hash": cfg.hash,
        "backbone_identity_hash": backbone_identity_hash(cfg),
        "adapter_identity_hash": adapter_identity_hash(cfg, num_candidates),
        "data_hash": sha256_array(prepared.data),
        "valid_hash": sha256_array(prepared.valid.astype(np.uint8)),
        "graph_hash": prepared.graph.graph_hash,
        "split_hash": hashlib.sha256(split_payload).hexdigest(),
        "split_names": [name for name, _ in split_items],
        "scaler_hash": hashlib.sha256(scaler_payload).hexdigest(),
        "num_nodes": int(prepared.num_nodes),
        "num_candidates": int(num_candidates),
    }


def move_input(inp: ModelInput, device: torch.device) -> ModelInput:
    return ModelInput(
        history=inp.history.to(device, non_blocking=True),
        observed=inp.observed.to(device, non_blocking=True),
        time_of_day=inp.time_of_day.to(device, non_blocking=True),
        day_of_week=inp.day_of_week.to(device, non_blocking=True),
    )


def move_target(target: TargetBatch, device: torch.device) -> TargetBatch:
    return TargetBatch(
        values_raw=target.values_raw.to(device, non_blocking=True),
        valid=target.valid.to(device, non_blocking=True),
        sample_ids=target.sample_ids.to(device, non_blocking=True),
    )


def _gpu_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def _epoch_seed(base_seed: int, epoch: int) -> int:
    return int(base_seed + 100_003 * epoch)


class _EpochLogger:
    def __init__(self, run: RunArtifacts, stage: str, device: torch.device) -> None:
        self.run = run
        self.stage = stage
        self.device = device
        self.started = time.perf_counter()

    def log(self, epoch: int, record: dict[str, object]) -> None:
        self.run.log_train(
            {
                "stage": self.stage,
                "epoch": epoch,
                "elapsed_seconds": time.perf_counter() - self.started,
                "gpu_peak_mb": _gpu_memory_mb(self.device),
                **record,
            }
        )


def _checkpoint_payload(
    *,
    model: nn.Module,
    state_dict: dict[str, object],
    stage: str,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    patience_counter: int,
    cfg: Config,
    hashes: dict[str, object],
    parent_checkpoint_hash: str | None,
    model_kind: str,
) -> dict[str, object]:
    return {
        "stage": stage,
        "model_kind": model_kind,
        "state_dict": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "amp": None,
        "epoch": epoch,
        # A real optimizer step, not a window count.  The contract uses this
        # value for logs/checkpoints, so skipped empty batches must not count.
        "global_step": int(global_step),
        "best_metric": best_metric,
        "patience_counter": patience_counter,
        "resolved_config": cfg.to_dict(),
        "hashes": hashes,
        "upstream_lock_hash": upstream_lock_hash(),
        "parent_checkpoint_hash": parent_checkpoint_hash,
        "parameter_report": parameter_report(model),
    }


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------


class BaseTrainer:
    def __init__(
        self,
        cfg: Config,
        prepared: PreparedData,
        run: RunArtifacts,
        device: torch.device,
        synthetic: bool = False,
    ) -> None:
        self.cfg = cfg
        self.prepared = prepared
        self.run = run
        self.device = device
        self.synthetic = synthetic
        self.stats: ScalerStats = prepared.scaler.to(device)
        self.graph = prepared.graph.to(device)
        self.model = build_backbone(
            cfg.backbone,
            num_nodes=prepared.num_nodes,
            in_steps=cfg.data.input_steps,
            out_steps=cfg.data.output_steps,
            steps_per_day=prepared.steps_per_day,
        )
        self.model.set_mode("full")
        self.model.to(device)

    def _loaders(self):
        train = make_loader(
            self.prepared,
            "train",
            self.cfg.training.initial_batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
            seed=self.cfg.seed,
        )
        val = make_loader(
            self.prepared,
            "val",
            self.cfg.training.eval_batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            seed=self.cfg.seed,
        )
        return train, val

    def _train_epoch(
        self, optimizer: torch.optim.Optimizer, epoch: int
    ) -> tuple[float, int, int]:
        """Return ``(weighted_loss, valid_elements, optimizer_steps)``."""
        self.model.train()
        total_loss = 0.0
        total_count = 0
        steps = 0
        loader = make_loader(
            self.prepared,
            "train",
            self.cfg.training.initial_batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
            seed=_epoch_seed(self.cfg.seed, epoch),
        )
        for batch in loader:
            inp = move_input(batch.model_input, self.device)
            target = move_target(batch.target, self.device)
            target_z = self.stats.standardize(target.values_raw)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.cfg, self.device):
                out = self.model.encode_and_predict(inp, self.graph)
                loss, count = base_mse_loss(out.base_pred_z, target_z, target.valid)
            if count == 0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip)
            optimizer.step()
            steps += 1
            total_loss += float(loss.detach()) * count
            total_count += count
        return total_loss / max(total_count, 1), total_count, steps

    @torch.no_grad()
    def _validate(self) -> dict[str, object]:
        self.model.eval()
        _, val_loader = self._loaders()
        accumulator = MetricAccumulator(
            self.cfg.data.mape_min_abs_target,
            self.cfg.evaluation.horizons,
            collect_per_sample=True,
        )
        for batch in val_loader:
            inp = move_input(batch.model_input, self.device)
            target = move_target(batch.target, self.device)
            out = self.model.encode_and_predict(inp, self.graph)
            pred_raw = self.stats.inverse(out.base_pred_z)
            accumulator.update(
                pred_raw,
                target.values_raw,
                target.valid,
                envelope=self.stats.envelope,
                sample_ids=target.sample_ids,
            )
        return accumulator.result()

    def fit(
        self,
        resume: str | Path | None = None,
        max_epochs: int | None = None,
        patience_override: int | None = None,
    ) -> TrainResult:
        max_epochs = max_epochs or stage_budget(self.cfg, "A")[0]
        patience = patience_override or stage_budget(self.cfg, "A")[1]
        hashes = make_hashes(self.prepared, self.cfg, num_candidates=1)
        optimizer = build_optimizer(
            self.model, stage_lr(self.cfg, "A"), self.cfg.training.weight_decay
        )
        start_epoch = 0
        best_metric: float | None = None
        patience_counter = 0
        global_step = 0
        parent_hash = None
        if resume is not None:
            payload = load_checkpoint(resume, map_location=self.device)
            if payload.get("stage") != "A":
                raise CheckpointError(f"expected stage A checkpoint, got {payload.get('stage')}")
            validate_resume(payload, hashes)
            load_state_dict_strict(self.model, payload["state_dict"], "backbone")
            optimizer.load_state_dict(payload["optimizer"])
            start_epoch = int(payload["epoch"]) + 1
            best_metric = payload.get("best_metric")
            patience_counter = int(payload.get("patience_counter", 0))
            global_step = int(payload.get("global_step", 0))
            parent_hash = checkpoint_sha256(resume)
            restore_rng_state(payload["rng"])
            self.model.set_mode("full")
        logger = _EpochLogger(self.run, "A", self.device)
        latest = None
        best_path = None
        epochs_run = 0
        for epoch in range(start_epoch, max_epochs):
            train_loss, count, steps = self._train_epoch(optimizer, epoch)
            if count == 0 or steps == 0:
                raise RuntimeError("no valid training samples in an epoch; refusing to train")
            global_step += steps
            metrics = self._validate()
            metric = metrics.get("raw_mae_all")
            if metric is None:
                raise RuntimeError("validation MAE is undefined; refusing to select a checkpoint")
            improved = best_metric is None or metric < best_metric - self.cfg.training.checkpoint_min_delta
            if improved:
                best_metric = float(metric)
                patience_counter = 0
            else:
                patience_counter += 1
            payload = _checkpoint_payload(
                model=self.model,
                state_dict=self.model.state_dict(),
                stage="A",
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                patience_counter=patience_counter,
                cfg=self.cfg,
                hashes=hashes,
                parent_checkpoint_hash=parent_hash,
                model_kind="base",
            )
            latest = save_checkpoint(self.run.path("checkpoint_latest.pt"), payload)
            if improved:
                best_path = save_checkpoint(self.run.path("checkpoint_best.pt"), payload)
            logger.log(
                epoch,
                {
                    "train_loss": train_loss,
                    "train_valid_samples": count,
                    "val": metrics,
                    "best_metric": best_metric,
                    "patience_counter": patience_counter,
                },
            )
            epochs_run += 1
            if patience_counter >= patience:
                break
        return TrainResult(
            run_dir=self.run.run_dir,
            stage="A",
            best_checkpoint=best_path,
            latest_checkpoint=latest,
            best_metric=best_metric,
            epochs_run=epochs_run,
        )


# ---------------------------------------------------------------------------
# Stage B / C
# ---------------------------------------------------------------------------


class CandidateTrainer:
    def __init__(
        self,
        cfg: Config,
        prepared: PreparedData,
        model: AdapterModel,
        run: RunArtifacts,
        device: torch.device,
        stage: str,
        synthetic: bool = False,
    ) -> None:
        if stage not in ("B", "C"):
            raise ValueError("candidate stage must be B or C")
        self.cfg = cfg
        self.prepared = prepared
        self.model = model
        self.run = run
        self.device = device
        self.stage = stage
        self.synthetic = synthetic
        self.stats = prepared.scaler.to(device)
        self.graph = prepared.graph.to(device)
        self.settings = OracleSettings.from_config(cfg)
        self.model.set_stage(stage)
        self.model.to(device)

    def _loader(self, epoch: int, train: bool):
        return make_loader(
            self.prepared,
            "train" if train else "val",
            self.cfg.training.initial_batch_size if train else self.cfg.training.eval_batch_size,
            shuffle=train,
            num_workers=self.cfg.training.num_workers,
            seed=_epoch_seed(self.cfg.seed, epoch) if train else self.cfg.seed,
        )

    def _train_epoch(
        self, optimizer: torch.optim.Optimizer, epoch: int
    ) -> tuple[float, float, int, int]:
        """Return ``(weighted_loss, weighted_separation, valid_elements, steps)``."""
        self.model.train()
        total_loss = 0.0
        total_sep = 0.0
        total_count = 0
        steps = 0
        for batch in self._loader(epoch, train=True):
            inp = move_input(batch.model_input, self.device)
            target = move_target(batch.target, self.device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.cfg, self.device):
                out = self.model.forward_candidates(inp, self.graph)
                oracle = build_oracle(out, target, self.stats, self.graph, self.settings)
                loss_kind = getattr(self.model, "loss_kind", "weighted")
                if loss_kind == "wta":
                    loss, count = wta_loss(oracle.error, oracle.valid_sample)
                    if count > 0 and getattr(self.model, "uses_score_selector", False):
                        if out.candidate_logits is None:
                            raise RuntimeError(
                                "MCL-WTA model did not return candidate_logits; the "
                                "score head cannot be trained"
                            )
                        score, _ = wta_score_loss(
                            out.candidate_logits, oracle.error, oracle.valid_sample
                        )
                        loss = loss + self.cfg.loss.wta_score * score
                    sep_value = 0.0
                elif loss_kind == "weighted":
                    loss, count = candidate_loss(
                        oracle,
                        epsilon=self.cfg.loss.epsilon_floor,
                        lambda_consistency=self.cfg.loss.consistency,
                    )
                    sep_value = 0.0
                    if count > 0 and getattr(self.model, "supports_separation", True):
                        sep, _ = separation_loss(
                            out.actions.normalized,
                            oracle.valid_sample,
                            margin=self.cfg.loss.separation_margin,
                        )
                        loss = loss + self.cfg.loss.separation * sep
                        sep_value = float(sep.detach())
                else:
                    raise RuntimeError(f"unknown candidate loss kind {loss_kind!r}")
            if count == 0:
                continue
            loss.backward()
            self._clip()
            optimizer.step()
            steps += 1
            total_loss += float(loss.detach()) * count
            total_sep += sep_value * count
            total_count += count
        return (
            total_loss / max(total_count, 1),
            total_sep / max(total_count, 1),
            total_count,
            steps,
        )

    def _clip(self) -> None:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, self.cfg.training.gradient_clip)

    @torch.no_grad()
    def _validate(self) -> dict[str, object]:
        """Validation metrics for candidate checkpoint selection.

        ``alpha == 1`` uses the whole-sample MAE oracle, as specified by the
        Stage C table.  ``alpha < 1`` uses the mean of the *best* training cost
        ``E + lambda*P``.  The MAE-oracle number is always reported separately
        so a cost-selected result is never mislabelled as an MAE oracle.
        """
        self.model.eval()
        mae_accumulator = MetricAccumulator(
            self.cfg.data.mape_min_abs_target,
            self.cfg.evaluation.horizons,
            collect_per_sample=True,
        )
        cost_accumulator = MetricAccumulator(
            self.cfg.data.mape_min_abs_target,
            self.cfg.evaluation.horizons,
            collect_per_sample=True,
        )
        cost_sum = 0.0
        cost_count = 0
        for batch in self._loader(0, train=False):
            inp = move_input(batch.model_input, self.device)
            target = move_target(batch.target, self.device)
            out = self.model.forward_candidates(inp, self.graph)
            pred_raw = self.stats.inverse(out.pred_z)

            mae_selection = oracle_mae(
                pred_raw,
                target,
                self.cfg.data.mape_min_abs_target,
                self.cfg.evaluation.horizons,
            )
            mae_accumulator.update(
                mae_selection.pred_raw,
                target.values_raw,
                target.valid,
                envelope=self.stats.envelope,
                sample_ids=target.sample_ids,
            )

            if self.cfg.oracle.alpha < 1.0:
                oracle = build_oracle(out, target, self.stats, self.graph, self.settings)
                index = select_whole_sample(
                    oracle.cost, "oracle_cost", oracle.valid_sample
                )
                selected = gather_candidates(pred_raw, index)
                cost_accumulator.update(
                    selected,
                    target.values_raw,
                    target.valid,
                    envelope=self.stats.envelope,
                    sample_ids=target.sample_ids,
                )
                cost_sum += float(oracle.cost.gather(1, index[:, None]).sum())
                cost_count += int(oracle.valid_sample.sum())

        # ``metrics`` mirrors the MAE oracle so downstream tables can use the
        # usual raw_mae_all key without ambiguity.
        metrics = mae_accumulator.result()
        metrics["candidate_oracle_mae_all"] = metrics["raw_mae_all"]
        if self.cfg.oracle.alpha < 1.0:
            cost_metrics = cost_accumulator.result()
            metrics["candidate_cost_selected_mae_all"] = cost_metrics["raw_mae_all"]
            metrics["candidate_best_cost_mean"] = (
                cost_sum / cost_count if cost_count else None
            )
            metrics["selection_metric"] = metrics["candidate_best_cost_mean"]
        else:
            metrics["candidate_cost_selected_mae_all"] = None
            metrics["candidate_best_cost_mean"] = None
            metrics["selection_metric"] = metrics["candidate_oracle_mae_all"]
        return metrics

    def fit(
        self,
        resume: str | Path | None = None,
        max_epochs: int | None = None,
        patience_override: int | None = None,
        parent_checkpoint: str | Path | None = None,
    ) -> TrainResult:
        max_epochs = max_epochs or stage_budget(self.cfg, self.stage)[0]
        patience = patience_override or stage_budget(self.cfg, self.stage)[1]
        hashes = make_hashes(self.prepared, self.cfg, num_candidates=self.model.num_candidates)
        optimizer = build_optimizer(
            self.model, stage_lr(self.cfg, self.stage), self.cfg.training.weight_decay
        )
        start_epoch = 0
        best_metric: float | None = None
        patience_counter = 0
        global_step = 0
        parent_hash = (
            checkpoint_sha256(parent_checkpoint) if parent_checkpoint is not None else None
        )
        if resume is not None:
            payload = load_checkpoint(resume, map_location=self.device)
            if payload.get("stage") != self.stage:
                raise CheckpointError(
                    f"expected stage {self.stage} checkpoint, got {payload.get('stage')}"
                )
            validate_resume(payload, hashes)
            load_state_dict_strict(self.model, payload["state_dict"], "adapter")
            optimizer.load_state_dict(payload["optimizer"])
            start_epoch = int(payload["epoch"]) + 1
            best_metric = payload.get("best_metric")
            patience_counter = int(payload.get("patience_counter", 0))
            global_step = int(payload.get("global_step", 0))
            parent_hash = checkpoint_sha256(resume)
            restore_rng_state(payload["rng"])
            self.model.set_stage(self.stage)
        logger = _EpochLogger(self.run, self.stage, self.device)
        latest = None
        best_path = None
        epochs_run = 0
        for epoch in range(start_epoch, max_epochs):
            train_loss, sep, count, steps = self._train_epoch(optimizer, epoch)
            if count == 0 or steps == 0:
                raise RuntimeError("no valid training samples in an epoch; refusing to train")
            global_step += steps
            metrics = self._validate()
            metric = metrics.get("selection_metric")
            if metric is None:
                raise RuntimeError("candidate selection metric is undefined")
            improved = best_metric is None or metric < best_metric - self.cfg.training.checkpoint_min_delta
            if improved:
                best_metric = float(metric)
                patience_counter = 0
            else:
                patience_counter += 1
            payload = _checkpoint_payload(
                model=self.model,
                state_dict=self.model.state_dict(),
                stage=self.stage,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                patience_counter=patience_counter,
                cfg=self.cfg,
                hashes=hashes,
                parent_checkpoint_hash=parent_hash,
                model_kind=getattr(self.model, "adapter_kind", "structured"),
            )
            latest = save_checkpoint(self.run.path("checkpoint_latest.pt"), payload)
            if improved:
                best_path = save_checkpoint(self.run.path("checkpoint_best.pt"), payload)
            logger.log(
                epoch,
                {
                    "train_loss": train_loss,
                    "train_separation": sep,
                    "train_valid_samples": count,
                    "val": metrics,
                    "best_metric": best_metric,
                    "patience_counter": patience_counter,
                },
            )
            epochs_run += 1
            if patience_counter >= patience:
                break
        return TrainResult(
            run_dir=self.run.run_dir,
            stage=self.stage,
            best_checkpoint=best_path,
            latest_checkpoint=latest,
            best_metric=best_metric,
            epochs_run=epochs_run,
        )


# ---------------------------------------------------------------------------
# Stage D
# ---------------------------------------------------------------------------


class RouterTrainer:
    def __init__(
        self,
        cfg: Config,
        prepared: PreparedData,
        model: AdapterModel,
        run: RunArtifacts,
        device: torch.device,
    ) -> None:
        if int(model.num_candidates) < 2:
            raise ValueError("router training requires K >= 2")
        self.cfg = cfg
        self.prepared = prepared
        self.model = model
        self.run = run
        self.device = device
        self.stats = prepared.scaler.to(device)
        self.graph = prepared.graph.to(device)
        self.settings = OracleSettings.from_config(cfg)
        self.model.set_stage("D")
        self.model.to(device)
        self._max_epochs: int | None = None

    def _loader(self, epoch: int, train: bool):
        return make_loader(
            self.prepared,
            "train" if train else "val",
            self.cfg.training.initial_batch_size if train else self.cfg.training.eval_batch_size,
            shuffle=train,
            num_workers=self.cfg.training.num_workers,
            seed=_epoch_seed(self.cfg.seed, epoch) if train else self.cfg.seed,
        )

    def _train_epoch(
        self, optimizer: torch.optim.Optimizer, epoch: int
    ) -> tuple[float, float, float, float, int, int]:
        """Return ``(loss, kl, regret, balance, valid_elements, steps)``."""
        self.model.train()  # router train; frozen candidate modules stay eval
        max_epochs = self._max_epochs or self.cfg.training.max_epochs["router"]
        beta = beta_schedule(epoch, max_epochs, self.cfg.loss.balance_max)
        total_loss = 0.0
        total_kl = 0.0
        total_regret = 0.0
        total_balance = 0.0
        total_count = 0
        steps = 0
        for batch in self._loader(epoch, train=True):
            inp = move_input(batch.model_input, self.device)
            target = move_target(batch.target, self.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                out = self.model.forward_candidates(inp, self.graph)
                oracle = build_oracle(out, target, self.stats, self.graph, self.settings)
            logits = self.model.router(out.h_global.detach())
            breakdown = router_loss(
                logits,
                oracle,
                beta=beta,
                kl_weight=self.cfg.loss.router_kl,
                regret_weight=self.cfg.loss.regret,
            )
            if breakdown.count == 0:
                continue
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.router.parameters() if p.requires_grad],
                self.cfg.training.gradient_clip,
            )
            optimizer.step()
            steps += 1
            total_loss += float(breakdown.total.detach()) * breakdown.count
            total_kl += float(breakdown.kl.detach()) * breakdown.count
            total_regret += float(breakdown.regret.detach()) * breakdown.count
            total_balance += float(breakdown.balance.detach()) * breakdown.count
            total_count += breakdown.count
        denominator = max(total_count, 1)
        return (
            total_loss / denominator,
            total_kl / denominator,
            total_regret / denominator,
            total_balance / denominator,
            total_count,
            steps,
        )

    @torch.no_grad()
    def _validate(self) -> dict[str, object]:
        self.model.eval()
        router_acc = MetricAccumulator(
            self.cfg.data.mape_min_abs_target,
            self.cfg.evaluation.horizons,
            collect_per_sample=True,
        )
        oracle_acc = MetricAccumulator(
            self.cfg.data.mape_min_abs_target,
            self.cfg.evaluation.horizons,
            collect_per_sample=True,
        )
        regret_sum = 0.0
        regret_count = 0
        for batch in self._loader(0, train=False):
            inp = move_input(batch.model_input, self.device)
            target = move_target(batch.target, self.device)
            top1 = self.model.predict_top1(inp, self.graph)
            router_pred_raw = self.stats.inverse(top1.pred_z)
            router_acc.update(
                router_pred_raw,
                target.values_raw,
                target.valid,
                envelope=self.stats.envelope,
                sample_ids=target.sample_ids,
            )
            out = self.model.forward_candidates(inp, self.graph)
            pred_raw = self.stats.inverse(out.pred_z)
            selection = oracle_mae(
                pred_raw,
                target,
                self.cfg.data.mape_min_abs_target,
                self.cfg.evaluation.horizons,
            )
            oracle_acc.update(
                selection.pred_raw,
                target.values_raw,
                target.valid,
                envelope=self.stats.envelope,
                sample_ids=target.sample_ids,
            )
            candidate_mae, _, valid = candidate_errors(pred_raw, target)
            router_mae = candidate_mae.gather(1, top1.index[:, None]).squeeze(1)
            regret = router_mae - candidate_mae.min(dim=-1).values
            regret_sum += float(regret[valid].sum())
            regret_count += int(valid.sum())
        router_metrics = router_acc.result()
        oracle_metrics_doc = oracle_acc.result()
        return {
            "router": router_metrics,
            "oracle_mae": oracle_metrics_doc,
            "router_raw_mae_all": router_metrics["raw_mae_all"],
            "oracle_mae_all": oracle_metrics_doc["raw_mae_all"],
            "router_regret_mean": regret_sum / regret_count if regret_count else None,
            "selection_metric": router_metrics["raw_mae_all"],
        }

    def fit(
        self,
        resume: str | Path | None = None,
        max_epochs: int | None = None,
        patience_override: int | None = None,
    ) -> TrainResult:
        max_epochs = max_epochs or stage_budget(self.cfg, "D")[0]
        patience = patience_override or stage_budget(self.cfg, "D")[1]
        self._max_epochs = max_epochs
        hashes = make_hashes(self.prepared, self.cfg, num_candidates=self.model.num_candidates)
        optimizer = build_optimizer(
            self.model.router, stage_lr(self.cfg, "D"), self.cfg.training.weight_decay
        )
        start_epoch = 0
        best_metric: float | None = None
        patience_counter = 0
        global_step = 0
        parent_hash = None
        if resume is not None:
            payload = load_checkpoint(resume, map_location=self.device)
            if payload.get("stage") != "D":
                raise CheckpointError(f"expected stage D checkpoint, got {payload.get('stage')}")
            validate_resume(payload, hashes)
            load_state_dict_strict(self.model, payload["state_dict"], "router model")
            optimizer.load_state_dict(payload["optimizer"])
            start_epoch = int(payload["epoch"]) + 1
            best_metric = payload.get("best_metric")
            patience_counter = int(payload.get("patience_counter", 0))
            global_step = int(payload.get("global_step", 0))
            parent_hash = checkpoint_sha256(resume)
            restore_rng_state(payload["rng"])
            self.model.set_stage("D")
        logger = _EpochLogger(self.run, "D", self.device)
        latest = None
        best_path = None
        epochs_run = 0
        for epoch in range(start_epoch, max_epochs):
            (
                train_loss,
                kl,
                regret,
                balance,
                count,
                steps,
            ) = self._train_epoch(optimizer, epoch)
            if count == 0 or steps == 0:
                raise RuntimeError("no valid training samples in an epoch; refusing to train")
            global_step += steps
            metrics = self._validate()
            metric = metrics.get("selection_metric")
            if metric is None:
                raise RuntimeError("Router@1 validation MAE is undefined")
            improved = best_metric is None or metric < best_metric - self.cfg.training.checkpoint_min_delta
            if improved:
                best_metric = float(metric)
                patience_counter = 0
            else:
                patience_counter += 1
            payload = _checkpoint_payload(
                model=self.model,
                state_dict=self.model.state_dict(),
                stage="D",
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                patience_counter=patience_counter,
                cfg=self.cfg,
                hashes=hashes,
                parent_checkpoint_hash=parent_hash,
                model_kind=getattr(self.model, "adapter_kind", "structured"),
            )
            latest = save_checkpoint(self.run.path("checkpoint_latest.pt"), payload)
            if improved:
                best_path = save_checkpoint(self.run.path("checkpoint_best.pt"), payload)
            logger.log(
                epoch,
                {
                    "train_loss": train_loss,
                    "train_kl": kl,
                    "train_regret": regret,
                    "train_balance": balance,
                    "train_valid_samples": count,
                    "val": metrics,
                    "best_metric": best_metric,
                    "patience_counter": patience_counter,
                    "beta": beta_schedule(epoch, max_epochs, self.cfg.loss.balance_max),
                },
            )
            epochs_run += 1
            if patience_counter >= patience:
                break
        return TrainResult(
            run_dir=self.run.run_dir,
            stage="D",
            best_checkpoint=best_path,
            latest_checkpoint=latest,
            best_metric=best_metric,
            epochs_run=epochs_run,
        )


__all__ = [
    "TrainResult",
    "BaseTrainer",
    "CandidateTrainer",
    "RouterTrainer",
    "resolve_device",
    "make_hashes",
    "backbone_identity_hash",
    "adapter_identity_hash",
    "move_input",
    "move_target",
    "autocast_context",
]
