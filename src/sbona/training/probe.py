"""Temporal F/R/Q route-probe protocol.

Within the official training interval ``[0, a)``:

* F = ``[0, 0.7a)`` fits all statistics and trains a probe-specific Base and
  candidate model;
* R = ``[0.7a, 0.85a)`` generates frozen candidate labels/costs and trains
  linear / MLP probes;
* Q = ``[0.85a, a)`` is used only for probe evaluation.

No random K-fold over the training interval is allowed: that would leak the
future labels that the candidate model was trained against.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from sbona.baselines import build_adapter
from sbona.config import Config
from sbona.contracts import GraphContext, ScalerStats
from sbona.data.dataset import PreparedData, make_loader
from sbona.data.scaler import fit_scaler
from sbona.data.windows import WindowScheme, build_subrange_scheme
from sbona.evaluation.artifacts import RunArtifacts, write_json
from sbona.evaluation.diversity import entropy_metrics, hard_usage
from sbona.objectives.oracle import OracleSettings, build_oracle
from sbona.training.checkpoint import load_checkpoint, load_state_dict_strict
from sbona.training.trainer import (
    BaseTrainer,
    CandidateTrainer,
    move_input,
    move_target,
)


class ProbeError(RuntimeError):
    pass


class SoftClassifier(nn.Module):
    def __init__(self, input_dim: int, num_candidates: int, kind: str = "linear") -> None:
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(input_dim, num_candidates)
        elif kind == "mlp":
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, num_candidates),
            )
        else:
            raise ValueError(f"unknown probe kind {kind!r}")

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _train_soft_probe(
    x_train: Tensor,
    w_train: Tensor,
    x_val: Tensor,
    w_val: Tensor,
    num_candidates: int,
    kind: str,
    epochs: int,
    lr: float,
    seed: int,
) -> SoftClassifier:
    torch.manual_seed(seed)
    model = SoftClassifier(x_train.shape[1], num_candidates, kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_loss = math.inf
    patience = 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = -(w_train * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = float(
                -(w_val * torch.log_softmax(val_logits, dim=-1)).sum(dim=-1).mean()
            )
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def _probe_predict(model: SoftClassifier, x: Tensor) -> Tensor:
    model.eval()
    return torch.argmax(model(x), dim=-1)


def _view(
    prepared: PreparedData,
    starts: dict[str, np.ndarray],
    scaler: ScalerStats,
) -> PreparedData:
    from sbona.data.windows import SplitBoundaries

    all_starts = np.concatenate([v for v in starts.values() if v.size]) if starts else np.array([], dtype=np.int64)
    boundaries = SplitBoundaries(
        num_steps=prepared.num_steps,
        train_end=int(all_starts.min()) if all_starts.size else 0,
        val_end=int(all_starts.max()) + prepared.scheme.output_steps
        if all_starts.size
        else 0,
    )
    scheme = WindowScheme(
        num_steps=prepared.num_steps,
        input_steps=prepared.scheme.input_steps,
        output_steps=prepared.scheme.output_steps,
        boundaries=boundaries,
        starts=starts,
    )
    return replace(prepared, scheme=scheme, scaler=scaler)


@torch.no_grad()
def _extract(
    model,
    prepared: PreparedData,
    split: str,
    scaler: ScalerStats,
    graph: GraphContext,
    settings: OracleSettings,
    device: torch.device,
    batch_size: int,
) -> dict[str, Tensor]:
    model.eval()
    loader = make_loader(prepared, split, batch_size, shuffle=False, seed=0)
    h_globals = []
    abs_sums = []
    counts = []
    soft_targets = []
    costs = []
    sample_ids = []
    actions = []
    for batch in loader:
        inp = move_input(batch.model_input, device)
        target = move_target(batch.target, device)
        out = model.forward_candidates(inp, graph)
        pred_raw = scaler.inverse(out.pred_z)
        error = (pred_raw - target.values_raw[:, None]).abs()
        mask = target.valid[:, None]
        abs_sum = (error * mask).sum(dim=(2, 3, 4))
        count = mask.expand_as(error).sum(dim=(2, 3, 4))[:, 0]
        oracle = build_oracle(out, target, scaler, graph, settings)
        h_globals.append(out.h_global.detach().cpu())
        abs_sums.append(abs_sum.detach().cpu())
        counts.append(count.detach().cpu())
        soft_targets.append(oracle.soft_target.detach().cpu())
        costs.append(oracle.cost.detach().cpu())
        sample_ids.append(target.sample_ids.detach().cpu())
        actions.append(out.actions.normalized.detach().cpu())
    return {
        "h_global": torch.cat(h_globals, dim=0),
        "abs_sum": torch.cat(abs_sums, dim=0),
        "count": torch.cat(counts, dim=0),
        "soft_target": torch.cat(soft_targets, dim=0),
        "cost": torch.cat(costs, dim=0),
        "sample_ids": torch.cat(sample_ids, dim=0),
        "actions": torch.cat(actions, dim=0),
    }


def _selected_mae(abs_sum: Tensor, count: Tensor, index: Tensor) -> float:
    selected = abs_sum.gather(1, index[:, None]).squeeze(1)
    denominator = float(count.sum())
    return float(selected.sum()) / denominator if denominator > 0 else float("nan")


def _evaluate_probe(data: dict[str, Tensor], index: Tensor, name: str) -> dict[str, object]:
    abs_sum = data["abs_sum"]
    count = data["count"]
    oracle_index = torch.argmin(abs_sum, dim=-1)
    oracle_mae = _selected_mae(abs_sum, count, oracle_index)
    selected_mae = _selected_mae(abs_sum, count, index)
    num_candidates = abs_sum.shape[1]
    top1 = float((index == oracle_index).float().mean())
    if num_candidates >= 2:
        top2_indices = torch.topk(-abs_sum, k=2, dim=-1).indices
        top2 = float((top2_indices == index[:, None]).any(dim=-1).float().mean())
    else:
        top2 = top1
    # Contract 12.2: near-ties within 1e-6 of the best cost/MAE count as
    # equivalent, and the fixed-argmin label is reported alongside.
    best_abs = abs_sum.min(dim=-1, keepdim=True).values
    tie_mask = (abs_sum - best_abs).abs() <= 1e-6
    selected_tie = tie_mask.gather(1, index[:, None]).squeeze(1)
    tie_aware_top1 = float(selected_tie.float().mean())
    tie_fraction = float(tie_mask.sum(dim=-1).gt(1).float().mean())
    return {
        "name": name,
        "selected_mae": selected_mae,
        "oracle_mae": oracle_mae,
        "regret": selected_mae - oracle_mae,
        "top1_accuracy": top1,
        "top1_accuracy_tie_aware": tie_aware_top1,
        "top2_accuracy": top2,
        "tie_fraction": tie_fraction,
        "branch_usage": hard_usage(index, num_candidates),
        "soft_label_entropy": entropy_metrics(data["soft_target"]),
        "samples": int(abs_sum.shape[0]),
    }


def _block_shuffle(x: Tensor, block: int, seed: int) -> Tensor:
    if block < 1:
        raise ProbeError("block must be >= 1")
    rng = np.random.default_rng(seed)
    num_blocks = math.ceil(x.shape[0] / block)
    order = rng.permutation(num_blocks)
    chunks = [x[i * block : (i + 1) * block] for i in range(num_blocks)]
    return torch.cat([chunks[i] for i in order], dim=0)[: x.shape[0]]


def run_route_probe(
    cfg: Config,
    prepared: PreparedData,
    run: RunArtifacts,
    device: torch.device,
    candidate_run: str | Path,
) -> dict[str, object]:
    """Run the full probe protocol and write ``probe_results.json``."""
    a = prepared.scheme.boundaries.train_end
    f_end = int(math.floor(0.7 * a))
    r_end = int(math.floor(0.85 * a))
    if f_end <= prepared.scheme.input_steps + prepared.scheme.output_steps:
        raise ProbeError("F interval is too short for a meaningful probe")
    if r_end <= f_end + prepared.scheme.output_steps:
        raise ProbeError("R interval is too short")
    if a <= r_end + prepared.scheme.output_steps:
        raise ProbeError("Q interval is too short")

    f_fit_end = int(math.floor(0.8 * f_end))
    r_train_end = int(math.floor(f_end + 0.8 * (r_end - f_end)))
    f_scheme = build_subrange_scheme(
        prepared.num_steps,
        prepared.scheme.input_steps,
        prepared.scheme.output_steps,
        {"train": (0, f_fit_end), "val": (f_fit_end, f_end)},
    )
    # Fits statistics on the whole F interval, not on R/Q.
    f_scaler = fit_scaler(
        prepared.data,
        prepared.valid,
        train_end=f_end,
        quantile=cfg.data.envelope_quantile,
        envelope_min_count=cfg.data.envelope_min_count,
    )
    f_view = _view(prepared, f_scheme.starts, f_scaler)
    f_view.steps_per_day = prepared.steps_per_day

    # 1) Probe-specific Base on F.
    base_run = RunArtifacts(run.path("probe_base"))
    base_trainer = BaseTrainer(cfg, f_view, base_run, device, synthetic=prepared.manifest.extra.get("synthetic", False))
    base_result = base_trainer.fit()
    if base_result.best_checkpoint is None:
        raise ProbeError("probe Base training produced no checkpoint")
    base_payload = load_checkpoint(base_result.best_checkpoint, map_location=device)

    # 2) Probe-specific candidates on F, loading the F-trained Base.
    model = build_adapter(
        "structured",
        cfg,
        num_nodes=prepared.num_nodes,
        steps_per_day=prepared.steps_per_day,
    )
    load_state_dict_strict(model.backbone, base_payload["state_dict"], "probe backbone")
    candidate_run_dir = RunArtifacts(run.path("probe_candidates"))
    candidate_trainer = CandidateTrainer(
        cfg, f_view, model, candidate_run_dir, device, "C"
    )
    candidate_result = candidate_trainer.fit(parent_checkpoint=base_result.best_checkpoint)
    if candidate_result.best_checkpoint is None:
        raise ProbeError("probe candidate training produced no checkpoint")
    candidate_payload = load_checkpoint(candidate_result.best_checkpoint, map_location=device)
    load_state_dict_strict(model, candidate_payload["state_dict"], "probe candidates")
    model.set_stage("D")
    model.eval()

    graph = prepared.graph.to(device)
    settings = OracleSettings.from_config(cfg)
    r_train_starts = np.arange(
        max(f_end, prepared.scheme.input_steps), r_train_end - prepared.scheme.output_steps + 1
    )
    r_val_starts = np.arange(
        max(r_train_end, prepared.scheme.input_steps), r_end - prepared.scheme.output_steps + 1
    )
    q_starts = np.arange(
        max(r_end, prepared.scheme.input_steps), a - prepared.scheme.output_steps + 1
    )
    if r_train_starts.size == 0 or r_val_starts.size == 0 or q_starts.size == 0:
        raise ProbeError("empty R/Q probe windows")

    r_train_view = _view(prepared, {"train": r_train_starts}, f_scaler)
    r_val_view = _view(prepared, {"train": r_val_starts}, f_scaler)
    q_view = _view(prepared, {"train": q_starts}, f_scaler)

    r_train = _extract(model, r_train_view, "train", f_scaler, graph, settings, device, cfg.training.eval_batch_size)
    r_val = _extract(model, r_val_view, "train", f_scaler, graph, settings, device, cfg.training.eval_batch_size)
    q_data = _extract(model, q_view, "train", f_scaler, graph, settings, device, cfg.training.eval_batch_size)

    x_train, w_train = r_train["h_global"], r_train["soft_target"]
    x_val, w_val = r_val["h_global"], r_val["soft_target"]
    num_candidates = int(q_data["abs_sum"].shape[1])
    epochs = cfg.training.max_epochs["probe"]
    lr = cfg.training.candidate_lr

    results: dict[str, object] = {}
    # Baselines from R_train labels only.
    cost_train = r_train["cost"]
    best_train = torch.argmin(cost_train, dim=-1)
    uniform = torch.randint(
        0,
        num_candidates,
        (q_data["abs_sum"].shape[0],),
        generator=torch.Generator().manual_seed(cfg.evaluation.bootstrap_seed),
    )
    results["uniform_random"] = _evaluate_probe(q_data, uniform, "uniform_random")
    frequent_index = int(torch.bincount(best_train, minlength=num_candidates).argmax())
    frequent = torch.full_like(uniform, frequent_index)
    results["most_frequent"] = _evaluate_probe(q_data, frequent, "most_frequent")
    mean_mae = r_train["abs_sum"] / r_train["count"].clamp_min(1)[:, None]
    best_mean_index = int(mean_mae.mean(dim=0).argmin())
    best_mean = torch.full_like(uniform, best_mean_index)
    results["lowest_mean_mae"] = _evaluate_probe(q_data, best_mean, "lowest_mean_mae")

    for kind in ("linear", "mlp"):
        probe = _train_soft_probe(
            x_train, w_train, x_val, w_val, num_candidates, kind, epochs, lr, cfg.seed
        )
        index = _probe_predict(probe, q_data["h_global"])
        results[f"probe_{kind}"] = _evaluate_probe(q_data, index, f"probe_{kind}")

        shuffled = _block_shuffle(x_train, cfg.evaluation.block_origins, cfg.seed + 991)
        probe_neg = _train_soft_probe(
            shuffled, w_train, x_val, w_val, num_candidates, kind, epochs, lr, cfg.seed
        )
        index_neg = _probe_predict(probe_neg, q_data["h_global"])
        results[f"probe_{kind}_block_shuffle"] = _evaluate_probe(
            q_data, index_neg, f"probe_{kind}_block_shuffle"
        )

    document = {
        "protocol": "F/R/Q temporal holdout",
        "ranges": {
            "train_end": a,
            "F": [0, f_end],
            "F_fit": [0, f_fit_end],
            "R": [f_end, r_end],
            "R_fit": [f_end, r_train_end],
            "Q": [r_end, a],
        },
        "windows": {
            "F_train": int(f_scheme.starts["train"].size),
            "F_val": int(f_scheme.starts["val"].size),
            "R_train": int(r_train_starts.size),
            "R_val": int(r_val_starts.size),
            "Q": int(q_starts.size),
        },
        "scaler": {
            "mu": f_scaler.mu,
            "sigma": f_scaler.sigma,
            "fit_end": f_end,
            "note": "F only; R/Q never contribute",
        },
        "settings": {
            "alpha": cfg.oracle.alpha,
            "temperature": cfg.oracle.temperature,
            "num_candidates": num_candidates,
            "probe_epochs": epochs,
            "block_shuffle_origins": cfg.evaluation.block_origins,
        },
        "results": results,
        "artifacts": {
            "base_checkpoint": str(base_result.best_checkpoint),
            "candidate_checkpoint": str(candidate_result.best_checkpoint),
            "candidate_run": str(candidate_run),
        },
    }
    write_json(run.path("probe_results.json"), document)
    return document


__all__ = ["ProbeError", "SoftClassifier", "run_route_probe"]
