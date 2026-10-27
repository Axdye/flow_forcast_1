"""Explicit training stages and optimizer construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from sbona.config import Config


@dataclass(frozen=True)
class StageSpec:
    name: str
    trainable_modules: frozenset[str]
    budget_key: str
    patience_key: str
    lr_key: str
    checkpoint_metric: str


STAGES: dict[str, StageSpec] = {
    "A": StageSpec("A", frozenset({"backbone"}), "base", "base", "base_lr", "raw_mae_all"),
    "B": StageSpec(
        "B",
        frozenset({"projector", "action_generator", "decoder"}),
        "single",
        "single",
        "candidate_lr",
        "single_raw_mae_all",
    ),
    "C": StageSpec(
        "C",
        frozenset({"projector", "action_generator", "decoder"}),
        "candidates",
        "candidates",
        "candidate_lr",
        "candidate_oracle_mae_all",
    ),
    "D": StageSpec("D", frozenset({"router"}), "router", "router", "router_lr", "router_raw_mae_all"),
}


def stage_spec(stage: str) -> StageSpec:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    return STAGES[stage]


def stage_budget(cfg: Config, stage: str) -> tuple[int, int]:
    spec = stage_spec(stage)
    return cfg.training.max_epochs[spec.budget_key], cfg.training.patience[spec.patience_key]


def stage_lr(cfg: Config, stage: str) -> float:
    spec = stage_spec(stage)
    return float(getattr(cfg.training, spec.lr_key))


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("optimizer would be empty: no trainable parameters")
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def optimizer_for_stage(model: nn.Module, cfg: Config, stage: str) -> torch.optim.Optimizer:
    return build_optimizer(model, stage_lr(cfg, stage), cfg.training.weight_decay)


def parameter_report(model: nn.Module) -> dict[str, object]:
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    return {
        "trainable_names": [name for name, _ in named],
        "trainable_parameters": int(sum(p.numel() for _, p in named)),
        "all_parameters": int(sum(p.numel() for p in model.parameters())),
    }


__all__ = [
    "StageSpec",
    "STAGES",
    "stage_spec",
    "stage_budget",
    "stage_lr",
    "build_optimizer",
    "optimizer_for_stage",
    "parameter_report",
]
