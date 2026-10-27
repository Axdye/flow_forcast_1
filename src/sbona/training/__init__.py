"""Training loops, stage budgets and checkpoint utilities."""

from __future__ import annotations

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
    STAGES,
    StageSpec,
    build_optimizer,
    optimizer_for_stage,
    parameter_report,
    stage_budget,
    stage_lr,
    stage_spec,
)
from sbona.training.trainer import (
    BaseTrainer,
    CandidateTrainer,
    RouterTrainer,
    TrainResult,
    autocast_context,
    make_hashes,
    move_input,
    move_target,
    resolve_device,
)

__all__ = [
    "CheckpointError",
    "checkpoint_sha256",
    "load_checkpoint",
    "load_state_dict_strict",
    "restore_rng_state",
    "save_checkpoint",
    "upstream_lock_hash",
    "validate_resume",
    "STAGES",
    "StageSpec",
    "build_optimizer",
    "optimizer_for_stage",
    "parameter_report",
    "stage_budget",
    "stage_lr",
    "stage_spec",
    "BaseTrainer",
    "CandidateTrainer",
    "RouterTrainer",
    "TrainResult",
    "autocast_context",
    "make_hashes",
    "move_input",
    "move_target",
    "resolve_device",
]
