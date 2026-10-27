"""Backbone registry.

Only STAEformer is implemented and runtime-checkable in this prototype.  The
other documented backbones intentionally fail loudly instead of pretending to
be supported: ``third_party/BACKBONE_CONTRACTS.md`` requires a pinned commit,
licence audit and an adapter/base equivalence test before a backbone may be
used in formal experiments.
"""

from __future__ import annotations

from sbona.backbones.base import (
    BackboneError,
    BackboneWrapper,
    UnsupportedBackboneError,
    count_trainable_parameters,
)
from sbona.backbones.staeformer import STAEformer, STAEformerAdapter
from sbona.config import BackboneConfig

SUPPORTED = ("staeformer",)
PENDING = ("stid", "graph_wavenet", "agcrn")


def build_backbone(
    cfg: BackboneConfig,
    num_nodes: int,
    in_steps: int,
    out_steps: int,
    steps_per_day: int = 288,
) -> BackboneWrapper:
    """Build the configured backbone or raise an explicit pending error."""
    if cfg.name == "staeformer":
        return STAEformerAdapter(
            cfg,
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
            steps_per_day=steps_per_day,
        )
    if cfg.name in PENDING:
        raise UnsupportedBackboneError(
            f"backbone {cfg.name!r} is pending: pin an immutable commit, vendor the "
            "minimal source with its licence, and pass the adapter/base equivalence "
            "test described in third_party/BACKBONE_CONTRACTS.md before use"
        )
    raise UnsupportedBackboneError(f"unknown backbone {cfg.name!r}")


__all__ = [
    "BackboneWrapper",
    "BackboneError",
    "UnsupportedBackboneError",
    "STAEformer",
    "STAEformerAdapter",
    "build_backbone",
    "count_trainable_parameters",
    "SUPPORTED",
    "PENDING",
]
