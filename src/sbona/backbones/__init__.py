"""Backbone registry.

Every entry below is a self-contained re-implementation of the pinned upstream
architecture on :class:`sbona.backbones.base.BackboneWrapper`.  The provenance
(pinned commit, licence, extraction points and documented deviations) lives in
``third_party/LOCK.md`` and ``third_party/BACKBONE_CONTRACTS.md``; the fidelity
notes for the adapters are in ``reviews/EXTRA_WORK.md``.

``PENDING`` is kept for architectures that were named in the design but are not
wired yet.  Building one raises :class:`UnsupportedBackboneError` instead of
pretending a half-ported model is usable.
"""

from __future__ import annotations

from sbona.backbones.agcrn import AGCRNAdapter
from sbona.backbones.base import (
    BackboneError,
    BackboneWrapper,
    UnsupportedBackboneError,
    count_trainable_parameters,
)
from sbona.backbones.d2stgnn import D2STGNNAdapter
from sbona.backbones.graph_wavenet import GraphWaveNetAdapter
from sbona.backbones.himnet import HimNetAdapter
from sbona.backbones.hyperd import HyperDAdapter
from sbona.backbones.pdformer import PDFormerAdapter
from sbona.backbones.staeformer import STAEformer, STAEformerAdapter
from sbona.backbones.stid import STIDAdapter
from sbona.backbones.stssdl import STSSDLAdapter
from sbona.config import BackboneConfig

#: Architectures with a registered adapter.
SUPPORTED = (
    "staeformer",
    "graph_wavenet",
    "agcrn",
    "stid",
    "d2stgnn",
    "pdformer",
    "himnet",
    "stssdl",
    "hyperd",
)
#: Documented architectures that still have no adapter.
PENDING: tuple[str, ...] = ()


def build_backbone(
    cfg: BackboneConfig,
    num_nodes: int,
    in_steps: int,
    out_steps: int,
    steps_per_day: int = 288,
) -> BackboneWrapper:
    """Build the configured backbone or raise an explicit pending error."""
    common = dict(
        num_nodes=num_nodes,
        in_steps=in_steps,
        out_steps=out_steps,
        steps_per_day=steps_per_day,
    )
    if cfg.name == "staeformer":
        return STAEformerAdapter(cfg, **common)
    if cfg.name == "graph_wavenet":
        return GraphWaveNetAdapter(cfg, **common)
    if cfg.name == "agcrn":
        return AGCRNAdapter(cfg, **common)
    if cfg.name == "stid":
        return STIDAdapter(cfg, **common)
    if cfg.name == "d2stgnn":
        return D2STGNNAdapter(cfg, **common)
    if cfg.name == "pdformer":
        return PDFormerAdapter(cfg, **common)
    if cfg.name == "himnet":
        return HimNetAdapter(cfg, **common)
    if cfg.name == "stssdl":
        return STSSDLAdapter(cfg, **common)
    if cfg.name == "hyperd":
        return HyperDAdapter(cfg, **common)
    if cfg.name in PENDING:
        raise UnsupportedBackboneError(
            f"backbone {cfg.name!r} is pending: pin an immutable commit, vendor the "
            "minimal source with its licence, and pass the adapter/base equivalence "
            "test described in third_party/BACKBONE_CONTRACTS.md before use"
        )
    raise UnsupportedBackboneError(f"unknown backbone {cfg.name!r}")


__all__ = [
    "AGCRNAdapter",
    "BackboneError",
    "BackboneWrapper",
    "D2STGNNAdapter",
    "GraphWaveNetAdapter",
    "HimNetAdapter",
    "HyperDAdapter",
    "PDFormerAdapter",
    "PENDING",
    "STAEformer",
    "STAEformerAdapter",
    "STIDAdapter",
    "STSSDLAdapter",
    "SUPPORTED",
    "UnsupportedBackboneError",
    "build_backbone",
    "count_trainable_parameters",
]
