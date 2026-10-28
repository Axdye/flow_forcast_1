"""Baseline factories.

``single`` (K=1) and ``structured`` share the *same* adapter math; Stage B and
Stage C differ only in K and in their independent initialization.  The
unstructured and independent-head variants are attribution controls.
"""

from __future__ import annotations

from dataclasses import replace

from sbona.backbones import build_backbone
from sbona.baselines.independent_heads import IndependentHeads
from sbona.baselines.mcl_wta import MCLWTA
from sbona.baselines.perturbation import gaussian_candidates
from sbona.baselines.single import SingleAdapter
from sbona.baselines.unstructured import UnstructuredDecoder
from sbona.config import Config
from sbona.models.actions import ActionGenerator
from sbona.models.adapter import AdapterModel, SBoN
from sbona.models.decoder import StructuredDecoder
from sbona.models.projector import FeatureProjector
from sbona.models.router import Router

ADAPTER_KINDS = ("structured", "single", "unstructured", "independent_heads", "mcl_wta")


def build_adapter(
    kind: str,
    cfg: Config,
    num_nodes: int,
    steps_per_day: int = 288,
    head_width: int = 64,
) -> AdapterModel:
    if kind not in ADAPTER_KINDS:
        raise ValueError(f"unknown adapter kind {kind!r}; choose from {ADAPTER_KINDS}")
    num_candidates = 1 if kind == "single" else cfg.adapter.num_candidates
    active_cfg = (
        replace(cfg, adapter=replace(cfg.adapter, num_candidates=1))
        if kind == "single"
        else cfg
    )
    backbone = build_backbone(
        active_cfg.backbone,
        num_nodes=num_nodes,
        in_steps=active_cfg.data.input_steps,
        out_steps=active_cfg.data.output_steps,
        steps_per_day=steps_per_day,
    )
    projector = FeatureProjector(backbone.node_feature_dim, active_cfg.adapter.hidden_dim)
    if kind == "mcl_wta":
        # No Stage-D Router: the score head is trained jointly in Stage C and is
        # itself the deployed selector.
        return MCLWTA(
            backbone=backbone,
            projector=projector,
            config=active_cfg,
            num_candidates=num_candidates,
            head_width=head_width,
        )
    router = Router(
        hidden_dim=active_cfg.adapter.hidden_dim,
        num_candidates=num_candidates,
    )
    if kind == "independent_heads":
        return IndependentHeads(
            backbone=backbone,
            projector=projector,
            router=router,
            config=active_cfg,
            num_candidates=num_candidates,
            head_width=head_width,
        )
    action_generator = ActionGenerator(
        hidden_dim=active_cfg.adapter.hidden_dim,
        num_candidates=num_candidates,
        output_steps=active_cfg.data.output_steps,
        ranges=active_cfg.adapter.action_ranges,
        sigma_min=active_cfg.adapter.sigma_min,
        anchor_seed=active_cfg.adapter.anchor_seed,
    )
    if kind == "unstructured":
        decoder = UnstructuredDecoder(
            latent_dim=5,
            rank=active_cfg.adapter.residual_dim,
            output_steps=active_cfg.data.output_steps,
            num_nodes=num_nodes,
            residual_scale=active_cfg.adapter.residual_scale,
        )
    else:
        decoder = StructuredDecoder(
            hidden_dim=active_cfg.adapter.hidden_dim,
            residual_dim=active_cfg.adapter.residual_dim,
            graph_hops=active_cfg.adapter.graph_hops,
            output_steps=active_cfg.data.output_steps,
            s_p=active_cfg.adapter.s_p,
            s_o=active_cfg.adapter.s_o,
            s_d=active_cfg.adapter.s_d,
            residual_scale=active_cfg.adapter.residual_scale,
            output_init_std=active_cfg.adapter.output_init_std,
        )
    stage = "B" if kind == "single" else "C"
    model = SBoN(
        backbone=backbone,
        projector=projector,
        action_generator=action_generator,
        decoder=decoder,
        router=router,
        config=active_cfg,
        stage=stage,
    )
    model.adapter_kind = kind
    model.supports_separation = kind == "structured"
    return model


__all__ = [
    "ADAPTER_KINDS",
    "build_adapter",
    "AdapterModel",
    "SingleAdapter",
    "UnstructuredDecoder",
    "IndependentHeads",
    "MCLWTA",
    "gaussian_candidates",
]
