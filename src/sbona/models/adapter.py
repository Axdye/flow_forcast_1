"""SBoN adapter: candidate generation and the single-decode Top-1 path."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from sbona.backbones.base import BackboneWrapper
from sbona.config import Config
from sbona.contracts import (
    CandidateOutput,
    GraphContext,
    ModelInput,
    Top1Output,
)
from sbona.models.actions import ActionGenerator
from sbona.models.decoder import StructuredDecoder
from sbona.models.projector import FeatureProjector
from sbona.models.router import Router

STAGE_TRAINABLE: dict[str, frozenset[str]] = {
    "A": frozenset({"backbone"}),
    "B": frozenset({"projector", "action_generator", "decoder"}),
    "C": frozenset({"projector", "action_generator", "decoder"}),
    "D": frozenset({"router"}),
    # Optional E is not part of the default method; it freezes the backbone
    # and jointly trains adapter + router.
    "E": frozenset({"projector", "action_generator", "decoder", "router"}),
}


@runtime_checkable
class AdapterModel(Protocol):
    num_candidates: int

    def forward_candidates(self, inp: ModelInput, graph: GraphContext) -> CandidateOutput:
        ...

    def predict_top1(self, inp: ModelInput, graph: GraphContext) -> Top1Output:
        ...

    def set_stage(self, stage: str) -> None:
        ...

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        ...


class SBoN(nn.Module):
    """Structured Best-of-N adapter.

    The class owns the backbone but never receives targets.  ``predict_top1``
    generates K five-dimensional actions (allowed), selects one with the
    router, and decodes only that action.
    """

    def __init__(
        self,
        backbone: BackboneWrapper,
        projector: FeatureProjector,
        action_generator: ActionGenerator,
        decoder: StructuredDecoder,
        router: Router,
        config: Config,
        stage: str = "C",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.action_generator = action_generator
        self.decoder = decoder
        self.router = router
        self.config = config
        self.num_candidates = int(config.adapter.num_candidates)
        if action_generator.num_candidates != self.num_candidates:
            raise ValueError("action generator K does not match config")
        if router.num_candidates != self.num_candidates:
            raise ValueError("router K does not match config")
        self._stage = stage
        self._backbone_mode = config.backbone.mode
        self.adapter_kind = "structured"
        self.supports_separation = True
        self.set_stage(stage)

    # ------------------------------------------------------------------
    # Stage / mode management
    # ------------------------------------------------------------------

    def set_stage(self, stage: str) -> None:
        if stage not in STAGE_TRAINABLE:
            raise ValueError(f"unknown stage {stage!r}")
        trainable = STAGE_TRAINABLE[stage]
        for name, module in self.named_children():
            for parameter in module.parameters():
                parameter.requires_grad_(name in trainable)
        mode = "full" if stage == "A" else self.config.backbone.mode
        self._backbone_mode = mode
        self.backbone.set_mode(mode)
        self._stage = stage
        self.train(self.training)

    def train(self, mode: bool = True) -> SBoN:
        super().train(mode)
        trainable = STAGE_TRAINABLE.get(self._stage, frozenset())
        for name, module in self.named_children():
            if name == "backbone":
                continue
            if name in trainable:
                module.train(mode)
            else:
                module.eval()
        self.backbone.set_mode(self._backbone_mode)
        return self

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, p in self.named_parameters() if p.requires_grad]

    @property
    def stage(self) -> str:
        return self._stage

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------

    def _encode(self, inp: ModelInput, graph: GraphContext):
        bb = self.backbone.encode_and_predict(inp, graph)
        h_node, h_global = self.projector(bb.node_hidden)
        return bb, h_node, h_global

    def forward_candidates(self, inp: ModelInput, graph: GraphContext) -> CandidateOutput:
        bb, h_node, h_global = self._encode(inp, graph)
        actions = self.action_generator(h_global)
        residual = self.decoder.forward_all(h_node, actions.physical, graph)
        pred_z = bb.base_pred_z[:, None, :, :, :] + residual
        return CandidateOutput(
            pred_z=pred_z,
            actions=actions,
            h_global=h_global,
            base_pred_z=bb.base_pred_z,
        )

    def predict_top1(self, inp: ModelInput, graph: GraphContext) -> Top1Output:
        bb, h_node, h_global = self._encode(inp, graph)
        actions = self.action_generator(h_global)
        batch_size = actions.physical.shape[0]
        if self.num_candidates == 1:
            index = torch.zeros(batch_size, dtype=torch.int64, device=actions.physical.device)
            logits = torch.zeros(
                (batch_size, 1), dtype=torch.float32, device=actions.physical.device
            )
        else:
            index, logits = self.router.select(h_global)
        selected = actions.physical[
            torch.arange(batch_size, device=actions.physical.device), index
        ]
        residual = self.decoder.forward_selected(h_node, selected, graph)
        pred_z = bb.base_pred_z + residual
        return Top1Output(
            pred_z=pred_z,
            index=index,
            logits=logits,
            selected_action=selected,
        )
