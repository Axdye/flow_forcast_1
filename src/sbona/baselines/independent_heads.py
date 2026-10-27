"""Independent residual-head baseline.

This is a capacity/attribution control, **not** the main structured method.  It
shares the backbone and projector with SBoN, keeps one independent residual
head per candidate, and uses the same router interface.  The candidate `actions`
field is a zero placeholder because this baseline has no structured actions; its
separation term is therefore disabled by the trainer for this method.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from sbona.backbones.base import BackboneWrapper
from sbona.config import Config
from sbona.contracts import ActionOutput, CandidateOutput, GraphContext, ModelInput, Top1Output
from sbona.models.projector import FeatureProjector
from sbona.models.router import Router


class ResidualHead(nn.Module):
    def __init__(self, hidden_dim: int, width: int, output_steps: int, residual_scale: float = 3.0) -> None:
        super().__init__()
        self.output_steps = output_steps
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, width),
            nn.GELU(),
            nn.Linear(width, output_steps),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_node: Tensor) -> Tensor:
        # [B,N,T] -> [B,T,N,1], bounded but otherwise free.
        out = self.net(h_node).permute(0, 2, 1).unsqueeze(-1)
        return (self.residual_scale * torch.tanh(out / self.residual_scale)).float()


class IndependentHeads(nn.Module):
    def __init__(
        self,
        backbone: BackboneWrapper,
        projector: FeatureProjector,
        router: Router,
        config: Config,
        num_candidates: int,
        head_width: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.router = router
        self.config = config
        self.num_candidates = num_candidates
        self.heads = nn.ModuleList(
            [
                ResidualHead(
                    config.adapter.hidden_dim,
                    head_width,
                    config.data.output_steps,
                    config.adapter.residual_scale,
                )
                for _ in range(num_candidates)
            ]
        )
        self._stage = "C"
        self._backbone_mode = config.backbone.mode
        self.adapter_kind = "independent_heads"
        self.supports_separation = False
        self.set_stage("C")

    def set_stage(self, stage: str) -> None:
        trainable = {
            "B": {"projector", "heads"},
            "C": {"projector", "heads"},
            "D": {"router"},
        }.get(stage)
        if trainable is None:
            raise ValueError(f"independent-head baseline does not support stage {stage!r}")
        for name, module in self.named_children():
            for parameter in module.parameters():
                parameter.requires_grad_(name in trainable)
        self._backbone_mode = self.config.backbone.mode
        self.backbone.set_mode(self._backbone_mode)
        self._stage = stage
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        trainable = {"B": {"projector", "heads"}, "C": {"projector", "heads"}, "D": {"router"}}.get(
            self._stage, frozenset()
        )
        for name, module in self.named_children():
            if name == "backbone":
                continue
            module.train(mode) if name in trainable else module.eval()
        self.backbone.set_mode(self._backbone_mode)
        return self

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def _encode(self, inp: ModelInput, graph: GraphContext):
        bb = self.backbone.encode_and_predict(inp, graph)
        h_node, h_global = self.projector(bb.node_hidden)
        return bb, h_node, h_global

    def _dummy_actions(self, h_global: Tensor) -> ActionOutput:
        batch_size = h_global.shape[0]
        zeros = h_global.new_zeros((batch_size, self.num_candidates, 5))
        return ActionOutput(
            physical=zeros,
            normalized=zeros,
            mu=h_global.new_zeros((batch_size, 5)),
            sigma=h_global.new_ones((batch_size, 5)),
        )

    def forward_candidates(self, inp: ModelInput, graph: GraphContext) -> CandidateOutput:
        bb, h_node, h_global = self._encode(inp, graph)
        residuals = torch.stack([head(h_node) for head in self.heads], dim=1)
        pred_z = bb.base_pred_z[:, None] + residuals
        return CandidateOutput(
            pred_z=pred_z,
            actions=self._dummy_actions(h_global),
            h_global=h_global,
            base_pred_z=bb.base_pred_z,
        )

    def predict_top1(self, inp: ModelInput, graph: GraphContext) -> Top1Output:
        bb, h_node, h_global = self._encode(inp, graph)
        batch_size = h_node.shape[0]
        if self.num_candidates == 1:
            index = torch.zeros(batch_size, dtype=torch.int64, device=h_node.device)
            logits = torch.zeros((batch_size, 1), device=h_node.device)
        else:
            index, logits = self.router.select(h_global)
        residual = torch.zeros_like(bb.base_pred_z)
        for candidate in range(self.num_candidates):
            mask = index == candidate
            indices = torch.where(mask)[0]
            if indices.numel() > 0:
                residual = residual.index_copy(
                    0, indices, self.heads[candidate](h_node.index_select(0, indices))
                )
        return Top1Output(
            pred_z=bb.base_pred_z + residual,
            index=index,
            logits=logits,
            selected_action=h_global.new_zeros((batch_size, 5)),
        )


__all__ = ["ResidualHead", "IndependentHeads"]
