"""TimeMCL-style winner-takes-all baseline.

**This is a transplant, not a reproduction of the official TimeMCL code.**
TimeMCL (ICML 2025) is a *multivariate probabilistic* forecaster: a mixture of
predictive distributions trained with a winner-takes-all (WTA) rule so that
different mixture components specialise on different samples, plus a score
branch used to weight/select components at inference.  This repository's
backbones are deterministic residual adapters on top of a shared base forecast,
so the transplant keeps the *training rule* (hard WTA over independent heads)
and the *score branch*, and drops the probabilistic head (mean/variance
mixtures, likelihood loss) that has no equivalent here.

What is kept
------------
``K`` independent residual heads (:class:`ResidualHead`, the same capacity
control as ``independent_heads``), one per candidate.  Each training sample's
gradient flows through **only the head with the smallest error** — the hard
minimum, not a soft weighting.  A :class:`ScoreHead` is trained with
cross-entropy against that winner, and at inference it picks the candidate
(``predict_top1``), mirroring TimeMCL's score-based selection.

What is deliberately absent
---------------------------
* No probabilistic/likelihood head, no variance output: :class:`CandidateOutput`
  carries deterministic predictions only.
* No Router and no Stage D.  The score head is trained *jointly* with the heads
  in Stage C; routing it afterwards would be a different method.

Attribution boundary
--------------------
``actions`` is the same zero placeholder used by ``independent_heads`` and the
separation term is disabled, so the only difference from that control is the
WTA training rule and the score head.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from sbona.backbones.base import BackboneWrapper
from sbona.baselines.independent_heads import ResidualHead
from sbona.config import Config
from sbona.contracts import ActionOutput, CandidateOutput, GraphContext, ModelInput, Top1Output
from sbona.models.projector import FeatureProjector


class ScoreHead(nn.Module):
    """Per-candidate score from the global state and the head's own response.

    The response summary is the mean absolute residual the head proposes; it is
    the cheapest signal that distinguishes an "aggressive" head from a
    conservative one without a second backbone pass.
    """

    def __init__(self, hidden_dim: int, width: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim + 1),
            nn.Linear(hidden_dim + 1, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_global: Tensor, residuals: Tensor) -> Tensor:
        """``h_global [B, D]``, ``residuals [B, K, T, N, 1]`` -> logits ``[B, K]``."""
        batch, num_candidates = residuals.shape[:2]
        summary = residuals.abs().flatten(2).mean(dim=-1)  # [B, K]
        features = torch.cat(
            [h_global[:, None, :].expand(batch, num_candidates, -1), summary[..., None]],
            dim=-1,
        )
        return self.net(features).squeeze(-1).float()


class MCLWTA(nn.Module):
    """Independent heads trained with hard WTA plus a jointly learned scorer."""

    def __init__(
        self,
        backbone: BackboneWrapper,
        projector: FeatureProjector,
        config: Config,
        num_candidates: int,
        head_width: int = 64,
        score_width: int = 64,
    ) -> None:
        super().__init__()
        if num_candidates < 2:
            raise ValueError("the MCL-WTA baseline needs K >= 2 candidate heads")
        self.backbone = backbone
        self.projector = projector
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
        self.score_head = ScoreHead(config.adapter.hidden_dim, score_width)
        self._stage = "C"
        self._backbone_mode = config.backbone.mode
        self.adapter_kind = "mcl_wta"
        self.supports_separation = False
        #: Tells :class:`CandidateTrainer` which objective to dispatch to.
        self.loss_kind = "wta"
        #: The score head is the deployed selector (no Stage-D Router).
        self.uses_score_selector = True
        self.set_stage("C")

    def set_stage(self, stage: str) -> None:
        trainable = {"B": {"projector", "heads", "score_head"}, "C": {"projector", "heads", "score_head"}}
        names = trainable.get(stage)
        if names is None:
            raise ValueError(
                "the MCL-WTA baseline trains its selector jointly in stages B/C; "
                f"stage {stage!r} (Router) does not apply"
            )
        for name, module in self.named_children():
            for parameter in module.parameters():
                parameter.requires_grad_(name in names)
        self._backbone_mode = self.config.backbone.mode
        self.backbone.set_mode(self._backbone_mode)
        self._stage = stage
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        trainable = {
            "B": {"projector", "heads", "score_head"},
            "C": {"projector", "heads", "score_head"},
        }.get(self._stage, frozenset())
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
        logits = self.score_head(h_global, residuals)
        return CandidateOutput(
            pred_z=bb.base_pred_z[:, None] + residuals,
            actions=self._dummy_actions(h_global),
            h_global=h_global,
            base_pred_z=bb.base_pred_z,
            candidate_logits=logits,
        )

    def predict_top1(self, inp: ModelInput, graph: GraphContext) -> Top1Output:
        bb, h_node, h_global = self._encode(inp, graph)
        residuals = torch.stack([head(h_node) for head in self.heads], dim=1)
        logits = self.score_head(h_global, residuals)
        # Deterministic argmax: the score head is the deployed policy.
        index = logits.argmax(dim=-1)
        selected = residuals.gather(
            1,
            index[:, None, None, None, None].expand(-1, 1, *residuals.shape[2:]),
        ).squeeze(1)
        return Top1Output(
            pred_z=bb.base_pred_z + selected,
            index=index,
            logits=logits,
            selected_action=h_global.new_zeros((h_global.shape[0], 5)),
        )


__all__ = ["MCLWTA", "ScoreHead"]
