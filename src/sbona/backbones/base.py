"""Backbone mode management and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from sbona.contracts import BackboneOutput, GraphContext, ModelInput


class BackboneError(RuntimeError):
    """Base class for backbone integration errors."""


class UnsupportedBackboneError(BackboneError):
    """Raised when an adapter has not been implemented/verified yet."""


class BackboneWrapper(nn.Module, ABC):
    """Owns exactly one upstream backbone and enforces Frozen/Partial/Full.

    ``train()`` is overridden because ``nn.Module.train`` would otherwise
    re-enable dropout and BatchNorm updates in a frozen backbone after the
    parent model is switched to train mode.  Callers must still re-apply
    ``set_mode`` after loading a checkpoint, as required by the contract.
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = "frozen",
        partial_patterns: tuple[str, ...] = (),
        node_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("frozen", "partial", "full"):
            raise BackboneError(f"unknown backbone mode {mode!r}")
        self.model = model
        self.mode = mode
        self.partial_patterns = tuple(partial_patterns)
        self._node_feature_dim = node_feature_dim
        self.set_mode(mode)

    @property
    def node_feature_dim(self) -> int:
        if self._node_feature_dim is None:
            raise BackboneError("node_feature_dim is not configured")
        return self._node_feature_dim

    def set_mode(self, mode: str) -> None:
        if mode not in ("frozen", "partial", "full"):
            raise BackboneError(f"unknown backbone mode {mode!r}")
        if mode == "partial" and not self.partial_patterns:
            raise BackboneError("partial mode requires at least one whitelist pattern")
        self.mode = mode
        for name, parameter in self.model.named_parameters():
            if mode == "full":
                parameter.requires_grad_(True)
            elif mode == "partial":
                parameter.requires_grad_(any(p in name for p in self.partial_patterns))
            else:
                parameter.requires_grad_(False)
        self._enforce_modes()

    def _enforce_modes(self) -> None:
        if self.mode == "full":
            self.model.train(self.training)
            return
        # Frozen backbone is always eval: dropout off, BN stats untouched.
        self.model.eval()
        if self.mode == "partial" and self.training:
            for name, module in self.model.named_modules():
                if name and any(p in name for p in self.partial_patterns):
                    module.train(True)

    def train(self, mode: bool = True) -> BackboneWrapper:
        super().train(mode)
        self._enforce_modes()
        return self

    def trainable_parameter_names(self) -> list[str]:
        return [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]

    def _context(self):
        return torch.no_grad() if self.mode == "frozen" else torch.enable_grad()

    @abstractmethod
    def encode_and_predict(self, inp: ModelInput, graph: GraphContext) -> BackboneOutput:
        """Run the backbone once and return base prediction + node hidden."""

    def extra_repr(self) -> str:
        return f"mode={self.mode}, trainable={len(self.trainable_parameter_names())}"


def check_input_against_graph(inp: ModelInput, graph: GraphContext) -> None:
    if inp.num_nodes != graph.num_nodes:
        raise BackboneError(
            f"input has {inp.num_nodes} nodes but graph has {graph.num_nodes}; "
            "node order / graph alignment is invalid"
        )
    if inp.history.device != graph.diffusion.device:
        # Backbones that do not use the graph still get a clear error rather
        # than a silent cross-device broadcast later.
        raise BackboneError("input and graph must be on the same device")


def count_trainable_parameters(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


__all__ = [
    "BackboneWrapper",
    "BackboneError",
    "UnsupportedBackboneError",
    "check_input_against_graph",
    "count_trainable_parameters",
]
