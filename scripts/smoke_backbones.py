"""Smoke forward for every registered backbone (tiny graph, B=2, N=5, L=T=12)."""

from __future__ import annotations

import sys
import traceback

import numpy as np
import torch

from sbona.backbones import SUPPORTED, build_backbone
from sbona.config import BackboneConfig
from sbona.contracts import GraphContext, ModelInput
from sbona.data.graph import build_graph_context

EDGES = np.array([[0, 1], [1, 0], [1, 2], [2, 3]], dtype=np.int64)


def main() -> int:
    torch.manual_seed(0)
    graph: GraphContext = build_graph_context(5, EDGES)
    batch, length, nodes, horizon = 2, 12, 5, 12
    inp = ModelInput(
        history=torch.randn(batch, length, nodes, 1),
        observed=torch.ones(batch, length, nodes, 1, dtype=torch.bool),
        time_of_day=torch.rand(batch, length),
        day_of_week=torch.randint(0, 7, (batch, length), dtype=torch.int64),
    )
    failures = 0
    for name in SUPPORTED:
        for mode in ("frozen", "partial", "full"):
            try:
                cfg = BackboneConfig.from_dict(
                    {"name": name, "mode": mode, "input_dim": 3, "dow_embedding_dim": 24}
                )
                wrapper = build_backbone(
                    cfg, num_nodes=nodes, in_steps=length, out_steps=horizon, steps_per_day=288
                )
                wrapper.train(mode != "frozen")
                out = wrapper.encode_and_predict(inp, graph)
                assert out.base_pred_z.shape == (batch, horizon, nodes, 1), out.base_pred_z.shape
                assert out.node_hidden.shape[:2] == (batch, nodes), out.node_hidden.shape
                assert torch.isfinite(out.base_pred_z).all(), "non-finite prediction"
                trainable = len(wrapper.trainable_parameter_names())
                print(
                    f"ok   {name:14s} {mode:8s} hidden={tuple(out.node_hidden.shape[1:])} "
                    f"trainable_tensors={trainable}"
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic script
                failures += 1
                print(f"FAIL {name:14s} {mode:8s} {type(exc).__name__}: {exc}")
                traceback.print_exc()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
