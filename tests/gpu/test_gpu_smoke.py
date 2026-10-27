"""Optional GPU smoke test.

No CUDA result is fabricated: the whole module is skipped when CUDA is absent.
The test only checks that a small forward/backward runs on the device and that
the public tensors stay finite.  It is not an A100 acceptance benchmark.
"""

from __future__ import annotations

import pytest
import torch

from sbona.baselines import build_adapter
from sbona.config import Config
from sbona.contracts import ModelInput, TargetBatch
from sbona.data.graph import build_graph_context

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available; no GPU smoke result is claimed", allow_module_level=True)


def test_small_forward_backward_on_cuda() -> None:
    cfg = Config.from_dict(
        {
            "schema_version": 1,
            "protocol_id": "gpu_smoke_v1",
            "method_id": "sbona_gpu_smoke",
            "seed": 1,
            "data": {
                "dataset": "synthetic",
                "synthetic": True,
                "synthetic_num_nodes": 5,
                "synthetic_num_steps": 120,
                "input_steps": 12,
                "output_steps": 12,
            },
            "backbone": {
                "name": "staeformer",
                "mode": "frozen",
                "input_embedding_dim": 8,
                "tod_embedding_dim": 8,
                "adaptive_embedding_dim": 8,
                "feed_forward_dim": 16,
                "num_heads": 4,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "adapter": {
                "num_candidates": 2,
                "hidden_dim": 8,
                "residual_dim": 4,
                "graph_hops": 1,
            },
            "training": {"precision": "bf16"},
        }
    )
    device = torch.device("cuda")
    graph = build_graph_context(5, []).to(device)
    model = build_adapter("structured", cfg, num_nodes=5).to(device)
    model.set_stage("C")
    model.train()
    inp = ModelInput(
        history=torch.randn(2, 12, 5, 1, device=device),
        observed=torch.ones(2, 12, 5, 1, dtype=torch.bool, device=device),
        time_of_day=torch.rand(2, 12, device=device),
        day_of_week=torch.zeros(2, 12, dtype=torch.int64, device=device),
    )
    target = TargetBatch(
        values_raw=torch.rand(2, 12, 5, 1, device=device) * 100.0,
        valid=torch.ones(2, 12, 5, 1, dtype=torch.bool, device=device),
        sample_ids=torch.arange(2, dtype=torch.int64, device=device),
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.forward_candidates(inp, graph)
    assert out.pred_z.shape == (2, 2, 12, 5, 1)
    assert torch.isfinite(out.pred_z).all()
    loss = ((out.pred_z - target.values_raw[:, None]) ** 2).mean()
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
