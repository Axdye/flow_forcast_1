"""Contract tests for every registered backbone adapter.

The models are instantiated on a 5-node graph with ``B=2``, ``L=T=12`` so the
whole matrix (9 architectures x 3 modes) runs on CPU in seconds.  These tests
check the *contract*, not numerical fidelity: shape, mode/gradient behaviour and
(where the architecture really consumes the road graph) graph sensitivity.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sbona.backbones import PENDING, SUPPORTED, build_backbone, count_trainable_parameters
from sbona.config import BackboneConfig
from sbona.contracts import GraphContext, ModelInput
from sbona.data.graph import build_graph_context

NODES = 5
BATCH = 2
STEPS = 12

#: Architectures whose prediction provably reads the road graph.  STID and
#: STAEformer use learned all-to-all node interactions, AGCRN learns an adaptive
#: adjacency and HimNet learns its own supports, so a road-graph swap must *not*
#: be asserted for them.
GRAPH_DEPENDENT = ("graph_wavenet", "d2stgnn", "pdformer", "stssdl", "hyperd")

EDGES = np.array([[0, 1], [1, 0], [1, 2], [2, 3]], dtype=np.int64)
EDGES_ALT = np.array(
    [[0, 2], [2, 0], [2, 4], [4, 1], [1, 3], [3, 0]], dtype=np.int64
)


def _input() -> ModelInput:
    generator = torch.Generator().manual_seed(7)
    return ModelInput(
        history=torch.randn(BATCH, STEPS, NODES, 1, generator=generator),
        observed=torch.ones(BATCH, STEPS, NODES, 1, dtype=torch.bool),
        time_of_day=torch.rand(BATCH, STEPS, generator=generator),
        day_of_week=torch.randint(0, 7, (BATCH, STEPS), generator=generator, dtype=torch.int64),
    )


def _wrapper(name: str, mode: str):
    cfg = BackboneConfig.from_dict(
        {"name": name, "mode": mode, "input_dim": 3, "dow_embedding_dim": 24}
    )
    torch.manual_seed(0)
    return build_backbone(
        cfg, num_nodes=NODES, in_steps=STEPS, out_steps=STEPS, steps_per_day=288
    )


def test_every_advertised_architecture_is_registered() -> None:
    expected = {
        "staeformer",
        "graph_wavenet",
        "agcrn",
        "stid",
        "d2stgnn",
        "pdformer",
        "himnet",
        "stssdl",
        "hyperd",
    }
    assert set(SUPPORTED) == expected
    # Nothing is allowed to sit in the "documented but unusable" state while
    # being advertised as supported.
    assert not set(PENDING) & set(SUPPORTED)


@pytest.mark.parametrize("name", SUPPORTED)
@pytest.mark.parametrize("mode", ("frozen", "partial", "full"))
def test_forward_contract(name: str, mode: str, tiny_graph: GraphContext) -> None:
    wrapper = _wrapper(name, mode)
    wrapper.train(mode != "frozen")
    out = wrapper.encode_and_predict(_input(), tiny_graph)

    assert out.base_pred_z.shape == (BATCH, STEPS, NODES, 1), out.base_pred_z.shape
    assert out.node_hidden.shape[:2] == (BATCH, NODES), out.node_hidden.shape
    assert out.node_hidden.shape[-1] == wrapper.node_feature_dim
    assert torch.isfinite(out.base_pred_z).all()
    assert torch.isfinite(out.node_hidden).all()

    trainable = list(wrapper.trainable_parameter_names())
    total, trainable_count = count_trainable_parameters(wrapper)
    assert total > 0
    if mode == "frozen":
        assert trainable == []
        assert trainable_count == 0
        assert not out.base_pred_z.requires_grad
    else:
        assert trainable
        assert trainable_count > 0
        assert out.base_pred_z.requires_grad


#: Parameters the upstream architectures create but never read, plus HyperD's
#: zero-initialised periodic table (see below).  They must not receive
#: gradients, and *only* these may be missing; any other detached branch is a
#: porting bug.
UNUSED_PARAMETER_PREFIXES = {
    # Upstream creates one residual conv per layer but only uses it when the
    # channel count changes; ``bn.7``/``gconv.7`` belong to the never-invoked
    # extra block.
    "graph_wavenet": ("residual_convs.", "bn.7.", "gconv.7."),
    # Upstream's forecast head drops the backcast branch and the distance
    # function never reads its time-slot embedding.
    "d2stgnn": (
        "layers.4.inh_layer.backcast_fc.",
        "layers.4.inh_layer.residual_decompose.",
        "dynamic_graph_constructor.distance_function.time_slot_embedding.",
    ),
    # With the documented "no prior" deviation the periodic table starts at
    # zero, so ``linear1`` only receives gradient through its bias.
    "hyperd": ("daily_emb.linear1.", "weekly_emb.linear1."),
}


@pytest.mark.parametrize("name", SUPPORTED)
def test_full_mode_backpropagates_to_the_prediction_path(name: str, tiny_graph: GraphContext) -> None:
    wrapper = _wrapper(name, "full")
    wrapper.train()
    out = wrapper.encode_and_predict(_input(), tiny_graph)
    out.base_pred_z.sum().backward()

    named = [(pname, parameter) for pname, parameter in wrapper.model.named_parameters() if parameter.requires_grad]
    assert named
    missing = [
        pname
        for pname, parameter in named
        if parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
    ]
    allowed = UNUSED_PARAMETER_PREFIXES.get(name, ())
    unexpected = [pname for pname in missing if not pname.startswith(allowed)]
    assert unexpected == [], f"{name}: parameters without gradient: {unexpected[:8]}"
    # At least 80% of the trainable tensors must move; the only dead tensors are
    # the documented upstream ones.
    assert 1.0 - len(missing) / len(named) >= 0.8
    # The last parameter is the output head/frequency layer: it must train.
    assert named[-1][0] not in missing


@pytest.mark.parametrize("name", GRAPH_DEPENDENT)
def test_graph_dependent_backbones_react_to_topology(name: str) -> None:
    graph_a = build_graph_context(NODES, EDGES)
    graph_b = build_graph_context(NODES, EDGES_ALT)
    wrapper = _wrapper(name, "frozen")
    wrapper.eval()
    if name == "hyperd":
        # HyperD's road graph enters through the learned daily/weekly table,
        # which starts at zero by the documented no-prior deviation.  Inject a
        # non-constant prior so the graph actually reaches the prediction.
        wrapper.set_periodic_prior(
            torch.rand(288, NODES, generator=torch.Generator().manual_seed(5)),
            torch.rand(7 * 288, NODES, generator=torch.Generator().manual_seed(6)),
        )
    inp = _input()
    with torch.no_grad():
        out_a = wrapper.encode_and_predict(inp, graph_a).base_pred_z
        out_b = wrapper.encode_and_predict(inp, graph_b).base_pred_z
    assert not torch.allclose(out_a, out_b), f"{name} ignored the road graph"


def test_hyperd_periodic_prior_restores_the_graph_and_linear1_gradient(tiny_graph: GraphContext) -> None:
    wrapper = _wrapper("hyperd", "full")
    generator = torch.Generator().manual_seed(11)
    wrapper.set_periodic_prior(torch.rand(288, NODES, generator=generator))
    wrapper.train()
    out = wrapper.encode_and_predict(_input(), tiny_graph)
    out.base_pred_z.sum().backward()
    linear1 = wrapper.model.daily_emb.linear1
    assert linear1.weight.grad is not None
    assert float(linear1.weight.grad.abs().sum()) > 0.0


def test_hyperd_periodic_prior_rejects_wrong_shape() -> None:
    wrapper = _wrapper("hyperd", "frozen")
    with pytest.raises(Exception, match="prior must be"):
        wrapper.set_periodic_prior(torch.zeros(12, NODES))


def test_unknown_backbone_name_is_rejected() -> None:
    with pytest.raises(Exception, match="backbone.name"):
        BackboneConfig.from_dict({"name": "not_a_model"})
