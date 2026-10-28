from __future__ import annotations

import numpy as np
import pytest

from sbona.config import Config, ConfigError
from sbona.data.graph import GraphError, build_graph_context
from sbona.evaluation.runner import metric_units


def _base_dict() -> dict:
    return {
        "schema_version": 1,
        "protocol_id": "flow_12x12_v1",
        "method_id": "sbona_base",
        "seed": 11,
        "data": {
            "dataset": "METRLA",
            "data_file": "METRLA/data.npz",
            "graph_file": "METRLA/adj_edges.npz",
        },
        "graph": {"topology": "derived_binary_undirected"},
        "backbone": {"name": "staeformer", "mode": "frozen"},
        "adapter": {"num_candidates": 8},
    }


def test_derived_topology_is_accepted_and_hashes_differently() -> None:
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    published = build_graph_context(3, edges, topology="published_binary_undirected")
    derived = build_graph_context(3, edges, topology="derived_binary_undirected")
    assert derived.graph_hash != published.graph_hash
    # Same edges and the same row-normalized diffusion.
    assert derived.num_edges == published.num_edges == 2
    assert np.allclose(
        derived.diffusion.to_dense().numpy(), published.diffusion.to_dense().numpy()
    )


def test_unknown_topology_is_rejected() -> None:
    with pytest.raises(GraphError, match="unknown topology"):
        build_graph_context(3, np.array([[0, 1]], dtype=np.int64), topology="gaussian")


def test_config_accepts_derived_topology() -> None:
    cfg = Config.from_dict(_base_dict())
    assert cfg.graph.topology == "derived_binary_undirected"


def test_config_rejects_unknown_topology() -> None:
    raw = _base_dict()
    raw["graph"]["topology"] = "gaussian"
    with pytest.raises(ConfigError, match="unknown graph.topology"):
        Config.from_dict(raw)


def test_unit_drives_metric_labels() -> None:
    raw = _base_dict()
    raw["data"]["unit"] = "speed (mph)"
    cfg = Config.from_dict(raw)
    assert cfg.data.unit == "speed (mph)"
    units = metric_units(cfg.data.unit)
    assert units["mae"] == "speed (mph)"
    assert units["rmse"] == "speed (mph)"
    assert units["mape"] == "percent"
    assert units["nvr"] == "fraction of valid elements"


def test_empty_unit_is_rejected() -> None:
    raw = _base_dict()
    raw["data"]["unit"] = "   "
    with pytest.raises(ConfigError, match="data.unit"):
        Config.from_dict(raw)
