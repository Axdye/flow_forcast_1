from __future__ import annotations

from pathlib import Path

import pytest

from sbona.config import Config, ConfigError, resolve_roots


def _base_dict() -> dict:
    return {
        "schema_version": 1,
        "protocol_id": "flow_12x12_v1",
        "method_id": "sbona_structured",
        "seed": 11,
        "data": {
            "dataset": "PEMS08",
            "data_file": "PEMS08/data.npz",
            "graph_file": "PEMS08/PEMS08.csv",
        },
        "graph": {"topology": "published_binary_undirected"},
        "backbone": {"name": "staeformer", "mode": "frozen"},
        "adapter": {"num_candidates": 8},
    }


def test_unknown_root_key_is_rejected() -> None:
    raw = _base_dict()
    raw["typo_key"] = 1
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict(raw)


def test_unknown_nested_key_is_rejected() -> None:
    raw = _base_dict()
    raw["data"]["unknown_field"] = 1
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict(raw)


def test_real_dataset_requires_graph_file_outside_identity() -> None:
    raw = _base_dict()
    raw["data"].pop("graph_file")
    with pytest.raises(ConfigError, match="data.graph_file"):
        Config.from_dict(raw)


def test_identity_graph_is_an_explicit_diagnostic() -> None:
    raw = _base_dict()
    raw["data"].pop("graph_file")
    raw["graph"]["topology"] = "identity"
    cfg = Config.from_dict(raw)
    assert cfg.graph.topology == "identity"


def test_alpha_below_one_needs_explicit_ablation_method_id() -> None:
    raw = _base_dict()
    raw["oracle"] = {"alpha": 0.5, "temperature": 0.5}
    raw["loss"] = {"consistency": 0.0}
    with pytest.raises(ConfigError, match="explicit ablation method_id"):
        Config.from_dict(raw)

    raw["method_id"] = "sbona_cost_only_ablation"
    cfg = Config.from_dict(raw)
    assert cfg.oracle.alpha == 0.5


def test_split_ratios_must_sum_to_one() -> None:
    raw = _base_dict()
    raw["data"]["split_ratios"] = [0.5, 0.2, 0.2]
    with pytest.raises(ConfigError, match="sum to 1"):
        Config.from_dict(raw)


def test_unimplemented_accumulation_is_rejected() -> None:
    raw = _base_dict()
    raw["training"] = {"accumulation": 2}
    with pytest.raises(ConfigError, match="accumulation"):
        Config.from_dict(raw)


def test_unfreezing_candidates_for_router_is_rejected() -> None:
    raw = _base_dict()
    raw["training"] = {"freeze_candidates_for_router": False}
    with pytest.raises(ConfigError, match="freezes candidates"):
        Config.from_dict(raw)


def test_horizons_must_fit_output_steps() -> None:
    raw = _base_dict()
    raw["data"]["output_steps"] = 6
    raw["evaluation"] = {"horizons": [3, 6, 12]}
    with pytest.raises(ConfigError, match="within 1..T"):
        Config.from_dict(raw)


def test_root_resolution_priority(tmp_path: Path) -> None:
    raw = _base_dict()
    raw["paths"] = {"data_root": str(tmp_path / "config-data"), "output_root": str(tmp_path / "config-out")}
    cfg = Config.from_dict(raw)

    data, output = resolve_roots(
        cfg,
        data_root=tmp_path / "cli-data",
        output_root=tmp_path / "cli-out",
        env={"SBONA_DATA_ROOT": "env-data", "SBONA_OUTPUT_ROOT": "env-out"},
    )
    assert data == tmp_path / "cli-data"
    assert output == tmp_path / "cli-out"

    data, output = resolve_roots(
        cfg,
        env={"SBONA_DATA_ROOT": "env-data", "SBONA_OUTPUT_ROOT": "env-out"},
    )
    assert data == Path("env-data")
    assert output == Path("env-out")

    data, output = resolve_roots(cfg, env={})
    assert data == tmp_path / "config-data"
    assert output == tmp_path / "config-out"


def test_no_legacy_absolute_paths_in_experiment_configs() -> None:
    root = Path(__file__).resolve().parents[2] / "configs" / "experiments"
    for path in sorted(root.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert "/root/Flow-RL" not in text
        assert "/Users/" not in text
