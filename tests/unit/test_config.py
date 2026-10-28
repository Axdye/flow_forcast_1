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


def test_backbone_params_default_to_the_official_values() -> None:
    raw = _base_dict()
    raw["backbone"] = {"name": "agcrn"}
    cfg = Config.from_dict(raw)
    # PEMS08 official configuration (BasicTS v0.5.8 ``AGCRN.yaml``).
    assert cfg.backbone.params == {
        "rnn_units": 64,
        "num_layers": 2,
        "cheb_k": 2,
        "embed_dim": 2,
    }


def test_backbone_params_override_keeps_other_defaults() -> None:
    raw = _base_dict()
    raw["backbone"] = {"name": "graph_wavenet", "params": {"blocks": 2, "dropout": 0.2}}
    cfg = Config.from_dict(raw)
    assert cfg.backbone.params["blocks"] == 2
    assert cfg.backbone.params["dropout"] == pytest.approx(0.2)
    assert cfg.backbone.params["layers"] == 2
    assert cfg.backbone.params["gcn_order"] == 2


def test_backbone_params_reject_unknown_keys() -> None:
    raw = _base_dict()
    raw["backbone"] = {"name": "stid", "params": {"not_a_knob": 1}}
    with pytest.raises(ConfigError, match="unknown key\\(s\\) in backbone.params"):
        Config.from_dict(raw)


def test_backbone_params_reject_another_architectures_knob() -> None:
    raw = _base_dict()
    # STAEformer is configured entirely by its flat fields; ``rnn_units`` is an
    # AGCRN knob and must not silently travel through ``params``.
    raw["backbone"] = {"name": "staeformer", "params": {"rnn_units": 64}}
    with pytest.raises(ConfigError, match="unknown key\\(s\\) in backbone.params"):
        Config.from_dict(raw)


def test_backbone_params_reject_wrong_types() -> None:
    raw = _base_dict()
    raw["backbone"] = {"name": "d2stgnn", "params": {"k_s": 2.5}}
    with pytest.raises(ConfigError, match="must be an integer"):
        Config.from_dict(raw)

    raw["backbone"] = {"name": "graph_wavenet", "params": {"add_adaptive_adj": 1}}
    with pytest.raises(ConfigError, match="must be a boolean"):
        Config.from_dict(raw)


def test_staeformer_input_dim_accepts_the_official_three_channel_projection() -> None:
    for input_dim in (1, 3):
        raw = _base_dict()
        raw["backbone"] = {"name": "staeformer", "mode": "frozen", "input_dim": input_dim}
        assert Config.from_dict(raw).backbone.input_dim == input_dim

    raw = _base_dict()
    raw["backbone"] = {"name": "staeformer", "input_dim": 2}
    with pytest.raises(ConfigError, match="input_dim must be 1 .* or 3"):
        Config.from_dict(raw)


def test_wta_score_weight_is_parsed_and_validated() -> None:
    raw = _base_dict()
    raw["loss"] = {"wta_score": 0.25}
    assert Config.from_dict(raw).loss.wta_score == pytest.approx(0.25)

    raw["loss"] = {"wta_score": -1.0}
    with pytest.raises(ConfigError, match="loss.wta_score must be >= 0"):
        Config.from_dict(raw)


def test_shipped_experiment_configs_still_parse() -> None:
    root = Path(__file__).resolve().parents[2] / "configs" / "experiments"
    paths = sorted(root.glob("*.yaml"))
    assert paths
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            import yaml

            Config.from_dict(yaml.safe_load(handle))


def test_every_backbone_has_a_pems08_experiment_config() -> None:
    import yaml

    from sbona.config import BACKBONE_PARAM_DEFAULTS, SUPPORTED_BACKBONES

    root = Path(__file__).resolve().parents[2] / "configs" / "experiments"
    for name in SUPPORTED_BACKBONES:
        path = root / f"pems08_{name}.yaml"
        assert path.is_file(), f"missing PEMS08 config for {name}"
        cfg = Config.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
        assert cfg.backbone.name == name
        # Each shipped config must pin the official defaults explicitly, so a
        # later change to ``BACKBONE_PARAM_DEFAULTS`` cannot silently change an
        # experiment that was already recorded.
        assert cfg.backbone.params == BACKBONE_PARAM_DEFAULTS[name]


def test_mcl_wta_config_selects_the_wta_baseline() -> None:
    import yaml

    path = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "pems08_mcl_wta.yaml"
    cfg = Config.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
    # The CLI appends the kind, so the stored run is ``sbona_mcl_wta``.
    assert cfg.method_id == "sbona"
    assert cfg.adapter.num_candidates >= 2
    assert cfg.loss.wta_score > 0

