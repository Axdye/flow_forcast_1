from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sbona.cli import main


def _config() -> dict:
    return {
        "schema_version": 1,
        "protocol_id": "synthetic_cli_v1",
        "method_id": "sbona",
        "seed": 5,
        "data": {
            "dataset": "synthetic",
            "input_steps": 12,
            "output_steps": 12,
            "split_ratios": [0.6, 0.2, 0.2],
            "steps_per_day": 24,
            "synthetic": True,
            "synthetic_num_nodes": 5,
            "synthetic_num_steps": 120,
            "synthetic_missing_fraction": 0.0,
            "day_of_week": False,
        },
        "graph": {"topology": "published_binary_undirected"},
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
        "training": {
            "precision": "fp32",
            "initial_batch_size": 4,
            "eval_batch_size": 4,
            "max_epochs": {"base": 1, "single": 1, "candidates": 1, "router": 1, "probe": 1},
            "patience": {"base": 1, "single": 1, "candidates": 1, "router": 1, "probe": 1},
            "num_workers": 0,
            "device": "cpu",
        },
        "evaluation": {
            "horizons": [3, 6, 12],
            "bootstrap_replicates": 20,
            "block_origins": 12,
        },
    }


@pytest.mark.slow
def test_cli_synthetic_end_to_end(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config(), sort_keys=False), encoding="utf-8")
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    common = [
        "--config",
        str(config_path),
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
        "--synthetic",
        "--device",
        "cpu",
    ]

    assert main(["prepare-data", *common]) == 0
    assert (output_root / "prepared").is_dir() or list(output_root.glob("prepared*"))

    assert main(["train-base", *common, "--run-id", "base"]) == 0
    assert (output_root / "base" / "checkpoint_best.pt").is_file()

    assert main(["train-single", *common, "--run-id", "single"]) == 0
    assert (output_root / "single" / "checkpoint_best.pt").is_file()

    assert main(["train-candidates", *common, "--run-id", "candidates"]) == 0
    assert (output_root / "candidates" / "checkpoint_best.pt").is_file()

    assert main(["train-router", *common, "--run-id", "router"]) == 0
    assert (output_root / "router" / "checkpoint_best.pt").is_file()

    assert main(["evaluate", *common, "--run", "base", "--split", "test"]) == 0
    assert (output_root / "base-eval-test" / "metrics.json").is_file()

    assert main(["evaluate", *common, "--run", "candidates", "--split", "test"]) == 0
    eval_dir = output_root / "candidates-eval-test"
    assert (eval_dir / "metrics.json").is_file()
    assert (eval_dir / "top1_test.npz").is_file()

    # Evaluation must not overwrite the source training run.
    assert (output_root / "candidates" / "metrics.json").is_file()
    assert (output_root / "experiments.csv").is_file()
