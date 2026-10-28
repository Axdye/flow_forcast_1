from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sbona.evaluation.artifacts import aggregate_experiments, validate_metrics_document

COLUMNS = [
    "run_id",
    "method",
    "dataset",
    "backbone",
    "split",
    "seed",
    "protocol_id",
    "data_hash",
    "metric",
    "value",
    "units",
    "deployable",
    "artifact",
]


def _write_csv(path: Path, rows: list[dict[str, str]], columns: list[str] = COLUMNS) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _row(artifact: Path, **overrides: str) -> dict[str, str]:
    row = {
        "run_id": "run-1",
        "method": "sbona_router",
        "dataset": "PEMS08",
        "backbone": "staeformer",
        "split": "test",
        "seed": "11",
        "protocol_id": "flow_12x12_v1",
        "data_hash": "abc",
        "metric": "raw_mae_all",
        "value": "1.0",
        "units": "flow per sampling interval (raw publisher units)",
        "deployable": "true",
        "artifact": str(artifact),
    }
    row.update(overrides)
    return row


def test_aggregate_reports_one_value_per_method(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")
    rows = [
        _row(artifact, method="sbona_base", value="1.0"),
        _row(artifact, method="sbona_structured_router", value="0.9"),
    ]
    manifest = tmp_path / "experiments.csv"
    _write_csv(manifest, rows)
    document = aggregate_experiments(manifest, tmp_path / "aggregate.json", tmp_path / "aggregate.csv")
    assert document["rows"] == 2
    assert document["selected_rows"] == 2
    assert document["split"] == "test"
    assert document["seed"] == 11
    by_method = {group["method"]: group["value"] for group in document["groups"]}
    assert by_method == {"sbona_base": 1.0, "sbona_structured_router": 0.9}
    assert all("std" not in group for group in document["groups"])


def test_aggregate_counts_one_seed_per_method_once(tmp_path: Path) -> None:
    """Two rows for one seed/method/metric must be an error, not n=2."""
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "experiments.csv"
    _write_csv(
        manifest,
        [
            _row(artifact, run_id="run-eval-a", value="0.80"),
            _row(artifact, run_id="run-eval-b", value="1.20"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate test row"):
        aggregate_experiments(manifest, tmp_path / "aggregate.json")


def test_aggregate_never_mixes_val_into_the_test_table(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "experiments.csv"
    _write_csv(
        manifest,
        [
            _row(artifact, method="sbona_base", split="val", value="0.80"),
            _row(artifact, method="sbona_router", split="test", value="1.20"),
        ],
    )
    document = aggregate_experiments(manifest, tmp_path / "aggregate.json")
    assert [group["method"] for group in document["groups"]] == ["sbona_router"]
    assert document["groups"][0]["value"] == 1.20

    with pytest.raises(ValueError, match="no split=train rows"):
        aggregate_experiments(manifest, tmp_path / "aggregate.json", split="train")


def test_aggregate_rejects_multiple_seeds(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "experiments.csv"
    _write_csv(
        manifest,
        [
            _row(artifact, seed="11", value="1.0"),
            _row(artifact, seed="22", value="3.0"),
        ],
    )
    with pytest.raises(ValueError, match="mixes 2 seeds"):
        aggregate_experiments(manifest, tmp_path / "aggregate.json")


def test_aggregate_requires_a_split_column(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "experiments.csv"
    row = {key: value for key, value in _row(artifact).items() if key != "split"}
    _write_csv(manifest, [row], columns=[c for c in COLUMNS if c != "split"])
    with pytest.raises(ValueError, match="missing columns: \\['split'\\]"):
        aggregate_experiments(manifest, tmp_path / "aggregate.json")


def test_oracle_rows_cannot_be_deployable(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "experiments.csv"
    _write_csv(manifest, [_row(artifact, method="oracle_mae_offline", deployable="true")])
    with pytest.raises(ValueError, match="cannot be marked deployable"):
        aggregate_experiments(manifest, tmp_path / "aggregate.json")


def test_mixed_units_for_one_metric_are_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "experiments.csv"
    _write_csv(
        manifest,
        [
            _row(artifact, method="sbona_base", units="percent"),
            _row(
                artifact,
                method="sbona_router",
                units="flow per sampling interval (raw publisher units)",
            ),
        ],
    )
    with pytest.raises(ValueError, match="inconsistent units"):
        aggregate_experiments(manifest, tmp_path / "aggregate.json")


def test_metrics_document_requires_schema_and_counts() -> None:
    validate_metrics_document(
        {
            "schema_version": 1,
            "protocol_id": "p",
            "metric_units": {},
            "counts": {},
        }
    )
    with pytest.raises(ValueError, match="missing keys"):
        validate_metrics_document({"schema_version": 1})
