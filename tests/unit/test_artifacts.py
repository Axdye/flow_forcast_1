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


def test_aggregate_groups_by_seed(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")
    rows = [
        _row(artifact, seed="11", value="1.0"),
        _row(artifact, seed="22", value="3.0"),
    ]
    manifest = tmp_path / "experiments.csv"
    _write_csv(manifest, rows)
    document = aggregate_experiments(manifest, tmp_path / "aggregate.json", tmp_path / "aggregate.csv")
    assert document["rows"] == 2
    group = document["groups"][0]
    assert group["seeds"] == 2
    assert group["mean"] == 2.0


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
            _row(artifact, units="percent"),
            _row(artifact, seed="22", units="flow per sampling interval (raw publisher units)"),
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
