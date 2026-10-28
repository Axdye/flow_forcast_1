"""End-to-end check that published raw files reproduce the audited artifacts.

This test is opt-in because it needs ~160 MB of downloads::

    python scripts/fetch_data.py --all --dest raw
    SBONA_RAW_DIR=raw python -m pytest tests/integration/test_raw_import_golden.py -q

It imports the *published* files into a temporary data root with
``scripts/import_raw.py`` and asserts that the prepared arrays, graph, split,
scaler and calendar hash equal ``tests/data/golden_prepared.json``.  The golden
values were recorded from the copies audited in ``reviews/DATA_PROVENANCE.md``,
so a match means the machine processed the same numbers we reviewed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sbona.config import load_config
from sbona.data.dataset import prepare_dataset
from sbona.data.manifest import sha256_array
from sbona.training.trainer import make_hashes

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads((Path(__file__).resolve().parents[1] / "data" / "golden_prepared.json").read_text())

DATASETS = {
    "pems03": "PEMS03",
    "pems04": "PEMS04",
    "pems07": "PEMS07",
    "pems08": "PEMS08",
    "metrla": "METRLA",
    "pemsbay": "PEMSBAY",
}

RAW_DIR = os.environ.get("SBONA_RAW_DIR")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not RAW_DIR,
        reason="set SBONA_RAW_DIR to a directory populated by scripts/fetch_data.py --all",
    ),
]


@pytest.mark.parametrize(("config_name", "dataset"), sorted(DATASETS.items()))
def test_published_raw_files_reproduce_audited_artifacts(
    config_name: str, dataset: str, tmp_path: Path
) -> None:
    raw_root = Path(RAW_DIR).resolve()
    raw_dir = raw_root / dataset
    if not raw_dir.is_dir():
        pytest.skip(f"{raw_dir} not downloaded; run scripts/fetch_data.py --dataset {dataset}")

    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "import_raw.py"),
            "--dataset",
            dataset,
            "--raw-dir",
            str(raw_root),
            "--data-root",
            str(data_root),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    cfg = load_config(REPO_ROOT / "configs" / "experiments" / f"{config_name}.yaml")
    prepared, _ = prepare_dataset(cfg, data_root, output_root)
    hashes = make_hashes(prepared, cfg, cfg.adapter.num_candidates)

    golden = GOLDEN[dataset]
    assert hashes["data_hash"] == golden["data_hash"], "raw flow values changed"
    assert hashes["valid_hash"] == golden["valid_hash"], "missing/zero mask changed"
    assert hashes["graph_hash"] == golden["graph_hash"], "topology or edge set changed"
    assert hashes["split_hash"] == golden["split_hash"], "window split changed"
    assert hashes["scaler_hash"] == golden["scaler_hash"], "train-only statistics changed"
    assert prepared.graph.node_order_hash == golden["node_order_hash"], "node order changed"
    assert prepared.num_nodes == golden["num_nodes"]
    assert {
        name: int(starts.size) for name, starts in prepared.scheme.starts.items()
    } == golden["windows"]

    with np.load(data_root / dataset / "calendar.npz", allow_pickle=False) as archive:
        calendar = np.asarray(archive["timestamps"])
    assert sha256_array(calendar) == golden["calendar_hash"], "tod/dow table changed"


def test_import_records_upstream_hashes(tmp_path: Path) -> None:
    """A successful import must leave a traceable RAW_SOURCE.json next to the data."""
    raw_root = Path(RAW_DIR).resolve() if RAW_DIR else None
    if raw_root is None or not (raw_root / "PEMS08").is_dir():
        pytest.skip("PEMS08 raw files are not downloaded")
    data_root = tmp_path / "data"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "import_raw.py"),
            "--dataset",
            "PEMS08",
            "--raw-dir",
            str(raw_root),
            "--data-root",
            str(data_root),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads((data_root / "PEMS08" / "RAW_SOURCE.json").read_text())
    assert record["series"]["sha256"] == GOLDEN["PEMS08"]["raw_series_sha256"]
    assert record["graph"]["sha256"] == GOLDEN["PEMS08"]["raw_graph_sha256"]
    assert record["graph"]["undirected_edges"] == 274
