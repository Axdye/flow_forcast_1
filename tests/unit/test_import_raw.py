"""Fast checks for the raw-file importer helpers (no downloads required)."""

from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "scripts" / "import_raw.py"


def _load_importer():
    spec = importlib.util.spec_from_file_location("sbona_import_raw", SPEC_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


importer = _load_importer()


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def test_binarize_undirected_unions_and_drops_diagonal() -> None:
    adjacency = np.array(
        [
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 3.0],
            [0.0, 0.0, 1.0],
        ]
    )
    edges = importer.binarize_undirected(adjacency)
    assert edges.tolist() == [[0, 1], [1, 2]]


def test_binarize_undirected_ignores_nan_and_negative() -> None:
    adjacency = np.array(
        [
            [0.0, np.nan, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert importer.binarize_undirected(adjacency).tolist() == [[1, 2]]


def test_read_adjacency_npz_accepts_both_orientations(tmp_path: Path) -> None:
    edges = np.array([[0, 1], [1, 2]])
    np.savez(tmp_path / "edge_index.npz", edge_index=edges.T)
    np.savez(tmp_path / "edges.npz", edges=edges)
    assert importer.read_adjacency(tmp_path / "edge_index.npz")[0].tolist() == edges.tolist()
    assert importer.read_adjacency(tmp_path / "edges.npz")[0].tolist() == edges.tolist()


class _Forbidden:
    def __reduce__(self):  # pragma: no cover - never actually executed
        return (eval, ("1 + 1",))


def test_restricted_unpickler_rejects_arbitrary_objects(tmp_path: Path) -> None:
    target = tmp_path / "graph.pkl"
    target.write_bytes(pickle.dumps(_Forbidden()))
    with pytest.raises(importer.ImportFailure):
        importer.read_adjacency(target)


def test_read_adjacency_dcrnn_tuple_layout(tmp_path: Path) -> None:
    ids = ["10", "20", "30"]
    id_to_ind = {node: index for index, node in enumerate(ids)}
    adjacency = np.zeros((3, 3))
    adjacency[0, 1] = 1.0
    target = tmp_path / "adj_mx.pkl"
    target.write_bytes(pickle.dumps((ids, id_to_ind, adjacency)))
    edges, read_ids = importer.read_adjacency(target)
    assert edges.tolist() == [[0, 1]]
    assert read_ids == ids


def test_read_node_ids_skips_header(tmp_path: Path) -> None:
    target = tmp_path / "ids.txt"
    target.write_text("sensor_id\n313344\n313349\n313438\n", encoding="utf-8")
    assert importer._read_node_ids(target, 3).tolist() == [313344, 313349, 313438]


# ---------------------------------------------------------------------------
# Calendar channel detection
# ---------------------------------------------------------------------------


def _channels(tod: np.ndarray, dow: np.ndarray) -> np.ndarray:
    nodes = 2
    series = np.zeros((tod.size, nodes, 3), dtype=np.float64)
    series[:, 0, 1] = tod
    series[:, 1, 1] = tod
    series[:, 0, 2] = dow
    series[:, 1, 2] = dow
    return series


def test_calendar_channels_accepts_real_tod_and_dow() -> None:
    steps = 4 * 288
    tod = (np.arange(steps) % 288) / 288.0
    dow = np.repeat(np.arange(4), 288).astype(np.float64)
    assert importer._calendar_channels_look_valid(_channels(tod, dow), 288)


def test_calendar_channels_rejects_occupancy_and_speed() -> None:
    steps = 4 * 288
    rng = np.random.default_rng(0)
    occupancy = rng.uniform(0.0, 0.8, size=steps)  # looks like a tod channel
    speed = rng.uniform(3.0, 85.0, size=steps)
    assert not importer._calendar_channels_look_valid(_channels(occupancy, speed), 288)


def test_calendar_channels_rejects_non_repeating_tod() -> None:
    steps = 4 * 288
    tod = np.linspace(0.0, 0.99, steps)  # monotone, not daily-periodic
    dow = np.repeat(np.arange(4), 288).astype(np.float64)
    assert not importer._calendar_channels_look_valid(_channels(tod, dow), 288)


# ---------------------------------------------------------------------------
# Series readers
# ---------------------------------------------------------------------------


def test_read_series_accepts_values_key(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.float64).reshape(4, 3)
    target = tmp_path / "series.npz"
    np.savez(target, values=values)
    series, timestamps, fmt = importer.read_series(target)
    assert series.shape == (4, 3)
    assert timestamps is None
    assert fmt == "npz"


def test_read_series_rejects_unknown_suffix(tmp_path: Path) -> None:
    target = tmp_path / "series.parquet"
    target.write_bytes(b"")
    with pytest.raises(importer.ImportFailure):
        importer.read_series(target)


def test_import_guard_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    existing = tmp_path / "data.npz"
    existing.write_bytes(b"audited")
    with pytest.raises(importer.ImportFailure):
        importer._guard_existing(existing, force=False)
    importer._guard_existing(existing, force=True)
