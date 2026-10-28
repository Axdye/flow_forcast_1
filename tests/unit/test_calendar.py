from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sbona.config import Config, ConfigError
from sbona.data.dataset import (
    TrafficWindowDataset,
    build_calendar,
    compute_prepare_signature,
    load_prepared,
    prepare_dataset,
)
from sbona.data.manifest import DataError, load_timestamps_file


def _calendar_table(num_steps: int) -> np.ndarray:
    index = np.arange(num_steps, dtype=np.float64)
    tod = (index % 288) / 288.0
    dow = ((index // 288) % 7).astype(np.float32)
    return np.stack([tod, dow], axis=1)


def test_build_calendar_accepts_explicit_table() -> None:
    table = _calendar_table(7 * 288)
    tod, dow, available = build_calendar(table, table.shape[0], 288)
    assert available is True
    assert tod.shape == (7 * 288,)
    assert dow.shape == (7 * 288,)
    assert set(np.unique(dow).tolist()) == set(range(7))
    assert float(tod.max()) < 1.0


def test_build_calendar_rejects_bad_time_of_day() -> None:
    table = _calendar_table(288)
    table[3, 0] = 1.0
    with pytest.raises(DataError, match="time_of_day"):
        build_calendar(table, table.shape[0], 288)


def test_build_calendar_rejects_fractional_day_of_week() -> None:
    table = _calendar_table(288)
    table[5, 1] = 0.5
    with pytest.raises(DataError, match="day_of_week"):
        build_calendar(table, table.shape[0], 288)


def test_load_timestamps_file_roundtrip(tmp_path: Path) -> None:
    stamps = np.datetime64("2018-01-01T00:00:00", "s") + np.arange(10) * np.timedelta64(5, "m")
    assert stamps.shape == (10,)
    np.savez_compressed(tmp_path / "cal.npz", timestamps=stamps)
    loaded = load_timestamps_file(tmp_path / "cal.npz", 10)
    assert np.array_equal(loaded, stamps)

    np.save(tmp_path / "cal.npy", stamps)
    loaded_npy = load_timestamps_file(tmp_path / "cal.npy", 10)
    assert np.array_equal(loaded_npy, stamps)


def test_load_timestamps_file_length_mismatch(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "cal.npz", timestamps=np.arange(9, dtype=np.int64)
    )
    with pytest.raises(DataError, match="does not match"):
        load_timestamps_file(tmp_path / "cal.npz", 10)


def test_config_rejects_calendar_file_with_timestamps_disabled() -> None:
    raw = {
        "schema_version": 1,
        "protocol_id": "flow_12x12_v1",
        "method_id": "sbona_base",
        "seed": 11,
        "data": {
            "dataset": "PEMS08",
            "data_file": "PEMS08/data.npz",
            "graph_file": "PEMS08/PEMS08.csv",
            "timestamps_file": "PEMS08/calendar.npz",
            "timestamps_available": False,
        },
        "graph": {"topology": "published_binary_undirected"},
        "backbone": {"name": "staeformer", "mode": "frozen"},
        "adapter": {"num_candidates": 8},
    }
    with pytest.raises(ConfigError, match="contradicts"):
        Config.from_dict(raw)


def _write_fixture(root: Path, num_steps: int, calendar: np.ndarray) -> dict:
    data_dir = root / "data" / "TOY"
    data_dir.mkdir(parents=True, exist_ok=True)
    time = np.arange(num_steps, dtype=np.float32)
    values = 50.0 + 10.0 * np.sin(time / 7.0)
    data = np.stack([values, values + 1.0], axis=1)[:, :, None]
    np.savez_compressed(data_dir / "data.npz", data=data)
    (data_dir / "graph.csv").write_text("from,to,cost\n0,1,1.0\n", encoding="utf-8")
    np.savez_compressed(data_dir / "calendar.npz", timestamps=calendar)
    return {
        "schema_version": 1,
        "protocol_id": "flow_12x12_v1",
        "method_id": "sbona_base",
        "seed": 11,
        "data": {
            "dataset": "TOY",
            "data_file": "TOY/data.npz",
            "graph_file": "TOY/graph.csv",
            "timestamps_file": "TOY/calendar.npz",
            "timestamps_available": True,
            "day_of_week": True,
        },
        "graph": {"topology": "published_binary_undirected"},
        "backbone": {"name": "staeformer", "mode": "frozen", "dow_embedding_dim": 8},
        "adapter": {"num_candidates": 2},
    }


def test_prepare_dataset_records_and_uses_calendar(tmp_path: Path) -> None:
    num_steps = 100
    start = np.datetime64("2018-01-01T00:00:00", "s")
    stamps = start + np.arange(num_steps) * np.timedelta64(5, "m")
    cfg = Config.from_dict(_write_fixture(tmp_path, num_steps, stamps))

    prepared, path = prepare_dataset(cfg, tmp_path / "data", tmp_path / "out")
    assert prepared.calendar_available is True
    assert "sha256=" in prepared.manifest.timestamps_source
    assert [f.role for f in prepared.manifest.files].count("timestamps") == 1

    loaded = load_prepared(path)
    dataset = TrafficWindowDataset(loaded, "train")
    sample = dataset[0]
    # 2018-01-01 is a Monday, so the first train window starts on day 0.
    assert int(sample.model_input.day_of_week.min()) == 0
    assert int(sample.model_input.day_of_week.max()) == 0

    signature = compute_prepare_signature(cfg, tmp_path / "data")
    np.savez_compressed(
        tmp_path / "data" / "TOY" / "calendar.npz",
        timestamps=start + np.arange(num_steps) * np.timedelta64(10, "m"),
    )
    assert compute_prepare_signature(cfg, tmp_path / "data") != signature
