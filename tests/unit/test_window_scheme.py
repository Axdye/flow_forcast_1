from __future__ import annotations

import numpy as np
import pytest

from sbona.data.windows import (
    WindowError,
    build_subrange_scheme,
    build_window_scheme,
    split_boundaries,
    validate_scheme,
)


def test_split_boundaries_and_target_ranges() -> None:
    bounds = split_boundaries(1000, (0.6, 0.2, 0.2))
    assert (bounds.train_end, bounds.val_end) == (600, 800)
    assert bounds.target_range("train") == (0, 600)
    assert bounds.target_range("val") == (600, 800)
    assert bounds.target_range("test") == (800, 1000)


def test_windows_never_leak_across_splits() -> None:
    scheme = build_window_scheme(1000, input_steps=12, output_steps=12)
    for split in ("train", "val", "test"):
        starts = scheme.starts[split]
        lo, hi = scheme.target_bounds(split)
        assert starts.size > 0
        assert starts.min() >= 12
        assert starts.min() >= lo
        assert starts.max() + 12 <= hi
        for start, history_start, target_end in scheme.iter_windows(split):
            assert history_start == start - 12
            assert target_end == start + 12
    assert scheme.starts["train"].max() + 12 <= scheme.starts["val"].min()
    assert scheme.starts["val"].max() + 12 <= scheme.starts["test"].min()


def test_validate_scheme_rejects_overlap() -> None:
    scheme = build_window_scheme(1000, input_steps=12, output_steps=12)
    scheme.starts["val"] = np.array([scheme.starts["train"].max()], dtype=np.int64)
    with pytest.raises(WindowError, match="overlap"):
        validate_scheme(scheme)


def test_too_short_split_fails_fast() -> None:
    with pytest.raises(WindowError):
        build_window_scheme(45, input_steps=12, output_steps=12)


def test_subrange_scheme_bounds_history() -> None:
    scheme = build_subrange_scheme(
        500,
        12,
        12,
        {"train": (0, 100), "probe": (100, 200)},
    )
    assert scheme.starts["train"].min() == 12
    assert scheme.starts["probe"].min() == 100
    assert scheme.starts["probe"].max() + 12 <= 200
