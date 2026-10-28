"""Deterministic window construction and split boundaries.

The only v1 split rule (design section 4.2): for ``M`` raw steps,
``a=floor(0.6 M)``, ``b=floor(0.8 M)``.  A window is identified by the absolute
index ``s`` of its first future step.  History is ``[s-L, s)`` and target is
``[s, s+T)``.  The *entire target* must lie inside one split; history may cross
from an earlier split because those observations are already in the past.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

SPLITS = ("train", "val", "test")


class WindowError(ValueError):
    """Raised when a split would produce an empty or leaking window set."""


@dataclass(frozen=True)
class SplitBoundaries:
    num_steps: int
    train_end: int
    val_end: int

    def target_range(self, split: str) -> tuple[int, int]:
        if split == "train":
            return 0, self.train_end
        if split == "val":
            return self.train_end, self.val_end
        if split == "test":
            return self.val_end, self.num_steps
        raise WindowError(f"unknown split {split!r}")


@dataclass(frozen=True)
class WindowScheme:
    num_steps: int
    input_steps: int
    output_steps: int
    boundaries: SplitBoundaries
    starts: dict[str, np.ndarray]

    def split_of_start(self, start: int) -> str:
        if 0 <= start < self.boundaries.train_end:
            return "train"
        if self.boundaries.train_end <= start < self.boundaries.val_end:
            return "val"
        if self.boundaries.val_end <= start < self.num_steps:
            return "test"
        raise WindowError(f"start {start} outside the time axis")

    def target_bounds(self, split: str) -> tuple[int, int]:
        return self.boundaries.target_range(split)

    def iter_windows(self, split: str) -> Iterator[tuple[int, int, int]]:
        """Yield ``(start, history_start, target_end)`` for one split."""
        for start in self.starts[split].tolist():
            yield int(start), int(start) - self.input_steps, int(start) + self.output_steps

    def to_dict(self) -> dict[str, object]:
        return {
            "num_steps": self.num_steps,
            "input_steps": self.input_steps,
            "output_steps": self.output_steps,
            "train_end": self.boundaries.train_end,
            "val_end": self.boundaries.val_end,
            "starts": {k: v.tolist() for k, v in self.starts.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> WindowScheme:
        train_end = int(data["train_end"])
        val_end = int(data["val_end"])
        num_steps = int(data["num_steps"])
        return cls(
            num_steps=num_steps,
            input_steps=int(data["input_steps"]),
            output_steps=int(data["output_steps"]),
            boundaries=SplitBoundaries(num_steps, train_end, val_end),
            starts={k: np.asarray(v, dtype=np.int64) for k, v in dict(data["starts"]).items()},
        )


def split_boundaries(num_steps: int, ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> SplitBoundaries:
    if num_steps < 3:
        raise WindowError("need at least 3 time steps")
    train_end = int(np.floor(ratios[0] * num_steps))
    val_end = int(np.floor((ratios[0] + ratios[1]) * num_steps))
    if not (0 < train_end < val_end < num_steps):
        raise WindowError(
            f"degenerate split for M={num_steps}: train_end={train_end}, val_end={val_end}"
        )
    return SplitBoundaries(num_steps=num_steps, train_end=train_end, val_end=val_end)


def build_window_scheme(
    num_steps: int,
    input_steps: int,
    output_steps: int,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    allow_empty: bool = False,
) -> WindowScheme:
    """Build all windows for the three splits.

    Train target in ``[0, a)``, val target in ``[a, b)``, test target in
    ``[b, M)``.  History always has ``s >= L`` so it stays in-bounds.
    """
    if input_steps < 1 or output_steps < 1:
        raise WindowError("input_steps and output_steps must be >= 1")
    bounds = split_boundaries(num_steps, ratios)
    starts: dict[str, np.ndarray] = {}
    for split in SPLITS:
        lo, hi = bounds.target_range(split)
        first = max(lo, input_steps)
        last = hi - output_steps
        if last < first:
            if allow_empty:
                starts[split] = np.arange(first, first, dtype=np.int64)
                continue
            raise WindowError(
                f"split {split} has no room for L={input_steps}, T={output_steps} "
                f"(target range [{lo}, {hi}))"
            )
        starts[split] = np.arange(first, last + 1, dtype=np.int64)
    scheme = WindowScheme(
        num_steps=num_steps,
        input_steps=input_steps,
        output_steps=output_steps,
        boundaries=bounds,
        starts=starts,
    )
    validate_scheme(scheme)
    return scheme


def build_subrange_scheme(
    num_steps: int,
    input_steps: int,
    output_steps: int,
    ranges: dict[str, tuple[int, int]],
) -> WindowScheme:
    """Explicit target ranges, used by the F/R/Q probe protocol.

    Keys are split names (``f``, ``r``, ``q`` in the probe).  History may start
    before ``range[0]`` but never before 0.
    """
    if input_steps < 1 or output_steps < 1:
        raise WindowError("input_steps and output_steps must be >= 1")
    starts: dict[str, np.ndarray] = {}
    for name, (lo, hi) in ranges.items():
        if not (0 <= lo < hi <= num_steps):
            raise WindowError(f"invalid range {name}={lo,hi}")
        first = max(lo, input_steps)
        last = hi - output_steps
        if last < first:
            raise WindowError(f"range {name} too small for L/T")
        starts[name] = np.arange(first, last + 1, dtype=np.int64)
    # Reuse the dataclass with synthetic boundaries for bookkeeping only.
    all_starts = np.concatenate(list(starts.values())) if starts else np.arange(0, 0)
    train_end = int(all_starts.min()) if all_starts.size else 0
    val_end = int(all_starts.max()) + output_steps if all_starts.size else 0
    bounds = SplitBoundaries(num_steps=num_steps, train_end=max(train_end, 0), val_end=min(val_end, num_steps))
    scheme = WindowScheme(
        num_steps=num_steps,
        input_steps=input_steps,
        output_steps=output_steps,
        boundaries=bounds,
        starts=starts,
    )
    for name, (_lo, hi) in ranges.items():
        if starts[name].size:
            if int(starts[name].min()) + output_steps > hi:
                raise WindowError(f"range {name} target exceeds upper bound")
    return scheme


def validate_scheme(scheme: WindowScheme) -> None:
    L, T = scheme.input_steps, scheme.output_steps
    for split in SPLITS:
        if split not in scheme.starts:
            raise WindowError(f"missing split {split}")
    # Target sets must be disjoint by construction; check overlap before the
    # per-split bounds so that a leaked window is always reported as leakage.
    train = scheme.starts["train"]
    val = scheme.starts["val"]
    test = scheme.starts["test"]
    if train.size and val.size:
        if int(train.max()) + T > int(val.min()):
            raise WindowError("train/val target windows overlap")
    if val.size and test.size:
        if int(val.max()) + T > int(test.min()):
            raise WindowError("val/test target windows overlap")
    for split in SPLITS:
        starts = scheme.starts[split]
        lo, hi = scheme.boundaries.target_range(split)
        if starts.size:
            if int(starts.min()) < L:
                raise WindowError(f"{split}: history would start before 0")
            if int(starts.min()) < lo or int(starts.max()) + T > hi:
                raise WindowError(f"{split}: target leaves its split")


__all__ = [
    "SPLITS",
    "WindowError",
    "SplitBoundaries",
    "WindowScheme",
    "split_boundaries",
    "build_window_scheme",
    "build_subrange_scheme",
    "validate_scheme",
]
