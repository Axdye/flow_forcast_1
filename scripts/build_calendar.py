"""Build a calendar file from a documented dataset start timestamp.

This is a data-provenance helper, not a model component.  It exists so that a
calendar is derived from a *recorded* start date instead of guessing the
weekday phase from the flow curve.  The generated ``.npz`` is read through
``data.timestamps_file`` and its hash is stored in the dataset manifest.

Example::

    python scripts/build_calendar.py \
        --start 2018-01-01 --steps 16992 --interval-minutes 5 \
        --note "PEMS04, publisher start date" \
        --out data/PEMS04/calendar.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_calendar(start: str, steps: int, interval_minutes: int) -> np.ndarray:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be >= 1")
    origin = np.datetime64(start, "s")
    offsets = np.arange(steps, dtype=np.int64) * np.timedelta64(interval_minutes, "m")
    return (origin + offsets).astype("datetime64[s]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="ISO start timestamp, e.g. 2018-01-01")
    parser.add_argument("--steps", type=int, required=True, help="number of raw time steps")
    parser.add_argument(
        "--interval-minutes", type=int, default=5, help="sampling interval in minutes"
    )
    parser.add_argument("--note", default="", help="free-form provenance note")
    parser.add_argument("--out", required=True, help="output .npz path")
    args = parser.parse_args()

    stamps = build_calendar(args.start, args.steps, args.interval_minutes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        timestamps=stamps,
        start=np.asarray(args.start),
        interval_minutes=np.int64(args.interval_minutes),
        note=np.asarray(args.note),
    )
    print(f"wrote {out} steps={stamps.shape[0]} {stamps[0]} .. {stamps[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
