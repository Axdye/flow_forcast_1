"""Build a calendar file from a documented dataset start date or its channels.

This is a data-provenance helper, not a model component.  The resulting
``.npz`` is read through ``data.timestamps_file`` and its hash is stored in the
dataset manifest.

Two modes:

1. From a recorded start timestamp (uniform sampling)::

       python scripts/build_calendar.py --start 2018-01-01 --steps 16992 \
           --out data/PEMS04/calendar.npz

2. From the tod/dow channels that some ``data.npz`` files already carry.  The
   ``--start-weekday`` argument only determines a *constant offset* applied to
   the stored day phase, so a phase-only channel (day index mod 7) can be turned
   into a real weekday.  ``--start-weekday 4`` means the first day is a Friday
   (Monday=0).  ``tod`` is taken verbatim, so intraday gaps are preserved::

       python scripts/build_calendar.py --from-data data/PEMS08/data.npz \
           --start-weekday 4 --out data/PEMS08/calendar.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def build_uniform(start: str, steps: int, interval_minutes: int) -> np.ndarray:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be >= 1")
    origin = np.datetime64(start, "s")
    offsets = np.arange(steps, dtype=np.int64) * np.timedelta64(interval_minutes, "m")
    return (origin + offsets).astype("datetime64[s]")


def build_table(
    data_path: Path,
    start_weekday: int,
    steps_per_day: int,
    tod_channel: int,
    dow_channel: int,
) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(data_path, allow_pickle=False) as archive:
        if "data" not in archive:
            raise ValueError(f"{data_path} must contain a 'data' array")
        array = np.asarray(archive["data"])
    return build_table_from_array(
        array,
        start_weekday=start_weekday,
        steps_per_day=steps_per_day,
        tod_channel=tod_channel,
        dow_channel=dow_channel,
        source=str(data_path),
    )


def build_table_from_array(
    array: np.ndarray,
    start_weekday: int,
    steps_per_day: int,
    tod_channel: int = 1,
    dow_channel: int = 2,
    source: str = "<array>",
) -> tuple[np.ndarray, dict[str, object]]:
    """Same as :func:`build_table` but without touching the filesystem."""
    if not 0 <= start_weekday <= 6:
        raise ValueError("start_weekday must be in 0..6 (Monday=0)")
    if array.ndim != 3 or array.shape[-1] <= max(tod_channel, dow_channel):
        raise ValueError(
            f"expected data [M, N, C] with channels {tod_channel}/{dow_channel}, got {array.shape}"
        )
    tod = np.asarray(array[:, 0, tod_channel], dtype=np.float64)
    phase_raw = np.asarray(array[:, 0, dow_channel], dtype=np.float64)
    if not np.isfinite(tod).all() or tod.min() < 0.0 or tod.max() >= 1.0:
        raise ValueError("time-of-day channel must be finite and in [0, 1)")
    phase = np.rint(phase_raw).astype(np.int64)
    if not np.allclose(phase_raw, phase, atol=1e-6) or phase.min() < 0 or phase.max() > 6:
        raise ValueError("day channel must contain integer days 0..6")
    offset = int((start_weekday - int(phase[0])) % 7)
    dow = (phase + offset) % 7
    table = np.stack([tod.astype(np.float32), dow.astype(np.float32)], axis=1)
    info = {
        "mode": "data_channels",
        "start_weekday": int(start_weekday),
        "start_weekday_name": WEEKDAY_NAMES[start_weekday],
        "day_offset_applied": offset,
        "steps_per_day": int(steps_per_day),
        "tod_channel": int(tod_channel),
        "dow_channel": int(dow_channel),
        "source": source,
    }
    return table, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="ISO start timestamp (uniform mode)")
    parser.add_argument("--steps", type=int, help="number of raw time steps (uniform mode)")
    parser.add_argument(
        "--interval-minutes", type=int, default=5, help="sampling interval in minutes"
    )
    parser.add_argument("--from-data", help="data.npz carrying tod/dow channels")
    parser.add_argument(
        "--start-weekday",
        type=int,
        default=None,
        help="weekday of the first row, Monday=0 (channel mode)",
    )
    parser.add_argument("--steps-per-day", type=int, default=288)
    parser.add_argument("--tod-channel", type=int, default=1)
    parser.add_argument("--dow-channel", type=int, default=2)
    parser.add_argument("--note", default="", help="free-form provenance note")
    parser.add_argument("--out", required=True, help="output .npz path")
    args = parser.parse_args()

    if args.from_data:
        if args.start_weekday is None:
            parser.error("--start-weekday is required with --from-data")
        table, info = build_table(
            Path(args.from_data),
            args.start_weekday,
            args.steps_per_day,
            args.tod_channel,
            args.dow_channel,
        )
        stamps: np.ndarray = table
    else:
        if args.start is None or args.steps is None:
            parser.error("either --from-data or --start/--steps is required")
        stamps = build_uniform(args.start, args.steps, args.interval_minutes)
        info = {
            "mode": "uniform",
            "start": args.start,
            "steps": int(args.steps),
            "interval_minutes": int(args.interval_minutes),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        timestamps=stamps,
        note=np.asarray(args.note),
        mode=np.asarray(str(info["mode"])),
    )
    print(f"wrote {out} rows={stamps.shape[0]} mode={info['mode']}")
    if info["mode"] == "data_channels":
        print(
            "  start_weekday="
            f"{info['start_weekday']}({info['start_weekday_name']}) "
            f"offset={info['day_offset_applied']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
