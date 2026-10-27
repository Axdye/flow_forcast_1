"""Latency / memory profiling helpers.

Numbers are only meaningful on the target GPU, with warm-up, synchronization,
the same precision and no co-running jobs.  This module never fabricates a GPU
result on CPU; ``device`` is reported in the output.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import torch


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no measurements")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def measure_latency(
    function: Callable[[], object],
    warmup: int = 100,
    iterations: int = 1000,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Return p50/p95/mean latency in milliseconds and peak memory if CUDA."""
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup >= 0 and iterations >= 1 required")
    device = torch.device(device)
    for _ in range(warmup):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        timings: list[float] = []
        for _ in range(iterations):
            start.record()
            function()
            end.record()
            torch.cuda.synchronize(device)
            timings.append(float(start.elapsed_time(end)))
        return {
            "device": str(device),
            "iterations": iterations,
            "warmup": warmup,
            "p50_ms": _percentile(timings, 0.5),
            "p95_ms": _percentile(timings, 0.95),
            "mean_ms": statistics.fmean(timings),
            "std_ms": statistics.pstdev(timings) if len(timings) > 1 else 0.0,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
        }

    timings = []
    for _ in range(iterations):
        begin = time.perf_counter()
        function()
        timings.append((time.perf_counter() - begin) * 1000.0)
    return {
        "device": str(device),
        "iterations": iterations,
        "warmup": warmup,
        "p50_ms": _percentile(timings, 0.5),
        "p95_ms": _percentile(timings, 0.95),
        "mean_ms": statistics.fmean(timings),
        "std_ms": statistics.pstdev(timings) if len(timings) > 1 else 0.0,
        "peak_allocated_mb": None,
        "peak_reserved_mb": None,
        "note": "CPU timing is not a GPU acceptance result",
    }


def profile_callable(
    function: Callable[[int], object],
    batch_sizes: tuple[int, ...] = (1, 64),
    warmup: int = 100,
    iterations: int = 1000,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Profile a callable that receives a batch size."""
    results: dict[str, object] = {"device": str(device), "batch_sizes": {}}
    for batch_size in batch_sizes:
        results["batch_sizes"][str(batch_size)] = measure_latency(
            lambda bs=batch_size: function(bs),
            warmup=warmup,
            iterations=iterations,
            device=device,
        )
    return results


@torch.no_grad()
def profile_model(
    model,
    make_input: Callable[[int], tuple[object, object]],
    batch_sizes: tuple[int, ...] = (1, 64),
    warmup: int = 100,
    iterations: int = 1000,
    device: torch.device | str = "cuda",
) -> dict[str, object]:
    """Profile ``model.predict_top1(*make_input(batch_size))`` on device."""
    model.eval()
    cache: dict[int, tuple[object, object]] = {}

    def run(batch_size: int):
        if batch_size not in cache:
            cache[batch_size] = make_input(batch_size)
        inp, graph = cache[batch_size]
        return model.predict_top1(inp, graph)

    return profile_callable(
        run,
        batch_sizes=batch_sizes,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )


__all__ = ["measure_latency", "profile_callable", "profile_model"]
