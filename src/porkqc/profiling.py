"""Small performance-measurement helpers (runtime, memory, FLOPs)."""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from typing import Callable

import numpy as np


@contextmanager
def timer(store: dict, key: str):
    """Context manager that records elapsed wall-clock seconds in ``store[key]``.

    Example:
        >>> t = {}
        >>> with timer(t, "step"):
        ...     _ = sum(range(1000))
        >>> t["step"] >= 0
        True
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        store[key] = time.perf_counter() - start


def time_function(fn: Callable, n_repeat: int = 20, n_warmup: int = 3) -> dict[str, float]:
    """Measure the latency of ``fn()`` over repeated calls.

    Warm-up calls are discarded because the first calls include one-off costs
    (memory allocation, PyTorch kernel selection) that do not occur on a
    running production line.

    Args:
        fn: Zero-argument callable to time.
        n_repeat: Number of timed calls.
        n_warmup: Number of untimed warm-up calls.

    Returns:
        Dict with mean, median, p95 and std latency in milliseconds and the
        implied throughput in images per second.
    """
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {
        "mean_ms": float(t.mean()),
        "median_ms": float(np.median(t)),
        "p95_ms": float(np.percentile(t, 95)),
        "std_ms": float(t.std()),
        "fps": float(1000 / np.median(t)),
    }


def peak_python_memory(fn: Callable) -> float:
    """Peak memory (MB) allocated by Python objects while running ``fn()``.

    Note: tracemalloc does not see memory allocated inside PyTorch's C++
    backend, so for the detector the model file size and parameter count are
    better memory indicators. It is accurate for the NumPy/scikit-learn baseline.
    """
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


def scale_flops(gflops_at_640: float, imgsz: int) -> float:
    """Estimate convolutional-network GFLOPs at another input size.

    For a fully convolutional network the operation count is proportional to
    the number of input pixels, i.e. to ``imgsz**2``.

    Args:
        gflops_at_640: GFLOPs reported for a 640 x 640 input.
        imgsz: New square input size in pixels.

    Returns:
        Estimated GFLOPs at ``imgsz``.
    """
    return gflops_at_640 * (imgsz / 640) ** 2
