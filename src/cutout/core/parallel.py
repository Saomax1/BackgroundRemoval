"""Spread full-resolution CPU work over every core.

The GPU runs the network on a 1-2K copy of the image. Everything at the photo's own
resolution (upsampling the matte, the edge refine, colour clean-up, PNG encoding)
runs on the CPU, and most of it is plain per-pixel NumPy, which uses one core.
This module cuts such work into horizontal strips and runs the strips on a shared
thread pool. NumPy, OpenCV and zlib release the GIL for large arrays, so the
strips really do run at the same time.

Only per-pixel (element-wise) steps are split: each output pixel depends on the same
input pixel only, so the result is bit-for-bit what the single-threaded code gives.
Filters that look at neighbours (blurs, resizes) stay whole-image calls; OpenCV
already runs those on all cores.

``CUTOUT_CPU_THREADS`` limits the number of threads (``1`` turns this off).
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

# Below this many pixels a strip is not worth a thread hand-off.
MIN_STRIP_PIXELS = 256 * 1024

_pool: ThreadPoolExecutor | None = None
_side_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()
_in_worker = threading.local()


def cpu_threads() -> int:
    """How many threads full-resolution work uses (all logical cores unless limited)."""
    env = os.environ.get("CUTOUT_CPU_THREADS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            log.warning("ignoring CUTOUT_CPU_THREADS=%r (not a number)", env)
    return max(1, os.cpu_count() or 1)


def _executor() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=cpu_threads(), thread_name_prefix="cutout-cpu")
        return _pool


def strips(height: int, width: int, n: int | None = None, max_pixels: int | None = None) -> list[tuple[int, int]]:
    """Row ranges ``[(y0, y1), ...]`` covering ``0..height`` for ``n`` threads.

    ``max_pixels`` caps the strip size (for steps whose temporary memory grows with it).
    """
    n = n or cpu_threads()
    if height <= 0:
        return []
    pixels = height * max(width, 1)
    by_size = max(1, pixels // MIN_STRIP_PIXELS)
    # A few more strips than threads keeps every core busy when strips finish unevenly.
    count = max(1, min(n * 3, by_size))
    if max_pixels:
        count = max(count, -(-pixels // max_pixels))
    count = min(count, height)
    step = -(-height // count)
    return [(y, min(height, y + step)) for y in range(0, height, step)]


def map_rows(fn: Callable[[int, int], T], height: int, width: int = 1) -> list[T]:
    """Call ``fn(y0, y1)`` for strips covering ``0..height`` in parallel; results in row order.

    ``fn`` must only touch rows ``y0:y1`` of the arrays it writes. Exceptions propagate.
    Called from inside a strip (nested), it runs serially, so it can never deadlock.
    """
    return map_parts(fn, strips(height, width))


def map_parts(fn: Callable[[int, int], T], parts: list[tuple[int, int]]) -> list[T]:
    """:func:`map_rows` over row ranges chosen by the caller (from :func:`strips`)."""
    if len(parts) <= 1 or cpu_threads() <= 1 or getattr(_in_worker, "active", False):
        return [fn(y0, y1) for y0, y1 in parts]

    def run(part: tuple[int, int]) -> T:
        _in_worker.active = True
        try:
            return fn(*part)
        finally:
            _in_worker.active = False

    return list(_executor().map(run, parts))


def submit(fn: Callable[..., T], *args, **kwargs):
    """Run ``fn`` in the background and return its ``Future`` (to overlap CPU work with the GPU).

    These jobs get their own small pool, so a job that itself calls :func:`map_rows`
    never waits on a strip queued behind it.
    """
    global _side_pool
    with _pool_lock:
        if _side_pool is None:
            _side_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cutout-prep")
    return _side_pool.submit(fn, *args, **kwargs)
