"""Shared benchmark helpers (plan §3.10.3).

Fixed input sizes so runs are comparable across machines and revisions:

    small:  512 px
    medium: 1024 px
    large:  2048 px
    xlarge: 4096 px
"""
from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

#: Canonical benchmark input sizes (plan §3.10.3).
INPUT_SIZES: dict[str, int] = {
    "small": 512,
    "medium": 1024,
    "large": 2048,
    "xlarge": 4096,
}


def make_image(side: int):
    """A deterministic RGB test image of ``side`` × ``side``."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed=side)
    data = rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)
    return Image.fromarray(data, mode="RGB")


@dataclass
class Samples:
    """Collected timings for one measurement."""

    name: str
    values: list[float] = field(default_factory=list)

    def add(self, seconds: float) -> None:
        self.values.append(float(seconds))

    @property
    def count(self) -> int:
        return len(self.values)

    def summary(self) -> dict:
        if not self.values:
            return {"name": self.name, "count": 0}
        ordered = sorted(self.values)
        return {
            "name": self.name,
            "count": len(ordered),
            "min": round(ordered[0], 4),
            "p50": round(_percentile(ordered, 0.50), 4),
            "p95": round(_percentile(ordered, 0.95), 4),
            "p99": round(_percentile(ordered, 0.99), 4),
            "max": round(ordered[-1], 4),
            "mean": round(statistics.fmean(ordered), 4),
        }


def _percentile(ordered: list[float], pct: float) -> float:
    """Nearest-rank percentile — delegated so there is exactly one such rule.

    `/metrics` (`observability.metrics.nearest_rank`), the benchmark and the
    repository's SQL (`task_repository._ceil_rank`) must agree, or a "p95" means
    three different things in three places.

    The formula this replaced, ``int(n * pct) - 1``, under-reported on exactly the
    small samples CI runs (--runs 2, --tasks 10): with n=2 it returned
    ``ordered[0]``, so the table printed a p95 *lower* than the p50.
    """
    from fiximg.infrastructure.observability.metrics import nearest_rank

    value = nearest_rank(ordered, pct)
    assert value is not None, "summary() only measures a non-empty sample"
    return value


@contextmanager
def temp_database():
    """Point the application at a throwaway SQLite file for the run."""
    from fiximg.config import settings
    from fiximg.infrastructure.db import engine

    with tempfile.TemporaryDirectory(prefix="fiximg_bench_") as tmp:
        db_path = os.path.join(tmp, "bench.db")
        original = {
            "db_path": settings.db_path,
            "engine_db": engine.DB_PATH,
            "engine_data": engine.ADMIN_DATA_DIR,
            "engine_conn": engine._conn,
        }
        os.makedirs(tmp, exist_ok=True)
        settings.db_path = db_path
        engine.DB_PATH = db_path
        engine.ADMIN_DATA_DIR = tmp
        engine._conn = None

        from fiximg.infrastructure.db.repositories import task_repository

        task_repository._DDL_DONE = False
        try:
            yield tmp
        finally:
            if engine._conn is not None:
                try:
                    engine._conn.close()
                except Exception:  # noqa: BLE001
                    pass
            settings.db_path = original["db_path"]
            engine.DB_PATH = original["engine_db"]
            engine.ADMIN_DATA_DIR = original["engine_data"]
            engine._conn = original["engine_conn"]
            task_repository._DDL_DONE = False


def synthetic_stage(delay_seconds: float = 0.0):
    """A stage that sleeps instead of running a model (platform benchmarking)."""
    from fiximg.inference.context import StageResult
    from fiximg.inference.stages.base import BaseStage

    class _SyntheticStage(BaseStage):
        name = "synthetic"
        version = "1.0"
        capabilities = frozenset({"synthetic"})

        def run(self, image, context):
            if delay_seconds:
                time.sleep(delay_seconds)
            return StageResult(image=image, metadata={"synthetic": True}, duration=delay_seconds)

    return _SyntheticStage()


def print_table(rows: list[dict], columns: list[str]) -> None:
    """Minimal fixed-width table printer (avoids a tabulate dependency)."""
    if not rows:
        print("(no samples)")
        return
    widths = {
        col: max(len(col), *(len(str(row.get(col, ""))) for row in rows)) for col in columns
    }
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))


def peak_rss_mb() -> float | None:
    """Peak resident set size of this process, in MB (best effort).

    Tries, in order: ``psutil`` (when installed), the POSIX ``resource`` module,
    then the Win32 ``K32GetProcessMemoryInfo`` call. Returns None when none of
    them is available, so a benchmark run degrades instead of failing.
    """
    try:
        import psutil  # noqa: PLC0415 — optional dependency

        return round(psutil.Process().memory_info().peak_wset / (1 << 20), 1)
    except Exception:  # noqa: BLE001 — psutil missing or no peak_wset on POSIX
        pass

    try:
        import resource  # noqa: PLC0415 — POSIX only

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:  # noqa: BLE001 — Windows has no `resource`
        pass

    try:
        import ctypes
        import ctypes.wintypes as wintypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # Explicit argtypes/restype matter: without them ctypes truncates the
        # HANDLE and the call silently fails with a zeroed structure.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_info = kernel32.K32GetProcessMemoryInfo
        get_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_info.restype = wintypes.BOOL
        get_current = kernel32.GetCurrentProcess
        get_current.restype = wintypes.HANDLE

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if get_info(get_current(), ctypes.byref(counters), counters.cb):
            return round(counters.PeakWorkingSetSize / (1 << 20), 1)
    except Exception:  # noqa: BLE001 — not Windows or the API is unavailable
        pass
    return None


def gpu_peak_mb() -> float | None:
    """Peak CUDA memory allocated in this process, in MB (None without CUDA)."""
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1 << 20), 1)
    except Exception:  # noqa: BLE001 — torch is optional for benchmarks
        pass
    return None


def ensure_importable() -> None:
    """Make ``src/`` importable when the package is not installed."""
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


__all__ = [
    "INPUT_SIZES",
    "Samples",
    "ensure_importable",
    "gpu_peak_mb",
    "make_image",
    "peak_rss_mb",
    "print_table",
    "synthetic_stage",
    "temp_database",
]
