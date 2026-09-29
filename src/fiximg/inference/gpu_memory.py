"""GPU memory probing, OOM detection and memory-aware device selection.

Plan §7.1 lists "GPU peak memory — 已有基础统计 → 加入每 task/stage 维度" as a KPI,
and the multi-GPU item (§4.3 Step 4) notes that capability routing alone is not
enough: without memory awareness a scheduler happily sends the next task to the
card that is already full.

This module provides the three pieces:

* :func:`device_memory` / :func:`all_devices` — free/total/allocated per device,
* :func:`select_device` — pick the candidate with the most headroom,
* :func:`is_oom_error` — recognise a CUDA out-of-memory failure so the runtime can
  retry on a different card instead of failing the task,
* :class:`MemorySampler` — measure the peak *delta* a stage caused.

Everything degrades cleanly without CUDA: probes return None, selection falls
back to the first candidate, and the sampler reports no peak. That keeps a
CPU-only deployment on exactly its previous behaviour.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from fiximg.infrastructure.observability.logging import get_logger, log_event

logger = get_logger("fiximg.gpu.memory")

#: Substrings that identify a CUDA out-of-memory failure across torch versions
#: and driver messages. Matching on text is the only portable option: torch
#: raises ``RuntimeError`` (and ``torch.cuda.OutOfMemoryError``, a subclass that
#: is not always importable).
_OOM_MARKERS = (
    "out of memory",
    "cuda_error_out_of_memory",
    "cuda error: out of memory",
    "hip out of memory",
    "cublas_status_alloc_failed",
)

#: Leave this much headroom when judging whether a device can take a task.
DEFAULT_HEADROOM_MB = 256


@dataclass(frozen=True, slots=True)
class DeviceMemory:
    """A snapshot of one device's memory, in MiB."""

    device: int
    total_mb: float
    free_mb: float
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0

    @property
    def used_mb(self) -> float:
        return max(0.0, self.total_mb - self.free_mb)

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "total_mb": round(self.total_mb, 1),
            "free_mb": round(self.free_mb, 1),
            "used_mb": round(self.used_mb, 1),
            "allocated_mb": round(self.allocated_mb, 1),
            "reserved_mb": round(self.reserved_mb, 1),
        }


def _torch():
    try:
        import torch  # noqa: PLC0415 — optional at import time

        return torch
    except Exception:  # noqa: BLE001 — CPU-only deployments may lack torch
        return None


def device_memory(device: int) -> DeviceMemory | None:
    """Memory snapshot for a CUDA device, or None when unavailable.

    ``device < 0`` means CPU, for which there is nothing to report.
    """
    torch = _torch()
    if torch is None or int(device) < 0 or not torch.cuda.is_available():
        return None
    try:
        index = int(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        stats = torch.cuda.memory_stats(index)
        return DeviceMemory(
            device=index,
            total_mb=total_bytes / (1024 * 1024),
            free_mb=free_bytes / (1024 * 1024),
            allocated_mb=torch.cuda.memory_allocated(index) / (1024 * 1024),
            reserved_mb=stats.get("reserved_bytes.all.current", 0) / (1024 * 1024),
        )
    except Exception as exc:  # noqa: BLE001 — a probe must never raise
        log_event(logger, "WARNING", "gpu memory probe failed",
                  device=device, error=str(exc))
        return None


def all_devices() -> list[DeviceMemory]:
    """Snapshot every visible CUDA device (empty list without CUDA)."""
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return []
    snapshots = []
    for index in range(torch.cuda.device_count()):
        snapshot = device_memory(index)
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots


def is_oom_error(exc: BaseException) -> bool:
    """True when ``exc`` is a CUDA/ROCm out-of-memory failure."""
    if exc is None:
        return False
    torch = _torch()
    if torch is not None:
        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _OOM_MARKERS)


def select_device(
    candidates: Sequence[int],
    *,
    required_mb: float = 0.0,
    headroom_mb: float = DEFAULT_HEADROOM_MB,
    probe: Callable[[int], DeviceMemory | None] | None = None,
) -> int:
    """Pick the candidate device with the most free memory.

    Falls back to the first candidate when no probe data is available (CPU-only,
    or a probe failure), so an unconfigured deployment is unaffected. When every
    device is short of ``required_mb`` the roomiest one is still returned: the
    caller may know better than a heuristic, and a hard refusal here would turn a
    transient memory spike into a hard failure.
    """
    options = [int(d) for d in candidates]
    if not options:
        return -1
    probe = probe or device_memory

    snapshots = [(device, probe(device)) for device in options]
    measured = [(device, snap) for device, snap in snapshots if snap is not None]
    if not measured:
        return options[0]

    wanted = float(required_mb) + float(headroom_mb)
    fitting = [(snap.free_mb, device) for device, snap in measured if snap.free_mb >= wanted]
    if fitting:
        return max(fitting)[1]

    # Nothing fits: hand back the roomiest device and say so.
    roomiest = max((snap.free_mb, device) for device, snap in measured)[1]
    log_event(
        logger, "WARNING", "no device has enough free memory; using the roomiest",
        required_mb=round(wanted, 1), chosen=roomiest,
        free_mb={device: round(snap.free_mb, 1) for device, snap in measured},
    )
    return roomiest


_lost_reported: set[str] = set()


def reset_lost_measurement_report() -> None:
    """Forget which measurement failures have already been logged (tests)."""
    _lost_reported.clear()


def _report_lost_measurement(stage_of_call: str, exc: Exception) -> None:
    """Say once, per kind of failure, that a VRAM figure is about to be missing.

    Sampling is diagnostic and must never fail a run, but "no number" and "a number
    that happens to be small" used to look the same from the outside. One WARNING per
    process keeps the log readable and makes the gap visible to whoever is looking at
    the memory figures.
    """
    from fiximg.infrastructure.observability.logging import get_logger, log_event

    key = f"{stage_of_call}:{type(exc).__name__}"
    if key in _lost_reported:
        return
    _lost_reported.add(key)
    log_event(
        get_logger("fiximg.gpu.memory"), "WARNING",
        "GPU memory measurement unavailable; stage metrics will omit it",
        failed_call=stage_of_call, error=str(exc)[:200],
    )


class MemorySampler:
    """Context manager measuring the peak CUDA allocation a block caused.

    ``peak_mb`` is the *delta* against the level at entry, which is what makes
    the number attributable to a single stage rather than to the whole process.
    Reset-and-measure is used rather than a difference of high-water marks, so a
    peak reached earlier in the run cannot inflate a later stage's figure.

    The reset is why this class also maintains :data:`_process_peak_mb`: PyTorch's
    peak counter is per-device and *since the last reset*, so a reader that calls
    ``max_memory_allocated()`` outside a sampler window — the stats endpoint, the
    models health view — gets "peak since the most recent stage started", which is
    smaller than a stage peak the same process just published. Folding the value
    into a process-wide maximum before and after each window keeps the reset local
    to the stage while the aggregate stays an upper bound on every number in it.
    """

    def __init__(self, device: int) -> None:
        self.device = int(device)
        self.peak_mb: float | None = None
        self._measuring = False
        self._lock = threading.Lock()

    def __enter__(self) -> MemorySampler:
        torch = _torch()
        self._torch = torch
        if torch is None or self.device < 0 or not torch.cuda.is_available():
            return self
        self._measuring = True
        # Anything the previous window left behind is lost the moment we reset.
        fold_process_peak(self.device, _peak_allocated_mb(torch, self.device))
        try:
            # The peak-stats calls need a CUDA context to exist. In a fresh process
            # the first `reset_peak_memory_stats()` raises "Invalid device argument",
            # and the first window is the one that loads the model — so the heaviest
            # stage of every run was the one reporting no memory at all, quietly.
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(self.device)
        except Exception as exc:  # noqa: BLE001 — sampling is diagnostic only
            self._measuring = False
            _report_lost_measurement("reset", exc)
        return self

    def __exit__(self, *_exc_info) -> None:
        # Only report a peak for a block that actually started measuring. Without
        # the flag a CPU-only host still answers `max_memory_allocated(0)` — with
        # 0.0 — so a stage on a device that does not exist would publish a
        # convincing "used no VRAM" reading instead of "not measured".
        torch = getattr(self, "_torch", None)
        if torch is None or not self._measuring or self.device < 0:
            return
        try:
            self.peak_mb = round(
                torch.cuda.max_memory_allocated(self.device) / (1024 * 1024), 1
            )
        except Exception as exc:  # noqa: BLE001
            self.peak_mb = None
            _report_lost_measurement("read", exc)
        fold_process_peak(self.device, self.peak_mb)

    def metrics(self) -> dict:
        """Stage metrics contribution (empty when nothing was measured)."""
        if self.peak_mb is None:
            return {}
        return {STAGE_PEAK_METRIC: self.peak_mb, "gpu_device": self.device}


#: Name the stage/task metric rows carry the sampler's number under. The stats
#: endpoint reads the persisted rows back through this same name, so the two ends
#: cannot disagree about which metric is the VRAM peak.
STAGE_PEAK_METRIC = "gpu_peak_mb"

#: High-water mark per device, in MiB, that no `reset_peak_memory_stats()` clears.
_process_peak_mb: dict[int, float] = {}
_process_peak_lock = threading.Lock()


def _peak_allocated_mb(torch, device: int) -> float | None:
    """PyTorch's own per-device peak, in MiB, or None if it cannot be read."""
    try:
        return round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1)
    except Exception:  # noqa: BLE001 — diagnostic only
        return None


def fold_process_peak(device: int, peak_mb: float | None) -> None:
    """Raise the remembered process peak for ``device`` to ``peak_mb``."""
    if device is None or device < 0 or peak_mb is None:
        return
    with _process_peak_lock:
        if peak_mb > _process_peak_mb.get(int(device), 0.0):
            _process_peak_mb[int(device)] = float(peak_mb)


def process_peak_mb(device: int) -> float | None:
    """The largest allocation this process has ever attributed to ``device``.

    Reads the folded history *and* the live counter: work that runs outside any
    :class:`MemorySampler` window (a warm-up, a model loaded by the CLI) only
    reaches PyTorch's own peak, and a reset would erase it.
    """
    if device < 0:
        return None
    with _process_peak_lock:
        folded = _process_peak_mb.get(int(device))
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        live = _peak_allocated_mb(torch, device)
        if live is not None:
            fold_process_peak(device, live)
            folded = max(folded or 0.0, live)
    return folded


def peak_memory_report() -> dict:
    """Process-wide peak per device, for the models API / stats endpoint."""
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return {}
    report: dict[str, float] = {}
    for index in range(torch.cuda.device_count()):
        peak = process_peak_mb(index)
        if peak is not None:
            report[str(index)] = peak
    return report


def current_device_peak_mb() -> float | None:
    """The process peak for the device this process is working on right now.

    Resolved through :func:`peak_memory_report` rather than by reading PyTorch's
    counter directly, so the stats endpoint and the per-stage metrics answer from
    one measurement and one seam — a CPU-only host gets None here and None there.
    """
    report = peak_memory_report()
    if not report:
        return None
    torch = _torch()
    try:
        return report.get(str(torch.cuda.current_device()))
    except Exception:  # noqa: BLE001 — device vanished between the two calls
        return max(report.values())


def memory_summary(devices: Iterable[int] | None = None) -> dict:
    """JSON-friendly memory view for ``/health/ready`` and ``/stats``."""
    if devices is None:
        snapshots = all_devices()
    else:
        snapshots = [s for s in (device_memory(d) for d in devices) if s is not None]
    return {
        "devices": [snap.to_dict() for snap in snapshots],
        "peak_allocated_mb": peak_memory_report(),
    }


__all__ = [
    "DEFAULT_HEADROOM_MB",
    "reset_lost_measurement_report",
    "DeviceMemory",
    "MemorySampler",
    "STAGE_PEAK_METRIC",
    "all_devices",
    "current_device_peak_mb",
    "device_memory",
    "fold_process_peak",
    "is_oom_error",
    "memory_summary",
    "peak_memory_report",
    "process_peak_mb",
    "select_device",
]
