"""Hardware GPU utilisation — what the card was *doing*, not what it held (plan §7.1).

:mod:`fiximg.inference.gpu_memory` answers "how much VRAM did this stage touch".
This answers "how hard was the device working while the stage ran", which is the
KPI §7.1 asks to be measured and the number §2.4's concurrency policy exists to
improve — a scheduler can hold a full set of slots while the GPU still spends
most of the window idle, and only a utilisation reading shows that.

PyTorch exposes no utilisation API, so the reading comes from the driver's own
query tool. That makes this optional instrumentation rather than a dependency:

* one background thread per process samples on an interval and keeps a bounded
  ring of timestamped readings, so attributing a stage window costs nothing at
  query time and a deployment never forks a tool per HTTP request. It samples
  only while a stage window is open or a reader has just asked: a driver query
  is a subprocess, and an idle worker would otherwise pay for one forever;
* a machine without the tool is reported as ``unavailable`` and the thread stops
  after a few failures. Nothing here fabricates a reading: a CPU-only host
  pretending 0 % utilisation would read as "the scheduler leaves the GPU idle",
  which is the opposite of the truth the series exists to show.

The stage metric and the metrics series are the same measurement — the window
aggregate — so `/health/ready`, the task report and `/metrics` cannot disagree
about what "utilisation" means.
"""
from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from fiximg.infrastructure.observability.logging import get_logger

logger = get_logger("fiximg.gpu")

#: Fields asked of the driver, in the order :func:`parse_query_output` expects.
QUERY_FIELDS = ("index", "utilization.gpu", "memory.used", "memory.total")
QUERY_CMD = [
    "nvidia-smi",
    f"--query-gpu={','.join(QUERY_FIELDS)}",
    "--format=csv,noheader,nounits",
]

DEFAULT_INTERVAL_SECONDS = 0.5
DEFAULT_TIMEOUT_SECONDS = 2.0
#: Readings kept per device-ish; at the default interval this is ~60 s of history.
MAX_SAMPLES = 128
#: Consecutive failed queries before the probe gives up for the process.
MAX_CONSECUTIVE_FAILURES = 3

#: Name the stage metric rows carry the window mean under. `max_metric` and the
#: task report read it back through this same name (the `gpu_peak_mb` pattern).
STAGE_UTILIZATION_METRIC = "gpu_util_pct"

#: Statuses :class:`UtilizationProbe` reports. `starting` means the thread is up
#: but no query has returned a row yet; `idle` means nobody asked for a reading.
STATUS_IDLE = "idle"
STATUS_STARTING = "starting"
STATUS_MEASURING = "measuring"
STATUS_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DeviceReading:
    """One instantaneous per-device reading from the driver."""

    device: int
    utilization_pct: int
    memory_used_mb: int
    memory_total_mb: int

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "utilization_pct": self.utilization_pct,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
        }


def parse_query_output(text: str) -> list[DeviceReading]:
    """Parse `nvidia-smi --format=csv,noheader,nounits` output.

    An unusable row is skipped, not fatal: MIG and vGPU rows answer `[N/A]` for
    utilisation, and one such row must not cost the readings for the healthy
    cards. A row that does not have exactly the queried field count is not a
    reading at all (a driver banner, an empty line), so it is dropped too.
    """
    rows: list[DeviceReading] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(QUERY_FIELDS):
            continue
        try:
            device = int(parts[0])
            pct = int(parts[1])
            used = int(parts[2])
            total = int(parts[3])
        except ValueError:
            continue
        rows.append(DeviceReading(device, pct, used, total))
    return rows


def run_query(
    cmd: Sequence[str] = QUERY_CMD, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> str:
    """One driver query. Propagates the tool's absence; the probe gives it meaning."""
    return subprocess.run(
        list(cmd), capture_output=True, text=True, timeout=timeout, check=False
    ).stdout


class UtilizationProbe:
    """Background sampler holding timestamped readings, aggregated per window.

    Injectable ``runner``/``clock`` keep the maths testable without a GPU and
    without wall-clock sleeps: the runner is whatever produces the driver's text.
    """

    def __init__(
        self,
        runner: Callable[[], str] | None = None,
        interval: float = DEFAULT_INTERVAL_SECONDS,
        clock: Callable[[], float] | None = None,
        max_failures: int = MAX_CONSECUTIVE_FAILURES,
        max_samples: int = MAX_SAMPLES,
    ) -> None:
        self._runner = runner or run_query
        self.interval = float(interval)
        self._clock = clock or time.monotonic
        self._max_failures = int(max_failures)
        self._samples: deque[tuple[float, DeviceReading]] = deque(maxlen=max_samples)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._failures = 0
        #: Open stage windows. The thread only forks the tool while one is open (or
        #: a reader asked), so an idle deployment pays nothing for the metric.
        self._windows = 0
        self._demand_until: float | None = None
        self.status = STATUS_IDLE

    # ---------------------------------------------------------------- lifecycle
    @property
    def thread_alive(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def ensure_running(self) -> None:
        """Start sampling once per process. Cheap to call from every read path.

        A probe that gave up stays given up: restarting it would fork a tool that
        is not there on every stage of every task.
        """
        with self._lock:
            if self._thread is not None or self.status == STATUS_UNAVAILABLE:
                return
            self._stop.clear()
            if self.status == STATUS_IDLE:
                self.status = STATUS_STARTING
            thread = threading.Thread(
                target=self._loop, name="gpu-utilization", daemon=True
            )
            self._thread = thread
        thread.start()

    def stop(self, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        """Stop the sampler and forget the readings (shutdown / tests)."""
        self._stop.set()
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            self._samples.clear()
            self._windows = 0
            self._demand_until = None
            self.status = STATUS_IDLE

    def _loop(self) -> None:
        thread = threading.current_thread()
        try:
            while not self._stop.is_set():
                if self._should_poll():
                    self.poll_once()
                self._stop.wait(self.interval)
        finally:
            with self._lock:
                if self._thread is thread:
                    self._thread = None

    def _should_poll(self) -> bool:
        """Poll while a stage window is open, or while a reader is asking."""
        with self._lock:
            if self._windows:
                return True
            if self._demand_until is not None and self._clock() < self._demand_until:
                return True
            self._demand_until = None
            return False

    def open_window(self) -> None:
        """Declare that a measurement interval is starting on this device."""
        with self._lock:
            self._windows += 1
            self._demand_until = self._clock() + 2 * self.interval

    def close_window(self) -> None:
        with self._lock:
            self._windows = max(0, self._windows - 1)

    def request_reading(self) -> None:
        """Keep polling for two intervals because somebody is looking."""
        with self._lock:
            self._demand_until = self._clock() + 2 * self.interval

    # --------------------------------------------------------------------- read
    def now(self) -> float:
        """The probe's own clock, so a caller's window cannot use a different one."""
        return self._clock()

    def _record_failure(self, reason: str) -> None:
        """Count one failed query; give up for the process after a few."""
        with self._lock:
            self._failures += 1
            failures = self._failures
            exhausted = failures >= self._max_failures
            first_report = failures == self._max_failures
            if exhausted:
                self.status = STATUS_UNAVAILABLE
                self._stop.set()
        if first_report:
            # Say so once, and stop asking: restarting would fork a tool that is
            # not installed at the start of every stage of every task.
            logger.warning(
                "gpu utilisation unavailable after %d query failure(s): %s. "
                "Utilisation stays unreported rather than reading 0.",
                failures, reason,
            )

    def poll_once(self) -> list[DeviceReading]:
        """Take one reading now and file it. Never raises; records the failure.

        A query that succeeds but yields no row is a failure too: that is what
        "the tool is installed, this machine has no device to report" looks like,
        and an empty answer repeated forever would fork a subprocess every tick.
        """
        try:
            rows = parse_query_output(self._runner())
        except Exception as exc:  # noqa: BLE001 — a probe must not break a stage
            self._record_failure(type(exc).__name__)
            return []
        if not rows:
            self._record_failure("no device rows")
            return []
        stamp = self._clock()
        with self._lock:
            self._failures = 0
            for row in rows:
                self._samples.append((stamp, row))
            self.status = STATUS_MEASURING
        return rows

    def window(self, started_at: float, ended_at: float, device: int) -> dict:
        """Aggregate the readings inside an execution window, in the clock's units.

        Empty when nothing falls inside it — including for a CPU device, which has
        no driver to ask. Callers must treat that as "not measured", not as zero.
        """
        if device is None or device < 0:
            return {}
        with self._lock:
            seen = [
                row.utilization_pct
                for stamp, row in self._samples
                if row.device == int(device) and started_at <= stamp <= ended_at
            ]
        if not seen:
            return {}
        return {
            STAGE_UTILIZATION_METRIC: round(sum(seen) / len(seen), 1),
            "gpu_util_max_pct": max(seen),
            "gpu_samples": len(seen),
            "gpu_device": int(device),
        }

    def describe(self, devices: Iterable[int] | None = None) -> dict:
        """JSON view for the readiness probe: status, freshness, latest readings.

        Asking is itself a reason to keep sampling — otherwise the thread idles
        until a stage opens a window — so scraping an idle deployment still gets
        readings. ``age_seconds`` names how old the reported reading is, so "no
        refresh yet" cannot be mistaken for "the card is idle right now".
        """
        self.request_reading()
        wanted = None if devices is None else {int(d) for d in devices if d >= 0}
        with self._lock:
            if wanted is not None and not wanted:
                return {
                    "status": self.status,
                    "interval_seconds": self.interval,
                    "devices": [],
                }
            picked: dict[int, tuple[float, DeviceReading]] = {}
            for stamp, row in reversed(self._samples):
                if wanted is not None and row.device not in wanted:
                    continue
                picked.setdefault(row.device, (stamp, row))
            rows = [picked[device] for device in sorted(picked)]
            newest = max((stamp for stamp, _ in rows), default=None)
            age = round(self._clock() - newest, 1) if newest is not None else None
        return {
            "status": self.status,
            "interval_seconds": self.interval,
            "age_seconds": age,
            "devices": [row.to_dict() for _stamp, row in rows],
            "busy_devices": sum(1 for _stamp, row in rows if row.utilization_pct > 0),
        }


#: Process-wide probe: one sampler thread, shared by the readiness endpoint and
#: by every stage window the runtime opens.
default_probe = UtilizationProbe()


def start_probe() -> UtilizationProbe:
    """Ensure the shared probe is sampling and hand it back."""
    default_probe.ensure_running()
    return default_probe


def describe(devices: Iterable[int] | None = None) -> dict:
    """Report the shared probe, starting it if nobody has asked yet.

    The readiness endpoint's entry point: it warms the sampler instead of forking
    the tool per request, so the first scrape answers `starting` and the ones after
    it answer with readings.
    """
    return start_probe().describe(devices)


def stage_window(started_at: float, ended_at: float, device: int) -> dict:
    """Aggregate the shared probe for one stage window (plan §7.1's per-stage KPI)."""
    return default_probe.window(started_at, ended_at, device)


__all__ = [
    "DeviceReading",
    "QUERY_CMD",
    "QUERY_FIELDS",
    "STAGE_UTILIZATION_METRIC",
    "UtilizationProbe",
    "default_probe",
    "describe",
    "parse_query_output",
    "run_query",
    "stage_window",
    "start_probe",
]
