"""GPU scheduler — concurrency policy instead of one global lock (plan §2.4).

V2 serialised every pipeline behind a process-wide ``threading.Lock``. That is
safe but leaves the GPU idle whenever a task is doing CPU work (resize,
scratch-mask post-processing, artifact writes).

V3 replaces "fully serial" with an explicit policy: each *capability* declares
how many concurrent executions are safe for its resource profile::

    policy = ConcurrencyPolicy(default=1, per_capability={"scratch_detection": 2})

Semantics:
  * a slot is acquired for the whole stage execution,
  * a stage declares the capabilities it needs; the strictest (smallest) limit
    among them wins,
  * limits are per *process*; multi-worker deployments get one scheduler each,
    which is the correct unit because each worker owns its own GPU context.

Deliberately conservative: the default remains 1, so enabling the scheduler is a
behaviour-preserving change until a policy is configured via
``FIXIMG_CONCURRENCY`` / ``FIXIMG_CONCURRENCY_<CAPABILITY>``.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections.abc import Iterable, Iterator

from fiximg.infrastructure.observability.logging import get_logger

logger = get_logger("fiximg.scheduler")


@dataclass
class ConcurrencyPolicy:
    """How many concurrent executions each capability may run."""

    default: int = 1
    per_capability: dict[str, int] = field(default_factory=dict)

    def limit_for(self, capabilities: Iterable[str]) -> int:
        """Strictest limit across the requested capabilities."""
        caps = list(capabilities)
        if not caps:
            return max(1, self.default)
        limits = [self.per_capability.get(cap, self.default) for cap in caps]
        return max(1, min(limits))

    @classmethod
    def from_env(cls) -> ConcurrencyPolicy:
        """Build a policy from ``settings.concurrency_default`` + per-capability vars.

        The base number comes from `settings`, not from a second read of the
        environment: `config.py` already resolves ``FIXIMG_CONCURRENCY`` (and the
        YAML ``concurrency_default``) with its own validation, so reading the
        variable here as well meant two sources for one knob — the YAML key was
        inert while still looking configured. The ``FIXIMG_CONCURRENCY_*``
        per-capability variables stay a raw read, because they are assembled at
        runtime from a prefix and have no YAML spelling.
        """
        from fiximg.config import settings

        try:
            default = int(getattr(settings, "concurrency_default", 1) or 1)
        except (TypeError, ValueError):
            default = 1

        per_capability: dict[str, int] = {}
        prefix = "FIXIMG_CONCURRENCY_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            capability = key[len(prefix):].strip().lower()
            if not capability:
                continue
            try:
                per_capability[capability] = int(value)
            except ValueError:
                continue
        return cls(default=default, per_capability=per_capability)

    def describe(self) -> dict:
        return {"default": self.default, "per_capability": dict(self.per_capability)}


class GpuScheduler:
    """Per-device, per-capability semaphores guarding execution slots.

    Slots are keyed ``(device, capability)`` so a multi-GPU deployment gets
    independent limits per card (plan §4.3 Step 4) while a single-device one
    behaves exactly as before.
    """

    def __init__(self, policy: ConcurrencyPolicy | None = None) -> None:
        self.policy = policy or ConcurrencyPolicy.from_env()
        self._semaphores: dict[tuple[int, str], threading.Semaphore] = {}
        self._lock = threading.Lock()
        self._active: dict[tuple[int, str], int] = {}
        self._acquired_total = 0

    # ------------------------------------------------------------- internals
    @staticmethod
    def _key(device: int, capability: str) -> tuple[int, str]:
        return int(device), capability

    def _semaphore(self, device: int, capability: str) -> threading.Semaphore:
        key = self._key(device, capability)
        with self._lock:
            sem = self._semaphores.get(key)
            if sem is None:
                limit = max(1, self.policy.per_capability.get(capability, self.policy.default))
                sem = threading.Semaphore(limit)
                self._semaphores[key] = sem
            return sem

    # ------------------------------------------------------------------- api
    @contextmanager
    def slot(self, capabilities: Iterable[str] = (), device: int | None = None) -> Iterator[None]:
        """Hold an execution slot for ``capabilities`` on ``device``.

        Locks are always taken in a deterministic order (device, then sorted
        capability) so a stage spanning several capabilities cannot deadlock.
        """
        resolved = int(device) if device is not None else -1
        caps = sorted(set(capabilities)) or ["__default__"]
        keys = [self._key(resolved, cap) for cap in caps]
        acquired: list[threading.Semaphore] = []
        try:
            for key in keys:
                sem = self._semaphore(*key)
                sem.acquire()
                acquired.append(sem)
                with self._lock:
                    self._active[key] = self._active.get(key, 0) + 1
            with self._lock:
                self._acquired_total += 1
            yield
        finally:
            for key in reversed(keys):
                with self._lock:
                    self._active[key] = max(0, self._active.get(key, 0) - 1)
                self._semaphores[key].release()

    def stats(self) -> dict:
        """Introspection for ``/api/v1/stats`` and the models API."""
        with self._lock:
            return {
                "policy": self.policy.describe(),
                # JSON has no tuple keys: render them as "device:capability".
                "active": {
                    f"{device}:{capability}": count
                    for (device, capability), count in self._active.items()
                },
                "acquired_total": self._acquired_total,
            }

    def reset(self) -> None:
        """Drop all semaphores (tests / config reload)."""
        with self._lock:
            self._semaphores.clear()
            self._active.clear()
            self._acquired_total = 0


#: Process-wide scheduler shared by the sync path and the inline worker.
default_scheduler = GpuScheduler()


__all__ = ["ConcurrencyPolicy", "GpuScheduler", "default_scheduler"]
