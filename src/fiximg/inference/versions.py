"""Versioned model registry — hot swap without downtime (plan §3.15).

V2's ``ModelManager`` held one handle per model, so "reload" had to unload
first: a stop-then-start that fails the model for the duration of the load and
leaves nothing to fall back on if the new weights are broken.

V3 keeps **two versions resident** and switches *routing* atomically:

    v1 READY
      ↓  build v2
      ↓  validate   (sha256 against models/manifest.yaml, when declared)
      ↓  warmup     (one dummy call removes first-request latency)
      ↓  health     (probe must report healthy)
      ↓  SWITCH     (single assignment under the lock — atomic)
      ↓  drain v1   (unloaded once the last in-flight call releases it)
      ↓  unload v1

    failure at any step → v2 discarded, v1 stays active (rollback is implicit
    because the switch never happened)

In-flight calls are tracked with a reference count, so a long-running inference
on v1 is never interrupted mid-flight: it finishes on the handle it started
with, and the old version is unloaded only after the last lease is released.

The registry never overwrites a ``.pth`` file while the service is running; a
new version is a *separate* resident handle loaded from a separate path.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from fiximg.domain.enums import ModelStatus
from fiximg.domain.errors import ModelUnavailableError
from fiximg.infrastructure.observability.logging import get_logger, log_event

logger = get_logger("fiximg.models.versions")


@dataclass
class Resident:
    """One loaded version of one model."""

    name: str
    version: str
    handle: Any
    weight_uri: str | None = None
    status: str = ModelStatus.READY
    loaded_at: float = field(default_factory=time.time)
    load_ms: int = 0
    #: number of in-flight inferences currently using this handle
    refcount: int = 0
    #: True once routing moved away from it (kept alive until refcount == 0)
    retiring: bool = False
    failure: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status,
            "weight_uri": self.weight_uri,
            "load_ms": self.load_ms,
            "refcount": self.refcount,
            "retiring": self.retiring,
            "failure": self.failure,
        }


class _Lease:
    """Context manager pinning a version for the duration of one inference."""

    def __init__(self, registry: VersionedModelRegistry, resident: Resident) -> None:
        self._registry = registry
        self.resident = resident

    @property
    def handle(self):
        return self.resident.handle

    @property
    def version(self) -> str:
        return self.resident.version

    def __enter__(self) -> _Lease:
        return self

    def __exit__(self, *_exc) -> None:
        self._registry.release(self.resident)


class VersionedModelRegistry:
    """Holds resident model versions and switches the active one atomically.

    Two residency policies, both covered by the plan:

    * ``keep_previous=True`` (default) — the version that was active before the
      last switch stays resident as the **rollback target** (the "双版本驻留"
      model). It is evicted only when a third version arrives, so
      :meth:`rollback` is an atomic switch rather than a reload.
    * ``keep_previous=False`` — the previous version is evicted as soon as its
      in-flight leases drain, trading rollback speed for GPU memory.

    Either way at most two versions are resident at a time.
    """

    def __init__(self, keep_previous: bool = True) -> None:
        self.keep_previous = keep_previous
        self._builders: dict[str, Callable[[str], Any]] = {}
        self._resident: dict[str, dict[str, Resident]] = {}
        self._active: dict[str, str] = {}
        self._previous: dict[str, str] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------- registration
    def register_builder(self, name: str, builder: Callable[[str], Any]) -> None:
        """Register a factory ``builder(version) -> handle`` for a model."""
        self._builders[name] = builder

    def has_builder(self, name: str) -> bool:
        return name in self._builders

    def known_models(self) -> list[str]:
        return sorted(set(self._builders) | set(self._resident))

    # ------------------------------------------------------------------ state
    def current_version(self, name: str) -> str | None:
        with self._lock:
            return self._active.get(name)

    def previous_version(self, name: str) -> str | None:
        with self._lock:
            return self._previous.get(name)

    def is_resident(self, name: str, version: str) -> bool:
        with self._lock:
            return version in self._resident.get(name, {})

    def residents(self, name: str) -> list[dict]:
        with self._lock:
            return [r.to_dict() for r in self._resident.get(name, {}).values()]

    def snapshot(self) -> dict:
        """Full registry state for ``GET /api/v1/models``."""
        with self._lock:
            return {
                "active": dict(self._active),
                "previous": dict(self._previous),
                "resident": {
                    name: [r.to_dict() for r in versions.values()]
                    for name, versions in self._resident.items()
                },
            }

    def current_handle(self, name: str):
        """The active handle, or None when nothing is loaded."""
        with self._lock:
            version = self._active.get(name)
            if version is None:
                return None
            resident = self._resident.get(name, {}).get(version)
            return resident.handle if resident else None

    # ------------------------------------------------------------------- lease
    @contextmanager
    def acquire(self, name: str) -> Iterator[_Lease]:
        """Pin the active version for one inference (never interrupts it).

        Raises :class:`ModelUnavailableError` when the model is not loaded; the
        caller decides whether to load it (lazy strategy) or fail.
        """
        with self._lock:
            version = self._active.get(name)
            resident = self._resident.get(name, {}).get(version) if version else None
            if resident is None:
                raise ModelUnavailableError(
                    f"Model not loaded: {name}", details={"model": name}
                )
            resident.refcount += 1
        lease = _Lease(self, resident)
        try:
            yield lease
        finally:
            self.release(resident)

    def release(self, resident: Resident) -> None:
        """Drop one lease and unload the version if it was retiring."""
        should_unload = False
        with self._lock:
            resident.refcount = max(0, resident.refcount - 1)
            if resident.retiring and resident.refcount == 0:
                should_unload = True
        if should_unload:
            self._unload_resident(resident)

    # ------------------------------------------------------------------ switch
    def activate(
        self,
        name: str,
        version: str,
        *,
        weight_uri: str | None = None,
        validator: Callable[[Resident], None] | None = None,
        warmer: Callable[[Resident], None] | None = None,
        health: Callable[[Resident], bool] | None = None,
        build: bool = True,
    ) -> dict:
        """Bring ``version`` live, keeping the current one until the switch.

        Returns a report dict; ``switched`` is False when the new version failed
        validation/warmup/health and the old one stayed active.
        """
        started = time.perf_counter()
        if not self.has_builder(name):
            raise ModelUnavailableError(
                f"No builder registered for model: {name}", details={"model": name}
            )

        with self._lock:
            existing = self._resident.get(name, {}).get(version)
            if existing is not None and not existing.retiring:
                # Already the active version: idempotent no-op.
                if self._active.get(name) == version:
                    return {
                        "name": name, "version": version, "switched": False,
                        "reason": "already active", "residents": self.residents(name),
                    }
                return self._switch(name, existing, started)

        # ---- build / validate / warmup / health, all OUTSIDE the lock so the
        # currently-active version keeps serving traffic during the whole flow.
        try:
            resident = existing or self._build(name, version, weight_uri)
            if validator is not None:
                validator(resident)
            if warmer is not None:
                warmer(resident)
            if health is not None and not health(resident):
                raise ModelUnavailableError(
                    f"Health probe failed for {name} {version}",
                    details={"model": name, "version": version},
                )
        except Exception as exc:  # noqa: BLE001 — a failed candidate must not
            # take down the running version; report and keep v1.
            log_event(
                logger, "ERROR", "model activation failed; keeping current version",
                model=name, version=version, error=str(exc),
            )
            with self._lock:
                failed = self._resident.get(name, {}).pop(version, None)
                if failed is not None:
                    failed.status = ModelStatus.UNHEALTHY
                    failed.failure = str(exc)
                    self._dispose(failed)
            # The failure belongs in the catalog too: a candidate that never came
            # up is the fact an operator needs from another node's API call.
            self._record_audit(
                name, version, status=ModelStatus.UNHEALTHY, error=str(exc)[:500]
            )
            return {
                "name": name,
                "version": version,
                "switched": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "active_version": self.current_version(name),
                "residents": self.residents(name),
            }

        with self._lock:
            return self._switch(name, resident, started)

    def _build(self, name: str, version: str, weight_uri: str | None) -> Resident:
        builder = self._builders.get(name)
        if builder is None:
            raise ModelUnavailableError(
                f"No builder registered for model: {name}", details={"model": name}
            )
        log_event(logger, "INFO", "loading model version", model=name, version=version)
        t0 = time.perf_counter()
        handle = builder(version)
        load_ms = int((time.perf_counter() - t0) * 1000)
        resident = Resident(
            name=name, version=version, handle=handle, weight_uri=weight_uri,
            status=ModelStatus.READY, load_ms=load_ms,
        )
        with self._lock:
            self._resident.setdefault(name, {})[version] = resident
        log_event(logger, "INFO", "model version loaded", model=name,
                  version=version, load_ms=load_ms)
        # §6: the load is an observation worth keeping, in the process that made it.
        self._record_audit(
            name, version, status=ModelStatus.READY, load_ms=load_ms,
            weight_uri=resident.weight_uri,
        )
        return resident

    def _record_audit(
        self,
        name: str,
        version: str,
        *,
        status: str,
        load_ms: int | None = None,
        error: str | None = None,
        weight_uri: str | None = None,
    ) -> None:
        """Upsert this version's row in `model_versions` (plan §6's ModelVersion).

        Only loads are recorded, never unloads: the row describes the deployment,
        and one process dropping a version says nothing about whether another still
        serves it. The manifest supplies the declared half (framework, sha256,
        weight path) so the row is a join of "what was declared" and "what happened".
        """
        declared = None
        try:
            from fiximg.inference.manifest import get_manifest

            declared = get_manifest().get(name)
        except Exception:  # noqa: BLE001 — the audit must not fail a load
            declared = None
        try:
            from fiximg.infrastructure.db.repositories import model_repository

            model_repository.record_load(
                name,
                version,
                status=status,
                framework=getattr(declared, "framework", None),
                weight_uri=weight_uri or getattr(declared, "weight_uri", None),
                sha256=getattr(declared, "sha256", None),
                load_ms=load_ms,
                error=error,
                metadata=getattr(declared, "metadata", None),
            )
        except Exception as exc:  # noqa: BLE001 — same reason
            log_event(
                logger, "WARNING", "model version audit failed",
                model=name, version=version, error=str(exc),
            )

    def _switch(self, name: str, resident: Resident, started: float) -> dict:
        """Point routing at ``resident`` and evict versions beyond the cap."""
        old_version = self._active.get(name)
        self._active[name] = resident.version
        if old_version and old_version != resident.version:
            self._previous[name] = old_version

        versions = list(self._resident.get(name, {}).values())
        # The keep set is exactly: the new version, plus (when configured) the
        # version it replaced — which is what makes rollback a switch instead of
        # a reload. Any third version is evicted, so memory stays bounded at two.
        keep: set[str] = {resident.version}
        if self.keep_previous and old_version and old_version != resident.version:
            keep.add(old_version)

        evicting = [r for r in versions if r.version not in keep]
        for other in evicting:
            other.retiring = True

        log_event(
            logger, "INFO", "model version switched",
            model=name, **{"from": old_version or "(none)", "to": resident.version},
            switch_ms=int((time.perf_counter() - started) * 1000),
            resident=sorted(keep),
        )

        # Unload outside the lock: dispose may release GPU memory slowly.
        for other in evicting:
            if other.refcount == 0:
                self._unload_resident(other)

        return {
            "name": name,
            "version": resident.version,
            "switched": old_version != resident.version,
            "previous_version": old_version,
            "active_version": resident.version,
            "switch_ms": int((time.perf_counter() - started) * 1000),
            "residents": self.residents(name),
        }

    # ----------------------------------------------------------------- unload
    def _dispose(self, resident: Resident) -> None:
        """Release whatever the handle owns, tolerating partial construction."""
        closer = getattr(resident.handle, "close", None) or getattr(
            resident.handle, "unload", None
        )
        if callable(closer):
            try:
                closer()
            except Exception as exc:  # noqa: BLE001 — disposal is best effort
                log_event(logger, "WARNING", "model disposal failed",
                          model=resident.name, version=resident.version, error=str(exc))

    def _unload_resident(self, resident: Resident) -> None:
        with self._lock:
            self._resident.get(resident.name, {}).pop(resident.version, None)
            if self._active.get(resident.name) == resident.version:
                self._active.pop(resident.name, None)
            if self._previous.get(resident.name) == resident.version:
                self._previous.pop(resident.name, None)
        self._dispose(resident)
        log_event(logger, "INFO", "model version unloaded",
                  model=resident.name, version=resident.version)

    def unload(self, name: str, version: str | None = None) -> list[str]:
        """Unload one version (default: every version of ``name``).

        Versions with in-flight leases are marked retiring instead of being
        dropped, so an active inference is never interrupted.
        """
        with self._lock:
            versions = dict(self._resident.get(name, {}))
        target = version or None
        unloaded: list[str] = []
        for candidate, resident in versions.items():
            if target is not None and candidate != target:
                continue
            if resident.refcount > 0:
                resident.retiring = True
                continue
            self._unload_resident(resident)
            unloaded.append(candidate)
        return unloaded

    # --------------------------------------------------------------- rollback
    def rollback(self, name: str) -> dict:
        """Switch back to the version that was active before the last switch.

        The plan's failure path in reverse: the previous version is still
        resident, so rollback is another atomic switch rather than a reload.
        """
        with self._lock:
            previous = self._previous.get(name)
            current = self._active.get(name)
            resident = self._resident.get(name, {}).get(previous) if previous else None

        if resident is None:
            return {
                "name": name, "switched": False,
                "reason": "no previous version resident",
                "active_version": current,
            }

        with self._lock:
            report = self._switch(name, resident, time.perf_counter())
        # After rolling back, the version we came from becomes the new fallback.
        with self._lock:
            if current and current != resident.version:
                self._previous[name] = current
        report["rolled_back"] = True
        log_event(logger, "WARNING", "model rolled back",
                  model=name, **{"to": resident.version})
        return report


__all__ = ["Resident", "VersionedModelRegistry"]
