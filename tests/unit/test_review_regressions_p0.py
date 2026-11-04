"""Regression tests for the v3.1.2 review findings (P0-1, P0-2, P0-3).

Each test here failed against the tag as shipped and passes after the fix. They
are written against the *observable* defect rather than the shape of the patch:
what a caller can see go wrong, not which line moved.

P0-1  versions.py — a version marked `retiring` can be switched back to active,
      and the next lease release then unloads the version that is serving.
P0-2  scheduler.py — `slot()` releases every *requested* key, including ones it
      never acquired, inflating the concurrency limit; and `reset()` during an
      in-flight slot raises KeyError on exit.
"""
from __future__ import annotations

import threading
import time

import pytest

from fiximg.inference.scheduler import ConcurrencyPolicy, GpuScheduler
from fiximg.inference.versions import VersionedModelRegistry


# ------------------------------------------------------------------ P0-1
class _Handle:
    def __init__(self, version: str, disposed: list[str]) -> None:
        self.version = version
        self._disposed = disposed

    def close(self) -> None:
        self._disposed.append(self.version)


@pytest.fixture()
def registry():
    disposed: list[str] = []
    reg = VersionedModelRegistry(keep_previous=True)
    reg.register_builder("demo", lambda v: _Handle(v, disposed))
    return reg, disposed


def test_switching_back_to_a_retiring_version_clears_the_flag(registry):
    """The reported defect: reactivating an evicted-but-pinned version.

    Sequence: v1 is serving and a long inference pins it; v2 then v3 arrive, so
    v1 leaves the keep set and is only alive because of that lease; the operator
    switches back to v1. v1 is now the *active* version again but is still marked
    `retiring`, so the moment the old lease is released the registry unloads the
    version that is serving traffic.
    """
    reg, disposed = registry
    reg.activate("demo", "1.0.0")

    lease_cm = reg.acquire("demo")
    lease_cm.__enter__()                  # held: do not let GC close it early
    try:
        reg.activate("demo", "2.0.0")
        reg.activate("demo", "3.0.0")     # v1 evicted from the keep set, pinned
        assert reg.is_resident("demo", "1.0.0")

        reg.activate("demo", "1.0.0")     # switch back to the pinned version
        assert reg.current_version("demo") == "1.0.0"
        # The version is active again, so it must not still be on its way out.
        assert {r["version"]: r["retiring"] for r in reg.residents("demo")}["1.0.0"] is False
    finally:
        lease_cm.__exit__(None, None, None)

    # The lease is gone, and the active version must still be there.
    assert reg.current_version("demo") == "1.0.0", "the active version was unloaded by a stale lease"
    assert reg.is_resident("demo", "1.0.0")
    assert disposed.count("1.0.0") <= 1
    with reg.acquire("demo") as live:
        assert live.version == "1.0.0"


def test_release_never_unloads_the_active_version(registry):
    """The invariant behind P0-1, asserted directly on `release`.

    Even if a stale `retiring` flag survives (a future caller setting it by hand,
    a version switched back outside `_switch`), dropping the last lease of the
    active version must not unload it. `release` decides "unload" from a snapshot
    taken under the lock and then unloads outside it; the version it is about to
    drop can become active in between.
    """
    reg, disposed = registry
    reg.activate("demo", "1.0.0")

    lease_cm = reg.acquire("demo")
    lease_cm.__enter__()
    try:
        reg.activate("demo", "2.0.0")
        resident = reg._resident["demo"]["1.0.0"]
        resident.retiring = True          # simulate the stale flag
        reg._active["demo"] = "1.0.0"     # ...while routing points back at it
    finally:
        lease_cm.__exit__(None, None, None)

    assert reg.current_version("demo") == "1.0.0"
    assert reg.is_resident("demo", "1.0.0"), "release unloaded the active version"
    with reg.acquire("demo") as live:
        assert live.version == "1.0.0"


# ------------------------------------------------------------------ P0-2
def test_slot_releases_only_the_semaphores_it_acquired():
    """A failure part-way through must not hand out capacity it never took.

    `slot()` loops over the requested keys, acquiring each; the `finally` then
    releases *every requested key*. When the second acquire raises, the first
    semaphore is released twice — once by the loop unwinding and once by the
    finally — so its internal count ends up above its limit and the capability
    runs more concurrently than the policy allows.
    """
    scheduler = GpuScheduler(ConcurrencyPolicy(default=1))
    with scheduler.slot(["b"]):
        pass                                  # materialise the 'b' semaphore

    real = scheduler._semaphore

    def fail_on_b(device, capability):
        if capability == "b":
            raise RuntimeError("simulated: state changed between keys")
        return real(device, capability)

    scheduler._semaphore = fail_on_b
    with pytest.raises(RuntimeError):
        with scheduler.slot(["a", "b"]):
            pass

    sem_b = scheduler._semaphores[(-1, "b")]
    limit = scheduler.policy.limit_for(["b"])
    assert sem_b._value <= limit, (
        f"a key that was never acquired was released anyway: value={sem_b._value} "
        f"limit={limit}"
    )


def test_the_observed_concurrency_never_exceeds_the_policy():
    """The user-visible consequence of the over-release above."""
    scheduler = GpuScheduler(ConcurrencyPolicy(default=1))
    with scheduler.slot(["b"]):
        pass

    real = scheduler._semaphore

    def fail_on_b(device, capability):
        if capability == "b":
            raise RuntimeError("simulated")
        return real(device, capability)

    scheduler._semaphore = fail_on_b
    with pytest.raises(RuntimeError):
        with scheduler.slot(["a", "b"]):
            pass
    scheduler._semaphore = real

    inside: list[int] = []
    peak = {"value": 0}
    lock = threading.Lock()

    def worker():
        with scheduler.slot(["b"]):
            with lock:
                inside.append(1)
                peak["value"] = max(peak["value"], len(inside))
            time.sleep(0.05)
            with lock:
                inside.pop()

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert peak["value"] == 1, f"policy says 1, observed {peak['value']}"


def test_reset_during_an_in_flight_slot_does_not_raise():
    """`reset()` drops the semaphore dict; the slot that is still inside it then
    has no semaphore to release, and the exit path raised KeyError — turning a
    clean shutdown into an exception on a worker thread."""
    scheduler = GpuScheduler(ConcurrencyPolicy(default=1))
    with scheduler.slot(["x"]):
        scheduler.reset()
    # Leaving the block must not raise, and must not resurrect state either.
    assert scheduler.stats()["acquired_total"] == 0


def test_a_reset_slot_leaves_the_scheduler_usable():
    scheduler = GpuScheduler(ConcurrencyPolicy(default=1))
    with scheduler.slot(["x"]):
        scheduler.reset()
    with scheduler.slot(["x"]):
        assert scheduler.stats()["active"]["-1:x"] == 1
    assert scheduler.stats()["active"]["-1:x"] == 0
