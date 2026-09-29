"""Model hot-swap tests (plan 搂3.15).

Pins the release flow the plan specifies:

    load v2 鈫?validate 鈫?warmup 鈫?health 鈫?atomic switch 鈫?drain v1 鈫?unload v1

and its failure path: a bad candidate must leave the running version serving.
"""
import contextlib
import threading
import time

import pytest

from fiximg.domain.errors import ModelUnavailableError
from fiximg.inference.versions import VersionedModelRegistry


class _FakeHandle:
    """A stand-in model handle that records disposal."""

    def __init__(self, version: str, dispose_log: list) -> None:
        self.version = version
        self._dispose_log = dispose_log

    def process(self, payload):  # pragma: no cover - not called in most tests
        return payload

    def close(self) -> None:
        self._dispose_log.append(self.version)


@pytest.fixture()
def harness():
    """A registry that keeps the previous version resident (rollback target)."""
    disposed: list[str] = []
    built: list[str] = []

    registry = VersionedModelRegistry(keep_previous=True)

    def builder(version: str) -> _FakeHandle:
        built.append(version)
        return _FakeHandle(version, disposed)

    registry.register_builder("demo", builder)
    return registry, built, disposed


@pytest.fixture()
def eager_harness():
    """A registry that evicts the previous version as soon as its leases drain."""
    disposed: list[str] = []
    built: list[str] = []

    registry = VersionedModelRegistry(keep_previous=False)

    def builder(version: str) -> _FakeHandle:
        built.append(version)
        return _FakeHandle(version, disposed)

    registry.register_builder("demo", builder)
    return registry, built, disposed


# ------------------------------------------------------------------ activation
def test_first_activation_switches_from_nothing(harness):
    registry, built, _ = harness
    report = registry.activate("demo", "1.0.0")

    assert report["switched"] is True
    assert report["previous_version"] is None
    assert registry.current_version("demo") == "1.0.0"
    assert built == ["1.0.0"]


def test_reactivating_the_same_version_is_a_noop(harness):
    registry, built, disposed = harness
    registry.activate("demo", "1.0.0")
    report = registry.activate("demo", "1.0.0")

    assert report["switched"] is False
    assert report["reason"] == "already active"
    assert built == ["1.0.0"]      # no rebuild
    assert disposed == []          # nothing unloaded


def test_unknown_model_raises_domain_error(harness):
    registry, _, _ = harness
    with pytest.raises(ModelUnavailableError):
        registry.activate("nope", "1.0.0")


# --------------------------------------------------------------------- switch
def test_switch_keeps_the_previous_version_as_the_rollback_target(harness):
    """`keep_previous=True` keeps v1 resident so rollback is a switch, not a load."""
    registry, _, disposed = harness
    registry.activate("demo", "1.0.0")
    report = registry.activate("demo", "2.0.0")

    assert report["switched"] is True
    assert report["previous_version"] == "1.0.0"
    assert report["active_version"] == "2.0.0"
    assert registry.current_version("demo") == "2.0.0"
    # v1 stays in memory as the fallback; nothing was disposed.
    assert registry.is_resident("demo", "1.0.0")
    assert disposed == []


def test_eager_policy_evicts_the_previous_version(eager_harness):
    """`keep_previous=False` trades rollback speed for GPU memory."""
    registry, _, disposed = eager_harness
    registry.activate("demo", "1.0.0")
    registry.activate("demo", "2.0.0")

    assert not registry.is_resident("demo", "1.0.0")
    assert disposed == ["1.0.0"]


def test_only_two_versions_stay_resident(harness):
    """A third version evicts the oldest, so memory stays bounded."""
    registry, _, disposed = harness
    registry.activate("demo", "1.0.0")
    registry.activate("demo", "2.0.0")
    registry.activate("demo", "3.0.0")

    assert registry.current_version("demo") == "3.0.0"
    assert registry.is_resident("demo", "2.0.0")
    assert not registry.is_resident("demo", "1.0.0")
    assert disposed == ["1.0.0"]


def test_current_handle_follows_the_switch(harness):
    registry, _, _ = harness
    registry.activate("demo", "1.0.0")
    assert registry.current_handle("demo").version == "1.0.0"

    registry.activate("demo", "2.0.0")
    assert registry.current_handle("demo").version == "2.0.0"


def test_switch_records_the_previous_version_for_rollback(harness):
    registry, _, _ = harness
    registry.activate("demo", "1.0.0")
    registry.activate("demo", "2.0.0")
    assert registry.previous_version("demo") == "1.0.0"


# ------------------------------------------------------------- release pipeline
def test_validation_failure_keeps_the_running_version(harness):
    registry, _, disposed = harness
    registry.activate("demo", "1.0.0")

    def validator(resident):
        raise ValueError("checksum mismatch")

    report = registry.activate("demo", "2.0.0", validator=validator)

    assert report["switched"] is False
    assert "checksum mismatch" in report["reason"]
    assert report["active_version"] == "1.0.0"
    assert registry.current_version("demo") == "1.0.0"
    # The failed candidate was disposed; the running one was not.
    assert disposed == ["2.0.0"]


def test_health_probe_failure_keeps_the_running_version(harness):
    registry, _, disposed = harness
    registry.activate("demo", "1.0.0")

    report = registry.activate("demo", "2.0.0", health=lambda _r: False)

    assert report["switched"] is False
    assert registry.current_version("demo") == "1.0.0"
    assert disposed == ["2.0.0"]


def test_warmup_runs_before_the_switch(harness):
    registry, _, _ = harness
    order: list[str] = []

    registry.activate(
        "demo", "1.0.0",
        warmer=lambda _r: order.append("warmup"),
    )
    assert order == ["warmup"]


def test_warmup_failure_aborts_the_switch(harness):
    registry, _, _ = harness
    registry.activate("demo", "1.0.0")

    def warmer(_resident):
        raise RuntimeError("cuda oom during warmup")

    report = registry.activate("demo", "2.0.0", warmer=warmer)
    assert report["switched"] is False
    assert registry.current_version("demo") == "1.0.0"


def test_successful_candidate_passes_every_gate(harness):
    registry, _, _ = harness
    registry.activate("demo", "1.0.0")
    seen: list[str] = []

    report = registry.activate(
        "demo", "2.0.0",
        validator=lambda _r: seen.append("validate"),
        warmer=lambda _r: seen.append("warmup"),
        health=lambda _r: (seen.append("health"), True)[1],
    )
    assert seen == ["validate", "warmup", "health"]
    assert report["switched"] is True


# ------------------------------------------------------------------- draining
def test_in_flight_lease_defers_unloading_the_old_version(eager_harness):
    registry, _, disposed = eager_harness
    registry.activate("demo", "1.0.0")

    with registry.acquire("demo") as lease:
        assert lease.version == "1.0.0"
        registry.activate("demo", "2.0.0")
        # The lease is still using v1: it must not be disposed yet.
        assert disposed == []
        assert registry.is_resident("demo", "1.0.0")

    # Releasing the lease drains it.
    assert disposed == ["1.0.0"]
    assert not registry.is_resident("demo", "1.0.0")


def test_lease_keeps_the_handle_it_started_with(harness):
    registry, _, _ = harness
    registry.activate("demo", "1.0.0")

    with registry.acquire("demo") as lease:
        old_handle = lease.handle
        registry.activate("demo", "2.0.0")
        # The in-flight call keeps operating on the version it acquired.
        assert lease.handle is old_handle
        assert lease.handle.version == "1.0.0"


def test_multiple_leases_drain_only_after_the_last_release(eager_harness):
    registry, _, disposed = eager_harness
    registry.activate("demo", "1.0.0")

    # Two overlapping leases on v1, released one at a time.
    with contextlib.ExitStack() as stack:
        stack.enter_context(registry.acquire("demo"))
        stack.enter_context(registry.acquire("demo"))

        registry.activate("demo", "2.0.0")
        assert disposed == []          # both leases still hold v1

        stack.close()                  # releases both
    assert disposed == ["1.0.0"]


def test_acquire_without_a_loaded_model_raises(harness):
    registry, _, _ = harness
    with pytest.raises(ModelUnavailableError):
        with registry.acquire("demo"):
            pass


def test_acquire_is_thread_safe(eager_harness):
    """Concurrent leases must all complete and drain exactly once."""
    registry, _, disposed = eager_harness
    registry.activate("demo", "1.0.0")

    barrier = threading.Barrier(4)

    def worker():
        with registry.acquire("demo"):
            barrier.wait(timeout=5)
            time.sleep(0.02)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert disposed == []
    registry.activate("demo", "2.0.0")
    assert disposed == ["1.0.0"]


# --------------------------------------------------------------------- unload
def test_unload_all_versions(harness):
    registry, _, disposed = harness
    registry.activate("demo", "1.0.0")
    registry.activate("demo", "2.0.0")

    unloaded = registry.unload("demo")
    assert "2.0.0" in unloaded
    assert registry.current_handle("demo") is None
    assert registry.current_version("demo") is None


def test_unload_single_version(harness):
    registry, _, _ = harness
    registry.activate("demo", "1.0.0")
    registry.activate("demo", "2.0.0")

    registry.unload("demo", "1.0.0")
    assert registry.current_version("demo") == "2.0.0"


def test_unload_defers_versions_with_in_flight_leases(harness):
    registry, _, disposed = harness
    registry.activate("demo", "1.0.0")

    with registry.acquire("demo"):
        registry.unload("demo")
        assert disposed == []
    assert disposed == ["1.0.0"]


def test_disposal_failure_does_not_break_the_switch(harness):
    registry, _, _ = harness
    registry.activate("demo", "1.0.0")

    class _AngryHandle(_FakeHandle):
        def close(self):
            raise RuntimeError("cannot free memory")

    registry._resident["demo"]["1.0.0"].handle = _AngryHandle("1.0.0", [])
    report = registry.activate("demo", "2.0.0")
    assert report["switched"] is True
    assert registry.current_version("demo") == "2.0.0"


# ------------------------------------------------------------------- rollback
def test_rollback_returns_to_the_previous_version(harness):
    registry, _, _ = harness
    registry.activate("demo", "1.0.0")
    registry.activate("demo", "2.0.0")

    # Keep v1 resident by holding a lease across the switch is not required:
    # rollback reloads it when it is no longer resident.
    report = registry.rollback("demo")
    assert report["switched"] is True
    assert report["rolled_back"] is True
    assert registry.current_version("demo") == "1.0.0"


def test_rollback_without_history_is_a_noop(harness):
    registry, _, _ = harness
    registry.activate("demo", "1.0.0")
    report = registry.rollback("demo")
    assert report["switched"] is False
    assert "no previous version" in report["reason"]


def test_rollback_swaps_the_fallback_back(harness):
    registry, _, _ = harness
    registry.activate("demo", "1.0.0")
    registry.activate("demo", "2.0.0")
    registry.rollback("demo")
    # After rolling back to 1.0.0, 2.0.0 becomes the new fallback.
    assert registry.previous_version("demo") == "2.0.0"


# ------------------------------------------------------------------ snapshots
def test_snapshot_reports_active_previous_and_resident(harness):
    registry, _, _ = harness
    registry.activate("demo", "1.0.0")
    registry.activate("demo", "2.0.0")

    snapshot = registry.snapshot()
    assert snapshot["active"]["demo"] == "2.0.0"
    assert snapshot["previous"]["demo"] == "1.0.0"
    # Both versions are resident: the active one and the rollback target.
    assert sorted(r["version"] for r in snapshot["resident"]["demo"]) == ["1.0.0", "2.0.0"]


def test_residents_expose_refcount_and_retiring(eager_harness):
    registry, _, _ = eager_harness
    registry.activate("demo", "1.0.0")

    with registry.acquire("demo"):
        registry.activate("demo", "2.0.0")
        entries = {r["version"]: r for r in registry.residents("demo")}
        assert entries["1.0.0"]["refcount"] == 1
        assert entries["1.0.0"]["retiring"] is True
        assert entries["2.0.0"]["retiring"] is False


def test_known_models_lists_registered_builders(harness):
    registry, _, _ = harness
    assert registry.known_models() == ["demo"]
    assert registry.has_builder("demo")
    assert not registry.has_builder("other")
