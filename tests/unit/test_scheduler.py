"""GPU scheduler tests (plan §2.4).

The scheduler replaces the single global execution lock with a per-capability
concurrency policy. These tests pin the policy maths and the actual
serialisation/overlap behaviour.
"""
import threading
import time

import pytest

from fiximg.inference.scheduler import ConcurrencyPolicy, GpuScheduler


# ------------------------------------------------------------------- policy
def test_default_policy_is_serial():
    policy = ConcurrencyPolicy()
    assert policy.default == 1
    assert policy.limit_for([]) == 1
    assert policy.limit_for(["restore"]) == 1


def test_policy_takes_the_strictest_limit():
    policy = ConcurrencyPolicy(default=2, per_capability={"restore": 1})
    assert policy.limit_for(["restore"]) == 1
    assert policy.limit_for(["colorize"]) == 2
    # A stage needing both capabilities is bound by the stricter one.
    assert policy.limit_for(["colorize", "restore"]) == 1


def test_policy_never_returns_less_than_one():
    policy = ConcurrencyPolicy(default=0, per_capability={"restore": -5})
    assert policy.limit_for(["restore"]) == 1
    assert policy.limit_for([]) == 1


def test_policy_from_env(monkeypatch):
    """The base number comes from settings, the per-capability ones from the env.

    `config.py` owns `FIXIMG_CONCURRENCY` (env, then YAML, then a validated
    default). The scheduler reading the variable again gave the knob two sources:
    the YAML `concurrency_default` looked configurable and did nothing.
    """
    from fiximg.config import settings

    monkeypatch.setattr(settings, "concurrency_default", 3, raising=False)
    monkeypatch.setenv("FIXIMG_CONCURRENCY_SCRATCH_DETECTION", "4")
    monkeypatch.setenv("FIXIMG_CONCURRENCY_BOGUS", "not-a-number")

    policy = ConcurrencyPolicy.from_env()
    assert policy.default == 3
    assert policy.per_capability["scratch_detection"] == 4
    assert "bogus" not in policy.per_capability


def test_policy_follows_the_setting_not_a_stale_env_read(monkeypatch):
    """The mutation this change exists to prevent: two sources, one knob."""
    from fiximg.config import settings

    monkeypatch.setattr(settings, "concurrency_default", 7, raising=False)
    assert ConcurrencyPolicy.from_env().default == 7


def test_policy_from_env_survives_garbage_default(monkeypatch):
    from fiximg.config import settings

    monkeypatch.setattr(settings, "concurrency_default", "abc", raising=False)
    assert ConcurrencyPolicy.from_env().default == 1
    monkeypatch.setattr(settings, "concurrency_default", 0, raising=False)
    assert ConcurrencyPolicy.from_env().default == 1


def test_policy_describe():
    policy = ConcurrencyPolicy(default=2, per_capability={"restore": 1})
    assert policy.describe() == {"default": 2, "per_capability": {"restore": 1}}


# ---------------------------------------------------------------- behaviour
def test_serial_capability_allows_only_one_holder():
    scheduler = GpuScheduler(ConcurrencyPolicy(default=1))
    inside = []
    peak = {"value": 0}
    lock = threading.Lock()

    def worker():
        with scheduler.slot(["restore"]):
            with lock:
                inside.append(1)
                peak["value"] = max(peak["value"], len(inside))
            time.sleep(0.05)
            with lock:
                inside.pop()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak["value"] == 1


def test_capability_with_higher_limit_overlaps():
    scheduler = GpuScheduler(ConcurrencyPolicy(default=1, per_capability={"detect": 3}))
    inside = []
    peak = {"value": 0}
    lock = threading.Lock()

    def worker():
        with scheduler.slot(["detect"]):
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
        thread.join()

    assert peak["value"] == 3


def test_slot_is_released_on_exception():
    scheduler = GpuScheduler(ConcurrencyPolicy(default=1))
    with pytest.raises(RuntimeError):
        with scheduler.slot(["restore"]):
            raise RuntimeError("boom")

    # If the semaphore leaked, this acquire would block forever.
    with scheduler.slot(["restore"], device=0):
        pass
    assert scheduler.stats()["active"]["0:restore"] == 0


def test_stats_report_policy_and_activity():
    scheduler = GpuScheduler(ConcurrencyPolicy(default=1, per_capability={"restore": 2}))
    with scheduler.slot(["restore"], device=0):
        stats = scheduler.stats()
        assert stats["active"]["0:restore"] == 1
        assert stats["policy"]["per_capability"] == {"restore": 2}

    assert scheduler.stats()["active"]["0:restore"] == 0
    assert scheduler.stats()["acquired_total"] == 1


def test_default_slot_uses_the_default_limit():
    scheduler = GpuScheduler(ConcurrencyPolicy(default=2))
    with scheduler.slot():
        assert scheduler.stats()["active"]["-1:__default__"] == 1


def test_slots_are_independent_per_device():
    """Two GPUs each get their own limit (plan §4.3 Step 4)."""
    scheduler = GpuScheduler(ConcurrencyPolicy(default=1))
    with scheduler.slot(["restore"], device=0):
        with scheduler.slot(["restore"], device=1):
            active = scheduler.stats()["active"]
            assert active["0:restore"] == 1
            assert active["1:restore"] == 1


def test_reset_clears_state():
    scheduler = GpuScheduler()
    with scheduler.slot(["restore"]):
        pass
    scheduler.reset()
    stats = scheduler.stats()
    assert stats["acquired_total"] == 0
    assert stats["active"] == {}
