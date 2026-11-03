"""Regression tests for the in-process rate limiter's key-capacity policy.

The v3.1.2 review found two problems, both confirmed:

* `reset()` on the Redis backend issued `FLUSHDB`, wiping the whole database the
  app shares with the task queue;
* `FIXIMG_SUBMIT_MAX=0` was documented as "disables" but `max(1, ...)` made it
  one-per-window, and the in-process limiter had no bound on its key dict — an
  unauthenticated username field could grow it without limit.

The first fix introduced a key cap that evicted the *oldest dict entry*. That is
wrong in the unsafe direction: an attacker who keeps creating fresh keys evicts
live counters (including a banned/limited key) and thereby resets an allowance.
The tests below pin the safe policy instead: expired counters go first, and when
the table is full of live counters a new key shares an overflow bucket; an
existing counter is never evicted to make room.
"""
from __future__ import annotations

import threading

import pytest

from fiximg.infrastructure.security import rate_limit as rlb


# ------------------------------------------------------------- redis reset
class _RecordingRedis:
    """A fake client that records every command it is given."""

    def __init__(self, keys=("fiximg:rl:a", "fiximg:rl:b", "queue:tasks")):
        self.calls: list[tuple] = []
        self._keys = list(keys)

    def flushdb(self):
        self.calls.append(("flushdb",))
        return True

    def delete(self, *keys):
        self.calls.append(("delete",) + tuple(keys))
        return len(keys)

    def scan_iter(self, match=None):
        self.calls.append(("scan_iter", match))
        prefix = (match or "").rstrip("*")
        return (k for k in self._keys if k.startswith(prefix))

    def pipeline(self):
        raise AssertionError("not used by these tests")


def test_reset_all_does_not_flush_the_shared_database():
    client = _RecordingRedis()
    backend = rlb._RedisBackend(client)
    backend.reset()
    assert ("flushdb",) not in client.calls, client.calls
    deleted = [k for call in client.calls if call[0] == "delete" for k in call[1:]]
    assert all(k.startswith("fiximg:rl:") for k in deleted), deleted
    assert "queue:tasks" not in deleted


def test_reset_one_key_deletes_only_that_key():
    client = _RecordingRedis()
    rlb._RedisBackend(client).reset("alice")
    assert client.calls == [("delete", "fiximg:rl:alice")], client.calls


# ------------------------------------------------------------- SUBMIT_MAX=0
def test_submit_max_zero_disables_the_limiter():
    limiter = rlb._build_submit_limiter(submit_max=0, window=60)
    for _ in range(5):
        assert limiter.hit("user:alice") is True, "0 must disable, not restrict"
    assert limiter.max_count == 0


def test_submit_max_positive_is_still_enforced():
    limiter = rlb._build_submit_limiter(submit_max=2, window=60)
    assert limiter.hit("user:alice") is True
    assert limiter.hit("user:alice") is True
    assert limiter.hit("user:alice") is False
    assert limiter.max_count == 2


# ------------------------------------------------------- key-capacity policy
def test_a_limited_key_stays_limited_while_an_attacker_churns_new_keys():
    """The bypass the old eviction policy allowed: churn resets the allowance."""
    limiter = rlb.SlidingWindowLimiter(3, 600, now=lambda: 0.0, max_keys=10)
    for _ in range(3):
        assert limiter.hit("victim") is True
    assert limiter.hit("victim") is False

    for index in range(1000):
        limiter.hit(f"flood-{index}")

    assert limiter.hit("victim") is False, "churn reset the victim's allowance"
    assert limiter.remaining("victim") == 0
    assert len(limiter._hits) <= 10, len(limiter._hits)


def test_an_already_limited_key_cannot_be_evicted_by_a_flood():
    backend = rlb._MemoryBackend(max_keys=4)
    for _ in range(2):
        assert backend.hit("victim", 2, 600) is True
    assert backend.hit("victim", 2, 600) is False

    for index in range(200):
        backend.hit(f"new-{index}", 2, 600)

    assert backend.hit("victim", 2, 600) is False, "the live counter was evicted"
    assert len(backend._hits) <= 4


def test_the_key_table_never_exceeds_max_keys():
    limiter = rlb.SlidingWindowLimiter(5, 600, now=lambda: 0.0, max_keys=8)
    for index in range(500):
        limiter.hit(f"key-{index}")
    assert len(limiter._hits) == 8


def test_expired_keys_are_reclaimed_before_capacity_is_applied():
    """Expired counters must not count against the cap."""
    clock = {"now": 0.0}
    limiter = rlb.SlidingWindowLimiter(2, 10, now=lambda: clock["now"], max_keys=3)
    for index in range(3):
        limiter.hit(f"old-{index}")
    assert len(limiter._hits) == 3

    clock["now"] = 11.0                      # every window has expired
    assert limiter.hit("newcomer") is True
    assert "newcomer" in limiter._hits
    assert len(limiter._hits) == 1


def test_the_overflow_bucket_does_not_reset_an_existing_counter():
    limiter = rlb.SlidingWindowLimiter(1, 600, now=lambda: 0.0, max_keys=2)
    assert limiter.hit("a") is True
    assert limiter.hit("b") is True
    # Capacity now full of live keys: further new keys go to one overflow bucket
    # with the same limit, and "a"/"b" keep their exhausted allowance.
    assert limiter.hit("c") is True            # first overflow slot
    assert limiter.hit("d") is False           # overflow exhausted
    assert limiter.hit("a") is False
    assert limiter.hit("b") is False
    assert len(limiter._hits) == 2


@pytest.mark.parametrize("bad", [0, -1, -100, True, "10", 1.5])
def test_max_keys_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        rlb.SlidingWindowLimiter(3, 60, max_keys=bad)


def test_max_keys_accepts_none_and_positive_integers():
    assert rlb.SlidingWindowLimiter(3, 60, max_keys=None).max_keys is None
    assert rlb.SlidingWindowLimiter(3, 60, max_keys=1).max_keys == 1


def test_a_bad_environment_value_falls_back_instead_of_failing_import(monkeypatch):
    assert rlb._env_positive_int("FIXIMG_TEST_MAX_KEYS", 7) == 7
    monkeypatch.setenv("FIXIMG_TEST_MAX_KEYS", "abc")
    assert rlb._env_positive_int("FIXIMG_TEST_MAX_KEYS", 7) == 7
    monkeypatch.setenv("FIXIMG_TEST_MAX_KEYS", "0")
    assert rlb._env_positive_int("FIXIMG_TEST_MAX_KEYS", 7) == 7
    monkeypatch.setenv("FIXIMG_TEST_MAX_KEYS", "-3")
    assert rlb._env_positive_int("FIXIMG_TEST_MAX_KEYS", 7) == 7
    monkeypatch.setenv("FIXIMG_TEST_MAX_KEYS", "12")
    assert rlb._env_positive_int("FIXIMG_TEST_MAX_KEYS", 7) == 12


# ----------------------------------------------------------- thread safety
def test_concurrent_hits_never_exceed_the_limit():
    """`hit` is a read-modify-write; without a lock two threads can both win."""
    limiter = rlb.SlidingWindowLimiter(5, 60)
    barrier = threading.Barrier(8)
    results: list[bool] = []
    lock = threading.Lock()

    def worker():
        barrier.wait(timeout=10)
        allowed = limiter.hit("same-key")
        with lock:
            results.append(allowed)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sum(results) == 5, f"expected exactly 5 winners, got {sum(results)}"
    assert limiter.remaining("same-key") == 0


def test_concurrent_hits_on_different_keys_are_all_counted():
    limiter = rlb.SlidingWindowLimiter(2, 60, max_keys=200)
    barrier = threading.Barrier(16)
    errors: list[BaseException] = []

    def worker(index: int):
        try:
            barrier.wait(timeout=10)
            limiter.hit(f"key-{index}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors, errors
    assert len(limiter._hits) == 16
