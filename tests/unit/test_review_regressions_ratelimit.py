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


# ------------------------------------------------- independent-review follow-ups
def test_the_overflow_counter_survives_many_other_limits():
    """A live overflow counter must not be evicted by unrelated (limit, window) pairs.

    The first implementation pruned the oldest overflow bucket once more than 64
    existed, which reset an overflow allowance an attacker could then reuse.
    """
    backend = rlb._MemoryBackend(max_keys=1)
    assert backend.hit("seed", 1, 600) is True          # table now full
    assert backend.hit("overflow-a", 1, 600) is True    # overflow slot
    assert backend.hit("overflow-b", 1, 600) is False   # exhausted

    # 70 distinct (max_count, window) pairs create 70 other overflow buckets.
    for index in range(70):
        backend.hit(f"other-{index}", 1 + index, 600 + index)

    assert backend.hit("overflow-c", 1, 600) is False, "the live overflow counter was reset"


def test_a_disabled_limiter_never_reads_as_exhausted():
    """FIXIMG_SUBMIT_MAX=0 means unlimited, including for remaining()."""
    limiter = rlb._build_submit_limiter(submit_max=0, window=60)
    assert limiter.remaining("user:alice") > 0
    assert limiter.allowed("user:alice") is True


def test_reset_of_one_key_drops_its_expiry_records():
    """`reset(key)` must not leave that key's heap entries behind."""
    limiter = rlb.SlidingWindowLimiter(5, 600, now=lambda: 0.0, max_keys=10)
    for _ in range(5):
        limiter.hit("k")
    limiter.hit("other")
    heap_size = len(limiter._store._heap)
    assert heap_size >= 6

    limiter.reset("k")
    assert len(limiter._store._heap) < heap_size
    assert all(item[2] != "k" for item in limiter._store._heap)


# --------------------------------------- Redis outage / shared-backend isolation
def test_the_memory_fallback_honours_the_configured_key_cap(monkeypatch):
    """A Redis outage must not lift FIXIMG_RATELIMIT_MAX_KEYS."""
    from fiximg.config import settings

    monkeypatch.setattr(rlb, "MAX_MEMORY_KEYS", 7)
    monkeypatch.setattr(settings, "redis_url", "", raising=False)
    rlb.reset_backend()
    try:
        backend = rlb.get_backend()
        assert backend.kind == "memory"
        assert backend.max_keys == 7, backend.max_keys
    finally:
        rlb.reset_backend()


def test_a_runtime_redis_downgrade_keeps_the_configured_key_cap(monkeypatch):
    """The fail-open path after repeated Redis errors must keep the same bound."""
    class _Broken:
        def ping(self):
            return True

        def pipeline(self):
            raise ConnectionError("down")

    monkeypatch.setattr(rlb, "MAX_MEMORY_KEYS", 5)
    rlb._BACKEND = rlb._RedisBackend(_Broken())
    rlb._BACKEND_KIND = "redis"
    try:
        for _ in range(3):
            rlb.get_backend().hit("k", 1, 60)
        backend = rlb.get_backend()
        assert backend.kind == "memory"
        assert backend.max_keys == 5, backend.max_keys
    finally:
        rlb.reset_backend()


def test_limiters_sharing_a_backend_do_not_share_keys(monkeypatch):
    """Registration and submission must not consume each other's allowance.

    With FIXIMG_REDIS_URL configured every limiter reads one shared backend, and
    the registration username is a caller-controlled string. Without a per-limiter
    namespace, a username like `user:alice` collided with the submission key for
    that user: hits from one counter consumed the other's allowance, and the
    shorter window could purge the longer one's history.
    """
    from fiximg.config import settings

    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:1/0", raising=False)
    rlb.reset_backend()          # connection fails -> shared memory backend
    try:
        registration = rlb._make_limiter(2, 600, namespace="register-username:")
        submission = rlb._make_limiter(60, 60, namespace="submit:")
        key = "user:alice"

        assert registration.hit(key) is True
        assert registration.hit(key) is True
        assert registration.hit(key) is False

        assert submission.hit(key) is True, "submission inherited the registration hits"
        assert submission.remaining(key) == 59
    finally:
        rlb.reset_backend()


def test_the_same_namespace_still_shares_a_counter(monkeypatch):
    """The namespace must scope, not disable, sharing (sanity check on the fix)."""
    from fiximg.config import settings

    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:1/0", raising=False)
    rlb.reset_backend()
    try:
        first = rlb._make_limiter(2, 600, namespace="shared:")
        second = rlb._make_limiter(60, 60, namespace="shared:")
        assert first.hit("k") is True
        assert first.hit("k") is True
        assert first.hit("k") is False
        assert second.remaining("k") == 58
    finally:
        rlb.reset_backend()


# ------------------------------------------- namespace-scoped reset (review finding)
def test_reset_of_a_namespaced_limiter_does_not_clear_other_scopes(monkeypatch):
    """limiter.reset() on a namespaced limiter clears only its own namespace."""
    from fiximg.config import settings

    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:1/0", raising=False)
    rlb.reset_backend()
    try:
        first = rlb._make_limiter(5, 60, namespace="alpha:")
        second = rlb._make_limiter(5, 60, namespace="beta:")
        assert first.hit("k") is True
        assert second.hit("k") is True
        assert first.remaining("k") == 4
        assert second.remaining("k") == 4

        first.reset()

        assert first.remaining("k") == 5, "the limiter own scope was not cleared"
        assert second.remaining("k") == 4, "another scope was cleared by reset()"
    finally:
        rlb.reset_backend()


class _ScanRedis:
    def __init__(self):
        self.deleted: list[str] = []
        self.matches: list[str] = []
        self.keys = ["fiximg:rl:alpha:user", "fiximg:rl:beta:user", "queue:tasks"]

    def scan_iter(self, match=None):
        self.matches.append(match)
        prefix = (match or "").rstrip("*")
        return (k for k in self.keys if k.startswith(prefix))

    def delete(self, *keys):
        self.deleted.extend(keys)
        return len(keys)

    def pipeline(self):
        raise AssertionError("not used")


def test_redis_reset_with_prefix_only_deletes_that_namespace():
    client = _ScanRedis()
    rlb._RedisBackend(client).reset(None, prefix="alpha:")
    assert client.matches == ["fiximg:rl:alpha:*"], client.matches
    assert client.deleted == ["fiximg:rl:alpha:user"], client.deleted


def test_env_non_negative_allows_zero_for_submit_max(monkeypatch):
    monkeypatch.setenv("FIXIMG_TEST_SUBMIT", "0")
    assert rlb._env_non_negative_int("FIXIMG_TEST_SUBMIT", 60) == 0
    monkeypatch.setenv("FIXIMG_TEST_SUBMIT", "-1")
    assert rlb._env_non_negative_int("FIXIMG_TEST_SUBMIT", 60) == 60
    monkeypatch.setenv("FIXIMG_TEST_SUBMIT", "abc")
    assert rlb._env_non_negative_int("FIXIMG_TEST_SUBMIT", 60) == 60
    monkeypatch.setenv("FIXIMG_TEST_SUBMIT", "7")
    assert rlb._env_non_negative_int("FIXIMG_TEST_SUBMIT", 60) == 7
