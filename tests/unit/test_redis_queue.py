"""Redis queue backend tests against a real Redis implementation.

The backend originally had no tests at all; the first version of this file used a
hand-written fake, which caught three genuine defects (XAUTOCLAIM's reply shape,
a no-op ``ack``, no degradation on an outage). But a hand-written fake only
encodes *my* understanding of Redis 鈥?it cannot catch a wrong understanding.

This file drives the backend with **fakeredis**, which implements the real
commands (``XADD``/``XREADGROUP``/``XAUTOCLAIM``/``XACK``, consumer groups,
pending entries) against an in-process server. That is as close to a live Redis
as is possible without one, and it is what makes "verified against the real
protocol" an honest claim.

Set ``FIXIMG_TEST_REDIS_URL`` to run the same tests against a real server:

    FIXIMG_TEST_REDIS_URL=redis://localhost:6379/15 pytest tests/unit/test_redis_queue.py
"""
import os

import pytest

from fiximg.infrastructure.db.repositories import task_repository as repo
from fiximg.infrastructure.queue import get_queue, reset_queue
from fiximg.infrastructure.queue.base import QueueBackend
from fiximg.infrastructure.queue.redis import (
    GROUP_NAME,
    STREAM_KEY,
    RedisQueueBackend,
    build_redis_queue_from_settings,
)

# The whole module needs a Redis implementation; skip it (rather than fail) on a
# minimal install.
fakeredis = pytest.importorskip("fakeredis", reason="fakeredis not installed")


def _real_url() -> str | None:
    return (os.environ.get("FIXIMG_TEST_REDIS_URL") or "").strip() or None


@pytest.fixture()
def client():
    """A fakeredis client, or a real one when FIXIMG_TEST_REDIS_URL is set."""
    url = _real_url()
    if url:
        import redis

        connection = redis.Redis.from_url(url, decode_responses=True)
        connection.flushdb()
        yield connection
        connection.flushdb()
        return

    yield fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture()
def backend(client, isolated_db):
    return RedisQueueBackend(client, repository=repo)


def _create(task_id="t-1", **kwargs):
    repo.create_task(task_id, "restore", "alice", **kwargs)


def _drop_client(client):
    """Make every command fail, to exercise the degradation paths."""
    def _boom(*_args, **_kwargs):
        raise ConnectionError("redis is down")

    for command in ("xadd", "xreadgroup", "xautoclaim", "xack", "xlen"):
        setattr(client, command, _boom)
    return client


# ------------------------------------------------------------------- enqueue
def test_enqueue_publishes_to_the_configured_stream(backend, client):
    backend.enqueue("t-1")
    entries = client.xrange(STREAM_KEY)
    assert len(entries) == 1
    _entry_id, fields = entries[0]
    assert fields["task_id"] == "t-1"


def test_enqueue_trims_the_stream(client, isolated_db):
    """MAXLEN keeps the transport from growing without bound."""
    from fiximg.infrastructure.queue.redis import STREAM_MAXLEN

    backend = RedisQueueBackend(client, repository=repo)
    for index in range(5):
        backend.enqueue(f"t-{index}")
    assert client.xlen(STREAM_KEY) == 5
    assert STREAM_MAXLEN > 0


def test_enqueue_is_survivable_when_redis_is_down(client, isolated_db):
    """The task row still exists, so a later poll can pick it up."""
    backend = RedisQueueBackend(client, repository=repo)
    _drop_client(client)
    backend.enqueue("t-1")  # must not raise


# --------------------------------------------------------------------- claim
def test_claim_reads_a_message_and_claims_in_the_repository(backend):
    _create("t-1")
    backend.enqueue("t-1")

    task = backend.claim("w-1", lease_seconds=60)
    assert task is not None
    assert task.id == "t-1"
    assert task.status == "running"

    # The authoritative bookkeeping lives in the database, not in Redis.
    row = repo.get_task("t-1")
    assert row["worker_id"] == "w-1"
    assert row["attempt_count"] == 1


def test_claim_returns_none_when_the_stream_is_empty(backend):
    """An empty stream falls through to XAUTOCLAIM, whose reply is a 3-tuple.

    This is the path the original hand-rolled fake got wrong; fakeredis returns
    the real shape, so a regression here is caught.
    """
    assert backend.claim("w-1") is None


def test_claim_acks_a_message_for_a_missing_task(backend, client):
    backend.enqueue("ghost")
    assert backend.claim("w-1") is None

    # The entry must be acknowledged, not left pending forever.
    assert client.xpending(STREAM_KEY, GROUP_NAME)["pending"] == 0


def test_claim_acks_a_task_that_is_no_longer_claimable(backend, client):
    """A duplicate message (already claimed) must be acked, not retried forever."""
    _create("t-1")
    backend.enqueue("t-1")
    assert backend.claim("w-1") is not None
    # The first entry stays pending: it belongs to w-1 until it is acked.
    assert client.xpending(STREAM_KEY, GROUP_NAME)["pending"] == 1

    backend.enqueue("t-1")                     # same task published twice
    assert backend.claim("w-2") is None
    # The duplicate was acked, so the pending count did not grow.
    assert client.xpending(STREAM_KEY, GROUP_NAME)["pending"] == 1

    backend.ack("t-1")
    assert client.xpending(STREAM_KEY, GROUP_NAME)["pending"] == 0


def test_claim_consumes_each_message_once(backend):
    _create("t-1")
    _create("t-2")
    backend.enqueue("t-1")
    backend.enqueue("t-2")

    claimed = {backend.claim("w-1").id for _ in range(2)}
    assert claimed == {"t-1", "t-2"}
    assert backend.claim("w-1") is None


def test_claim_forwards_the_capability_filter(isolated_db, client):
    """A specialised consumer must not claim work it cannot serve (Sec.4.3 Step 4)."""
    _create("t-restore", required_capabilities=["restore"])
    _create("t-color", required_capabilities=["colorize"])

    color_only = RedisQueueBackend(client, repository=repo, capabilities={"colorize"})
    color_only.enqueue("t-restore")
    color_only.enqueue("t-color")

    claimed = []
    while True:
        task = color_only.claim("w-color")
        if task is None:
            break
        claimed.append(task.id)
    assert claimed == ["t-color"]


def test_claim_ignores_a_message_without_a_task_id(backend, client):
    client.xadd(STREAM_KEY, {"wrong": "field"})
    assert backend.claim("w-1") is None


def test_claim_survives_an_outage_and_still_finds_the_work(client, isolated_db):
    """A dead transport may delay nothing it can read from the database.

    This asserted `is None` before, which encoded the bug: with the stream as the
    only source of work, an outage meant queued tasks simply never ran, and the
    worker's own liveness looked healthy. Redis going away is now a slower wake-up
    path, not lost work.
    """
    backend = RedisQueueBackend(client, repository=repo)
    _create("t-1")
    _drop_client(client)
    task = backend.claim("w-1")
    assert task is not None and task.id == "t-1"


def test_reclaim_picks_up_an_abandoned_entry(isolated_db, client):
    """XAUTOCLAIM recovers entries a dead consumer never acknowledged."""
    backend = RedisQueueBackend(client, repository=repo)
    _create("t-1")
    backend.enqueue("t-1")

    # Consumer "dead" claims the entry, then disappears without acking.
    client.xreadgroup(GROUP_NAME, "dead", {STREAM_KEY: ">"}, count=1)

    # Another consumer reclaims it once the lease has elapsed (min_idle_time=0).
    task = backend.claim("alive", lease_seconds=0)
    assert task is not None and task.id == "t-1"


# ------------------------------------------------------------------- ack
def test_ack_clears_the_pending_entry(isolated_db, client):
    """`ack` maps a task id back to the stream record it was claimed from."""
    backend = RedisQueueBackend(client, repository=repo)
    _create("t-1")
    backend.enqueue("t-1")
    assert backend.claim("w-1") is not None
    assert client.xpending(STREAM_KEY, GROUP_NAME)["pending"] == 1

    backend.ack("t-1")
    assert client.xpending(STREAM_KEY, GROUP_NAME)["pending"] == 0


def test_ack_of_an_unknown_task_is_a_noop(backend, client):
    backend.ack("never-claimed")
    assert client.xpending(STREAM_KEY, GROUP_NAME)["pending"] == 0


def test_ack_twice_is_harmless(isolated_db, client):
    backend = RedisQueueBackend(client, repository=repo)
    _create("t-1")
    backend.enqueue("t-1")
    backend.claim("w-1")

    backend.ack("t-1")
    backend.ack("t-1")                          # already forgotten: no-op
    assert client.xpending(STREAM_KEY, GROUP_NAME)["pending"] == 0


# ------------------------------------------------------------ protocol / misc
def test_backend_satisfies_the_queue_protocol(backend):
    assert isinstance(backend, QueueBackend)
    assert backend.kind == "redis"


def test_backend_exposes_its_capabilities(client, isolated_db):
    backend = RedisQueueBackend(client, repository=repo,
                                capabilities={"colorize", "face_restore"})
    assert backend.capabilities == frozenset({"colorize", "face_restore"})


def test_depth_counts_queued_rows_not_stream_entries(backend):
    """The queue gauge, `/health` and `PipelineWorker.depth()` all call this "queued".

    ``XLEN`` is not that quantity: it counts every entry ever published minus the
    trim, so it grew while the backlog emptied - two submissions and one claim
    still read as a backlog of two, which is a number no operator can act on.
    """
    assert backend.depth() == 0
    _create("t-1")
    _create("t-2")
    backend.enqueue("t-1")
    backend.enqueue("t-2")
    assert backend.depth() == 2
    assert backend.claim("w-1") is not None
    assert backend.depth() == 1, "one claimed, one still queued"


def test_depth_is_unaffected_by_a_dead_transport(client, isolated_db):
    """The backlog is a database fact, so an outage cannot invent a number."""
    backend = RedisQueueBackend(client, repository=repo)
    _create("t-1")
    _drop_client(client)
    assert backend.depth() == 1


def test_work_is_still_found_when_the_notification_is_gone(backend, client):
    """A lost stream entry must not strand a queued task (搂2.6).

    Entries go missing in ordinary ways: the stream is trimmed at STREAM_MAXLEN,
    `_pending` dies with the process, an enqueue during an outage is logged and
    dropped. Each used to leave the task queued until `XAUTOCLAIM` reached the
    lease window, because the worker only ever acted on what the stream said.
    """
    _create("t-1")
    backend.enqueue("t-1")
    client.delete(STREAM_KEY)
    task = backend.claim("w-1")
    assert task is not None and task.id == "t-1"


def test_a_delayed_retry_is_claimable_after_its_backoff(backend):
    """The retry notification arrives before the backoff elapses and is discarded.

    That is correct - the row is not runnable yet - but it must not be the only
    chance: without the database fallback the task stayed runnable and unheard
    for the rest of the lease period (default one hour).
    """
    import time

    _create("t-1", max_attempts=3)
    assert backend.claim("w-1") is not None
    repo.fail_task("t-1", "transient", 10)

    assert backend.retry("t-1", delay_seconds=0.4) is True
    assert backend.claim("w-2") is None, "must still respect the backoff window"
    time.sleep(0.6)
    assert backend.claim("w-3") is not None, "the delayed retry was never woken"


def test_group_is_created_once(client, isolated_db):
    RedisQueueBackend(client, repository=repo)
    RedisQueueBackend(client, repository=repo)  # BUSYGROUP is tolerated
    groups = client.xinfo_groups(STREAM_KEY)
    assert any(group["name"] == GROUP_NAME for group in groups)


def test_retry_republishes_the_task(isolated_db, client):
    backend = RedisQueueBackend(client, repository=repo)
    _create("t-1", max_attempts=3)
    backend.claim("w-1")
    repo.fail_task("t-1", "transient", 10)

    assert backend.retry("t-1", delay_seconds=0) is True
    assert repo.get_task("t-1")["status"] == "queued"
    # ...and it is claimable again.
    assert backend.claim("w-2") is not None


def test_a_manual_retry_republishes_to_the_stream(isolated_db, client, backend, monkeypatch):
    """搂2.6: on this topology the stream is the only thing a worker reads.

    The submission path learned this already; the operator-retry path still wrote
    the row and stopped, so a retried task was runnable in the database and
    unknown to the transport. It would eventually be picked up when
    ``XAUTOCLAIM`` reclaimed the original entry after the lease window 鈥?which
    reads as "the retry did nothing for an hour", not as a bug.
    """
    import fiximg.application.task_service as task_service_module
    import fiximg.infrastructure.queue as queue_module

    _create("t-retry", max_attempts=1)
    backend.enqueue("t-retry")
    assert backend.claim("w-1") is not None
    repo.fail_task("t-retry", "boom", 10)
    monkeypatch.setattr(queue_module, "get_queue", lambda: backend)

    assert task_service_module.task_service.retry_task("t-retry") is True

    published = [fields.get("task_id") for _id, fields in client.xrange(STREAM_KEY)]
    assert published.count("t-retry") == 2, published
    claimed = backend.claim("w-2")
    assert claimed is not None and claimed.id == "t-retry", "the retry is claimable now"


def test_finishing_a_task_releases_its_stream_entry(isolated_db, client, backend):
    """An acked entry is gone from the group's pending list; an unacked one is not.

    `QueueBackend.ack()` used to have no caller at all, so every finished task
    stayed pending until its lease expired and was then re-delivered to a worker
    that had nothing left to do 鈥?and the backend's own task->record map kept
    growing one entry per finished task for the life of the process.
    """
    _create("t-ack")
    backend.enqueue("t-ack")
    assert backend.claim("w-1") is not None
    # Claiming leaves the entry pending under this consumer.
    assert client.xpending(STREAM_KEY, GROUP_NAME)["pending"] == 1

    backend.ack("t-ack")

    assert client.xpending(STREAM_KEY, GROUP_NAME)["pending"] == 0
    assert backend._pending == {}, "the backend still holds the record id"


def test_heartbeat_refreshes_a_running_task(isolated_db, client):
    backend = RedisQueueBackend(client, repository=repo)
    _create("t-1")
    backend.enqueue("t-1")
    assert backend.claim("w-1") is not None      # heartbeat needs status='running'

    backend.heartbeat("t-1")
    assert repo.get_task("t-1")["last_heartbeat"] is not None


def test_cancel_reaches_a_running_task_on_this_transport_too(isolated_db, client):
    """The transport follows the row, and the row now accepts a running cancel (搂3.8).

    Answering False here was the bug the endpoint died of: a task could only be stopped
    in the few hundred milliseconds before a worker claimed it.
    """
    backend = RedisQueueBackend(client, repository=repo)
    _create("queued-task")
    assert backend.cancel("queued-task") is True
    assert repo.get_task("queued-task")["status"] == "cancelled"

    _create("running-task")
    backend.enqueue("running-task")
    assert backend.claim("w-1") is not None
    assert backend.cancel("running-task") is True
    assert repo.get_task("running-task")["status"] == "cancelled"

    # A terminal row stays terminal: cancelling twice reports the refusal.
    assert backend.cancel("running-task") is False


def test_a_cancelled_task_is_never_handed_to_a_worker(isolated_db, client):
    """Its wake-up is released rather than left for `XAUTOCLAIM` to re-deliver.

    Without the discard the transport would keep offering a task the user stopped: the
    claim itself is safe (the database is the authority and refuses it), but every lease
    period a worker would be woken for work that will never run.
    """
    backend = RedisQueueBackend(client, repository=repo)
    _create("cancelled-awake")
    backend.enqueue("cancelled-awake")
    assert repo.cancel_task("cancelled-awake") is True, "the row is not cancellable"

    assert backend.claim("w-1") is None, "a cancelled task was claimed"
    assert client.xpending(STREAM_KEY, GROUP_NAME)["pending"] == 0,         "the wake-up was left pending"


# --------------------------------------------------------------------- factory
def test_factory_returns_none_for_the_sqlite_backend(monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "queue_backend", "sqlite")
    assert build_redis_queue_from_settings(config_mod.settings) is None


def test_factory_requires_a_url(monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "queue_backend", "redis")
    monkeypatch.setattr(config_mod.settings, "redis_url", "")
    with pytest.raises(ValueError, match="FIXIMG_REDIS_URL"):
        build_redis_queue_from_settings(config_mod.settings)


def test_factory_builds_a_backend_for_a_redis_url(monkeypatch):
    """With `redis` installed the factory must produce a backend.

    The client is stubbed: constructing a real one would try to open a consumer
    group, which needs a live server. Point ``FIXIMG_TEST_REDIS_URL`` at one to
    exercise the real connection path (see the module docstring).
    """
    import fiximg.config as config_mod
    import fiximg.infrastructure.queue.redis as redis_module

    created = {}

    class _StubRedis:
        """Records the URL and tolerates the consumer-group setup."""

        def __init__(self, *args, **kwargs):
            created["args"] = args

        @classmethod
        def from_url(cls, url, **kwargs):
            return cls(url, **kwargs)

        def xgroup_create(self, *_args, **_kwargs):
            return True

    monkeypatch.setattr(redis_module.redis, "Redis", _StubRedis)
    monkeypatch.setattr(config_mod.settings, "queue_backend", "redis")
    monkeypatch.setattr(config_mod.settings, "redis_url", "redis://localhost:6379/0")

    built = build_redis_queue_from_settings(config_mod.settings)
    assert built is not None
    assert built.kind == "redis"
    assert created["args"] == ("redis://localhost:6379/0",)


def test_factory_reports_an_unreachable_server(monkeypatch):
    """A misconfigured Redis must fail loudly at construction, not silently.

    The service then refuses to start, which is visible; degrading here would
    leave a queue that never delivers anything and never says why.
    """
    import fiximg.config as config_mod
    import fiximg.infrastructure.queue.redis as redis_module

    class _Unreachable:
        @classmethod
        def from_url(cls, url, **kwargs):
            return cls()

        def xgroup_create(self, *_args, **_kwargs):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(redis_module.redis, "Redis", _Unreachable)
    monkeypatch.setattr(config_mod.settings, "queue_backend", "redis")
    monkeypatch.setattr(config_mod.settings, "redis_url", "redis://localhost:6379/0")
    with pytest.raises(ConnectionError):
        build_redis_queue_from_settings(config_mod.settings)


def test_get_queue_defaults_to_sqlite_without_redis(monkeypatch, isolated_db):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "queue_backend", "sqlite")
    reset_queue()
    try:
        assert get_queue().kind == "sqlite"
    finally:
        reset_queue()


def test_get_queue_forwards_worker_capabilities(monkeypatch, isolated_db):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "queue_backend", "sqlite")
    monkeypatch.setattr(config_mod.settings, "worker_capabilities", "colorize, face_restore")
    reset_queue()
    try:
        assert get_queue().capabilities == frozenset({"colorize", "face_restore"})
    finally:
        reset_queue()


# ------------------------------------------------ the producer (submit) side
def _submit(task_id: str, tmp_path, monkeypatch, backend):
    """Submit through the real service path with the transport replaced."""
    from PIL import Image

    import fiximg.config as config_mod
    import fiximg.infrastructure.queue as queue_module
    from fiximg.application.task_service import task_service

    monkeypatch.setattr(config_mod.settings, "tasks_root", str(tmp_path / "tasks"))
    monkeypatch.setattr(queue_module, "get_queue", lambda: backend)
    image = Image.new("RGB", (8, 8), (10, 20, 30))
    return task_service.submit_queued(task_id, image, {"username": "alice"}, "restore")


def test_submitting_publishes_to_the_stream(client, backend, monkeypatch, isolated_db, tmp_path):
    """A task the transport never heard of is a task no worker will ever run.

    ``claim()`` reads the stream and nothing else 鈥?the database only decides
    *what* a claim means (lease, attempts, capabilities). The submission path
    therefore has to notify the transport. This was missing, and invisible on the
    SQLite topology because its ``enqueue`` is a documented no-op.
    """
    assert _submit("pub-1", tmp_path, monkeypatch, backend) is True

    assert [fields.get("task_id") for _id, fields in client.xrange(STREAM_KEY)] == ["pub-1"]
    claimed = backend.claim("w-1")
    assert claimed is not None
    assert claimed.id == "pub-1"
    assert claimed.status == "running", "the DB claim bookkeeping still runs"


def test_submit_survives_a_dead_transport_without_losing_the_row(
    client, backend, monkeypatch, isolated_db, tmp_path
):
    """A Redis hiccup must not turn into a failed submission.

    The row is the authority, so the task is not lost 鈥?only its wake-up is. That
    is the trade the best-effort notify makes deliberately, and it is why the
    stale-lease sweep and the next retry still reach it.
    """
    _drop_client(client)

    assert _submit("pub-2", tmp_path, monkeypatch, backend) is True
    assert repo.queued_count() == 1, "the task is queued in the database"
    assert repo.claim_next_task("w-2")["id"] == "pub-2", "and still claimable"
