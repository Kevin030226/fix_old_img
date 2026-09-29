"""Redis Streams queue (plan §3.1 Step 1, P2).

Status: **implemented, exercised against a fake client in
``tests/unit/test_redis_queue.py``; not yet validated against a live server.**
The transport is implemented against Redis Streams consumer groups, which give
the same claim/ack/retry semantics the SQLite backend provides via
``BEGIN IMMEDIATE``:

* ``XADD`` on submit,
* ``XREADGROUP`` for an exclusive claim (plus ``XAUTOCLAIM`` for lease recovery),
* ``XACK`` on success,
* and a database claim behind every poll, because the stream is only the wake-up.

Task *state* stays in the database — Redis only carries the "which task is due"
signal, so the API and result endpoints keep working unchanged. Selecting it is
``FIXIMG_QUEUE_BACKEND=redis`` plus ``FIXIMG_REDIS_URL``; ``redis`` must be
installed (``pip install 'fiximg[redis]'``).

This module is import-safe without the ``redis`` package: nothing is imported at
module scope.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fiximg.domain.tasks import Task
from fiximg.infrastructure.db.repositories import task_repository as task_repo
from fiximg.infrastructure.observability.logging import get_logger, log_event

#: `redis` stays None until the optional import succeeds. Declaring it `Any`
#: first is what makes this pattern stable in *both* environments: rebinding a
#: Module-typed name to None needs a `type: ignore` when the package is
#: installed, and that same ignore is reported as unused when it is not — so the
#: gate would depend on what happens to be in the interpreter.
redis: Any = None
try:  # optional dependency: importing this module must not require `redis`
    import redis
except ImportError:  # pragma: no cover - depends on the environment
    pass

logger = get_logger("fiximg.queue.redis")

STREAM_KEY = "fiximg:tasks"
GROUP_NAME = "fiximg:workers"

#: Bound on the transport stream: the database holds the authoritative queue, so
#: the stream is a notification channel and must not grow without limit.
STREAM_MAXLEN = 10_000


class RedisQueueBackend:
    """Redis Streams transport; task state remains in the database."""

    kind = "redis"

    def __init__(self, client, stream: str = STREAM_KEY, group: str = GROUP_NAME,
                 consumer: str = "worker", repository=None,
                 capabilities: Iterable[str] | None = None) -> None:
        self._client = client
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._repo = repository or task_repo
        #: Capabilities this consumer serves; empty means "everything".
        self.capabilities = frozenset(capabilities or ())
        #: task_id -> stream record id, so `ack(task_id)` can actually XACK.
        #: Without this a completed task's message stays pending until the lease
        #: expires and gets reclaimed, which is both noisy and slow.
        self._pending: dict[str, str] = {}
        self._ensure_group()

    # ------------------------------------------------------------- internals
    def _ensure_group(self) -> None:
        try:
            self._client.xgroup_create(self._stream, self._group, id="0", mkstream=True)
        except Exception as exc:  # noqa: BLE001 — BUSYGROUP means it already exists
            if "BUSYGROUP" not in str(exc):
                raise

    # -------------------------------------------------------------- protocol
    def enqueue(self, task_id: str) -> None:
        """Publish a task to the stream (best effort).

        The task row already exists in the database, so a Redis outage delays
        pickup rather than losing work — which is why this degrades instead of
        raising into the worker loop.
        """
        try:
            self._client.xadd(self._stream, {"task_id": task_id},
                              maxlen=STREAM_MAXLEN, approximate=True)
        except Exception as exc:  # noqa: BLE001 — transport is best effort
            log_event(logger, "WARNING", "redis enqueue failed",
                      task_id=task_id, error=str(exc))

    def claim(self, worker_id: str, lease_seconds: float = 3600.0) -> Task | None:
        """Claim one pending entry, falling back to reclaiming abandoned ones.

        The stream is a *wake-up*, the database is the queue: an entry is consumed
        and discarded whenever the row behind it is not runnable yet, so a worker
        that only ever listened to the stream could miss work permanently. Ask the
        database directly once the stream has come up empty (§2.6).
        """
        try:
            entries = self._client.xreadgroup(
                self._group, worker_id, {self._stream: ">"}, count=1, block=1000
            )
            if not entries:
                entries = self._reclaim(worker_id, lease_seconds)
        except Exception as exc:  # noqa: BLE001 — transport is best effort
            log_event(logger, "WARNING", "redis claim failed", error=str(exc))
            entries = []
        for _stream, records in entries or []:
            for record_id, fields in records:
                task_id = fields.get("task_id")
                if not task_id:
                    self._client.xack(self._stream, self._group, record_id)
                    continue
                # The authoritative claim (attempt/lease bookkeeping) is still
                # performed transactionally in the database, including the
                # capability filter for specialised workers (§4.3 Step 4).
                row = self._claim_row(worker_id, lease_seconds)
                if row is not None:
                    self._pending[task_id] = record_id
                    return Task.from_row(row)
                # Nothing runnable behind a real notification: three ways this
                # happens, all normal - a retry published before its backoff has
                # elapsed, a second worker that got there first, or a task the API
                # already cancelled. The row stays the authority either way.
                self._client.xack(self._stream, self._group, record_id)
        # A wake-up-independent claim. Entries can be discarded legitimately, the
        # stream is trimmed at STREAM_MAXLEN, `_pending` dies with the process, and
        # an enqueue during an outage is logged and dropped - each of those leaves a
        # queued task nobody told anyone about, which used to wait for the lease
        # window. The SQLite backend polls the same table, so this also makes the
        # two transports behave alike.
        row = self._claim_row(worker_id, lease_seconds)
        return Task.from_row(row) if row is not None else None

    def _claim_row(self, worker_id: str, lease_seconds: float):
        """The authoritative claim: attempt/lease bookkeeping, capability filter."""
        return self._repo.claim_next_task(
            worker_id, lease_seconds=lease_seconds,
            capabilities=self.capabilities or None,
        )

    def _reclaim(self, worker_id: str, lease_seconds: float):
        """Recover entries abandoned by a dead consumer.

        ``XAUTOCLAIM`` returns ``(next_cursor, [(id, fields), ...], deleted_ids)``
        — a *three*-element reply, unlike ``XREADGROUP``'s two-element one. It is
        normalised here to the ``XREADGROUP`` shape so the caller has one path.
        """
        reply = self._client.xautoclaim(
            self._stream, self._group, worker_id,
            min_idle_time=int(lease_seconds * 1000), count=1,
        )
        if not reply:
            return []
        if len(reply) == 3:
            _cursor, records, _deleted = reply
            return [(self._stream, records or [])]
        return reply

    def ack(self, task_id: str) -> None:
        """Acknowledge the stream entry for a finished task.

        Without this the entry stays pending until ``XAUTOCLAIM`` picks it up
        after the lease expires, so a completed task would be re-claimed and
        re-checked (and re-acked) on every lease period.
        """
        record_id = self._pending.pop(task_id, None)
        if record_id is None:
            return None
        try:
            self._client.xack(self._stream, self._group, record_id)
        except Exception as exc:  # noqa: BLE001 — best effort
            log_event(logger, "WARNING", "redis ack failed",
                      task_id=task_id, error=str(exc))
        return None

    def retry(self, task_id: str, delay_seconds: float = 0.0) -> bool:
        if not self._repo.retry_task(task_id, delay_seconds=delay_seconds):
            return False
        self.enqueue(task_id)
        return True

    def cancel(self, task_id: str) -> bool:
        return self._repo.cancel_task(task_id)

    def heartbeat(self, task_id: str) -> None:
        self._repo.heartbeat_task(task_id)

    def depth(self) -> int:
        """Queued rows nobody has claimed - the same quantity the SQLite backend reports.

        ``XLEN`` is not that number: it counts every entry ever published (minus
        the trim), so a stream that has served 500 completed tasks reports a
        backlog of 500. The gauge, ``/health`` and ``PipelineWorker.depth()`` all
        present this as "queued", so one of them would be lying - and the DB was
        already the authority for the claim, so it is the authority here too.
        """
        return self._repo.queued_count()


def build_redis_queue_from_settings(settings_obj, capabilities: Iterable[str] | None = None):
    """Create the backend when configured; None otherwise."""
    if getattr(settings_obj, "queue_backend", "sqlite") != "redis":
        return None
    url = (getattr(settings_obj, "redis_url", "") or "").strip()
    if not url:
        raise ValueError("FIXIMG_REDIS_URL is required when queue_backend=redis")
    if redis is None:  # pragma: no cover - depends on deployment
        raise ImportError(
            "redis is required for the Redis queue backend (pip install 'fiximg[redis]')"
        )
    # Constructing the client opens a consumer group, so an unreachable server
    # fails here — at startup — rather than silently leaving a queue that never
    # delivers anything.
    client = redis.Redis.from_url(url, socket_timeout=1.0, socket_connect_timeout=1.0)
    return RedisQueueBackend(client, capabilities=capabilities)


__all__ = [
    "GROUP_NAME",
    "STREAM_KEY",
    "RedisQueueBackend",
    "build_redis_queue_from_settings",
]
