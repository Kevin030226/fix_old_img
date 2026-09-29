"""SQLite-backed queue (plan §5.5).

Thin adapter over :mod:`fiximg.infrastructure.db.repositories.task_repository`:
the transport *is* the ``tasks`` table, exactly as in V2, but now hidden behind
the :class:`QueueBackend` contract so the worker never writes SQL directly.
"""
from __future__ import annotations

from collections.abc import Iterable

from fiximg.domain.tasks import Task
from fiximg.infrastructure.db.repositories import task_repository as task_repo


class SqliteQueueBackend:
    """DB-backed queue using atomic ``BEGIN IMMEDIATE`` claim."""

    kind = "sqlite"

    def __init__(self, repository=None, capabilities: Iterable[str] | None = None) -> None:
        self._repo = repository or task_repo
        #: Capabilities this consumer serves (plan §4.3 Step 4). Empty = all.
        self.capabilities = frozenset(capabilities or ())

    def enqueue(self, task_id: str) -> None:
        """No-op: ``create_task`` already inserts the row as ``queued``."""
        return None

    def claim(self, worker_id: str, lease_seconds: float = 3600.0) -> Task | None:
        row = self._repo.claim_next_task(
            worker_id, lease_seconds=lease_seconds,
            capabilities=self.capabilities or None,
        )
        return Task.from_row(row) if row else None

    def ack(self, task_id: str) -> None:
        """Terminal success is recorded by ``finish_task``; nothing to release."""
        return None

    def retry(self, task_id: str, delay_seconds: float = 0.0) -> bool:
        return self._repo.retry_task(task_id, delay_seconds=delay_seconds)

    def cancel(self, task_id: str) -> bool:
        return self._repo.cancel_task(task_id)

    def heartbeat(self, task_id: str) -> None:
        self._repo.heartbeat_task(task_id)

    def depth(self) -> int:
        return self._repo.queued_count()


__all__ = ["SqliteQueueBackend"]
