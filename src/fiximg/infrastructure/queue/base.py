"""QueueBackend protocol (plan §5.5).

The V2 queue *is* the ``tasks`` table, which works but hard-codes SQLite into
every call site. V3 defines the transport contract so Redis Streams (P2) can be
dropped in without touching the worker:

    enqueue → claim(worker_id) → ack | retry(delay) | cancel

The SQLite implementation (:mod:`fiximg.infrastructure.queue.sqlite`) keeps the
existing atomic-claim SQL; it simply presents it behind this interface.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from fiximg.domain.tasks import Task


@runtime_checkable
class QueueBackend(Protocol):
    """Transport contract for the task queue."""

    #: backend identifier surfaced by /health and /stats
    kind: str
    #: capabilities this consumer serves; empty means "everything" (§4.3 Step 4)
    capabilities: frozenset[str]

    def enqueue(self, task_id: str) -> None:
        """Make a task eligible for claiming."""
        ...

    def claim(self, worker_id: str, lease_seconds: float = 3600.0) -> Task | None:
        """Atomically claim the next eligible task, or None when idle."""
        ...

    def ack(self, task_id: str) -> None:
        """Mark a claimed task as successfully finished (releases the lease)."""
        ...

    def retry(self, task_id: str, delay_seconds: float = 0.0) -> bool:
        """Return a claimed/failed task to the queue; False when attempts exhausted."""
        ...

    def cancel(self, task_id: str) -> bool:
        """Cancel a not-yet-running task; False when it is already running."""
        ...

    def heartbeat(self, task_id: str) -> None:
        """Refresh the lease of a running task."""
        ...

    def depth(self) -> int:
        """Number of tasks waiting to be claimed."""
        ...


__all__ = ["QueueBackend"]
