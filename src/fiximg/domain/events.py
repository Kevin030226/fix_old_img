"""Task event model (plan §2.5 / §3.8).

Events are the append-only narrative of a task: created → enqueued → stage
started → progress → completed/failed. They are persisted in ``system_events``
and streamed to clients over SSE, which removes the need to poll
``GET /tasks/{id}`` while a large image is processing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fiximg.domain.enums import EventType
from fiximg.domain.tasks import decode_json_column


#: Events after which the server sends nothing more, so the SSE stream may close.
#:
#: `task.completed` is deliberately absent: with `FIXIMG_EVAL_MODE=async` the
#: metrics arrive afterwards as `task.evaluated` (plan §3.16), and a client that
#: stopped listening at "completed" would lose them. The route grants that case a
#: bounded grace window instead - which is why this set, and not a second literal
#: list in the transport layer, is the single definition of "closing".
STREAM_CLOSING_EVENTS: frozenset[str] = frozenset({
    EventType.TASK_FAILED,
    EventType.TASK_CANCELLED,
    EventType.TASK_EVALUATED,
})


@dataclass(slots=True)
class TaskEvent:
    """A single point-in-time fact about a task."""

    event_type: str
    task_id: str | None = None
    user_id: str | None = None
    level: str = "info"
    message: str | None = None
    data: dict = field(default_factory=dict)
    created_at: str | None = None
    seq: int | None = None

    @property
    def closes_stream(self) -> bool:
        """True for the events after which the SSE stream may be closed.

        Named for the question the transport actually asks. It used to be
        ``is_terminal`` and included ``task.completed``, which is wrong in
        ``FIXIMG_EVAL_MODE=async``: the metrics event comes after it.
        """
        return self.event_type in STREAM_CLOSING_EVENTS

    @classmethod
    def from_row(cls, row: Any) -> TaskEvent:
        raw = dict(row)
        # `data_json` is one of the repository's JSON columns: text on SQLite, already
        # a structure on PostgreSQL.
        payload = decode_json_column(raw.get("data_json") or raw.get("data"), {})
        return cls(
            event_type=raw.get("event_type", ""),
            task_id=raw.get("task_id"),
            user_id=raw.get("user_id"),
            level=raw.get("level") or "info",
            message=raw.get("message"),
            data=payload or {},
            created_at=raw.get("created_at"),
            seq=raw.get("id"),
        )


__all__ = ["STREAM_CLOSING_EVENTS", "TaskEvent"]
