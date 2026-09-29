"""Queue transport backends (plan §5.5).

    from fiximg.infrastructure.queue import get_queue

    queue = get_queue()
    task = queue.claim(worker_id="w-1")
    ...

Selection is by ``FIXIMG_QUEUE_BACKEND`` (``sqlite`` default, ``redis`` P2).
"""
from __future__ import annotations

from fiximg.infrastructure.queue.base import QueueBackend
from fiximg.infrastructure.queue.sqlite import SqliteQueueBackend

_queue: QueueBackend | None = None


def _worker_capabilities() -> frozenset[str]:
    """Capabilities this consumer serves (plan §4.3 Step 4).

    Empty means "everything", which is the single-worker default.
    """
    from fiximg.config import settings

    raw = getattr(settings, "worker_capabilities", "") or ""
    return frozenset(c.strip() for c in raw.split(",") if c.strip())


def get_queue() -> QueueBackend:
    """Resolve the configured queue transport once per process."""
    global _queue
    if _queue is not None:
        return _queue
    from fiximg.config import settings

    capabilities = _worker_capabilities()
    if getattr(settings, "queue_backend", "sqlite") == "redis":
        from fiximg.infrastructure.queue.redis import build_redis_queue_from_settings

        built = build_redis_queue_from_settings(settings, capabilities=capabilities)
        if built is not None:
            _queue = built
            return _queue
    _queue = SqliteQueueBackend(capabilities=capabilities)
    return _queue


def reset_queue() -> None:
    """Drop the cached transport (tests / config reload)."""
    global _queue
    _queue = None


__all__ = ["QueueBackend", "SqliteQueueBackend", "get_queue", "reset_queue"]
