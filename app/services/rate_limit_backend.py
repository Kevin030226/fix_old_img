"""Shared rate-limit backend selection (plan section 22, P3).

The V1 in-memory sliding-window limiter only works inside one process. When
`FIXIMG_REDIS_URL` is set (and the `redis` package is installed), limiter
counters move to Redis so every API process shares the same allowance — the
plan's "Redis Rate Limit" step. Any Redis failure degrades gracefully to the
in-memory limiter so a cache outage can never take registration down.

The public surface mirrors config/ratelimit.SlidingWindowLimiter: hit(),
allowed(), remaining(), reset() — call sites do not change.
"""
import time
from collections import deque

from app.core.logging import get_logger

logger = get_logger("fiximg.ratelimit")

# Resolution happens once; tests can force a re-resolution via reset_backend().
_BACKEND = None
_BACKEND_KIND = None


class _MemoryBackend:
    """The V1 sliding window (unchanged semantics)."""

    kind = "memory"

    def __init__(self) -> None:
        self._hits: dict = {}

    def _purge(self, key: str, now_ts: float, window: float) -> None:
        dq = self._hits.get(key)
        if not dq:
            return
        cutoff = now_ts - window
        while dq and dq[0] <= cutoff:
            dq.popleft()
        if not dq:
            self._hits.pop(key, None)

    def hit(self, key: str, max_count: int, window: float) -> bool:
        now_ts = time.monotonic()
        self._purge(key, now_ts, window)
        dq = self._hits.setdefault(key, deque())
        if len(dq) >= max_count:
            return False
        dq.append(now_ts)
        return True

    def allowed(self, key: str, max_count: int, window: float) -> bool:
        now_ts = time.monotonic()
        self._purge(key, now_ts, window)
        return len(self._hits.get(key, deque())) < max_count

    def remaining(self, key: str, max_count: int, window: float) -> int:
        now_ts = time.monotonic()
        self._purge(key, now_ts, window)
        return max(0, max_count - len(self._hits.get(key, deque())))

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._hits.clear()
        else:
            self._hits.pop(key, None)


class _RedisBackend:
    """Sliding window backed by a Redis sorted set per key.

    Each hit is a member with its timestamp as score; the window is enforced
    by ZREMRANGEBYSCORE + ZCARD. Operations are best-effort: a failed command
    falls back to allowing the request (fail-open) and the process-wide
    backend flips back to memory after repeated failures.
    """

    kind = "redis"

    def __init__(self, client) -> None:
        self._client = client
        self._prefix = "fiximg:rl:"
        self._failures = 0

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def _run(self, op, *args):
        try:
            result = op(*args)
            self._failures = 0
            return result
        except Exception as exc:  # noqa: BLE001 — degrade to fail-open
            self._failures += 1
            if self._failures == 1:
                logger.warning("redis rate limiter unavailable (%s); failing open", exc)
            if self._failures >= 3:
                _switch_to_memory()
            return None

    def hit(self, key: str, max_count: int, window: float) -> bool:
        import random

        now_ms = int(time.time() * 1000)
        window_ms = int(window * 1000)
        member = f"{now_ms}-{random.randrange(1 << 30)}"

        def _op(client):
            pipe = client.pipeline()
            pipe.zremrangebyscore(self._key(key), 0, now_ms - window_ms)
            pipe.zcard(self._key(key))
            pipe.zadd(self._key(key), {member: now_ms})
            pipe.expire(self._key(key), max(int(window) + 1, 1))
            _, count, _, _ = pipe.execute()
            return int(count) < max_count

        result = self._run(_op, self._client)
        return True if result is None else result

    def allowed(self, key: str, max_count: int, window: float) -> bool:
        now_ms = int(time.time() * 1000)

        def _op(client):
            pipe = client.pipeline()
            pipe.zremrangebyscore(self._key(key), 0, now_ms - int(window * 1000))
            pipe.zcard(self._key(key))
            _, count = pipe.execute()
            return int(count) < max_count

        result = self._run(_op, self._client)
        return True if result is None else result

    def remaining(self, key: str, max_count: int, window: float) -> int:
        now_ms = int(time.time() * 1000)

        def _op(client):
            pipe = client.pipeline()
            pipe.zremrangebyscore(self._key(key), 0, now_ms - int(window * 1000))
            pipe.zcard(self._key(key))
            _, count = pipe.execute()
            return max(0, max_count - int(count))

        result = self._run(_op, self._client)
        return max_count if result is None else result

    def reset(self, key: str | None = None) -> None:
        def _op(client):
            if key is None:
                client.flushdb()
            else:
                client.delete(self._key(key))
            return True

        self._run(_op, self._client)


def _switch_to_memory() -> None:
    global _BACKEND, _BACKEND_KIND
    _BACKEND = _MemoryBackend()
    _BACKEND_KIND = "memory"


def get_backend():
    """Resolve the rate-limit backend once: Redis when configured, else memory."""
    global _BACKEND, _BACKEND_KIND
    if _BACKEND is not None:
        return _BACKEND
    from app.core.config import settings

    url = (settings.redis_url or "").strip()
    if url:
        try:
            import redis  # noqa: PLC0415 — optional dependency

            client = redis.Redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=0.5)
            client.ping()
            _BACKEND = _RedisBackend(client)
            _BACKEND_KIND = "redis"
            logger.info("rate limiter backend: redis")
            return _BACKEND
        except Exception as exc:  # noqa: BLE001 — never block startup on cache
            logger.warning("redis unavailable (%s); rate limiter uses memory", exc)
    _switch_to_memory()
    return _BACKEND


def reset_backend() -> str:
    """Forget the resolved backend (tests / config reload); returns new kind."""
    global _BACKEND, _BACKEND_KIND
    _BACKEND = None
    _BACKEND_KIND = None
    return get_backend().kind


class SharedSlidingWindowLimiter:
    """Drop-in replacement for SlidingWindowLimiter using the shared backend.

    Keeps the constructor signature (max_count, window_seconds, now=) and the
    hit()/allowed()/remaining()/reset() API of the V1 limiter, so call sites
    only need to swap the class — or use the factory below.
    """

    def __init__(self, max_count: int, window_seconds: int, now=None) -> None:
        if max_count < 1:
            raise ValueError("max_count must be >= 1")
        if window_seconds < 1:
            raise ValueError("window_seconds must be >= 1")
        self.max_count = max_count
        self.window_seconds = window_seconds
        self._now = now  # injectable clock honoured by the memory backend

    @property
    def backend_kind(self) -> str:
        return get_backend().kind

    def hit(self, key: str) -> bool:
        return get_backend().hit(key, self.max_count, self.window_seconds)

    def allowed(self, key: str) -> bool:
        return get_backend().allowed(key, self.max_count, self.window_seconds)

    def remaining(self, key: str) -> int:
        return get_backend().remaining(key, self.max_count, self.window_seconds)

    def reset(self, key: str | None = None) -> None:
        get_backend().reset(key)


def make_limiter(max_count: int, window_seconds: int) -> SharedSlidingWindowLimiter:
    """Create a limiter bound to the configured (shared) backend."""
    return SharedSlidingWindowLimiter(max_count, window_seconds)
