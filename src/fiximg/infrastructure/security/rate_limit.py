"""Rate limiting: in-process limiter + optional shared Redis backend (plan §22).

V3 consolidates what used to live in two modules (`config/ratelimit.py` and
`app/services/rate_limit_backend.py`) into one:

* ``SlidingWindowLimiter`` — the dependency-free in-process sliding window used
  for single-process demos; the clock is injectable for tests.
* ``_RedisBackend`` / ``SharedSlidingWindowLimiter`` — used when
  ``FIXIMG_REDIS_URL`` is set, so every API process shares one allowance. Any
  Redis failure degrades gracefully to the in-memory limiter (fail-open): a
  cache outage must never take registration down.
* ``client_ip`` + the three module-level registration limiters consumed by the
  auth router.

Both implementations expose the same surface — ``hit()``, ``allowed()``,
``remaining()``, ``reset()`` — so call sites never care which one is active.
"""
import os
import time
from collections import deque

from fiximg.infrastructure.observability.logging import get_logger

logger = get_logger("fiximg.ratelimit")

# Resolution happens once; tests can force a re-resolution via reset_backend().
_BACKEND: "_MemoryBackend | _RedisBackend | None" = None
_BACKEND_KIND: str | None = None


class SlidingWindowLimiter:
    """In-process sliding-window limiter: one timestamp deque per key."""

    def __init__(self, max_count, window_seconds, now=None):
        if max_count < 1:
            raise ValueError("max_count must be >= 1")
        if window_seconds < 1:
            raise ValueError("window_seconds must be >= 1")
        self.max_count = max_count
        self.window_seconds = window_seconds
        self._hits: dict = {}  # key -> deque[float]
        self._now = now  # injectable clock for tests

    def _now_ts(self):
        return self._now() if self._now is not None else time.monotonic()

    def _purge(self, key, now_ts):
        dq = self._hits.get(key)
        if not dq:
            return
        cutoff = now_ts - self.window_seconds
        while dq and dq[0] <= cutoff:
            dq.popleft()
        if not dq:
            self._hits.pop(key, None)

    def allowed(self, key):
        """Check whether a request would be allowed, without recording it."""
        now_ts = self._now_ts()
        self._purge(key, now_ts)
        return len(self._hits.get(key, deque())) < self.max_count

    def hit(self, key):
        """Record one request: True when allowed, False when over the limit."""
        now_ts = self._now_ts()
        self._purge(key, now_ts)
        dq = self._hits.setdefault(key, deque())
        if len(dq) >= self.max_count:
            return False
        dq.append(now_ts)
        return True

    def remaining(self, key):
        """Remaining allowance inside the current window."""
        now_ts = self._now_ts()
        self._purge(key, now_ts)
        return max(0, self.max_count - len(self._hits.get(key, deque())))

    def reset(self, key=None):
        if key is None:
            self._hits.clear()
        else:
            self._hits.pop(key, None)


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
    from fiximg.config import settings

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
    """Create a limiter that always reads the shared backend.

    The factory to use for any scope beyond the three module-level
    registration/submission limiters. It returns :class:`SharedSlidingWindowLimiter`,
    which resolves ``get_backend()`` on *every* call: correct on the Redis
    backend, correct on the memory fallback, and still correct after a runtime
    backend switch (which is what a Redis outage triggers). Instantiating
    :class:`SlidingWindowLimiter` directly would instead pin a per-process window
    and silently multiply the quota by the number of replicas — the failure is
    invisible in a single-process test, which is why this is the documented path
    rather than a comment.
    """
    return SharedSlidingWindowLimiter(max_count, window_seconds)


# --------------------------------------------------------------- registration
def client_ip(request) -> str:
    """Client IP; forwarding headers are trusted only from configured proxies.

    ``FIXIMG_TRUSTED_PROXIES`` is a comma-separated peer allowlist — a request
    arriving from any other peer has its ``X-Forwarded-For`` ignored, so the
    header cannot be spoofed to bypass per-IP limiting.
    """
    client = getattr(request, "client", None)
    peer = client.host if client is not None else "unknown"
    trusted = {
        item.strip()
        for item in os.environ.get("FIXIMG_TRUSTED_PROXIES", "").split(",")
        if item.strip()
    }
    if peer in trusted:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return peer


#: Registration rate-limit window (seconds); tunable for calibration.
REGISTER_WINDOW = int(os.environ.get("FIXIMG_REGISTER_WINDOW", 600))


def _make_limiter(max_count: int, window_seconds: int = REGISTER_WINDOW):
    """Redis-backed limiter when configured, otherwise the in-process window."""
    from fiximg.config import settings

    if (getattr(settings, "redis_url", "") or "").strip():
        return SharedSlidingWindowLimiter(max_count, window_seconds)
    return SlidingWindowLimiter(max_count, window_seconds)


register_ip_limiter = _make_limiter(int(os.environ.get("FIXIMG_REGISTER_MAX", 5)))
register_global_limiter = _make_limiter(int(os.environ.get("FIXIMG_REGISTER_GLOBAL_MAX", 20)))
register_username_limiter = _make_limiter(int(os.environ.get("FIXIMG_REGISTER_USERNAME_MAX", 3)))

# ---------------------------------------------------------------- submissions
#: Task-submission rate limit (plan §2.5's `RATE_LIMITED` code, which the API
#: documents but could not previously emit on a JSON route).
#:
#: One limiter, keyed by the authenticated principal and falling back to the client
#: address. Queue *capacity* is a different protection and is already handled by
#: `QUEUE_FULL` at enqueue time, so this one is only about one caller flooding the
#: planner and the GPU queue. `FIXIMG_SUBMIT_MAX=0` disables it.
SUBMIT_WINDOW = int(os.environ.get("FIXIMG_SUBMIT_WINDOW", 60))
SUBMIT_MAX = int(os.environ.get("FIXIMG_SUBMIT_MAX", 60))

#: Rebuilt by whoever needs a clean allowance (test fixtures); read through the
#: module attribute so a replacement is seen by the call sites.
submit_limiter = _make_limiter(max(1, SUBMIT_MAX), SUBMIT_WINDOW)


def submission_key(principal, request) -> tuple[str, str]:
    """``(key, scope)`` for a submission attempt.

    The principal wins because a shared NAT egress must not make a hundred users
    look like one abuser; the address is the fallback for a request that reached
    the route without an identity, which is the only case where IP is the right key.
    """
    user_id = getattr(principal, "user_id", None)
    if user_id:
        return f"user:{user_id}", "principal"
    return f"ip:{client_ip(request)}", "client-address"
