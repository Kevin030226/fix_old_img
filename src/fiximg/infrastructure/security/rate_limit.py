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
import heapq
import os
import threading
import time
from collections import deque

from fiximg.infrastructure.observability.logging import get_logger

logger = get_logger("fiximg.ratelimit")

#: `remaining()` value for a limiter that is switched off.
_UNLIMITED_REMAINING = 2**31 - 1


def _env_positive_int(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back with a warning.

    A bad knob must not abort import: `FIXIMG_RATELIMIT_MAX_KEYS=abc` used to
    raise ValueError while the module was being imported, which surfaces as a
    traceback from an unrelated import. Invalid values fall back to the default
    and say so.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    if value < 1:
        logger.warning("%s=%r must be >= 1; using %d", name, raw, default)
        return default
    return value


#: How many keys an in-process limiter tracks before it starts using its overflow
#: bucket. The registration limiters are keyed by a username taken from an
#: unauthenticated form, so an attacker controls the key set; this is what keeps
#: that from being a memory-exhaustion primitive. It applies to *every* memory
#: backend, including the one a Redis outage falls back to.
def _env_non_negative_int(name: str, default: int) -> int:
    """Read an int >= 0 from the environment; 0 has a meaning of its own.

    `FIXIMG_SUBMIT_MAX=0` disables the submission limit, so it cannot go through
    `_env_positive_int`; a negative or unparsable value is still a mistake and
    falls back to the default with a warning rather than aborting the import.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    if value < 0:
        logger.warning("%s=%r must be >= 0; using %d", name, raw, default)
        return default
    return value


MAX_MEMORY_KEYS = _env_positive_int("FIXIMG_RATELIMIT_MAX_KEYS", 10_000)

# Resolution happens once; tests can force a re-resolution via reset_backend().
_BACKEND: "_MemoryBackend | _RedisBackend | None" = None
_BACKEND_KIND: str | None = None


def _validate_max_keys(value, *, name: str = "max_keys"):
    """Validate a key cap. ``None`` means "no cap" where that is allowed."""
    if value is None:
        return None
    # bool is an int subclass; True as a cap is a typo, not a policy.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer >= 1 or None, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1 when set, got {value!r}")
    return value


class _BoundedWindowStore:
    """Per-key sliding windows with a bounded key set and an overflow bucket.

    Capacity policy (the security-relevant part):

    1. expired counters are purged *first* — a min-heap of expiry times makes
       that amortised O(log n) instead of a full dict scan per call;
    2. a key that is already tracked always keeps its own window;
    3. a *new* key is admitted only while the table is below ``max_keys``;
    4. when the table is full of live keys, new keys share a single overflow
       bucket per (limit, window) instead of evicting a live counter. Evicting
       live counters would let an attacker reset an allowance by flooding fresh
       keys — the exact bypass a limiter must not have. The trade-off is that a
       legitimate newcomer can be throttled while the table is full; that is the
       safe direction, and it is bounded by the same window as any other key.

    All state is guarded by one lock: ``hit`` is a read-modify-write on deques
    that used to be unsynchronised, so two threads could both observe the last
    slot as free and both take it.
    """

    def __init__(self, max_keys=None) -> None:
        self.max_keys = _validate_max_keys(max_keys)
        self.hits: dict[str, deque] = {}          # key -> deque[float]
        self._heap: list[tuple[float, int, str, float]] = []  # (expiry, seq, key, window)
        self._seq = 0
        self._overflow: dict[tuple[int, float], deque] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- internals
    def _purge_key_locked(self, key: str, now: float, window: float) -> deque | None:
        dq = self.hits.get(key)
        if dq is None:
            return None
        cutoff = now - window
        while dq and dq[0] <= cutoff:
            dq.popleft()
        if not dq:
            self.hits.pop(key, None)
            return None
        return dq

    def _remember_locked(self, key: str, now: float, window: float) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (now + window, self._seq, key, window))

    def _purge_expired_locked(self, now: float) -> None:
        heap = self._heap
        while heap and heap[0][0] <= now:
            _expiry, _seq, key, window = heapq.heappop(heap)
            dq = self.hits.get(key)
            if dq is None:
                continue
            cutoff = now - window
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if not dq:
                self.hits.pop(key, None)

    def _at_capacity(self, key: str) -> bool:
        return (self.max_keys is not None
                and key not in self.hits
                and len(self.hits) >= self.max_keys)

    def _overflow_locked(self, max_count: int, window: float) -> deque:
        bucket = self._overflow.get((max_count, window))
        if bucket is None:
            bucket = deque()
            self._overflow[(max_count, window)] = bucket
        # Deliberately no eviction here. The keys are (max_count, window) pairs
        # taken from the configured limiters — a small, fixed set that a client
        # cannot influence — so this table is already bounded. Evicting the
        # oldest bucket "to keep it small" would reset a live overflow counter,
        # which is exactly the bypass this class exists to prevent.
        return bucket

    # -------------------------------------------------------------------- api
    def hit(self, key: str, max_count: int, window: float, now: float) -> bool:
        with self._lock:
            self._purge_expired_locked(now)
            dq = self._purge_key_locked(key, now, window)
            if dq is None:
                if self._at_capacity(key):
                    bucket = self._overflow_locked(max_count, window)
                    cutoff = now - window
                    while bucket and bucket[0] <= cutoff:
                        bucket.popleft()
                    if len(bucket) >= max_count:
                        return False
                    bucket.append(now)
                    return True
                dq = deque()
                self.hits[key] = dq
            if len(dq) >= max_count:
                return False
            dq.append(now)
            self._remember_locked(key, now, window)
            return True

    def allowed(self, key: str, max_count: int, window: float, now: float) -> bool:
        with self._lock:
            self._purge_expired_locked(now)
            dq = self._purge_key_locked(key, now, window)
            if dq is None:
                if self._at_capacity(key):
                    bucket = self._overflow_locked(max_count, window)
                    cutoff = now - window
                    while bucket and bucket[0] <= cutoff:
                        bucket.popleft()
                    return len(bucket) < max_count
                return True
            return len(dq) < max_count

    def remaining(self, key: str, max_count: int, window: float, now: float) -> int:
        with self._lock:
            self._purge_expired_locked(now)
            dq = self._purge_key_locked(key, now, window)
            if dq is None:
                if self._at_capacity(key):
                    bucket = self._overflow_locked(max_count, window)
                    cutoff = now - window
                    while bucket and bucket[0] <= cutoff:
                        bucket.popleft()
                    return max(0, max_count - len(bucket))
                return max_count
            return max(0, max_count - len(dq))

    def reset(self, key: str | None = None, prefix: str | None = None) -> None:
        """Clear one key, one namespace prefix, or everything.

        `prefix` exists so a namespaced limiter can clear its own counters without
        wiping the other limiters that share this store. The overflow bucket is a
        shared pressure valve keyed by (limit, window) rather than by key, so it is
        only cleared by a full reset — documented rather than silently global.
        """
        with self._lock:
            if key is not None:
                self.hits.pop(key, None)
                self._heap = [item for item in self._heap if item[2] != key]
                heapq.heapify(self._heap)
            elif prefix:
                for victim in [name for name in self.hits if name.startswith(prefix)]:
                    self.hits.pop(victim, None)
                self._heap = [
                    item for item in self._heap if not item[2].startswith(prefix)
                ]
                heapq.heapify(self._heap)
            else:
                self.hits.clear()
                self._heap.clear()
                self._overflow.clear()


class SlidingWindowLimiter:
    """In-process sliding-window limiter: one timestamp deque per key."""

    def __init__(self, max_count, window_seconds, now=None, max_keys=None):
        if max_count < 1:
            raise ValueError("max_count must be >= 1")
        if window_seconds < 1:
            raise ValueError("window_seconds must be >= 1")
        self.max_count = max_count
        self.window_seconds = window_seconds
        self._now = now  # injectable clock for tests
        self._store = _BoundedWindowStore(max_keys)
        # Public-ish handle kept for introspection/tests: len(limiter._hits).
        self._hits = self._store.hits
        self.max_keys = self._store.max_keys

    def _now_ts(self):
        return self._now() if self._now is not None else time.monotonic()

    def allowed(self, key):
        """Check whether a request would be allowed, without recording it."""
        return self._store.allowed(key, self.max_count, self.window_seconds, self._now_ts())

    def hit(self, key):
        """Record one request: True when allowed, False when over the limit."""
        return self._store.hit(key, self.max_count, self.window_seconds, self._now_ts())

    def remaining(self, key):
        """Remaining allowance inside the current window."""
        return self._store.remaining(key, self.max_count, self.window_seconds, self._now_ts())

    def reset(self, key=None, prefix=None):
        self._store.reset(key, prefix)


class _MemoryBackend:
    """The V1 sliding window, bounded by `max_keys`."""

    kind = "memory"

    #: How many distinct keys a process tracks. Bounds the memory an unauthenticated
    #: key source (a username form field, a spoofable address) can make this dict
    #: grow to; the same reasoning as `SlidingWindowLimiter.max_keys`.
    max_keys = 10_000

    def __init__(self, max_keys: int | None = None) -> None:
        self._store = _BoundedWindowStore(
            self.max_keys if max_keys is None else max_keys
        )
        self.max_keys = self._store.max_keys
        self._hits = self._store.hits

    def hit(self, key: str, max_count: int, window: float) -> bool:
        return self._store.hit(key, max_count, window, time.monotonic())

    def allowed(self, key: str, max_count: int, window: float) -> bool:
        return self._store.allowed(key, max_count, window, time.monotonic())

    def remaining(self, key: str, max_count: int, window: float) -> int:
        return self._store.remaining(key, max_count, window, time.monotonic())

    def reset(self, key: str | None = None, prefix: str | None = None) -> None:
        self._store.reset(key, prefix)

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

    def reset(self, key: str | None = None, prefix: str | None = None) -> None:
        """Clear allowance — only for keys this limiter owns.

        `FLUSHDB` was used for "reset everything", which wipes the whole database:
        this limiter shares Redis with the task queue and any cache a deployment
        put there, so a rate-limit helper could destroy unrecoverable state. The
        keys are prefixed (`fiximg:rl:`), so the narrow form is also the correct
        one — a reset means "forget my counters", not "empty the server".
        """

        def _op(client):
            if key is not None:
                client.delete(self._key(key))
                return True
            match = f"{self._prefix}{prefix}*" if prefix else f"{self._prefix}*"
            batch: list[str] = []
            for found in client.scan_iter(match=match):
                batch.append(found)
                if len(batch) >= 256:
                    client.delete(*batch)
                    batch.clear()
            if batch:
                client.delete(*batch)
            return True

        self._run(_op, self._client)


def _switch_to_memory() -> None:
    global _BACKEND, _BACKEND_KIND
    # The fallback must keep the configured memory bound: a Redis outage is not a
    # reason to let the in-process key table grow without the operator's limit.
    _BACKEND = _MemoryBackend(max_keys=MAX_MEMORY_KEYS)
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

    def reset(self, key: str | None = None, prefix: str | None = None) -> None:
        get_backend().reset(key, prefix)


def make_limiter(
    max_count: int, window_seconds: int, namespace: str = ""
) -> "SharedSlidingWindowLimiter | _NamespacedLimiter":
    """Create a limiter that always reads the shared backend.

    The factory to use for any scope beyond the three module-level
    registration/submission limiters. It resolves ``get_backend()`` on *every*
    call: correct on the Redis backend, correct on the memory fallback, and still
    correct after a runtime backend switch (which is what a Redis outage
    triggers). Instantiating :class:`SlidingWindowLimiter` directly would instead
    pin a per-process window and silently multiply the quota by the number of
    replicas — the failure is invisible in a single-process test, which is why this
    is the documented path rather than a comment.

    With a ``namespace`` the result is a :class:`_NamespacedLimiter` wrapper (same
    API) whose counters cannot collide with another limiter sharing the backend;
    without one it is a :class:`SharedSlidingWindowLimiter`.
    """
    limiter = SharedSlidingWindowLimiter(max_count, window_seconds)
    # `namespace` scopes the keys inside the shared backend. Two limiters with the
    # same key string but different scopes (registration vs submission, two
    # different callers) must not consume each other's allowance; without it the
    # backend cannot tell them apart.
    return _NamespacedLimiter(limiter, namespace) if namespace else limiter


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
REGISTER_WINDOW = _env_positive_int("FIXIMG_REGISTER_WINDOW", 600)


class _NamespacedLimiter:
    """Prefixes every key with a scope, so limiters cannot share an allowance.

    The shared backend stores counters by the bare key string. Registration uses
    the submitted username and submission uses `user:<name>`: without a scope, a
    caller could make one limiter count against another (and a shorter window
    would purge a longer one's history), which is both a quota bypass and a way
    to exhaust somebody else's allowance. Each limiter therefore owns a prefix.

    Keys are internal to this module: `submission_key()` and the registration
    keys stay what callers and tests see; only the backend sees the prefix.
    """

    def __init__(self, inner, namespace: str) -> None:
        if not namespace:
            raise ValueError("a namespaced limiter needs a non-empty namespace")
        self._inner = inner
        self._namespace = namespace

    @property
    def max_count(self):
        return self._inner.max_count

    @property
    def window_seconds(self):
        return self._inner.window_seconds

    @property
    def backend_kind(self):
        return self._inner.backend_kind

    def _scoped(self, key) -> str:
        return f"{self._namespace}{key}"

    def hit(self, key) -> bool:
        return self._inner.hit(self._scoped(key))

    def allowed(self, key) -> bool:
        return self._inner.allowed(self._scoped(key))

    def remaining(self, key) -> int:
        return self._inner.remaining(self._scoped(key))

    def reset(self, key=None) -> None:
        # A no-argument reset on a namespaced limiter means "clear my scope", not
        # "clear every limiter that happens to share this backend". The inner
        # limiter receives the namespace as a prefix.
        if key is None:
            self._inner.reset(None, prefix=self._namespace)
        else:
            self._inner.reset(self._scoped(key))


def _make_limiter(max_count: int, window_seconds: int = REGISTER_WINDOW, namespace: str = ""):
    """Redis-backed limiter when configured, otherwise the in-process window.

    `namespace` scopes the keys inside the backend; the module-level limiters
    below each pass one so a shared backend cannot merge their counters.
    """
    from fiximg.config import settings

    if (getattr(settings, "redis_url", "") or "").strip():
        limiter = SharedSlidingWindowLimiter(max_count, window_seconds)
    else:
        limiter = SlidingWindowLimiter(max_count, window_seconds, max_keys=MAX_MEMORY_KEYS)
    return _NamespacedLimiter(limiter, namespace) if namespace else limiter


register_ip_limiter = _make_limiter(
    _env_positive_int("FIXIMG_REGISTER_MAX", 5), namespace="register-ip:"
)
register_global_limiter = _make_limiter(
    _env_positive_int("FIXIMG_REGISTER_GLOBAL_MAX", 20), namespace="register-global:"
)
register_username_limiter = _make_limiter(
    _env_positive_int("FIXIMG_REGISTER_USERNAME_MAX", 3), namespace="register-username:"
)

# ---------------------------------------------------------------- submissions
#: Task-submission rate limit (plan §2.5's `RATE_LIMITED` code, which the API
#: documents but could not previously emit on a JSON route).
#:
#: One limiter, keyed by the authenticated principal and falling back to the client
#: address. Queue *capacity* is a different protection and is already handled by
#: `QUEUE_FULL` at enqueue time, so this one is only about one caller flooding the
#: planner and the GPU queue. `FIXIMG_SUBMIT_MAX=0` disables it.
SUBMIT_WINDOW = _env_positive_int("FIXIMG_SUBMIT_WINDOW", 60)
SUBMIT_MAX = _env_non_negative_int("FIXIMG_SUBMIT_MAX", 60)

#: A limiter that never refuses: what `FIXIMG_SUBMIT_MAX=0` selects.
#: `docs/deployment.md` documents 0 as "disables", and `max(1, ...)` turned that
#: into "one per window" — stricter than the default 60 and the opposite of the
#: instruction. `None` is the disabled state, so call sites can skip the check.


class _DisabledLimiter:
    """Stand-in used when a limit is switched off: always allows, never counts."""

    kind = "disabled"

    def __init__(self, window_seconds: int) -> None:
        self.max_count = 0
        self.window_seconds = window_seconds

    @property
    def backend_kind(self) -> str:
        return "disabled"

    def hit(self, key) -> bool:
        return True

    def allowed(self, key) -> bool:
        return True

    def remaining(self, key) -> int:
        """Unlimited: a caller must never read "disabled" as "exhausted".

        Returning 0 here meant an operator who set FIXIMG_SUBMIT_MAX=0 to turn
        the limit *off* produced a value that any `remaining()`-based gate would
        read as "no allowance left". The sentinel is deliberately large and
        documented rather than magical.
        """
        return _UNLIMITED_REMAINING

    def reset(self, key=None) -> None:
        return None


def _build_submit_limiter(submit_max: int, window: int = SUBMIT_WINDOW):
    """The submission limiter for a configured maximum; 0 means disabled.

    Split out of the module-level construction so the documented meaning of 0
    is testable without reloading the module (the env var is read once, at
    import, like every other knob here).
    """
    if int(submit_max) <= 0:
        return _DisabledLimiter(window)
    return _make_limiter(int(submit_max), window, namespace="submit:")


#: Rebuilt by whoever needs a clean allowance (test fixtures); read through the
#: module attribute so a replacement is seen by the call sites.
submit_limiter = _build_submit_limiter(SUBMIT_MAX, SUBMIT_WINDOW)


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
