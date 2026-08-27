"""In-memory sliding-window rate limiter (for single-process demos).

Goal: provide lightweight, dependency-free rate limiting for public-entry endpoints such as registration.
- Sliding-window semantics: requests exceeding the cap within the window are rejected; expired counts are evicted automatically.
- The clock is injectable (now=) for unit tests, avoiding real-time dependence.

⚠️ Limitation: in-process memory counters reset on restart and are not shared across workers/instances. Use shared storage such as Redis for cluster-wide consistency in production. The current deployment is a single process (one uvicorn instance, one worker), so in-memory limiting is sufficient.
"""
import os
import time
from collections import deque


class RateLimitExceeded(Exception):
    """Semantic placeholder exception.

    By default, callers return 429 directly when hit() returns False; this exception is kept for a future exception-based style.
    """

    pass


class SlidingWindowLimiter:
    """Sliding-window rate limiter: maintains a timestamp deque per key."""

    def __init__(self, max_count, window_seconds, now=None):
        if max_count < 1:
            raise ValueError("max_count must be >= 1")
        if window_seconds < 1:
            raise ValueError("window_seconds must be >= 1")
        self.max_count = max_count
        self.window_seconds = window_seconds
        self._hits = {}  # key -> deque[float]
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
        """Only check whether the request is allowed, without recording it."""
        now_ts = self._now_ts()
        self._purge(key, now_ts)
        return len(self._hits.get(key, deque())) < self.max_count

    def hit(self, key):
        """Record one request: returns True if allowed, False if over the limit."""
        now_ts = self._now_ts()
        self._purge(key, now_ts)
        dq = self._hits.setdefault(key, deque())
        if len(dq) >= self.max_count:
            return False
        dq.append(now_ts)
        return True

    def remaining(self, key):
        """Return the remaining allowance within the window."""
        now_ts = self._now_ts()
        self._purge(key, now_ts)
        return max(0, self.max_count - len(self._hits.get(key, deque())))

    def reset(self, key=None):
        if key is None:
            self._hits.clear()
        else:
            self._hits.pop(key, None)


def client_ip(request):
    """Get the client IP; forward headers are trusted only when trusted proxies (FIXIMG_TRUSTED_PROXIES) are explicitly configured."""
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


# Tunable via environment variables; defaults target single-process demo + intranet scenarios.
REGISTER_WINDOW = int(os.environ.get("FIXIMG_REGISTER_WINDOW", 600))

register_ip_limiter = SlidingWindowLimiter(
    int(os.environ.get("FIXIMG_REGISTER_MAX", 5)), REGISTER_WINDOW
)
register_global_limiter = SlidingWindowLimiter(
    int(os.environ.get("FIXIMG_REGISTER_GLOBAL_MAX", 20)), REGISTER_WINDOW
)
register_username_limiter = SlidingWindowLimiter(
    int(os.environ.get("FIXIMG_REGISTER_USERNAME_MAX", 3)), REGISTER_WINDOW
)
