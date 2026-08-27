"""内存滑动窗口限流器（单进程演示用）。

设计目标：为公开注册等入口提供轻量、无外部依赖的限流能力。
- 采用滑动窗口语义：窗口内请求数超过上限即拒绝，过期计数自动淘汰。
- 时钟可注入（now=），便于单元测试，避免依赖真实时间。

⚠️ 局限：本实现为进程内内存计数，重启即清零，且多 worker / 多实例间
不共享。生产环境应改用 Redis 等共享存储，使限流在集群范围内一致。
当前部署为单进程（uvicorn 单实例 + 单并发），内存限流足够。
"""
import os
import time
from collections import deque


class RateLimitExceeded(Exception):
    """语义占位异常。

    本模块默认由调用方在 hit() 返回 False 时直接返回 429，不主动抛出；
    保留该异常以便未来需要“抛异常”风格时使用。
    """

    pass


class SlidingWindowLimiter:
    """滑动窗口限流器：维护每个 key 的时间戳队列。"""

    def __init__(self, max_count, window_seconds, now=None):
        if max_count < 1:
            raise ValueError("max_count 必须 >= 1")
        if window_seconds < 1:
            raise ValueError("window_seconds 必须 >= 1")
        self.max_count = max_count
        self.window_seconds = window_seconds
        self._hits = {}  # key -> deque[float]
        self._now = now  # 可注入时钟，便于测试

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
        """仅检查是否被允许，不记录本次请求。"""
        now_ts = self._now_ts()
        self._purge(key, now_ts)
        return len(self._hits.get(key, deque())) < self.max_count

    def hit(self, key):
        """记录一次请求：允许返回 True，超限返回 False。"""
        now_ts = self._now_ts()
        self._purge(key, now_ts)
        dq = self._hits.setdefault(key, deque())
        if len(dq) >= self.max_count:
            return False
        dq.append(now_ts)
        return True

    def remaining(self, key):
        """返回窗口内剩余可用次数。"""
        now_ts = self._now_ts()
        self._purge(key, now_ts)
        return max(0, self.max_count - len(self._hits.get(key, deque())))

    def reset(self, key=None):
        if key is None:
            self._hits.clear()
        else:
            self._hits.pop(key, None)


def client_ip(request):
    """获取客户端 IP；仅在显式配置可信代理（FIXIMG_TRUSTED_PROXIES）时信任转发头。"""
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


# ===================== 注册接口的限流器实例 =====================
# 可通过环境变量调参；默认值面向“单进程演示 + 内网”场景。
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
