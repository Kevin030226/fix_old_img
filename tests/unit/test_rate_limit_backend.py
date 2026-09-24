"""Unit tests for the shared rate-limit backend (plan §22, P3)."""
import pytest

from app.services import rate_limit_backend as rlb


@pytest.fixture(autouse=True)
def memory_backend(monkeypatch):
    """Tests run on the deterministic memory backend (no redis server)."""
    monkeypatch.setattr(rlb, "_BACKEND", None, raising=False)
    monkeypatch.setattr(rlb, "_BACKEND_KIND", None, raising=False)
    from app.core.config import settings

    monkeypatch.setattr(settings, "redis_url", "", raising=False)
    yield
    monkeypatch.setattr(rlb, "_BACKEND", None, raising=False)
    monkeypatch.setattr(rlb, "_BACKEND_KIND", None, raising=False)


def test_memory_backend_sliding_window():
    limiter = rlb.make_limiter(2, 60)
    assert limiter.hit("k") is True
    assert limiter.hit("k") is True
    assert limiter.hit("k") is False
    assert limiter.remaining("k") == 0
    limiter.reset("k")
    assert limiter.hit("k") is True


def test_backend_kind_is_memory_without_redis():
    assert rlb.get_backend().kind == "memory"
    assert rlb.make_limiter(1, 10).backend_kind == "memory"


def test_redis_url_without_package_falls_back(monkeypatch):
    """A configured Redis that cannot be reached must not break limiting."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:1/0", raising=False)
    kind = rlb.reset_backend()
    assert kind == "memory"
    limiter = rlb.make_limiter(1, 10)
    assert limiter.hit("x") is True
    assert limiter.hit("x") is False


def test_redis_import_error_degrades(monkeypatch):
    """Simulate a machine without the redis package installed."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "redis_url", "redis://redis:6379/0", raising=False)

    import builtins

    real_import = builtins.__import__

    def _no_redis(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("No module named 'redis'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_redis)
    assert rlb.reset_backend() == "memory"


def test_backend_flip_to_memory_after_repeated_failures():
    class _BrokenRedis:
        def ping(self):
            return True

        def pipeline(self):
            raise ConnectionError("down")

    backend = rlb._RedisBackend(_BrokenRedis())
    # fail-open while degraded...
    assert backend.hit("k", 1, 60) is True
    assert backend.hit("k", 1, 60) is True
    # ...and after 3 failures the process-wide backend flips back to memory
    assert rlb._BACKEND_KIND == "memory" or rlb.get_backend().kind == "memory"


def test_constructor_validates_limits():
    with pytest.raises(ValueError):
        rlb.make_limiter(0, 60)
    with pytest.raises(ValueError):
        rlb.make_limiter(5, 0)
