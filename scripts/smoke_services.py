"""Connectivity smoke test for the external services (plan §3.13, §4.3).

The gap this closes is narrow but real: the Redis / PostgreSQL / MinIO adapters
have complete interfaces and (for Redis and S3) fake-client contract tests, but
nothing ever talked to a *real* server. This script does, so an operator can
verify a deployment in one command instead of discovering a misconfiguration in
production.

It only checks connectivity and the handful of operations each adapter needs —
it does not run a full task.

Usage::

    docker compose -f docker/compose.yaml --profile platform up -d
    python scripts/smoke_services.py

Exit codes: 0 = everything configured is reachable, 1 = at least one failure.
A service that is not configured is reported as SKIP, not as a failure.
"""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

#: Seconds a service probe may spend connecting. This script exists to answer
#: "is the deployment reachable?" — so it must never be the thing that hangs.
PROBE_CONNECT_TIMEOUT = 5


class Result:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.rows.append((name, status, detail))

    @property
    def failed(self) -> bool:
        return any(status == FAIL for _n, status, _d in self.rows)

    def report(self) -> None:
        width = max(len(name) for name, _s, _d in self.rows) if self.rows else 0
        for name, status, detail in self.rows:
            print(f"{name:<{width}}  {status:<4}  {detail}")


def check_database(result: Result) -> None:
    """SQLite is always checked; a non-SQLite URL is reported as unsupported."""
    url = (os.environ.get("FIXIMG_DATABASE_URL") or "").strip()
    if not url or url.startswith("sqlite"):
        from fiximg.infrastructure.db.engine import apply_database_url, init_db, get_conn

        try:
            apply_database_url()
            init_db()
            get_conn().execute("SELECT 1").fetchone()
            result.add("database (sqlite)", PASS, "schema ready")
        except Exception as exc:  # noqa: BLE001
            result.add("database (sqlite)", FAIL, f"{type(exc).__name__}: {exc}")
        return

    from fiximg.infrastructure.db.engine import UnsupportedDatabaseError, resolve_database

    try:
        kind, target = resolve_database(url)
    except UnsupportedDatabaseError as exc:
        result.add("database (url)", FAIL, str(exc).splitlines()[0])
        return

    if kind != "postgresql":
        result.add("database (url)", FAIL, f"unexpected scheme resolved to {kind}")
        return

    # Open a real connection and run a trivial query: this proves the driver,
    # the credentials and the network path, not just that the URL parses. The
    # timeout is the point of the keyword — an operator running a smoke test
    # needs an answer, and psycopg waits forever without it (see
    # connection.DEFAULT_CONNECT_TIMEOUT).
    try:
        from fiximg.infrastructure.db.connection import open_postgres

        connection = open_postgres(target, connect_timeout=PROBE_CONNECT_TIMEOUT)
        try:
            connection.execute("SELECT 1").fetchone()
        finally:
            connection.close()
        result.add("database (postgresql)", PASS, "connection + SELECT 1")
    except Exception as exc:  # noqa: BLE001
        result.add("database (postgresql)", FAIL, f"{type(exc).__name__}: {exc}")


def check_redis(result: Result) -> None:
    url = (os.environ.get("FIXIMG_REDIS_URL") or "").strip()
    if not url:
        result.add("redis", SKIP, "FIXIMG_REDIS_URL not set")
        return
    try:
        import redis
    except ImportError:
        result.add("redis", FAIL, "redis package missing (pip install 'fiximg[redis]')")
        return
    try:
        client = redis.Redis.from_url(url, socket_timeout=2.0, socket_connect_timeout=2.0)
        client.ping()
        from fiximg.infrastructure.queue.redis import GROUP_NAME, RedisQueueBackend, STREAM_KEY

        backend = RedisQueueBackend(client)
        # Exercise the real commands the worker uses, including XAUTOCLAIM.
        backend.enqueue("smoke-test-task")
        depth = backend.depth()
        backend.claim("smoke-test-worker", lease_seconds=1)
        result.add(
            "redis", PASS,
            f"ping ok, stream={STREAM_KEY}, group={GROUP_NAME}, depth={depth}",
        )
    except Exception as exc:  # noqa: BLE001
        result.add("redis", FAIL, f"{type(exc).__name__}: {exc}")


def check_object_storage(result: Result) -> None:
    from fiximg.config import settings

    backend = getattr(settings, "storage_backend", "local")
    if backend != "s3":
        result.add("object storage", SKIP, f"FIXIMG_STORAGE_BACKEND={backend}")
        return
    try:
        from fiximg.infrastructure.storage.s3 import build_s3_store_from_settings

        store = build_s3_store_from_settings(settings)
        if store is None:
            result.add("object storage", FAIL, "s3 configured but the store could not be built")
            return
        # A round trip proves the credentials, the bucket and the prefix.
        ref = store.put_bytes(b"ok", "smoke/probe.txt", "text/plain")
        payload = store.get_bytes(ref)
        store.delete(ref)
        result.add("object storage", PASS if payload == b"ok" else FAIL,
                   f"bucket={getattr(settings, 'storage_bucket', '?')} round-trip")
    except ImportError as exc:
        result.add("object storage", FAIL, f"boto3 missing: {exc}")
    except Exception as exc:  # noqa: BLE001
        result.add("object storage", FAIL, f"{type(exc).__name__}: {exc}")


def check_worker_queue(result: Result) -> None:
    """The configured queue backend must be constructible."""
    try:
        from fiximg.config import settings
        from fiximg.infrastructure.db.dialect import default_dialect
        from fiximg.infrastructure.queue import get_queue, reset_queue

        reset_queue()
        queue = get_queue()
        # `kind` names the *implementation family*, not the engine: the SQL queue
        # is still called "sqlite" when `FIXIMG_DATABASE_URL` points at
        # PostgreSQL, which reads like a contradiction unless the driver is shown.
        result.add(
            "queue backend", PASS,
            f"kind={queue.kind}, driver={default_dialect().name}, depth={queue.depth()}",
        )
        result.add("queue setting", PASS if queue.kind == getattr(settings, "queue_backend", "sqlite")
                   else FAIL, f"FIXIMG_QUEUE_BACKEND={getattr(settings, 'queue_backend', '?')}")
    except Exception as exc:  # noqa: BLE001
        result.add("queue backend", FAIL, f"{type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    result = Result()
    check_database(result)
    check_redis(result)
    check_object_storage(result)
    check_worker_queue(result)
    result.report()

    if result.failed:
        print("\nAt least one configured service is unreachable.")
        return 1
    print("\nEvery configured service is reachable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
