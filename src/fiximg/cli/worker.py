"""Standalone GPU worker process (plan §3.2 `cli/worker.py`).

    python worker.py                     # poll forever, one GPU task at a time
    FIXIMG_DEVICE=auto python worker.py

The API/Gradio process only enqueues (tasks table); this process claims rows via
the DB-backed queue and runs the inference runtime. Keeping GPU models out of the
API process means models load once here regardless of how many uvicorn workers
serve HTTP (plan §13).
"""
from __future__ import annotations

import os
import signal
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from fiximg.config import settings  # noqa: E402
from fiximg.infrastructure.observability.logging import get_logger, log_event  # noqa: E402

logger = get_logger("fiximg.worker.main")


def prepare_database() -> tuple[str, str]:
    """Point this process at the configured database and make sure it is usable.

    :func:`apply_database_url` is not optional here. The API reaches it through
    ``create_app()``, and the migration runner calls it explicitly, but a standalone
    worker that skipped it would open the *default* SQLite file while the API it
    serves writes to PostgreSQL: it claims nothing, blocks on an empty queue, and
    every task stays ``queued`` with both processes healthy. Measured on a real
    split-topology boot, where the worker's own readiness line named
    ``admin_data/fixoldimg.db`` although the environment named another database.

    Returns ``(kind, target)`` so the caller can log the database it really opened.
    """
    from fiximg.infrastructure.db.engine import apply_database_url, init_db, resolve_database
    from fiximg.infrastructure.db.repositories.task_repository import ensure_schema

    apply_database_url()
    init_db()
    ensure_schema()
    return resolve_database()


def main() -> int:
    from fiximg.inference.runtime import PipelineOrchestrator
    from fiximg.inference.worker import PipelineWorker

    kind, target = prepare_database()

    orchestrator = PipelineOrchestrator()
    worker = PipelineWorker(
        orchestrator,
        poll_seconds=settings.worker_poll_seconds,
        worker_id=os.environ.get("FIXIMG_WORKER_ID") or f"worker-{os.getpid()}",
    )
    worker.start()

    def _shutdown(signum, _frame):
        log_event(logger, "INFO", "worker shutting down", signal=signum)
        worker.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log_event(
        logger, "INFO", "standalone gpu worker ready",
        device=settings.device, db_kind=kind, db=target,
    )
    print(
        f"[worker] ready (device={settings.device}, db={kind}:{target}); "
        "Ctrl+C to stop.",
        flush=True,
    )

    # Keep the main thread alive; SIGINT/SIGTERM trigger the graceful stop.
    try:
        if hasattr(signal, "pause"):
            signal.pause()
        else:
            _wait_forever()
    except KeyboardInterrupt:
        worker.stop()
    return 0


def _wait_forever() -> None:
    import threading

    threading.Event().wait()


if __name__ == "__main__":
    sys.exit(main())
