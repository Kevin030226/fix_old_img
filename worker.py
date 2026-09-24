"""Standalone GPU worker process (plan sections 12 and 28).

    python worker.py                     # poll forever, one GPU task at a time
    FIXIMG_DEVICE=auto python worker.py

The API/Gradio process only enqueues (tasks table); this process claims rows
via the DB-backed queue and runs the PipelineOrchestrator. Keeping GPU models
out of the API process means models load once here regardless of how many
uvicorn workers serve HTTP (plan section 13).
"""
import os
import signal
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from app.core.config import settings  # noqa: E402
from app.core.logging import get_logger, log_event  # noqa: E402

logger = get_logger("fiximg.worker.main")


def main() -> int:
    from app.db import init_db
    from app.inference.orchestrator import PipelineOrchestrator
    from app.inference.worker import PipelineWorker
    from app.repositories.task_repository import ensure_schema

    init_db()
    ensure_schema()

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
        device=settings.device, db=settings.db_path,
    )
    print(
        f"[worker] ready (device={settings.device}, db={settings.db_path}); "
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
