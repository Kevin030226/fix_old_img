"""Health endpoints: liveness and readiness (V1 behavior preserved).

/health/ready additionally reports the DB-queue worker state (plan sections
12/28): inline worker thread presence (or standalone-worker assumption), queue
depth and stale-task requeue settings.
"""
from fastapi import APIRouter
from fastapi.responses import Response

from app.core.config import settings
from app.db import get_conn
from app.inference.model_manager import model_manager

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


def _worker_report() -> dict:
    """Best-effort view of the GPU worker and the DB-backed queue."""
    report = {
        "topology": "inline" if settings.inline_worker else "standalone",
        "poll_seconds": settings.worker_poll_seconds,
        "max_queue": settings.worker_max_queue,
    }
    try:
        from app.repositories.task_repository import queued_count

        report["queued"] = queued_count()
    except Exception:  # noqa: BLE001
        report["queued"] = None
    if settings.inline_worker:
        try:
            from app.factory import get_inline_worker

            worker = get_inline_worker()
            report["worker_running"] = worker.is_running() if worker else False
        except Exception:  # noqa: BLE001
            report["worker_running"] = False
    else:
        # With FIXIMG_INLINE_WORKER=false the worker is a separate process;
        # liveness there is observed via stale-running requeue + queue depth.
        report["worker_running"] = None
    return report


@router.get("/health/ready")
async def readiness():
    """Database readiness probe; also reports model-manager and worker state."""
    try:
        get_conn().execute("SELECT 1").fetchone()
        return {
            "status": "ready",
            "database": "ok",
            "models": model_manager.health(),
            "worker": _worker_report(),
        }
    except Exception as exc:  # noqa: BLE001
        return Response(
            content=f'{{"status":"not_ready","database":"error","detail":"{type(exc).__name__}"}}',
            status_code=503,
            media_type="application/json",
        )
