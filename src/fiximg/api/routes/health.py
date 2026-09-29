"""Health endpoints (plan §3.8).

V2 exposed ``/health`` and ``/health/ready``. V3 keeps those (Docker healthcheck,
existing clients) and adds the canonical ``/api/v1/health/live`` and
``/api/v1/health/ready`` pair. All four stay **public** — a probe must not need
a credential, and the readiness body never contains secrets.

``ready`` additionally reports the queue worker state (plan §12/§28): inline
worker thread presence (or standalone-worker assumption), queue depth and the
lease/retry settings.
"""
from fastapi import APIRouter
from fiximg.api.schemas.diagnostics import LivenessResponse, ReadinessResponse
from fastapi.responses import JSONResponse

from fiximg.config import settings
from fiximg.infrastructure.db.engine import get_conn
from fiximg.inference.model_manager import model_manager

router = APIRouter(tags=["health"])


def _liveness() -> dict:
    return {"status": "ok"}


def _worker_report() -> dict:
    """Best-effort view of the GPU worker and the DB-backed queue."""
    report = {
        "topology": "inline" if settings.inline_worker else "standalone",
        "poll_seconds": settings.worker_poll_seconds,
        "max_queue": settings.worker_max_queue,
        "lease_seconds": settings.worker_lease_seconds,
        "max_attempts": settings.task_max_attempts,
        # §4.3 Step 4: which GPU this process is pinned to, and what it serves.
        "pinned_gpu": settings.worker_gpu or None,
        "capabilities": [
            c.strip() for c in (settings.worker_capabilities or "").split(",") if c.strip()
        ] or None,
    }
    try:
        from fiximg.infrastructure.queue import get_queue

        report["queue_backend"] = get_queue().kind
        # `kind` names the transport *family* — the value `FIXIMG_QUEUE_BACKEND`
        # takes — so it stays "sqlite" even when that queue table lives in
        # PostgreSQL. Say which server holds it too, or the probe reads as a
        # SQLite deployment on a production PostgreSQL node.
        from fiximg.infrastructure.db.dialect import default_dialect

        report["queue_driver"] = default_dialect().name
        report["queued"] = get_queue().depth()
    except Exception:  # noqa: BLE001 — a probe must never raise
        report["queued"] = None
    if settings.inline_worker:
        try:
            from fiximg.app_factory import get_inline_worker

            worker = get_inline_worker()
            report["worker_running"] = worker.is_running() if worker else False
        except Exception:  # noqa: BLE001
            report["worker_running"] = False
    else:
        # With FIXIMG_INLINE_WORKER=false the worker is another process, and this
        # one cannot ask it anything. What it *can* read is what that process writes
        # while it works: a heartbeat per running task, and the backlog it has not
        # taken. Three honest answers follow — alive while it holds work, unknown
        # while nothing needs it, and not-consuming when work waits unconsumed.
        report["worker_running"] = _standalone_liveness(report)
    return report


def _standalone_liveness(report: dict) -> bool | None:
    """Derive worker liveness for a deployment whose worker is another process."""
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    try:
        beats = task_repo.worker_heartbeats()
        oldest_queued = task_repo.oldest_queued_age_seconds()
    except Exception:  # noqa: BLE001 — a probe must never raise
        report["workers_reporting"] = 0
        report["workers_silent"] = 0
        report["oldest_queued_seconds"] = None
        report["worker_liveness"] = "unknown"
        return None

    # A worker heartbeats each loop iteration, so a beat older than a few polls
    # means the process is wedged rather than slow. Floor it because `poll` alone
    # (0.5 s) would flag a worker mid-stage-write as silent.
    window = max(3.0, 3 * float(settings.worker_poll_seconds))
    reporting = [b for b in beats if b["heartbeat_age_seconds"] is not None
                 and b["heartbeat_age_seconds"] <= window]
    silent = [b for b in beats if b not in reporting]

    report["workers_reporting"] = len(reporting)
    report["workers_silent"] = len(silent)
    report["oldest_queued_seconds"] = oldest_queued

    if reporting:
        report["worker_liveness"] = "alive"
        return True
    if silent:
        report["worker_liveness"] = "silent"
        return False
    if oldest_queued is None:
        # Nothing running and nothing waiting: the worker may be perfectly healthy
        # and idle. An earlier version of this probe had no word for that case and
        # a dashboard could not tell it from a dead fleet.
        report["worker_liveness"] = "idle-or-unknown"
        return None
    waited_too_long = oldest_queued > max(4 * window, 60.0)
    report["worker_liveness"] = "backlogged" if waited_too_long else "starting"
    return False if waited_too_long else None


def _gpu_report() -> dict:
    """GPU topology, scheduler policy and memory (plan §2.4 / §4.3 / §7.1)."""
    from fiximg.inference.gpu import GpuTopology
    from fiximg.inference.gpu_memory import memory_summary
    from fiximg.inference.gpu_utilization import describe as utilization_describe
    from fiximg.inference.scheduler import default_scheduler

    try:
        topology = GpuTopology.from_settings()
        described = topology.describe()
        # §7.1: per-device free/used VRAM plus the process high-water mark.
        described["memory"] = memory_summary(topology.devices())
        # §7.1's other KPI: was the card working? Read from the shared background
        # probe so an HTTP scrape never forks the driver tool itself.
        described["utilization"] = utilization_describe(topology.devices())
        described["memory_aware"] = settings.gpu_memory_aware
        described["headroom_mb"] = settings.gpu_memory_headroom_mb
    except Exception as exc:  # noqa: BLE001 — a probe must never raise
        described = {"error": type(exc).__name__}
    try:
        scheduler = default_scheduler.stats()
    except Exception:  # noqa: BLE001
        scheduler = {}
    return {"topology": described, "scheduler": scheduler}


def _plugin_report() -> dict:
    """Third-party plugins discovered at startup (plan §3.14.1)."""
    from fiximg.inference import plugins as plugins_module

    discovery = plugins_module.last_discovery()
    if discovery is None:
        return {"discovery": "not-run", "enabled": plugins_module.plugins_enabled()}
    report = discovery.to_dict()
    report["enabled"] = plugins_module.plugins_enabled()
    return report


def _readiness() -> dict:
    """Readiness body, or a 503 payload when a dependency is down."""
    try:
        get_conn().execute("SELECT 1").fetchone()
    except Exception as exc:  # noqa: BLE001 — report, never raise
        return {
            "_status_code": 503,
            "status": "not_ready",
            "database": "error",
            "detail": type(exc).__name__,
        }

    from fiximg.application.model_service import model_service

    return {
        "status": "ready",
        "database": "ok",
        "models": model_manager.health(),
        "model_health": model_service.health()["backends"],
        "storage_backend": settings.storage_backend,
        "worker": _worker_report(),
        "gpu": _gpu_report(),
        "plugins": _plugin_report(),
    }


@router.get("/health", response_model=LivenessResponse, summary="Liveness")
async def health():
    """Liveness alias kept for V2 deployments and container healthchecks."""
    return _liveness()


@router.get("/health/ready", response_model=ReadinessResponse, summary="Readiness")
async def readiness():
    """Readiness alias kept for V2 deployments."""
    payload = _readiness()
    status_code = payload.pop("_status_code", 200)
    return JSONResponse(content=payload, status_code=status_code)


@router.get("/api/v1/health/live", response_model=LivenessResponse, summary="Liveness")
async def live():
    """Canonical liveness probe (plan §3.8): the process is up."""
    return _liveness()


@router.get("/api/v1/health/ready", response_model=ReadinessResponse,
        summary="Readiness")
async def ready():
    """Canonical readiness probe (plan §3.8): dependencies are usable."""
    payload = _readiness()
    status_code = payload.pop("_status_code", 200)
    return JSONResponse(content=payload, status_code=status_code)
