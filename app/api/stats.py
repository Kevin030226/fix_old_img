"""Performance stats endpoint (plan section 30).

Aggregates from the tasks / task_stages tables: total counts, success rate and
per-stage P50/P95 durations — the data recorded by the orchestrator since
Phase 4 — without any extra instrumentation.
"""
from fastapi import APIRouter, Depends

from app.api.security import require_api_token
from app.repositories import task_repository as task_repo

router = APIRouter(
    prefix="/api/v1/stats",
    tags=["stats"],
    dependencies=[Depends(require_api_token)],
)


@router.get("")
async def stats(days: int = 7):
    """Task totals/success-rate plus per-stage P50/P95 timings.

    `days` limits the window to recent N days (0 = all time).
    Also reports plan §30 instrumentation from the ModelManager: model load
    times and (when CUDA is active) the GPU memory peak of this process.
    """
    window = days if days and days > 0 else None
    payload = task_repo.task_stats(window_days=window)
    try:
        from app.inference.model_manager import model_manager

        payload["models"] = model_manager.gpu_stats()
    except Exception:  # noqa: BLE001 — stats must never fail because of GPU probes
        payload["models"] = {"load_times_ms": {}}
    return payload
