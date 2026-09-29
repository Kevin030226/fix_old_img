"""Performance stats endpoint (plan §2.10 / §30).

Aggregates from the tasks / task_stages tables — total counts, success rate and
per-stage P50/P95 durations recorded by the runtime — and adds the in-process
metric registry (queue depth/wait, task and stage durations, model load and
inference time, retries).
"""
from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse

from fiximg.api.schemas.diagnostics import RecentEventView, RecentEventsResponse, StatsResponse
from fiximg.api.security import require_api_token
from fiximg.application.task_service import task_service
from fiximg.domain.events import TaskEvent
from fiximg.infrastructure.db.repositories import task_repository as task_repo

router = APIRouter(
    prefix="/api/v1/stats",
    tags=["stats"],
    dependencies=[Depends(require_api_token)],
)


@router.get("", response_model=StatsResponse, summary="Performance stats")
async def stats(days: int = Query(default=7, ge=0, le=365)):
    """Task totals/success-rate, per-stage timings and live metrics.

    ``days`` limits the DB window to the recent N days (0 = all time).
    """
    window = days if days and days > 0 else None
    payload = task_service.stats_snapshot(window_days=window)

    try:
        from fiximg.inference.model_manager import model_manager

        payload["models"] = model_manager.gpu_stats()
    except Exception:  # noqa: BLE001 — stats must never fail because of GPU probes
        payload["models"] = {"load_times_ms": {}}

    # The peak a *worker* process reached is not visible in this process's CUDA
    # counters — an API node that never runs a stage reports 0.0 beside stage rows
    # saying 334.8 MB, which is the contradiction a reader cannot resolve. Fold in
    # the largest persisted stage figure so the endpoint answers for the deployment.
    try:
        from fiximg.inference.gpu_memory import STAGE_PEAK_METRIC

        recorded = task_repo.max_metric(STAGE_PEAK_METRIC)
        if recorded is not None:
            local = payload["models"].get("gpu_memory_peak_mb") or 0.0
            payload["models"]["gpu_memory_peak_mb"] = max(float(local), recorded)
    except Exception:  # noqa: BLE001
        pass

    # §2.10: in-process counters/histograms (uptime, queue wait, retries...).
    try:
        from fiximg.infrastructure.observability.metrics import metrics

        payload["metrics"] = metrics.snapshot()
    except Exception:  # noqa: BLE001
        payload["metrics"] = {}

    # §2.4: current GPU concurrency policy and in-flight slots.
    try:
        from fiximg.inference.scheduler import default_scheduler

        payload["scheduler"] = default_scheduler.stats()
    except Exception:  # noqa: BLE001
        payload["scheduler"] = {}

    return payload


@router.get("/metrics", summary="Prometheus text exposition",
            response_class=PlainTextResponse,
            responses={200: {"description": "Prometheus text format",
                             "content": {"text/plain": {}}}})
async def prometheus_metrics():
    """Metrics in Prometheus text format (no extra dependency).

    Two scopes, both labelled with ``scope=``:

    * ``scope="process"`` — what *this* process recorded. On an API node that is
      `task_submit_total` and `queue_depth`; the worker records the rest, in its
      own memory.
    * ``scope="deployment"`` — aggregated from the rows every process writes, so
      the stage durations, queue wait, retries, VRAM and utilisation a Prometheus
      scraper needs are answerable for the deployment rather than for whichever
      replica answered the scrape. Before this existed the endpoint exported two of
      the eleven declared names on the documented ``api`` + ``worker`` topology,
      while `docs/deployment.md` lists `gpu_utilization_percent{stage,device}`
      among its series.

    `model_inference_seconds` and `artifact_io_seconds` are not persisted, so they
    stay process-local; the output says so in a comment rather than leaving a
    reader to conclude they were never measured.

    ``response_class`` is declared because the handler returns a raw response:
    without it the OpenAPI document offered this endpoint's body as
    `application/json`, which is what the diagnostics schema gate then caught.
    """
    from fiximg.infrastructure.observability.metrics import metrics

    try:
        deployment = task_repo.deployment_metrics()
    except Exception:  # noqa: BLE001 - observability must not fail the scrape
        deployment = None

    return PlainTextResponse(
        metrics.render_prometheus(deployment), media_type="text/plain"
    )


@router.get(
    "/events",
    response_model=RecentEventsResponse,
    summary="Recent deployment events",
)
async def recent_events(
    limit: int = Query(default=50, ge=1, le=500),
    level: str | None = Query(default=None, description="Filter by level, e.g. error"),
):
    """The most recent events across every task, newest first (plan §2.10).

    ``GET /tasks/{id}/events`` streams one task's lifecycle, so a rejected
    submission (no task row to attach to), a worker restart or a queue that is
    refusing work are all invisible through it. This is the deployment-wide view
    an operator opens when a task is missing rather than slow.
    """
    rows = task_repo.list_events(limit=limit, level=level)
    events = []
    for row in rows:
        event = TaskEvent.from_row(row)
        events.append(
            RecentEventView(
                seq=event.seq,
                event=event.event_type or "message",
                level=event.level,
                task_id=event.task_id,
                user_id=event.user_id,
                message=event.message,
                data=event.data,
                created_at=event.created_at,
            )
        )
    return RecentEventsResponse(events=events)
