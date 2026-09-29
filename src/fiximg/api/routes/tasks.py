"""Task API router (plan §3.8).

    POST   /api/v1/tasks                    create a queued task (multipart upload)
    GET    /api/v1/tasks                    list tasks (paginated)
    GET    /api/v1/tasks/{task_id}          status / progress / current_stage
    GET    /api/v1/tasks/{task_id}/events   live task events (SSE)
    GET    /api/v1/tasks/{task_id}/artifacts
    GET    /api/v1/tasks/{task_id}/report
    GET    /api/v1/tasks/{task_id}/result   the finished image
    POST   /api/v1/tasks/{task_id}/cancel
    POST   /api/v1/tasks/{task_id}/retry

Every response is a declared Pydantic model, so the OpenAPI document is
complete and a renamed field breaks the build instead of a client (§2.5).
"""
import asyncio
import io
import json
import os

from fastapi import APIRouter, Depends, File, Form, Header, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from fiximg.api.dependencies import Principal, require_principal
from fiximg.infrastructure.db import timestamps
from fiximg.api.schemas.common import Page, Pagination
from fiximg.api.security import require_api_token
from fiximg.api.schemas.task import (
    ArtifactView,
    StageView,
    TaskArtifactsResponse,
    TaskCreateForm,
    TaskCreateResponse,
    TaskEventSchema,
    TaskOptionsSchema,
    TaskReportResponse,
    TaskStatusResponse,
)
from fiximg.application.pipeline_modes import PIPELINE_MODES
from fiximg.application.task_service import task_service
from fiximg.config import settings
from fiximg.domain.enums import EventType, TaskStatus
from fiximg.domain.events import TaskEvent
from fiximg.domain.tasks import decode_json_column
from fiximg.domain.errors import (
    ImageTooLargeError,
    InvalidImageError,
    InvalidOptionsError,
    InvalidRequestError,
    RateLimitedError,
    TaskNotCancellableError,
    UnsupportedTaskTypeError,
    UploadTooLargeError,
)
from fiximg.infrastructure.security import rate_limit

# Every task route requires the API bearer token (fiximg/api/security.py): the
# Gradio UI uses the service layer in-process and is unaffected. The create
# route additionally resolves the acting Principal (plan §2.5 point 3).
router = APIRouter(
    prefix="/api/v1/tasks",
    tags=["tasks"],
    dependencies=[Depends(require_api_token)],
)

_ALLOWED_TYPES = {cfg["task_type"] for cfg in PIPELINE_MODES.values()}
_MAX_UPLOAD_BYTES = settings.max_upload_mb * 1024 * 1024

#: How long an SSE stream may stay open without a closing event (plan §3.8).
#: Which events close it is decided by the domain
#: (:data:`fiximg.domain.events.STREAM_CLOSING_EVENTS`), not by a list here, so a
#: new closing event cannot be added in one place and missed in the other.
_SSE_TIMEOUT_SECONDS = 900.0
_SSE_POLL_SECONDS = 0.5

#: With ``FIXIMG_EVAL_MODE=async`` the metrics arrive *after* ``task.completed``
#: (plan §3.16), so the stream waits this long for ``task.evaluated`` before
#: closing — a bounded grace period, so a failed evaluator cannot pin a client.
_SSE_POST_COMPLETION_GRACE = 60.0


# --------------------------------------------------------------------- helpers
def _decode_image(payload: bytes, what: str = "image"):
    from PIL import Image as PILImage

    if not payload:
        raise InvalidImageError(f"Empty {what} upload")
    if len(payload) > _MAX_UPLOAD_BYTES:
        raise UploadTooLargeError(
            f"{what.capitalize()} exceeds the {settings.max_upload_mb} MB upload limit"
        )
    try:
        return PILImage.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise InvalidImageError(f"Invalid {what} file") from exc


def _parse_options(raw: str | None) -> TaskOptionsSchema:
    try:
        return TaskOptionsSchema.parse(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise InvalidOptionsError(str(exc)) from exc


def _stage_views(raw) -> list[StageView]:
    out = []
    for item in raw or []:
        if isinstance(item, dict):
            out.append(
                StageView(
                    stage_name=item.get("stage_name", ""),
                    stage_order=item.get("stage_order"),
                    status=item.get("status") or "pending",
                    duration_ms=item.get("duration_ms"),
                    message=item.get("message"),
                    stage_version=item.get("stage_version"),
                    metrics=decode_json_column(item.get("metrics_json"), {}),
                )
            )
    return out


def _artifact_location(row: dict) -> str:
    """Where a client is told the bytes live: a store key, never a server path.

    The stored ``path`` is absolute (``<tasks_root>/<year>/<month>/<task>/...``)
    because that is what ``GET /result`` opens. Putting it on the wire published the
    deployment's filesystem layout — drive letter, user directory, custom
    ``tasks_root`` — to every token holder, and no API client can use it: an API
    client downloads through ``/result``, it does not open the server's disk. A path
    that is not under ``tasks_root`` (a V2 row migrated in place) is reduced to its
    file name, which identifies the artifact without describing the host.
    """
    key = row.get("uri")
    if key:
        return str(key)
    path = str(row.get("path") or "")
    if not path:
        return ""
    try:
        relative = os.path.relpath(path, settings.tasks_root)
    except ValueError:  # different drive on Windows
        return os.path.basename(path)
    relative = relative if not relative.startswith("..") else os.path.basename(path)
    # Always forward slashes: the value is a key-shaped reference, and a client on
    # another platform must be able to read a Windows deployment's without seeing
    # separators it cannot use.
    return relative.replace("\\", "/")


def _artifact_views(rows) -> list[ArtifactView]:
    return [
        ArtifactView(
            kind=r.get("kind", ""),
            uri=_artifact_location(r),
            mime_type=r.get("mime_type"),
            width=r.get("width"),
            height=r.get("height"),
            size_bytes=r.get("size_bytes"),
            sha256=r.get("sha256"),
            model_version=r.get("model_version"),
        )
        for r in rows or []
    ]


def _status_from_row(row: dict) -> TaskStatusResponse:
    stages = _stage_views(decode_json_column(row.get("stages"), []))
    metrics = decode_json_column(row.get("metrics"), {})
    return TaskStatusResponse(
        task_id=row["id"],
        status=TaskStatus(row["status"]),
        progress=int(row.get("progress") or 0),
        current_stage=row.get("current_stage"),
        error_message=row.get("error_message"),
        error_code=row.get("error_code"),
        duration_ms=row.get("duration_ms"),
        attempt_count=row.get("attempt_count"),
        max_attempts=row.get("max_attempts"),
        priority=row.get("priority"),
        stages=stages,
        metrics=metrics if isinstance(metrics, dict) else {},
    )


# -------------------------------------------------------------------- create
@router.post(
    "",
    response_model=TaskCreateResponse,
    status_code=202,
    summary="Queue a processing task",
    # Multipart cannot be expressed as a JSON body, so the documented shape is
    # injected explicitly (plan §2.5 point 2) — without this the OpenAPI
    # document would show the create endpoint with no request schema at all.
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {"schema": TaskCreateForm.model_json_schema()}
            },
        }
    },
)
async def create_task(
    request: Request,
    type: str = Form(..., description="restore | restore_scratch | detect_scratch | colorize | auto_restore"),
    image: UploadFile = File(...),
    options: str = Form(None),
    priority: int = Form(
        0,
        description=f"0–{settings.priority_max}; higher is claimed first within the queue",
    ),
    ground_truth: UploadFile = File(None),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    principal: Principal = Depends(require_principal),
):
    """Queue a task; the GPU worker picks it up asynchronously (plan §3.8).

    Fields: ``type`` (see :class:`~fiximg.api.schemas.task.TaskCreateForm`),
    ``image`` (multipart), optional ``options`` JSON object, optional
    ``priority``, optional ``ground_truth`` reference photo, and an optional
    ``Idempotency-Key`` header making the call safely retryable (plan §2.6).

    Submissions are rate limited per principal (``FIXIMG_SUBMIT_MAX`` per
    ``FIXIMG_SUBMIT_WINDOW``); the reply is the canonical envelope with
    ``RATE_LIMITED``, and it happens **before** the image is decoded, because the
    decode is the part a flood is paying for.
    """
    key, scope = rate_limit.submission_key(principal, request)
    limiter = rate_limit.submit_limiter
    if not limiter.hit(key):
        # Read the limits off the limiter that just decided, not off the module
        # constants: they can disagree (a rebuilt limiter, `FIXIMG_SUBMIT_MAX=0`
        # clamped to 1), and a `details.limit` the caller cannot reconcile with the
        # behaviour they are seeing is worse than no number at all.
        raise RateLimitedError(
            f"Too many task submissions ({limiter.max_count} per "
            f"{limiter.window_seconds}s); try again later",
            details={
                "limit": limiter.max_count,
                "window_seconds": limiter.window_seconds,
                "scope": scope,
                "remaining": limiter.remaining(key),
            },
        )
    if type not in _ALLOWED_TYPES:
        raise UnsupportedTaskTypeError(
            f"Unknown task type: {type}",
            details={"allowed": sorted(_ALLOWED_TYPES)},
        )
    # `priority` decides who gets a GPU next, so it is checked against the limit the
    # deployment is actually running (`FIXIMG_PRIORITY_MAX`, read here rather than at
    # import time so a rebuilt settings object cannot make the reply describe a limit
    # that no longer exists) and refused rather than clamped: a caller that asked for
    # 20 and quietly got 10 would believe a queue position it did not receive.
    if not 0 <= priority <= settings.priority_max:
        raise InvalidRequestError(
            f"priority must be between 0 and {settings.priority_max}",
            details={"priority": priority, "min": 0, "max": settings.priority_max},
        )
    task_options = _parse_options(options)

    pil_image = _decode_image(await image.read())
    if max(pil_image.size) > settings.max_image_side:
        raise ImageTooLargeError(
            f"Image long side exceeds {settings.max_image_side} px",
            details={"max_side": settings.max_image_side, "size": list(pil_image.size)},
        )

    # §16 Ground Truth mode: optional reference photo enables LPIPS evaluation.
    gt_image = None
    if ground_truth is not None and ground_truth.filename:
        gt_image = _decode_image(await ground_truth.read(), "ground truth")

    # §2.5 point 3: the task is attributed to the authenticated principal, not
    # to a hard-coded placeholder account.
    explicit_options = task_options.model_dump(exclude_defaults=True)
    task_id = task_service.enqueue(
        pil_image,
        {"username": principal.user_id},
        type,
        options=explicit_options or None,
        ground_truth_image=gt_image,
        idempotency_key=idempotency_key,
        priority=priority,
    )
    row_priority = task_service.task_priority(task_id)
    return TaskCreateResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        # The same instant convention the row is stored with (plan §2.7), so a
        # client that compares this against `GET /tasks/{id}` sees one clock.
        created_at=timestamps.now(),
        #: Read from the row rather than echoed from the form: what the queue will
        #: actually order by is the stored value, and a submission that lost it on the
        #: way in (an idempotent replay returns the earlier task, whose priority is its
        #: own) has to be told the truth about the task it got. `None` is kept as
        #: `null` rather than defaulted to 0, because "this build could not read it back"
        #: is not the same statement as "the task is FIFO".
        priority=row_priority,
    )


# ---------------------------------------------------------------------- read
@router.get("", response_model=Page[TaskStatusResponse], summary="List tasks")
async def list_tasks(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    user_id: str | None = Query(default=None),
):
    """Paginated task list, newest first."""
    rows = task_service.list_tasks(limit=limit + offset, user_id=user_id)
    window = rows[offset: offset + limit]
    return Page[TaskStatusResponse](
        items=[_status_from_row(row) for row in window],
        pagination=Pagination(limit=limit, offset=offset, total=None),
    )


@router.get("/{task_id}", response_model=TaskStatusResponse, summary="Task status")
async def get_task(task_id: str):
    return _status_from_row(task_service.require_task(task_id))


@router.get(
    "/{task_id}/artifacts",
    response_model=TaskArtifactsResponse,
    summary="Task artifacts",
)
async def get_task_artifacts(task_id: str):
    task_service.require_task(task_id)
    return TaskArtifactsResponse(
        task_id=task_id,
        artifacts=_artifact_views(task_service.get_task_artifacts(task_id)),
    )


@router.get("/{task_id}/report", response_model=TaskReportResponse, summary="Task report")
async def get_report(task_id: str):
    row = task_service.require_task(task_id)
    stages = _stage_views(decode_json_column(row.pop("stages", None), []))
    metrics = decode_json_column(row.pop("metrics", None), {})

    # §10 auto_restore: surface the analyzer-driven planner decisions. Read through
    # the service, because the web UI shows the same block and two copies of
    # "where the run wrote its decisions" drift.
    planner_decisions = task_service.planner_decisions(row.get("result_path"))

    return TaskReportResponse(
        task_id=row["id"],
        task_type=row["task_type"],
        status=TaskStatus(row["status"]),
        duration_ms=row.get("duration_ms"),
        evaluation_text=row.get("evaluation_text"),
        stages=stages,
        metrics=metrics if isinstance(metrics, dict) else {},
        planner_decisions=planner_decisions,
        artifacts=_artifact_views(task_service.get_task_artifacts(task_id)),
    )


@router.get(
    "/{task_id}/result",
    summary="Download the finished image",
    # The body is a PNG, so it has no JSON model to name. Declaring the media type is
    # what keeps the OpenAPI document honest — without it FastAPI advertises
    # `application/json` for a route that answers with a FileResponse.
    response_class=FileResponse,
    responses={200: {"description": "The restored image",
                     "content": {"image/png": {"schema": {"type": "string",
                                                          "format": "binary"}}}},
               410: {"description": "Result reclaimed by the retention TTL"}},
)
async def get_result(task_id: str):
    path = task_service.task_result_path(task_id)
    return FileResponse(path, media_type="image/png", filename=os.path.basename(path))


# ------------------------------------------------------------------ lifecycle
@router.post("/{task_id}/cancel", response_model=TaskStatusResponse, summary="Cancel a task")
async def cancel_task(task_id: str):
    task_service.require_task(task_id)
    if not task_service.cancel_task(task_id):
        raise TaskNotCancellableError(
            "Task not cancellable (missing, or already finished)",
            details={"task_id": task_id},
        )
    return _status_from_row(task_service.require_task(task_id))


@router.post("/{task_id}/retry", response_model=TaskStatusResponse, summary="Retry a task")
async def retry_task(task_id: str):
    task_service.retry_task(task_id)
    return _status_from_row(task_service.require_task(task_id))


# ------------------------------------------------------------------------ SSE
@router.get(
    "/{task_id}/events",
    summary="Stream task events (SSE)",
    # Same reason as `/result`: the payload is an event stream, not a model. Each
    # `data:` line is a `TaskEvent` object, documented by name here.
    response_class=StreamingResponse,
    responses={200: {"description": "Server-Sent Events; each data line is a TaskEvent",
                     "content": {"text/event-stream": {}}}},
)
async def stream_events(
    task_id: str,
    last_event_id: int = Query(default=0, alias="last_event_id"),
    follow: bool = Query(default=True),
):
    """Server-Sent Events stream of a task's lifecycle (plan §2.5 point 4).

    Replays persisted events after ``last_event_id`` and then tails new ones,
    closing on a terminal event. Replaces polling ``GET /tasks/{id}`` for large
    images, where a client would otherwise poll for minutes.

    Termination depends on ``FIXIMG_EVAL_MODE``: with ``inline`` evaluation,
    ``task.completed`` is the last event; with ``async`` (plan §3.16) the stream
    keeps a bounded window open for the follow-up ``task.evaluated`` so clients
    receive the metrics without reconnecting.
    """
    task_service.require_task(task_id)

    async def event_source():
        cursor = int(last_event_id or 0)
        loop = asyncio.get_event_loop()
        deadline = loop.time() + _SSE_TIMEOUT_SECONDS
        completed_at: float | None = None
        while True:
            events = await asyncio.to_thread(
                task_service.get_task_events, task_id, cursor, 200
            )
            for raw in events:
                cursor = max(cursor, int(raw.get("id") or 0))
                # One path from a stored row to the wire: the domain model reads it
                # and decides what the row means, the schema only spells the payload.
                event = TaskEvent.from_row(raw)
                payload = TaskEventSchema(
                    event=event.event_type or "message",
                    task_id=event.task_id,
                    level=event.level,
                    message=event.message,
                    data=event.data,
                    created_at=event.created_at,
                    seq=event.seq,
                )
                yield (
                    f"id: {payload.seq}\nevent: {payload.event}\n"
                    f"data: {payload.model_dump_json()}\n\n"
                )
                if event.closes_stream:
                    return
                if event.event_type == EventType.TASK_COMPLETED:
                    completed_at = loop.time()

            if completed_at is not None:
                # `task.completed` is not unconditionally terminal when
                # evaluation is deferred: `task.evaluated` carries the metrics
                # and belongs to the same lifecycle. Wait for it (bounded), even
                # for a client that asked not to follow — otherwise a
                # `follow=false` caller would miss the metrics entirely.
                #
                # We wait for the *event*, not for the metrics to appear: the
                # evaluator writes the metric rows before the event, so closing
                # on "metrics exist" would return in the gap between the two.
                if settings.eval_mode != "async":
                    return
                if loop.time() - completed_at > _SSE_POST_COMPLETION_GRACE:
                    return
                await asyncio.sleep(_SSE_POLL_SECONDS)
                continue

            if not follow or loop.time() > deadline:
                return
            await asyncio.sleep(_SSE_POLL_SECONDS)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
