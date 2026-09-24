"""Task API router (plan section 12).

    POST /api/v1/tasks                 create a queued task (file upload)
    GET  /api/v1/tasks/{task_id}       status / progress / current_stage
    POST /api/v1/tasks/{task_id}/cancel
    GET  /api/v1/tasks/{task_id}/result
    GET  /api/v1/tasks/{task_id}/report
"""
import io
import json
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.api.security import require_api_token
from app.inference.orchestrator import new_task_id
from app.services.pipeline_modes import PIPELINE_MODES
from app.services.task_service import task_service

# Every task route requires the API bearer token (app/api/security.py): the
# Gradio UI uses the service layer in-process and is unaffected.
router = APIRouter(
    prefix="/api/v1/tasks",
    tags=["tasks"],
    dependencies=[Depends(require_api_token)],
)

_ALLOWED_TYPES = {cfg["task_type"] for cfg in PIPELINE_MODES.values()}

# Upload guard (plan section 22 file-size limit). Gradio enforces the same
# limit for UI submissions via max_file_size in app/factory.py.
from app.core.config import settings as _settings  # noqa: E402

_MAX_UPLOAD_BYTES = _settings.max_upload_mb * 1024 * 1024
_KNOWN_OPTIONS = {"hr", "face_enhance", "auto_colorize"}


@router.post("")
async def create_task(
    type: str = Form(...),
    image: UploadFile = File(...),
    options: str = Form(None),
    ground_truth: UploadFile = File(None),
):
    """Queue a task; the GPU worker picks it up asynchronously."""
    if type not in _ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown task type: {type}")

    # options: JSON object, e.g. {"hr": true, "face_enhance": true} (plan §12).
    task_options = {}
    if options:
        try:
            parsed = json.loads(options)
            if not isinstance(parsed, dict):
                raise ValueError
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="options must be a JSON object") from exc
        task_options = {k: v for k, v in parsed.items() if k in _KNOWN_OPTIONS}

    task_id = new_task_id()
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(payload) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {_settings.max_upload_mb} MB upload limit",
        )

    from PIL import Image as PILImage

    try:
        pil_image = PILImage.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    # §16 Ground Truth mode: optional reference photo enables LPIPS evaluation.
    gt_image = None
    if ground_truth is not None and ground_truth.filename:
        gt_payload = await ground_truth.read()
        if len(gt_payload) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Ground truth exceeds the {_settings.max_upload_mb} MB upload limit",
            )
        try:
            gt_image = PILImage.open(io.BytesIO(gt_payload)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="Invalid ground truth image") from exc

    from app.core.config import settings

    if max(pil_image.size) > settings.max_image_side:
        raise HTTPException(status_code=400, detail="Image too large")

    ok = task_service.submit_queued(
        task_id, pil_image, {"username": "api"}, type, options=task_options or None,
        ground_truth_image=gt_image,
    )
    if not ok:
        # No row was created; the queue-depth check rejects before insertion.
        raise HTTPException(status_code=503, detail="Task queue saturated; try again later")
    return JSONResponse({"task_id": task_id, "status": "queued"}, status_code=202)


def _parse_json_field(value, default):
    """SQLite json_group_* aggregates come back as JSON strings; decode them."""
    if value is None or value == "":
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return default
    return value


@router.get("/{task_id}")
async def get_task(task_id: str):
    row = task_service.get_task(task_id)
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    stages = _parse_json_field(row.pop("stages", None), [])
    metrics = _parse_json_field(row.pop("metrics", None), {})
    return {
        "task_id": row["id"],
        "status": row["status"],
        "progress": row["progress"],
        "current_stage": row.get("current_stage"),
        "error_message": row.get("error_message"),
        "duration_ms": row.get("duration_ms"),
        "stages": stages,
        "metrics": metrics,
    }


@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str):
    ok = task_service.cancel_task(task_id)
    if not ok:
        raise HTTPException(status_code=409, detail="Task not cancellable (missing or already running)")
    return {"task_id": task_id, "status": "cancelled"}


@router.get("/{task_id}/result")
async def get_result(task_id: str):
    row = task_service.get_task(task_id)
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    if row["status"] != "completed" or not row.get("result_path"):
        raise HTTPException(status_code=409, detail=f"Task not completed (status={row['status']})")
    path = row["result_path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=410, detail="Result expired")
    return FileResponse(path, media_type="image/png", filename=os.path.basename(path))


@router.get("/{task_id}/report")
async def get_report(task_id: str):
    row = task_service.get_task(task_id)
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    stages = _parse_json_field(row.pop("stages", None), [])
    metrics = _parse_json_field(row.pop("metrics", None), {})
    report = {
        "task_id": row["id"],
        "task_type": row["task_type"],
        "status": row["status"],
        "duration_ms": row.get("duration_ms"),
        "evaluation_text": row.get("evaluation_text"),
        "stages": stages,
        "metrics": metrics,
    }
    # §10 auto_restore: surface the analyzer-driven planner decisions.
    report_path = row.get("result_path")
    if report_path:
        import os

        run_dir = os.path.dirname(os.path.dirname(report_path))
        decisions_path = os.path.join(run_dir, "report.json")
        if os.path.exists(decisions_path):
            try:
                with open(decisions_path, encoding="utf-8") as f:
                    full = json.load(f)
                report["planner_decisions"] = full.get("planner_decisions")
            except (OSError, ValueError):
                pass
    return report
