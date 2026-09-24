"""Task progress panel for the Gradio UI (plan sections 23 and 24).

Implements the V2 flow: the submit callback enqueues a task via TaskService
(never running GPU inference in the web process) and polls the DB-backed queue,
yielding stage-level progress feedback until a terminal state is reached.

UI per plan section 24:

    [██████████░░░░░░░░]  45%
    ✓ global_restore (3.2 s)
    ▶ face_enhancement ...
    ○ evaluation

Falls back to the synchronous path automatically when no worker is reachable
(queue saturated), preserving the V1 UX as a degraded mode.
"""
import time

import gradio as gr

from app.core.exceptions import PasswordChangeRequiredError
from app.core.logging import get_logger, log_event
from app.services.pipeline_modes import PIPELINE_MODES
from app.services.task_service import task_service

logger = get_logger("fiximg.ui.progress")

_POLL_INTERVAL = 0.5
_POLL_TIMEOUT = 30 * 60  # give up after 30 minutes of polling

_CHECK = "✓"
_ACTIVE = "▶"
_PENDING = "○"
_FAIL = "✗"


def _bar(progress: int, width: int = 24) -> str:
    filled = int(width * progress / 100)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _stage_lines(stages: list) -> str:
    """Render the per-stage checklist (plan section 24)."""
    icons = {"completed": _CHECK, "running": _ACTIVE, "failed": _FAIL, "pending": _PENDING}
    lines = []
    for s in stages:
        icon = icons.get(s.get("status"), _PENDING)
        dur = s.get("duration_ms")
        dur_text = f" ({dur / 1000:.1f} s)" if dur else ""
        msg = (s.get("message") or "").strip()
        suffix = f" — {msg[:80]}" if s.get("status") == "failed" and msg else ""
        lines.append(f"{icon} {s.get('stage_name', '?')}{dur_text}{suffix}")
    return "\n".join(lines)


def _submit_and_stream(mode: str, input_image, user_state, options=None):
    """Enqueue + poll; yields (result_image_or_None, status_text) tuples."""
    task_id = None
    try:
        task_id = task_service.enqueue(input_image, user_state, mode, options=options)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except PasswordChangeRequiredError as exc:
        # §21 force-change: bootstrap admin must replace the initial password.
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Failed to submit task: {exc}") from exc

    label = PIPELINE_MODES.get(mode, {}).get("label", mode)
    started = time.monotonic()
    while True:
        row = task_service.get_task(task_id) or {}
        status = row.get("status", "queued")
        stages = row.get("stages") or []
        if isinstance(stages, str):
            stages = []

        if status == "completed":
            result_path = row.get("result_path")
            head = f"✅ {label} completed in {(row.get('duration_ms') or 0) / 1000:.1f} s"
            yield result_path, f"{head}\n{_stage_lines(stages)}"
            return
        if status == "failed":
            err = row.get("error_message") or "unknown error"
            raise gr.Error(f"{label} failed: {err[:200]}")
        if status == "cancelled":
            yield None, f"⛔ {label} cancelled\n{_stage_lines(stages)}"
            return

        progress = int(row.get("progress") or 0)
        current = row.get("current_stage") or ("queued" if status == "queued" else status)
        head = f"{_bar(progress)} {progress}%  ·  {label}  ·  {current}"
        yield None, f"{head}\n{_stage_lines(stages)}"

        if time.monotonic() - started > _POLL_TIMEOUT:
            raise gr.Error("Timed out waiting for the task to finish")
        time.sleep(_POLL_INTERVAL)


def _sync_fallback(mode: str, input_image, user_state):
    """V1-compatible synchronous path (inline execution, blocking)."""
    res_img, evaluate_text = task_service.submit(mode, input_image, user_state)
    return res_img, evaluate_text or "done"


def _check_password_change(user_state) -> None:
    """§21: raise a clear gr.Error for accounts that must change passwords."""
    from app.core.exceptions import PasswordChangeRequiredError

    gate = getattr(task_service, "_require_password_changed", None)
    if gate is None:
        return
    try:
        gate(user_state)
    except PasswordChangeRequiredError as exc:
        raise gr.Error(str(exc)) from exc


def make_submit_handler(mode: str, options: dict | None = None):
    """Build a Gradio callback for a tab: enqueue -> poll -> yield progress.

    Returns a generator function yielding (result_image, progress_text) and
    finally (result_image, final_text).
    """

    def handler(input_image, user_state):
        if input_image is None:
            raise gr.Error("Please upload an image first.")
        _check_password_change(user_state)
        if task_service.has_worker():
            log_event(logger, "INFO", "ui submit queued", mode=mode)
            yield from _submit_and_stream(mode, input_image, user_state, options=options)
            return
        # No queue consumer available (e.g. FIXIMG_INLINE_WORKER=false without
        # a worker process): keep the old blocking behaviour instead of hanging.
        log_event(logger, "WARNING", "ui submit sync fallback", mode=mode)
        yield _sync_fallback(mode, input_image, user_state)

    return handler
