"""Task history panel for the Gradio UI (plan §3.9.3).

V2 kept a ``processing_history`` table but the UI never showed it, so a user
could not revisit an earlier result or re-run it. This module renders the
existing task records as a browsable table:

    History
    ├── thumbnails (a page of result previews)
    ├── table (task id, type, created, duration, status, metrics)
    ├── pagination (page size + prev/next + "page N of M")
    ├── Before / After slider
    ├── Original / Result / Mask / Face crop previews  (plan §3.9.2)
    └── re-run

The panel reads through :mod:`fiximg.application.task_service` only — it never
touches the database or the pipeline directly.
"""
from __future__ import annotations

import os

import gradio as gr

from fiximg.application.pipeline_modes import PIPELINE_MODES
from fiximg.application.task_service import task_service
from fiximg.domain.tasks import decode_json_column
from fiximg.infrastructure.db import timestamps
from fiximg.infrastructure.observability.logging import get_logger, log_event
from fiximg.ui.components import comparison_slider, preview_row

logger = get_logger("fiximg.ui.history")

#: Column order of the history table (kept in sync with COLUMNS below).
COLUMNS = ["Task ID", "Type", "Created", "Duration", "Status", "Progress", "Stages", "Metrics"]

#: Rows per page by default, and the sizes offered in the UI.
DEFAULT_PAGE_SIZE = 20
PAGE_SIZES = [10, 20, 50, 100]

#: Long side of a gallery thumbnail, in pixels.
THUMBNAIL_SIDE = 180

#: task_type → UI mode, so a history row can be re-run through the same path.
_TASK_TYPE_TO_MODE: dict[str, str] = {}
for _mode, _cfg in PIPELINE_MODES.items():
    _TASK_TYPE_TO_MODE.setdefault(_cfg["task_type"], _mode)


def _duration_text(duration_ms) -> str:
    if not duration_ms:
        return "—"
    seconds = duration_ms / 1000
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


def _when_text(value) -> str:
    """Render a stored instant on the viewer's wall clock.

    The database keeps canonical UTC (plan §2.7), and showing that raw would put
    every history row eight hours off for a UTC+8 viewer — a *correct* string the
    user reads as wrong. Values the codec cannot read (V1 rows predate it) pass
    through untouched rather than blanking the row out.
    """
    if not value:
        return ""
    try:
        return timestamps.parse(value).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return str(value)


def _metrics_text(metrics) -> str:
    """Compact PSNR/SSIM summary — the numbers users actually compare."""
    if not isinstance(metrics, dict) or not metrics:
        return ""
    parts = []
    for key in ("psnr", "ssim", "mae", "lpips", "niqe"):
        if key in metrics:
            parts.append(f"{key.upper()}={metrics[key]}")
    return "  ".join(parts) if parts else ""


def _stages_text(stages) -> str:
    if not isinstance(stages, list) or not stages:
        return ""
    return " → ".join(s.get("stage_name", "?") for s in stages)


def _visible_user_id(user_state) -> str | None:
    """Users see their own history; administrators see everyone's.

    Delegates to :mod:`fiximg.ui.state` so the rule has one implementation
    instead of being restated per panel (plan §3.2).
    """
    from fiximg.ui.state import visible_user_id

    return visible_user_id(user_state)


# ------------------------------------------------------------------ thumbnails
def thumbnail_cache_dir() -> str:
    """Where gallery thumbnails are cached (next to the artifact storage root)."""
    from fiximg.config import settings

    return os.path.join(os.path.dirname(settings.tasks_root.rstrip("/\\")), "thumbs")


def make_thumbnail(path: str, side: int = THUMBNAIL_SIDE) -> str | None:
    """Return a cached thumbnail of ``path``, or None when unusable.

    A gallery renders every row at once, so serving multi-megapixel originals
    makes the panel crawl. Thumbnails are cached by (path, mtime, size) so a
    re-render does not re-encode anything.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        stat = os.stat(path)
        cache_dir = thumbnail_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        target = os.path.join(cache_dir, f"{stem}_{stat.st_mtime_ns}_{side}.png")
        if os.path.exists(target):
            return target

        from PIL import Image

        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((side, side), Image.Resampling.LANCZOS)
            thumb.save(target, format="PNG", optimize=True)
        return target
    except Exception as exc:  # noqa: BLE001 — a thumbnail is cosmetic
        log_event(logger, "WARNING", "thumbnail failed", path=path, error=str(exc))
        return None


def _artifact_path(task_id: str, *kinds: str) -> str | None:
    """First existing artifact of any of ``kinds`` for a task."""
    try:
        rows = task_service.get_task_artifacts(task_id)
    except Exception:  # noqa: BLE001 — previews must never break the panel
        return None
    for kind in kinds:
        for row in rows or []:
            if row.get("kind") != kind:
                continue
            path = row.get("uri") or row.get("path")
            if path and os.path.exists(path):
                return path
    return None


def _first_face_crop(task_id: str) -> str | None:
    """First aligned face from the detection stage (a directory of crops)."""
    try:
        return task_service.first_face_crop(task_id)
    except Exception:  # noqa: BLE001 — previews must never break the panel
        return None


# ----------------------------------------------------------------------- table
def list_history(user_state, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE):
    """Render one page of history.

    Returns ``(rows, gallery, page_label, page_state, status_text)``.
    """
    try:
        page_size = max(1, min(int(page_size or DEFAULT_PAGE_SIZE), 200))
    except (TypeError, ValueError):
        page_size = DEFAULT_PAGE_SIZE
    try:
        page = max(1, int(page or 1))
    except (TypeError, ValueError):
        page = 1

    user_id = _visible_user_id(user_state)
    try:
        # Fetch one row past this page so "is there a next page?" needs no COUNT,
        # and so the last page can be derived when the caller asks for too far.
        tasks = task_service.list_tasks(limit=page_size * page + 1, user_id=user_id)
    except Exception as exc:  # noqa: BLE001 — the panel must never break the UI
        log_event(logger, "ERROR", "history listing failed", error=str(exc))
        return [], [], "Page 1 of 1", 1, f"⚠ Could not load history: {type(exc).__name__}"

    total_fetched = len(tasks)
    last_page = max(1, -(-total_fetched // page_size))
    page = min(page, last_page)

    start = (page - 1) * page_size
    window = tasks[start:start + page_size]
    has_more = total_fetched > start + page_size

    rows, gallery = [], []
    for task in window:
        task_id = task.get("id", "")
        rows.append(
            [
                task_id,
                task.get("task_type", ""),
                _when_text(task.get("created_at")),
                _duration_text(task.get("duration_ms")),
                task.get("status", ""),
                f"{int(task.get('progress') or 0)}%",
                _stages_text(decode_json_column(task.get("stages"), [])),
                _metrics_text(decode_json_column(task.get("metrics"), {})),
            ]
        )
        thumb = make_thumbnail(task.get("result_path") or "")
        if thumb:
            gallery.append((thumb, f"{task.get('task_type', '')} · {task_id}"))

    label = f"Page {page} of {page + 1 if has_more else page}"
    scope = "all users" if user_id is None else f"user '{user_id}'"
    status = (
        f"{len(rows)} row(s) on this page — {scope}, newest first. "
        f"Page size {page_size}."
    )
    return rows, gallery, label, page, status


def next_page(user_state, page_size, page):
    """Advance one page."""
    return list_history(user_state, int(page or 1) + 1, page_size)


def prev_page(user_state, page_size, page):
    """Go back one page (never below 1)."""
    return list_history(user_state, max(1, int(page or 1) - 1), page_size)


# --------------------------------------------------------------------- preview
def load_history_entry(user_state, page, page_size, evt: gr.SelectData):
    """Populate every preview from the selected history row.

    Returns ``(input, result, mask, face_crop, before_after, mask_compare, detail,
    download)`` — plan §3.9.2 asks for more than the two-state original/result
    slider, and the download control belongs to whichever row is selected.
    """
    task_id = _task_id_from_event(evt, user_state, page, page_size)
    if not task_id:
        return (None, None, None, None, gr.update(), gr.update(),
                "Select a row to preview it.", gr.update(visible=False))

    row = task_service.get_task(task_id)
    if not row:
        return (None, None, None, None, gr.update(), gr.update(),
                f"Task {task_id} no longer exists.", gr.update(visible=False))

    input_path = row.get("input_path")
    result_path = row.get("result_path")
    before = input_path if input_path and os.path.exists(input_path) else None
    after = result_path if result_path and os.path.exists(result_path) else None
    mask = _artifact_path(task_id, "mask")
    face_crop = _first_face_crop(task_id)

    before_after = gr.update(value=(before, after)) if (before and after) else gr.update()
    # The mask comparison only makes sense when a mask exists.
    mask_compare = (
        gr.update(value=(before, mask), visible=True) if (before and mask)
        else gr.update(visible=False)
    )

    detail = [
        f"Task: {task_id}",
        f"Type: {row.get('task_type')}   Status: {row.get('status')}   "
        f"Duration: {_duration_text(row.get('duration_ms'))}",
    ]
    stages = decode_json_column(row.get("stages"), [])
    if stages:
        detail.append("Stages: " + _stages_text(stages))
    metrics = decode_json_column(row.get("metrics"), {})
    if metrics:
        detail.append("Metrics: " + _metrics_text(metrics))
    if row.get("error_code"):
        detail.append(f"Error code: {row['error_code']}")
    if row.get("error_message"):
        detail.append(f"Error: {row['error_message'][:300]}")
    if not after:
        detail.append(
            "(No result file — it may have been reclaimed after the retention window.)"
        )
    # Offered only for a result that is actually on disk: this panel is also where
    # a reclaimed task gets explained, and a dead download link answers worse than
    # no link at all.
    download = gr.update(value=result_path, visible=bool(after))
    return before, after, mask, face_crop, before_after, mask_compare, "\n".join(detail), download


def rerun_history_entry(user_state, page, page_size, evt: gr.SelectData):
    """Re-enqueue the selected task through the same pipeline mode."""
    from PIL import Image

    def _current(status: str):
        rows, gallery, label, current, _ = list_history(user_state, page, page_size)
        return rows, gallery, label, current, status

    task_id = _task_id_from_event(evt, user_state, page, page_size)
    if not task_id:
        return _current("Select a row to re-run it.")

    row = task_service.get_task(task_id)
    if not row:
        return _current(f"Task {task_id} no longer exists.")

    input_path = row.get("input_path")
    if not input_path or not os.path.exists(input_path):
        return _current("⚠ Cannot re-run: the input image is no longer available.")

    mode = _TASK_TYPE_TO_MODE.get(row.get("task_type"), row.get("task_type"))
    try:
        image = Image.open(input_path).convert("RGB")
        options = decode_json_column(row.get("options_json"), {}) or {}
        # Internal keys (e.g. the ground-truth path) are not re-sent: they point
        # at artifacts of the *old* run.
        options = {k: v for k, v in options.items() if not k.startswith("_")}
        # The re-run keeps the row's queue position, so "Re-run" and "Retry" (which
        # re-queues the same row) start from the same priority. A task somebody paid
        # the `FIXIMG_PRIORITY_MAX` budget for should not silently drop to FIFO because
        # it was submitted again.
        new_id = task_service.enqueue(
            image, user_state, mode, options=options or None,
            priority=int(row.get("priority") or 0),
        )
    except Exception as exc:  # noqa: BLE001 — surface the reason in the caption
        log_event(logger, "ERROR", "history re-run failed", task_id=task_id, error=str(exc))
        return _current(f"⚠ Re-run failed: {exc}")

    log_event(logger, "INFO", "history re-run queued", source_task=task_id, task_id=new_id)
    return _current(f"🔁 Re-queued {row.get('task_type')} as {new_id}.")


def retry_history_entry(user_state, page, page_size, evt: gr.SelectData):
    """Requeue the selected task under its own id (plan §3.8 ``/tasks/{id}/retry``).

    Deliberately different from "Re-run selected": re-run submits a **new** task
    from the stored input, while retry resets the attempt budget and re-queues the
    **same** task id, so the history row, its artifacts and its metrics keep
    describing one job. The endpoint has supported this since V2; the panel simply
    had no way to reach it, so "retry" in the UI silently meant "duplicate".
    """

    def _current(status: str):
        rows, gallery, label, current, _ = list_history(user_state, page, page_size)
        return rows, gallery, label, current, status

    task_id = _task_id_from_event(evt, user_state, page, page_size)
    if not task_id:
        return _current("Select a row to retry it.")

    row = task_service.get_task(task_id)
    if not row:
        return _current(f"Task {task_id} no longer exists.")
    if row.get("status") not in ("failed", "cancelled"):
        return _current(
            f"↻ {task_id} is {row.get('status')}; retry only applies to a failed or "
            "cancelled task — use Re-run to start a fresh one."
        )

    try:
        requeued = task_service.retry_task(task_id)
    except Exception as exc:  # noqa: BLE001 — the caption is the user-facing answer
        log_event(logger, "ERROR", "history retry failed",
                  task_id=task_id, error=str(exc))
        return _current(f"⚠ Retry failed: {exc}")

    if not requeued:
        return _current(f"↻ {task_id} was not requeued (claimed or finished in the meantime).")
    log_event(logger, "INFO", "history retry queued", task_id=task_id)
    return _current(f"↻ Re-queued {task_id} under the same id; attempt count reset.")


def _task_id_from_event(evt, user_state, page, page_size) -> str | None:
    """Resolve the task id of the clicked Dataframe row (page-relative)."""
    index = getattr(evt, "index", None)
    if index is None:
        return None
    row_index = index[0] if isinstance(index, (tuple, list)) else index
    try:
        row_index = int(row_index)
    except (TypeError, ValueError):
        return None

    rows, _gallery, _label, _page, _status = list_history(user_state, page, page_size)
    if 0 <= row_index < len(rows):
        return rows[row_index][0]
    return None


def build_history_panel(user_state):
    """Build the History tab contents and wire its callbacks.

    Returns the components the page-load handler needs (nothing today; kept for
    symmetry with the other panels).
    """
    page_state = gr.State(1)

    with gr.Row():
        refresh_button = gr.Button("Refresh", elem_classes="clear-button")
        page_size_dropdown = gr.Dropdown(
            choices=PAGE_SIZES, value=DEFAULT_PAGE_SIZE, label="Rows per page", scale=1,
        )
        prev_button = gr.Button("◀ Prev", elem_classes="clear-button")
        page_label = gr.Textbox(value="Page 1 of 1", label="", interactive=False, scale=1)
        next_button = gr.Button("Next ▶", elem_classes="clear-button")
        rerun_button = gr.Button("Re-run selected", elem_classes="start-button")
        retry_button = gr.Button("Retry failed (same task)", elem_classes="clear-button")

    history_status = gr.Textbox(label="History", lines=1, interactive=False)

    # §3.9.3: thumbnails. A Dataframe cannot show images, so the gallery carries
    # the visual index and the table carries the numbers.
    gallery = gr.Gallery(
        label="Result thumbnails (this page)", columns=6, height=200,
        object_fit="contain", elem_id="history_gallery",
    )

    table = gr.Dataframe(
        headers=COLUMNS,
        datatype=["str"] * len(COLUMNS),
        interactive=False,
        wrap=True,
        elem_id="history_table",
        label="Recent tasks (click a row to preview)",
    )

    gr.Markdown("#### Before / After")
    # The same two components every tab builds, from `ui.components` — the panel
    # used to construct them inline, which left the helper unused and the two
    # definitions free to drift.
    before_after = comparison_slider(
        "history_compare", label="Drag the handle: original ◀ ▶ restored"
    )
    mask_compare = comparison_slider(
        "history_mask_compare",
        label="Original ◀ ▶ detected mask (produced by scratch detection)",
        visible=False,
    )

    history_input, history_result, history_mask, history_face = preview_row(
        ["Original", "Restored", "Mask", "Face crop"]
    )
    history_detail = gr.Textbox(label="Task detail", lines=5, interactive=False)
    download_button = gr.DownloadButton(
        "Download this result", size="sm", visible=False,
        elem_id="history_download",
    )

    # The five outputs every list-producing callback must return.
    list_outputs = [table, gallery, page_label, page_state, history_status]

    refresh_button.click(
        list_history, inputs=[user_state, page_state, page_size_dropdown],
        outputs=list_outputs,
    )
    page_size_dropdown.change(
        list_history, inputs=[user_state, page_state, page_size_dropdown],
        outputs=list_outputs,
    )
    prev_button.click(
        prev_page, inputs=[user_state, page_size_dropdown, page_state],
        outputs=list_outputs,
    )
    next_button.click(
        next_page, inputs=[user_state, page_size_dropdown, page_state],
        outputs=list_outputs,
    )
    table.select(
        load_history_entry,
        inputs=[user_state, page_state, page_size_dropdown],
        outputs=[history_input, history_result, history_mask, history_face,
                 before_after, mask_compare, history_detail, download_button],
    )
    retry_button.click(
        retry_history_entry,
        inputs=[user_state, page_state, page_size_dropdown],
        outputs=list_outputs,
    )
    rerun_button.click(
        rerun_history_entry,
        inputs=[user_state, page_state, page_size_dropdown],
        outputs=list_outputs,
    )
    return None


__all__ = [
    "COLUMNS",
    "DEFAULT_PAGE_SIZE",
    "PAGE_SIZES",
    "build_history_panel",
    "list_history",
    "load_history_entry",
    "make_thumbnail",
    "next_page",
    "prev_page",
    "rerun_history_entry",
    "retry_history_entry",
    "thumbnail_cache_dir",
]
