"""History service: the admin panel's view of what the pipeline actually ran.

This used to read the V1 ``history`` table, on the stated assumption that the
V1→V2 migration was still in a "migration window" during which the old table would
keep being written. That window never closed. The only writer of ``history`` was
``engine.append_history()``, whose sole caller was a test, so V2/V3 never wrote a
single row — and an audit that removed the function recorded the table as
"read-only V1 legacy" while noting in the same sentence that the admin panel reads
it. So the two admin tabs that use this service (*Task Management* and *Photo
Archive*) were reading a table nothing populates: on a fresh install, and on every
install since, their Refresh buttons returned "No records" forever while dozens of
tasks sat in ``tasks``.

The fix is to read the table that is written. ``tasks`` is the authoritative
record, it holds every field these panels display, and ``list_tasks`` already
names the history panel as an intended caller. Keeping a second store in sync
would be two answers to one question, which is the thing this codebase has spent
forty batches removing.

``history`` is not gone: ``cli/migrate_v1.py`` still imports from it, which is what
the table is now *for*.

Field names are kept in the panel's own vocabulary (``timestamp``, ``user``,
``type``, ``psnr``…) so ``admin_panel`` is unchanged, and ``output_path`` is
carried as an alias of ``result_path`` because the archive detail view asks for
that name.
"""
from __future__ import annotations

from fiximg.infrastructure.db import engine as _db
from fiximg.infrastructure.db.repositories import task_repository as _tasks
from fiximg.infrastructure.db.repositories.task_repository import decode_json_column

#: The panel shows one input name; the store keeps a full path.
_QUALITY = ("psnr", "ssim", "mae")


def _as_panel_record(row: dict) -> dict:
    """Map a `tasks` row onto the shape the admin panel already expects.

    `metrics` is read through the shared decoder rather than a local `json.loads`:
    it is a JSON column, and `tests/unit/test_json_columns.py` requires every reader
    to go through one implementation — a second one is how the `metric_value` /
    `metric_text` pair stops being folded the same way in two places.
    """
    metrics = decode_json_column(row.get("metrics"), {})
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "id": row.get("id"),
        "timestamp": row.get("created_at", ""),
        "user": row.get("user_id", ""),
        "type": row.get("task_type", ""),
        "status": row.get("status", ""),
        "duration_ms": row.get("duration_ms"),
        "input_path": row.get("input_path", "") or "",
        # The panel asks for `output_path`; the task table calls it `result_path`.
        "output_path": row.get("result_path", "") or "",
        **{name: metrics.get(name, "") for name in _QUALITY},
    }


def list_history(limit: int = 50):
    """Newest-first task records, for the admin panel.

    `limit` still caps the rows returned; it used to cap the legacy table's growth
    via `FIXIMG_HISTORY_MAX`, which configured a table that no longer grows.
    """
    return [_as_panel_record(row) for row in _tasks.list_tasks(limit=limit)]


def get_history_record(history_id: str):
    """One record by its task id, or ``None``."""
    row = _tasks.get_task(history_id)
    return _as_panel_record(row) if row else None


def history_stats() -> str:
    """The panel's two-line summary, counted over tasks rather than legacy rows.

    It used to read `history`, so it reported "Total tasks: 0" on a deployment that
    had just finished one — a number that is not merely stale but contradicts the
    task list three centimetres above it.
    """
    rows = _tasks.list_tasks(limit=1000)
    users = {r.get("user_id") for r in rows if r.get("user_id")}
    return f"Total tasks: {len(rows)}\nUsers: {len(users)}"


def clear_history():
    """Delete the legacy V1 `history` rows.

    **This does not delete task records.** Task rows are the queue's state and the
    source of `/api/v1/stats`, so removing them is a retention decision, not a
    button press; the button is labelled "Clear All Records", which reads as though
    it would. The legacy table is all it clears, and on a V3 install that table is
    empty, so the button reports success while changing nothing.
    """
    _db.clear_history()


def archive_input_dir() -> str:
    """Where the V1 admin panel looks for an archived input copy."""
    return _db.archive_input_dir()


def archive_output_dir() -> str:
    return _db.archive_output_dir()


purge_stale_archives = _db.purge_stale_archives
