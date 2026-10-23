"""History row formatting, without a web framework in scope (plan §3.9.3).

These are the rules the history table is made of: how a stored instant is shown on
the viewer's wall clock, how a duration and a metrics dict read, and what the detail
pane says about a task. They used to live inside the Gradio panel, which meant they
could only be tested by building a `gr.Blocks`; the native UI renders the table from
`GET /ui/api/history`, so the rules are pure functions here and both the route and
its tests use them.

The formatting choices are unchanged from the panel on purpose — a migration that
changed what a column says would be indistinguishable from a bug:

* storage is UTC (plan §2.7) and a UTC+8 viewer must not read a time eight hours off,
  so the instant is rendered on the viewer's wall clock;
* a V1 row predates the codec, and blanking its date would lose the row — an
  unparseable instant passes through untouched;
* a duration of zero or None is an em dash, because "0.0 s" for a task that never ran
  is a wrong number rather than a missing one.
"""
from __future__ import annotations

from fiximg.infrastructure.db import timestamps

#: Column order of the history table. The route and the table template both read it.
COLUMNS = (
    "Task ID", "Type", "Created", "Duration", "Status", "Progress", "Stages", "Metrics",
)

#: Rows per page by default, and the sizes the page offers.
DEFAULT_PAGE_SIZE = 20
PAGE_SIZES = (10, 20, 50, 100)

#: Long side of a gallery thumbnail, in pixels.
THUMBNAIL_SIDE = 180

#: Metric keys the table shows, in the order a reader compares them.
_METRIC_ORDER = ("psnr", "ssim", "mae", "lpips", "niqe")


def duration_text(duration_ms) -> str:
    """A stored duration as a reader expects it, or an em dash when there is none."""
    if not duration_ms:
        return "—"
    seconds = duration_ms / 1000
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


def when_text(value) -> str:
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


def metrics_text(metrics) -> str:
    """Compact PSNR/SSIM summary — the numbers users actually compare."""
    if not isinstance(metrics, dict) or not metrics:
        return ""
    parts = []
    for key in _METRIC_ORDER:
        if key in metrics:
            parts.append(f"{key.upper()}={metrics[key]}")
    return "  ".join(parts) if parts else ""


def stages_text(stages) -> str:
    """The plan a run executed, as a single arrow-separated line."""
    if not isinstance(stages, list) or not stages:
        return ""
    return " → ".join(stage.get("stage_name", "?") for stage in stages)


def detail_text(row: dict, *, result_present: bool = True) -> str:
    """The detail pane's text for one task.

    Includes the reason a result is not there, because this pane is also where a
    reclaimed task gets explained — a blank preview with no sentence beside it reads
    as a broken page rather than as a retention policy.
    """
    from fiximg.domain.tasks import decode_json_column

    row = row or {}
    lines = [
        f"Task: {row.get('id', '')}",
        f"Type: {row.get('task_type')}   Status: {row.get('status')}   "
        f"Duration: {duration_text(row.get('duration_ms'))}",
    ]
    stages = decode_json_column(row.get("stages"), [])
    if stages:
        lines.append("Stages: " + stages_text(stages))
    metrics = decode_json_column(row.get("metrics"), {})
    if metrics:
        lines.append("Metrics: " + metrics_text(metrics))
    if row.get("error_code"):
        lines.append(f"Error code: {row['error_code']}")
    if row.get("error_message"):
        lines.append(f"Error: {str(row['error_message'])[:300]}")
    if not result_present:
        lines.append(
            "(No result file — it may have been reclaimed after the retention window.)"
        )
    return "\n".join(lines)


def rerun_options(row: dict) -> dict:
    """The options a re-run should send: the row's own, minus internal keys.

    Internal keys (the ground-truth path) point at artifacts of the *old* run, so
    forwarding them would have the new task read a file that describes a different
    submission.
    """
    from fiximg.domain.tasks import decode_json_column

    stored = decode_json_column((row or {}).get("options_json"), {}) or {}
    if not isinstance(stored, dict):
        return {}
    return {key: value for key, value in stored.items() if not key.startswith("_")}


__all__ = [
    "COLUMNS",
    "DEFAULT_PAGE_SIZE",
    "PAGE_SIZES",
    "THUMBNAIL_SIDE",
    "detail_text",
    "duration_text",
    "metrics_text",
    "rerun_options",
    "stages_text",
    "when_text",
]
