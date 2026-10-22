"""Progress, stage and option formatting for the web UI (plan §3.9.1 / §24).

These are the rules the progress panel is made of, kept as pure functions with no
web framework in scope. The Gradio panel used to own them, which meant every rule
in this module could only be checked by building a `gr.Blocks` and reading its
config; the native UI renders the same strings server-side (for the no-JS
fallback and for `GET /ui/api/tasks/{id}`) and the browser only displays them, so
the rules are testable as what they are.

Nothing here talks to a database, a queue or a browser: every input is a plain
dict or scalar, and every output is a string, a number or a JSON-able dict.
"""
from __future__ import annotations

import time

#: How the stage checklist marks each state. `skipped` is deliberately not
#: `completed`: a stage that never ran must not read as a success.
_CHECK = "✓"
_ACTIVE = "▶"
_PENDING = "○"
_FAIL = "✗"
_SKIP = "⊘"

_ICONS = {
    "completed": _CHECK,
    "running": _ACTIVE,
    "failed": _FAIL,
    "pending": _PENDING,
    "skipped": _SKIP,
}

#: Below this the extrapolation has less signal than noise: a four-stage plan's
#: first tick is 25 %, and one tick is not a basis for a minute-scale promise.
_ETA_MIN_PROGRESS = 20
#: …and neither is a percentage measured over a few milliseconds. A run that has
#: been going for less than this shows no ETA rather than a number that would be
#: wrong by orders of magnitude as soon as the model finishes loading.
_ETA_MIN_ELAPSED_SECONDS = 1.0


def bar(progress: int, width: int = 24) -> str:
    """The §3.9.1 bar: filled cells proportional to `progress`, percent-clamped."""
    filled = int(width * max(0, min(int(progress or 0), 100)) / 100)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def duration_text(seconds: float) -> str:
    """A duration a person reads: seconds under a minute, then minutes and seconds."""
    total = int(round(seconds))
    if total < 60:
        return f"{total} s"
    minutes, rest = divmod(total, 60)
    return f"{minutes} min {rest} s"


def eta_text(progress: int, elapsed_seconds: float) -> str:
    """§3.9.1's ETA line: what is left, extrapolated from the progress the runtime set.

    Deliberately derived from the two numbers this process already trusts — the
    runtime's own percent (which is plan-aware) and the wall time the run has been
    going — rather than from historical per-stage timings: those describe *other*
    images, and a stage's duration here varies more with the picture than with the
    stage name. Until both signals are meaningful the answer is `ETA  --`, which is a
    statement about the estimate, not a small estimate.
    """
    if progress < _ETA_MIN_PROGRESS or elapsed_seconds < _ETA_MIN_ELAPSED_SECONDS:
        return "ETA  --"
    return f"ETA  ~{duration_text(elapsed_seconds * (100 - progress) / progress)}"


def queue_text(depth: int | None) -> str:
    """The queue readout §3.9.1 puts next to the bar."""
    return "Queue  ?" if depth is None else f"Queue  {depth}"


def progress_head(label: str, current: str, progress: int, elapsed_seconds: float,
                  depth: int | None) -> str:
    """One progress frame's first line (plan §3.9.1: bar, percent, stage, ETA, queue)."""
    return "  ·  ".join((
        bar(progress), f"{progress}%", label, current,
        eta_text(progress, elapsed_seconds), queue_text(depth),
    ))


def stage_icon(status: str | None) -> str:
    """The checklist marker for one stage status."""
    return _ICONS.get(status or "pending", _PENDING)


def stage_lines(stages: list) -> str:
    """Render the per-stage checklist (plan section 24).

    A skipped stage's message is the reason it did nothing (no dlib, no aligned
    face), which is the part a user needs to see; a failure's message is the same
    courtesy. Both are truncated, because a stage message is a log line.
    """
    lines = []
    for stage in stages or []:
        if not isinstance(stage, dict):
            continue
        icon = stage_icon(stage.get("status"))
        duration = stage.get("duration_ms")
        duration_suffix = f" ({duration / 1000:.1f} s)" if duration else ""
        message = (stage.get("message") or "").strip()
        reason = (
            f" — {message[:80]}"
            if stage.get("status") in ("failed", "skipped") and message else ""
        )
        lines.append(
            f"{icon} {stage.get('stage_name', '?')}{duration_suffix}{reason}"
        )
    return "\n".join(lines)


def stage_rows(stages: list) -> list[dict]:
    """The checklist as structured data, for a client that renders it itself.

    Same rules as :func:`stage_lines` — one source, two renderings — so the browser
    and the text fallback cannot disagree about what a stage did.
    """
    rows = []
    for stage in stages or []:
        if not isinstance(stage, dict):
            continue
        duration = stage.get("duration_ms")
        message = (stage.get("message") or "").strip()
        rows.append({
            "name": stage.get("stage_name") or "?",
            "status": stage.get("status") or "pending",
            "icon": stage_icon(stage.get("status")),
            "duration_ms": duration,
            "duration_text": f"{duration / 1000:.1f} s" if duration else "",
            "message": message[:200],
            # Only a failure or a skip owes the reader an explanation.
            "reason": message[:200] if stage.get("status") in ("failed", "skipped") else "",
        })
    return rows


def elapsed_since_run_start(row: dict, submitted_at: float) -> float:
    """Seconds this task has been executing, or since submit if it never started.

    The ETA is aged against when the *run* started, not when the user pressed
    submit: a task that sat in the queue for ten minutes would otherwise get an
    "ETA ~10 min" at its first 25 % tick, for work that has not begun.
    """
    started_at = (row or {}).get("started_at")
    if started_at:
        from fiximg.infrastructure.db import timestamps

        try:
            return max(0.0, timestamps.age_seconds(started_at))
        except Exception:  # noqa: BLE001 — a malformed instant is not the user's problem
            pass
    return max(0.0, time.monotonic() - submitted_at)


# ---------------------------------------------------------------- options
def switch_options(mode: str, names: list[str], values) -> dict:
    """Turn checkbox states into task options.

    Two kinds of mode, because two kinds of plan:

    * task-type plans (`restore` / `restore_scratch`): every switch is sent as
      typed. The drawn defaults reproduce the planner's own defaults, so a submit
      with nobody touching a box runs the same pipeline as no options at all.
    * analysis-derived plans (`auto`): a checked box *permits* what the analysis
      decided, so it sends nothing; only an unchecked one sends a value, and it sends
      `false`. Sending `true` there would instruct the planner to run the stage
      even when the image says it is not needed, which is not what the user ticked.

    `hr` is not a pipeline switch on either kind: the face stages read it themselves,
    so it always goes as typed.
    """
    from fiximg.application.pipeline_modes import ANALYSIS_DECIDED_MODES, PLAN_SWITCHES

    options = dict(zip(names, values, strict=True))
    if mode not in ANALYSIS_DECIDED_MODES:
        return options
    return {
        name: value
        for name, value in options.items()
        if name not in PLAN_SWITCHES or not value
    }


def describe_preview(preview: dict) -> str:
    """Render `TaskService.auto_preview()` as the analysis row's text.

    The numbers come from `format_analysis_summary`, which is the same formatter the
    worker writes into its log — so the row, the log line and the run report cannot
    disagree about what the analyzer said. The stage names are the ones the task row
    will carry, for the same reason.
    """
    if not preview.get("available"):
        return f"Analysis unavailable: {preview.get('reason', 'unknown error')}"

    lines = [
        f"Analysis: {preview['summary']}",
        "Planned pipeline: " + (" → ".join(preview["stages"]) or "(nothing)"),
    ]
    if preview["forced"]:
        lines.append("Forced by your switches, not by the image: "
                     + ", ".join(preview["forced"]))
    if preview["declined"]:
        lines.append("Declined by your switches: " + ", ".join(preview["declined"]))
    return "\n".join(lines)


# ----------------------------------------------------------------- failures
#: What the user can actually do about this refusal. The domain message says what
#: the system found; this says which control to change, which is the part a person
#: looking at a photo does not need explained twice.
_HR_SCRATCH_HINT = (
    'Untick "High-resolution face path" to restore this photo, or use one '
    "without scratches. The upstream project does not publish the "
    "mapping_Patch_Attention weights the HR scratch branch needs, and the "
    "vendored loader would run that network uninitialised, so the request is "
    "refused rather than answered with noise."
)

#: A stable fragment of the backend's refusal, so the queued path can recognise
#: *this* refusal rather than any missing model. The error code alone is not enough:
#: a run with HR ticked that fails on an unrelated missing weight must not be told
#: the HR scratch branch is the problem. Matching prose is a compromise — the
#: alternative is misattributing failures — so the fragment is the backend's own
#: directory name, and it is asserted against the backend in the tests.
_HR_REFUSAL_MARKER = "mapping_Patch_Attention"


def hr_scratch_message() -> str:
    """The actionable sentence for the HR-scratch refusal."""
    return ("High-resolution face path is unavailable for this photo. "
            + _HR_SCRATCH_HINT)


def describe_failure(exc: Exception) -> str:
    """A user-facing sentence for a run that cannot proceed.

    Both submission paths go through this, so the queued and synchronous routes
    cannot describe the same refusal differently.
    """
    from fiximg.domain.errors import ModelUnavailableError

    details = getattr(exc, "details", None) or {}
    if isinstance(exc, ModelUnavailableError) and details.get("hr"):
        return hr_scratch_message()
    return str(exc).strip() or exc.__class__.__name__


def failure_text(row: dict, options: dict | None) -> str:
    """The queued path's version of :func:`describe_failure`.

    Here the worker has already caught the failure, so it arrives as a row rather
    than as an exception, and the `details` dict that identifies the HR refusal is
    gone. So the same sentence is chosen from the error code, the options this
    submission sent, and the backend's own marker in the message.
    """
    from fiximg.domain.errors import ErrorCode

    message = (row or {}).get("error_message") or "unknown error"
    if ((row or {}).get("error_code") == ErrorCode.MODEL_UNAVAILABLE
            and (options or {}).get("hr")
            and _HR_REFUSAL_MARKER in message):
        return hr_scratch_message()
    return message[:200]


# ------------------------------------------------------------- task framing
def task_label(task_type: str) -> str:
    """The human name for a task type, from the one mode table."""
    from fiximg.application.pipeline_modes import PIPELINE_MODES

    for cfg in PIPELINE_MODES.values():
        if cfg["task_type"] == task_type:
            return cfg["label"]
    return task_type


def completed_head(task_type: str, duration_ms: int | None) -> str:
    """The first line of a finished run's status text."""
    return (f"✅ {task_label(task_type)} completed in "
            f"{(duration_ms or 0) / 1000:.1f} s")


def cancelled_head(task_type: str) -> str:
    """The first line of a cancelled run's status text."""
    return f"⛔ {task_label(task_type)} cancelled"


def failure_head(task_type: str) -> str:
    """The first line of a failed run's status text."""
    return f"⚠ {task_label(task_type)} failed"


def status_text(row: dict, *, queue_depth: int | None = None,
                submitted_at: float | None = None,
                decisions: dict | None = None,
                options: dict | None = None) -> dict:
    """One poll frame as structured data, for `GET /ui/api/tasks/{id}`.

    The single place that decides what a task row *means*, so the browser, the
    no-JS fallback and the tests cannot describe the same row differently. The
    caller supplies the two things a row cannot know about itself: the queue depth
    (a deployment quantity) and the analysis decisions (read from the run's own
    report file).

    `text` is the human-readable frame the Gradio panel used to yield; it is kept
    because it is what a client shows when it does not want to lay the parts out
    itself, and because the assertions about it are the record of what the panel
    promised.
    """
    from fiximg.domain.tasks import decode_json_column

    row = row or {}
    status = row.get("status") or "queued"
    task_type = row.get("task_type") or ""
    stages = decode_json_column(row.get("stages"), [])
    progress = int(row.get("progress") or 0)
    frame = {
        "task_id": row.get("id"),
        "status": status,
        "progress": progress,
        "current_stage": row.get("current_stage"),
        "task_type": task_type,
        "label": task_label(task_type),
        "stages": stage_rows(stages),
        "stage_text": stage_lines(stages),
        "error_code": row.get("error_code"),
        "error_message": row.get("error_message"),
        "duration_ms": row.get("duration_ms"),
        "queue_depth": queue_depth,
        "result_ready": False,
        "text": "",
        "failed": status == "failed",
        "cancelled": status == "cancelled",
        "completed": status == "completed",
    }

    if status == "completed":
        body = (row.get("evaluation_text") or "").strip()
        if decisions:
            chosen = ", ".join(
                f"{key}={value}" for key, value in sorted(decisions.items())
                if not isinstance(value, dict)
            )
            body = f"{body}\nPipeline decisions: {chosen}" if body else \
                f"Pipeline decisions: {chosen}"
        tail = stage_lines(stages)
        frame["result_ready"] = bool(row.get("result_path"))
        frame["text"] = "\n\n".join(
            part for part in (completed_head(task_type, row.get("duration_ms")),
                              body, tail) if part
        )
        return frame

    if status == "failed":
        frame["text"] = (f"{failure_head(task_type)}\n"
                         f"{failure_text(row, options)}")
        return frame

    if status == "cancelled":
        frame["text"] = "\n".join(
            part for part in (cancelled_head(task_type), stage_lines(stages)) if part
        )
        return frame

    current = row.get("current_stage") or ("queued" if status == "queued" else status)
    head = progress_head(
        task_label(task_type), current, progress,
        elapsed_since_run_start(row, submitted_at if submitted_at is not None
                                else time.monotonic()),
        queue_depth,
    )
    frame["text"] = "\n".join(part for part in (head, stage_lines(stages)) if part)
    return frame


__all__ = [
    "bar",
    "cancelled_head",
    "completed_head",
    "describe_failure",
    "describe_preview",
    "duration_text",
    "elapsed_since_run_start",
    "eta_text",
    "failure_head",
    "failure_text",
    "hr_scratch_message",
    "progress_head",
    "queue_text",
    "stage_icon",
    "stage_lines",
    "stage_rows",
    "status_text",
    "switch_options",
    "task_label",
]
