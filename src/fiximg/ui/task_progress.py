"""Task progress panel for the Gradio UI (plan sections 23 and 24).

Implements the V2 flow: the submit callback enqueues a task via TaskService
(never running GPU inference in the web process) and polls the DB-backed queue,
yielding stage-level progress feedback until a terminal state is reached.

UI per plan section 24 and the §3.9.1 layout:

    [██████████░░░░░░░░]  45%  ·  Restore (no scratches)  ·  global_restore
      ·  ETA  ~55 s  ·  Queue  2
    ✓ global_restore (3.2 s)
    ▶ face_enhancement ...
    ○ evaluation

The input area's option checkboxes (§3.9.1's "□ Face Enhance / □ Auto Colorize /
□ High Resolution") are built by :func:`build_switches` from the mode's declared
switches, and their values reach the planner as task options.

Falls back to the synchronous path automatically when no worker is reachable
(queue saturated), preserving the V1 UX as a degraded mode.
"""
import time

import gradio as gr

from fiximg.domain.errors import (
    AppError,
    ErrorCode,
    ModelUnavailableError,
    PasswordChangeRequiredError,
)
from fiximg.domain.tasks import decode_json_column
from fiximg.infrastructure.observability.logging import get_logger, log_event
from fiximg.application.pipeline_modes import PIPELINE_MODES
from fiximg.application.task_service import task_service

logger = get_logger("fiximg.ui.progress")

_POLL_INTERVAL = 0.5
_POLL_TIMEOUT = 30 * 60  # give up after 30 minutes of polling

_CHECK = "✓"
_ACTIVE = "▶"
_PENDING = "○"
_FAIL = "✗"
_SKIP = "⊘"


def _bar(progress: int, width: int = 24) -> str:
    filled = int(width * progress / 100)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


#: Below this the extrapolation has less signal than noise: a four-stage plan's
#: first tick is 25 %, and one tick is not a basis for a minute-scale promise.
_ETA_MIN_PROGRESS = 20
#: …and neither is a percentage measured over a few milliseconds. A run that has
#: been going for less than this shows no ETA rather than a number that would be
#: wrong by orders of magnitude as soon as the model finishes loading.
_ETA_MIN_ELAPSED_SECONDS = 1.0


def _duration_text(seconds: float) -> str:
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
    stage name. Until both signals are meaningful the answer is `ETA --`, which is a
    statement about the estimate, not a small estimate.
    """
    if progress < _ETA_MIN_PROGRESS or elapsed_seconds < _ETA_MIN_ELAPSED_SECONDS:
        return "ETA  --"
    return f"ETA  ~{_duration_text(elapsed_seconds * (100 - progress) / progress)}"


def queue_text(depth: int | None) -> str:
    """The queue readout §3.9.1 puts next to the bar."""
    return "Queue  ?" if depth is None else f"Queue  {depth}"


def progress_head(label: str, current: str, progress: int, elapsed_seconds: float,
                  depth: int | None) -> str:
    """One progress frame's first line (plan §3.9.1: bar, percent, stage, ETA, queue)."""
    return "  ·  ".join((
        _bar(progress), f"{progress}%", label, current,
        eta_text(progress, elapsed_seconds), queue_text(depth),
    ))


def _stage_lines(stages: list) -> str:
    """Render the per-stage checklist (plan section 24)."""
    icons = {
        "completed": _CHECK, "running": _ACTIVE, "failed": _FAIL,
        "pending": _PENDING, "skipped": _SKIP,
    }
    lines = []
    for s in stages:
        icon = icons.get(s.get("status"), _PENDING)
        dur = s.get("duration_ms")
        dur_text = f" ({dur / 1000:.1f} s)" if dur else ""
        msg = (s.get("message") or "").strip()
        # A skipped stage's message is the reason it did nothing (no dlib, no
        # aligned face), which is the part a user needs to see.
        suffix = (
            f" — {msg[:80]}"
            if s.get("status") in ("failed", "skipped") and msg else ""
        )
        lines.append(f"{icon} {s.get('stage_name', '?')}{dur_text}{suffix}")
    return "\n".join(lines)


def _submit_and_stream(mode: str, input_image, user_state, options=None):
    """Enqueue + poll; yields ``(result_image, status_text, comparison, download)``.

    The third value feeds the Before/After ``gr.ImageSlider``. Progress ticks
    yield ``gr.update()`` so the slider keeps whatever it already shows, and only
    the terminal tick sets a value — otherwise every poll would reset it. The
    download control follows the same rule: hidden until there is a result.
    """
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
        # `stages` is one of the repository's JSON columns: JSON **text** on SQLite,
        # already a list on PostgreSQL. Treating the text as "no stages" was not a
        # cosmetic slip — on the default engine the whole §24 checklist disappeared
        # from every frame, because a string is truthy and iterating it gives out
        # characters rather than stage rows.
        stages = decode_json_column(row.get("stages"), [])

        if status == "completed":
            result_path = row.get("result_path")
            head = f"✅ {label} completed in {(row.get('duration_ms') or 0) / 1000:.1f} s"
            # The box this text goes to is labelled "Difference metrics vs. degraded
            # original (PSNR/SSIM/MAE)" (and, on the auto tab, "Evaluation &
            # automatic pipeline decisions"). The queued path used to put only the
            # progress bar and stage checklist in it, so the numbers the label
            # promises appeared solely in the synchronous fallback. `evaluation_text`
            # is the same string the sync path returns; the decisions line is what
            # `GET /tasks/{id}/report` shows.
            body = (row.get("evaluation_text") or "").strip()
            decisions = task_service.planner_decisions(result_path)
            if decisions:
                chosen = ", ".join(f"{k}={v}" for k, v in sorted(decisions.items())
                                   if not isinstance(v, dict))
                body = f"{body}\nPipeline decisions: {chosen}" if body else \
                    f"Pipeline decisions: {chosen}"
            tail = _stage_lines(stages)
            text = "\n\n".join(part for part in (body, tail) if part)
            yield (result_path, f"{head}\n{text}", _comparison(input_image, result_path),
                   gr.update(value=result_path, visible=True))
            return
        if status == "failed":
            raise gr.Error(f"{label} failed: {_failure_text(row, options)}")
        if status == "cancelled":
            yield (None, f"⛔ {label} cancelled\n{_stage_lines(stages)}",
                   gr.update(), gr.update(visible=False))
            return

        progress = int(row.get("progress") or 0)
        current = row.get("current_stage") or ("queued" if status == "queued" else status)
        # The ETA is aged against when the *run* started, not when the user pressed
        # submit: a task that sat in the queue for ten minutes would otherwise get an
        # "ETA ~10 min" at its first 25 % tick, for work that has not begun.
        head = progress_head(
            label, current, progress,
            _elapsed_since_run_start(row, started), task_service.queue_depth(),
        )
        # A progress frame leaves the download control untouched (it keeps whatever
        # an earlier frame set), exactly like the comparison slider.
        yield None, f"{head}\n{_stage_lines(stages)}", gr.update(), gr.update()

        if time.monotonic() - started > _POLL_TIMEOUT:
            raise gr.Error("Timed out waiting for the task to finish")
        time.sleep(_POLL_INTERVAL)


def _elapsed_since_run_start(row: dict, submitted_at: float) -> float:
    """Seconds this task has been executing, or since submit if it never started."""
    started_at = row.get("started_at")
    if started_at:
        from fiximg.infrastructure.db import timestamps

        try:
            return max(0.0, timestamps.age_seconds(started_at))
        except Exception:  # noqa: BLE001 — a malformed instant is not the user's problem
            pass
    return max(0.0, time.monotonic() - submitted_at)


def _comparison(before, after):
    """Before/After slider update, or a no-op when either side is missing."""
    if before is None or after is None:
        return gr.update()
    return gr.update(value=(before, after))


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


def _hr_scratch_message() -> str:
    return ("High-resolution face path is unavailable for this photo. "
            + _HR_SCRATCH_HINT)


def _describe_failure(exc: Exception) -> str:
    """A user-facing sentence for a run that cannot proceed.

    Both submission paths go through this, so the queued and synchronous routes
    cannot describe the same refusal differently. The defect was that only one of
    them described it at all.
    """
    details = getattr(exc, "details", None) or {}
    if isinstance(exc, ModelUnavailableError) and details.get("hr"):
        return _hr_scratch_message()
    return str(exc).strip() or exc.__class__.__name__


#: A stable fragment of the backend's refusal, so the queued path can recognise
#: *this* refusal rather than any missing model. The error code alone is not enough:
#: a run with HR ticked that fails on an unrelated missing weight must not be told
#: the HR scratch branch is the problem. Matching prose is a compromise — the
#: alternative is misattributing failures — so the fragment is the backend's own
#: directory name, and it is asserted against the backend in the tests.
_HR_REFUSAL_MARKER = "mapping_Patch_Attention"


def _failure_text(row: dict, options: dict | None) -> str:
    """The queued path's version of :func:`_describe_failure`.

    Here the worker has already caught the failure, so it arrives as a row rather
    than as an exception, and the `details` dict that identifies the HR refusal is
    gone. So the same sentence is chosen from the error code, the options this
    submission sent, and the backend's own marker in the message.
    """
    message = row.get("error_message") or "unknown error"
    if ((row.get("error_code") == ErrorCode.MODEL_UNAVAILABLE)
            and (options or {}).get("hr")
            and _HR_REFUSAL_MARKER in message):
        return _hr_scratch_message()
    return message[:200]


def _sync_fallback(mode: str, input_image, user_state, options: dict | None = None):
    """V1-compatible synchronous path (inline execution, blocking).

    A domain failure is allowed to propagate: `make_submit_handler` translates it
    in one place for both routes, so the synchronous path cannot end up describing
    a refusal differently from the queued one.
    """
    res_img, evaluate_text = task_service.submit(mode, input_image, user_state,
                                                 options=options)
    return (res_img, evaluate_text or "done", _comparison(input_image, res_img),
            gr.update(value=res_img, visible=bool(res_img)))


def _check_password_change(user_state) -> None:
    """§21: raise a clear gr.Error for accounts that must change passwords."""
    from fiximg.domain.errors import PasswordChangeRequiredError

    gate = getattr(task_service, "_require_password_changed", None)
    if gate is None:
        return
    try:
        gate(user_state)
    except PasswordChangeRequiredError as exc:
        raise gr.Error(str(exc)) from exc


def build_switches(mode: str) -> list:
    """The §3.9.1 option checkboxes for one tab, in the order the handler reads them.

    An empty list for a mode with no switches, so the caller can splice the result
    into `inputs=[...]` without special-casing the colourisation and detection tabs.
    """
    from fiximg.application.pipeline_modes import SWITCHES, switch_label

    return [
        gr.Checkbox(label=switch_label(mode, name), value=default)
        for name, default in SWITCHES.get(mode, {}).items()
    ]


def switch_options(mode: str, names: list[str], values: tuple) -> dict:
    """Turn checkbox states into task options.

    Two kinds of mode, because two kinds of plan:

    * task-type plans (``restore`` / ``restore_scratch``): every switch is sent as
      typed. The drawn defaults reproduce the planner's own defaults, so a submit
      with nobody touching a box runs the same pipeline as no options at all.
    * analysis-derived plans (``auto``): a checked box *permits* what the analysis
      decided, so it sends nothing; only an unchecked one sends a value, and it sends
      ``false``. Sending ``true`` there would instruct the planner to run the stage
      even when the image says it is not needed, which is not what the user ticked.

    ``hr`` is not a pipeline switch on either kind: the face stages read it themselves,
    so it always goes as typed.
    """
    from fiximg.application.pipeline_modes import (
        ANALYSIS_DECIDED_MODES,
        PLAN_SWITCHES,
    )

    options = dict(zip(names, values, strict=True))
    if mode not in ANALYSIS_DECIDED_MODES:
        return options
    return {
        name: value
        for name, value in options.items()
        if name not in PLAN_SWITCHES or not value
    }


def build_preview(mode: str):
    """The §3.9.1 "Auto Analysis" row, or None for a tab whose plan the image does not steer.

    Only Auto Restore derives its pipeline from the picture, so only there does the
    analysis have a decision to explain — and its captions ("only when the analysis finds
    a face") are otherwise unverifiable from the UI.
    """
    from fiximg.application.pipeline_modes import ANALYSIS_DECIDED_MODES

    if mode not in ANALYSIS_DECIDED_MODES:
        return None
    return gr.Textbox(
        label="Auto analysis (before you submit)",
        value="",
        lines=2,
        interactive=False,
        elem_id=f"{mode}_analysis_preview",
    )


def describe_preview(preview: dict) -> str:
    """Render `TaskService.auto_preview()` as the row's text.

    The numbers come from `format_analysis_summary`, which is the same formatter the
    worker writes into its log — so the row, the log line and the run report cannot
    disagree about what the analyzer said. The stage names are the ones the task row will
    carry, for the same reason.
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


def make_preview_handler(mode: str):
    """Build the callback that fills the analysis row from the current upload.

    Takes ``(image, *switch_values)`` in the same order as
    :func:`make_submit_handler`, because the switches are part of the answer: refusing
    the face chain has to show a plan without it.
    """
    from fiximg.application.pipeline_modes import SWITCHES

    names = list(SWITCHES.get(mode, {}))

    def handler(input_image, *switch_values):
        if input_image is None:
            return gr.update(value="")
        if len(switch_values) != len(names):
            raise gr.Error(
                f"{mode} expects {len(names)} option(s), got {len(switch_values)}"
            )
        options = switch_options(mode, names, switch_values) if names else None
        return gr.update(value=describe_preview(task_service.auto_preview(input_image, options)))

    return handler


def make_submit_handler(mode: str):
    """Build a Gradio callback for a tab: enqueue -> poll -> yield progress.

    Yields ``(result_image, progress_text, comparison_slider_update, download)`` on
    every tick, ending with the result and the Before/After pair.

    The callback takes ``(image, user_state, *switch_values)``, where the switch
    values are exactly :func:`build_switches` order, and maps them with
    :func:`switch_options`. An unchecked box always means ``false``, never "whatever
    this task type defaults to" — a control the user can see must not have a second,
    invisible meaning.
    """
    from fiximg.application.pipeline_modes import SWITCHES

    names = list(SWITCHES.get(mode, {}))

    def handler(input_image, user_state, *switch_values):
        if input_image is None:
            raise gr.Error("Please upload an image first.")
        if len(switch_values) != len(names):
            # A tab wired with the wrong number of checkboxes would otherwise
            # `zip` away the switch the user last ticked and submit a plan they did
            # not ask for. Fail loudly instead of guessing which control is missing.
            raise gr.Error(
                f"{mode} expects {len(names)} option(s), got {len(switch_values)}"
            )
        _check_password_change(user_state)
        options = switch_options(mode, names, switch_values) if names else None
        try:
            if task_service.has_worker():
                log_event(logger, "INFO", "ui submit queued", mode=mode,
                          options=options)
                yield from _submit_and_stream(mode, input_image, user_state,
                                              options=options)
                return
            # No queue consumer available (e.g. FIXIMG_INLINE_WORKER=false without
            # a worker process): keep the old blocking behaviour instead of hanging.
            log_event(logger, "WARNING", "ui submit sync fallback", mode=mode,
                      hint="running inference in the API process because "
                           "has_worker() is false; set "
                           "FIXIMG_HAS_EXTERNAL_WORKER=true when a separate "
                           "worker consumes the queue",
                      consequence="this process loads its own copy of the models, "
                                  "which on one GPU ends in CUDA OOM")
            yield _sync_fallback(mode, input_image, user_state, options=options)
            return
        except (AppError, gr.Error) as exc:
            # The reason is written into the progress textbox and the callback
            # *returns* — it does not raise.
            #
            # Two earlier attempts were verified in the browser as `tester` on the
            # Auto Restore tab with the HR box ticked, and both failed to reach the
            # page. Raising produced the three result components flipping to the
            # generic "error" badge with no text anywhere, including inside the
            # shadow roots the toast renders in. Yielding the sentence and then
            # raising still failed: the value was in the textarea, but the error
            # state supersedes the rendered content, so the textbox stayed blank.
            #
            # So this does not rely on Gradio's error UI at all. The textbox is an
            # ordinary output, it renders whenever the components are not in an error
            # state, and it is where the user is already reading progress. The
            # leading marker keeps it legible as a failure without the red styling.
            text = (_describe_failure(exc) if isinstance(exc, AppError)
                    else str(exc).strip())
            yield (None, f"⚠ {text}", gr.update(), gr.update(visible=False))
            return

    return handler
