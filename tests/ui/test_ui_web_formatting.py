"""The web UI's pure formatting rules (plan §3.9.1 / §24 / §3.9.3).

These were Gradio panel internals: the only way to check them was to build a
`gr.Blocks` and read its config file, so the rules and the framework were tested
together. The native UI renders from the same rules server-side, which makes them
ordinary functions with ordinary tests — the properties below are the ones the
Gradio panel promised, moved over unchanged.

The ETA cases are the interesting ones: they encode "do not promise a number you
cannot support", which is a product decision that a refactor can quietly lose.
"""
from __future__ import annotations

import pytest

from fiximg.application.pipeline_modes import (
    ANALYSIS_DECIDED_MODES,
    PIPELINE_MODES,
    PLAN_SWITCHES,
    SWITCHES,
    SWITCH_LABELS,
    SWITCH_LABELS_BY_MODE,
    switch_label,
)
from fiximg.infrastructure.db import timestamps
from fiximg.ui import history as hist
from fiximg.ui import progress as tp

#: Two analyses for the Auto tab's switches to be measured against: one where the
#: analyzer wants no optional stage, one where it wants both. §10's thresholds put
#: `scratch_score` above the cut in each, so the restoration stage is not what is
#: under test here.
_ANALYSIS_CLEAN = {"width": 800, "height": 600, "is_grayscale": False,
                   "face_count": 0, "blur_score": 0.2, "scratch_score": 0.1}
_ANALYSIS_BUSY = {"width": 800, "height": 600, "is_grayscale": True,
                  "face_count": 2, "blur_score": 0.6, "scratch_score": 0.8}


# ------------------------------------------------------------------------ ETA
@pytest.mark.parametrize("progress,elapsed,expected", [
    (50, 30.0, "ETA  ~30 s"),
    (25, 60.0, "ETA  ~3 min 0 s"),
    (75, 15.0, "ETA  ~5 s"),
    # Not enough signal: the first meaningful tick of a four-stage plan is 25 %,
    # and a percentage measured over half a second is not a basis for a promise.
    (10, 600.0, "ETA  --"),
    (50, 0.4, "ETA  --"),
    (0, 120.0, "ETA  --"),
])
def test_eta_extrapolates_only_when_there_is_signal(progress, elapsed, expected):
    assert tp.eta_text(progress, elapsed) == expected


def test_an_empty_queue_is_reported_as_zero_not_as_unknown():
    assert tp.queue_text(0) == "Queue  0"
    assert tp.queue_text(2) == "Queue  2"
    assert tp.queue_text(None) == "Queue  ?"


def test_the_progress_head_carries_every_field_the_plan_asks_for():
    """§3.9.1: bar + percent, stage, ETA and queue in one line."""
    line = tp.progress_head("Restore (no scratches)", "face_enhancement", 50, 40.0, 3)

    assert line.startswith("[")
    assert "50%" in line
    assert "Restore (no scratches)" in line
    assert "face_enhancement" in line
    assert "ETA  ~40 s" in line
    assert "Queue  3" in line
    # One separator style, so the line cannot silently drop a field into the bar.
    assert line.count("·") == 5, line


def test_the_bar_is_clamped_so_a_bad_percent_cannot_break_the_layout():
    assert tp.bar(0) == "[" + "░" * 24 + "]"
    assert tp.bar(50) == "[" + "█" * 12 + "░" * 12 + "]"
    assert tp.bar(100) == "[" + "█" * 24 + "]"
    # A percent outside 0..100 is a bug upstream, not a reason to draw 30 cells.
    assert tp.bar(-10) == tp.bar(0)
    assert tp.bar(140) == tp.bar(100)


def test_the_eta_is_aged_against_the_run_not_against_the_submit():
    """A task queued for ten minutes must not open with "ETA ~10 min".

    `started_at` is the worker's own mark; while the task sat in the queue the run
    had not begun, so the progress-to-time ratio has to be measured from it.
    """
    row_started_30s_ago = timestamps.deadline(-30.0)
    assert tp.elapsed_since_run_start({"started_at": row_started_30s_ago}, 0.0) == pytest.approx(
        30.0, abs=2.0
    )
    # No run start yet: fall back to the poll's own clock.
    import time

    assert tp.elapsed_since_run_start({"status": "queued"}, time.monotonic() - 12.0) == (
        pytest.approx(12.0, abs=1.0)
    )
    assert tp.elapsed_since_run_start({"started_at": "not-a-timestamp"},
                                      time.monotonic() - 12.0) >= 0.0


# ------------------------------------------------------------------- the panel
def test_stage_lines_render_each_state_with_its_own_marker():
    stages = [
        {"stage_name": "global_restore", "status": "completed", "duration_ms": 3200},
        {"stage_name": "face_enhancement", "status": "running", "duration_ms": None},
        {"stage_name": "evaluation", "status": "pending", "duration_ms": None},
    ]
    lines = tp.stage_lines(stages)
    assert "✓ global_restore (3.2 s)" in lines
    assert "▶ face_enhancement" in lines
    assert "○ evaluation" in lines

    failed = [{"stage_name": "global_restore", "status": "failed",
               "duration_ms": 100, "message": "Subprocess failed"}]
    assert "✗ global_restore (0.1 s) — Subprocess failed" in tp.stage_lines(failed)


def test_skipped_stage_is_marked_and_explains_itself():
    """`skipped` is not `completed`.

    `StageStatus.SKIPPED` was declared for several batches with no writer, so every
    skip was persisted as `completed`: the panel showed a green tick for a stage
    that never ran, the task claimed more than it did, and a photo whose face chain
    was skipped because dlib is missing looked exactly like one where enhancement
    succeeded. The reason has to travel with the status, or the reader cannot tell a
    skip from a success.
    """
    out = tp.stage_lines([
        {"stage_name": "face_detection", "status": "skipped",
         "duration_ms": None, "message": "dlib is not installed"},
        {"stage_name": "global_restore", "status": "completed", "duration_ms": 3200},
    ])

    skip_line = next(ln for ln in out.splitlines() if "face_detection" in ln)
    done_line = next(ln for ln in out.splitlines() if "global_restore" in ln)

    assert skip_line.startswith("⊘"), ascii(skip_line)
    assert "dlib is not installed" in skip_line
    # A skip carries no duration, and must not borrow one.
    assert "s)" not in skip_line
    # And it is visibly not the green tick.
    assert done_line.startswith("✓"), ascii(done_line)
    assert skip_line[0] != done_line[0]


def test_a_skip_without_a_reason_still_marks_itself():
    """The status is the load-bearing part; the reason is a courtesy."""
    out = tp.stage_lines([
        {"stage_name": "warp_back", "status": "skipped", "duration_ms": None},
    ])
    assert out.strip().startswith("⊘"), ascii(out)
    assert "warp_back" in out


def test_the_stage_rows_and_the_stage_text_describe_the_same_frames():
    """Two renderings of one rule, so the browser and the text cannot disagree."""
    stages = [
        {"stage_name": "global_restore", "status": "completed", "duration_ms": 3200},
        {"stage_name": "face_detection", "status": "skipped", "duration_ms": None,
         "message": "dlib is not installed"},
        {"stage_name": "evaluation", "status": "failed", "duration_ms": 40,
         "message": "the evaluator raised"},
    ]
    rows = tp.stage_rows(stages)
    text = tp.stage_lines(stages)
    assert [row["icon"] for row in rows] == ["✓", "⊘", "✗"]
    for row in rows:
        assert row["name"] in text
    # Only the two that owe an explanation carry one.
    assert rows[0]["reason"] == ""
    assert "dlib" in rows[1]["reason"]
    assert "evaluator" in rows[2]["reason"]


def test_the_progress_frame_reports_a_queued_task_without_a_fabricated_eta():
    frame = tp.status_text(
        {"id": "t-1", "task_type": "restore", "status": "queued", "progress": 0,
         "stages": []},
        queue_depth=4,
    )
    assert frame["status"] == "queued"
    assert "Queue  4" in frame["text"]
    assert "ETA  --" in frame["text"]
    assert frame["completed"] is False
    assert frame["result_ready"] is False


def test_the_progress_frame_reports_a_completed_task_with_its_metrics_and_stages():
    frame = tp.status_text(
        {"id": "t-2", "task_type": "restore", "status": "completed", "progress": 100,
         "duration_ms": 4200, "result_path": "/tmp/out.png",
         "evaluation_text": "PSNR: 20",
         "stages": [{"stage_name": "global_restore", "status": "completed",
                     "duration_ms": 4200}]},
        decisions={"colorize": False},
    )
    assert frame["completed"] is True
    assert frame["result_ready"] is True
    assert "completed in 4.2 s" in frame["text"]
    assert "PSNR: 20" in frame["text"]
    assert "Pipeline decisions: colorize=False" in frame["text"]
    assert "✓ global_restore (4.2 s)" in frame["text"]


def test_the_progress_frame_renders_the_stages_the_repository_actually_returns(
    isolated_db,
):
    """The panel's own double was more generous than production.

    `get_task` returns `stages` as JSON **text** on SQLite (psycopg decodes
    `json_agg` for us, `json_group_array` is a string), and the panel treated any
    string as "no stages yet". Every panel test built its rows as lists, so on the
    default engine the §24 checklist never appeared in a single frame while the
    suite stayed green.
    """
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    task_repo.create_task("panel-1", "restore", "alice")
    task_repo.start_task("panel-1")
    task_repo.record_stage("panel-1", 0, "global_restore", "running")
    real_row = task_repo.get_task("panel-1")
    assert isinstance(real_row["stages"], str), (
        "this test is only worth writing if SQLite hands back JSON text"
    )
    frame = tp.status_text(real_row)
    assert "▶ global_restore" in frame["text"], frame["text"]

    task_repo.finish_stage("panel-1", 0, "completed", 3200, None)
    done = tp.status_text({**task_repo.get_task("panel-1"), "status": "completed",
                           "progress": 100, "result_path": "/tmp/out.png",
                           "evaluation_text": "PSNR: 20", "duration_ms": 4200})
    assert "✓ global_restore (3.2 s)" in done["text"], done["text"]


# ------------------------------------------------------------------- switches
def test_the_switches_a_tab_offers_are_option_keys_the_api_accepts():
    """A checkbox the backend cannot parse would be rejected by `TaskOptionsSchema`.

    Derived from the schema rather than from a list kept beside the UI, so adding a
    switch that the options parser does not know about fails here instead of at the
    user's submit.
    """
    from fiximg.api.schemas.task import TaskOptionsSchema

    accepted = set(TaskOptionsSchema.model_fields)
    offered = {name for names in SWITCHES.values() for name in names}

    assert offered, "no tab offers any switch"
    assert offered <= accepted, f"UI offers unknown options: {sorted(offered - accepted)}"
    # §3.9.1 names these three; a switch nobody can toggle is a layout regression.
    assert offered == {"face_enhance", "auto_colorize", "hr"}, sorted(offered)


def test_every_offered_switch_default_is_a_boolean():
    for mode, names in SWITCHES.items():
        for name, default in names.items():
            assert isinstance(default, bool), f"{mode}.{name} defaults to {default!r}"


def test_tabs_without_switches_send_no_options():
    """Colourisation has no face chain to turn off, so it has nothing to send."""
    assert SWITCHES["colorize"] == {}
    assert SWITCHES["detect"] == {}


def test_the_restoration_tabs_send_every_switch_as_typed():
    names = list(SWITCHES["restore"])
    assert tp.switch_options("restore", names, (False, True, False)) == {
        "face_enhance": False, "auto_colorize": True, "hr": False,
    }


def test_a_checked_auto_box_sends_nothing_because_the_analysis_is_still_the_authority():
    """The Auto tab's two states are "as the image decided" and "never", not "always".

    Sending `true` for a checked box would tell the planner to run the face chain on
    a face-less photo — the opposite of what a tab whose whole premise is automatic
    analysis has promised.
    """
    names = list(SWITCHES["auto"])
    assert tp.switch_options("auto", names, (True, True, False)) == {"hr": False}
    assert tp.switch_options("auto", names, (True, False, True)) == {
        "auto_colorize": False, "hr": True,
    }
    # `hr` is not a pipeline switch: the face stages read it themselves, so it always
    # goes as typed, checked or not.
    assert tp.switch_options("auto", names, (True, True, True)) == {"hr": True}


def test_the_two_states_of_an_auto_box_are_two_different_plans():
    """A checkbox whose checked and unchecked states plan the same pipeline is decoration.

    The relational check ("as drawn = as if nothing were sent") cannot see a planner
    that reads an absent switch the same way as a refusal: both sides of the
    comparison collapse together. This one measures the two box states against each
    other, so the collapse is caught from the UI's side too.
    """
    from fiximg.inference.planner import PipelinePlanner

    planner = PipelinePlanner()
    names = list(SWITCHES["auto"])
    on = tp.switch_options("auto", names, (True, True, False))
    off = tp.switch_options("auto", names, (False, False, False))

    allowed = [n for n, _ in planner.plan_auto_restore(_ANALYSIS_BUSY, on).stages]
    refused = [n for n, _ in planner.plan_auto_restore(_ANALYSIS_BUSY, off).stages]
    assert allowed == [
        "global_restore", "face_detection", "face_enhancement", "warp_back", "colorization",
    ], allowed
    assert refused == ["global_restore"], refused
    # The checked state must also stay unforced: nothing the analysis did not ask for.
    assert [n for n, _ in planner.plan_auto_restore(_ANALYSIS_CLEAN, on).stages] == [
        "global_restore"
    ]


def test_the_switches_as_drawn_submit_the_same_plan_as_no_options_at_all():
    """Opening a tab must not silently change what runs.

    The defaults are the UI's claim about the backend's defaults, so the check is
    against the planner with an empty options dict — an independent source. A default
    flipped without anyone touching a box would submit a different pipeline than
    "no preferences".
    """
    from fiximg.inference.planner import PipelinePlanner

    offered = sorted(m for m, names in SWITCHES.items() if names)
    assert offered == ["auto", "restore", "restore_scratch"], offered
    planner = PipelinePlanner()

    for mode in offered:
        names = list(SWITCHES[mode])
        drawn = tp.switch_options(mode, names, tuple(SWITCHES[mode].values()))
        if mode in ANALYSIS_DECIDED_MODES:
            for analysis in (_ANALYSIS_CLEAN, _ANALYSIS_BUSY):
                assert [n for n, _ in planner.plan_auto_restore(analysis, drawn).stages] == [
                    n for n, _ in planner.plan_auto_restore(analysis).stages
                ], f"{mode}: drawn defaults change the plan for {analysis}"
        else:
            task_type = PIPELINE_MODES[mode]["task_type"]
            assert [n for n, _ in planner.plan(task_type, drawn).stages] == [
                n for n, _ in planner.plan(task_type, {}).stages
            ], f"{mode}: drawn defaults change the plan"


def test_every_analysis_decided_mode_offers_the_switches_it_relabels():
    """A mode listed as analysis-derived must offer both plan switches and a caption.

    Otherwise the decline-only mapping silently applies to keys that are not on the
    tab (so nothing changes), or a caption is written for a checkbox nobody builds.
    """
    assert ANALYSIS_DECIDED_MODES <= set(SWITCHES), sorted(ANALYSIS_DECIDED_MODES)
    for mode in sorted(ANALYSIS_DECIDED_MODES):
        offered = set(SWITCHES[mode])
        assert PLAN_SWITCHES <= offered, f"{mode} is analysis-decided but offers no switch"
        assert set(SWITCH_LABELS_BY_MODE.get(mode, {})) <= offered, f"{mode}: stale caption"


def test_the_decline_only_mapping_is_applied_exactly_where_the_label_says():
    """The table the UI reads and the table the captions live in must agree."""
    for mode in sorted(SWITCHES):
        if mode not in ANALYSIS_DECIDED_MODES:
            assert mode not in SWITCH_LABELS_BY_MODE, f"{mode} rewrites captions it does not gate"


def test_the_auto_tab_words_its_switches_differently_from_the_restoration_tabs():
    """The same key means "when the image says so" here and "run it" there.

    Reusing the restoration wording would promise a stage the checkbox does not
    schedule.
    """
    auto = [switch_label("auto", name) for name in SWITCHES["auto"]]
    restore = [switch_label("restore", name) for name in SWITCHES["restore"]]
    assert auto[:2] != restore[:2], (auto, restore)
    assert auto[0] == "Face enhancement (only when the analysis finds a face)"
    assert restore[0] == "Face enhancement (detect, enhance, warp back)"
    # The third switch is the same control on both, and says so.
    assert auto[2] == restore[2] == SWITCH_LABELS["hr"]


def test_every_switch_caption_is_a_sentence_not_a_key():
    for mode, names in SWITCHES.items():
        for name in names:
            caption = switch_label(mode, name)
            assert caption != name, f"{mode}.{name} has no caption"
            assert " " in caption, f"{mode}.{name}: {caption!r} is not a sentence"


# --------------------------------------------------------------------- analysis
def test_describe_preview_reports_the_plan_the_run_will_make():
    preview = {
        "available": True,
        "summary": "800x600, colour, 2 face(s)",
        "stages": ["global_restore", "face_detection"],
        "forced": [],
        "declined": ["colorization"],
    }
    text = tp.describe_preview(preview)
    assert "Analysis: 800x600, colour, 2 face(s)" in text
    assert "Planned pipeline: global_restore → face_detection" in text
    assert "Declined by your switches: colorization" in text
    assert "Forced" not in text


def test_describe_preview_says_why_it_has_no_answer():
    """An unavailable analysis is a sentence, not an empty box."""
    text = tp.describe_preview({"available": False, "reason": "OSError: unreadable"})
    assert text == "Analysis unavailable: OSError: unreadable"
    assert tp.describe_preview({"available": False}) == "Analysis unavailable: unknown error"


def test_describe_preview_names_an_empty_plan_rather_than_showing_nothing():
    text = tp.describe_preview({
        "available": True, "summary": "s", "stages": [], "forced": [], "declined": [],
    })
    assert "Planned pipeline: (nothing)" in text


# --------------------------------------------------------------------- failures
def test_the_hr_scratch_refusal_is_explained_as_an_action():
    """The domain message says what the system found; this says what to change."""
    from fiximg.domain.errors import ModelUnavailableError

    exc = ModelUnavailableError("weights missing", details={"hr": True})
    text = tp.describe_failure(exc)
    assert "High-resolution face path is unavailable" in text
    assert "Untick" in text
    # An unrelated missing model must not be described as the HR refusal.
    plain = ModelUnavailableError("ddcolor weights missing")
    assert "High-resolution" not in tp.describe_failure(plain)


def test_the_queued_path_describes_the_hr_refusal_the_same_way():
    """The worker catches the failure, so the same sentence is chosen from the row."""
    from fiximg.domain.errors import ErrorCode

    row = {"error_code": ErrorCode.MODEL_UNAVAILABLE,
           "error_message": "mapping_Patch_Attention weights are not published"}
    assert "High-resolution" in tp.failure_text(row, {"hr": True})
    # Without the option, or without the marker, it is an ordinary failure.
    assert "High-resolution" not in tp.failure_text(row, {"hr": False})
    other = {"error_code": ErrorCode.MODEL_UNAVAILABLE, "error_message": "some other weight"}
    assert tp.failure_text(other, {"hr": True}) == "some other weight"


def test_a_failure_message_is_truncated_to_what_a_status_box_holds():
    long_message = "x" * 500
    assert len(tp.failure_text({"error_message": long_message}, {})) == 200


# --------------------------------------------------------------------- history
def test_duration_text_formats_seconds_minutes_and_the_absent_case():
    assert hist.duration_text(None) == "—"
    assert hist.duration_text(0) == "—"
    assert hist.duration_text(4200) == "4.2 s"
    assert hist.duration_text(95_000) == "1m 35s"


def test_metrics_text_is_compact_and_ordered():
    text = hist.metrics_text({"mae": 1.2, "psnr": 24.0, "ssim": 0.9, "other": 1})
    assert text == "PSNR=24.0  SSIM=0.9  MAE=1.2"
    assert hist.metrics_text({}) == ""
    assert hist.metrics_text("not a dict") == ""


def test_stages_text_is_an_arrow_separated_plan():
    assert hist.stages_text([{"stage_name": "a"}, {"stage_name": "b"}]) == "a → b"
    assert hist.stages_text([]) == ""
    assert hist.stages_text("nope") == ""


def test_when_text_renders_the_viewers_wall_clock():
    """Storage is UTC; a person reads local (plan §2.7)."""
    from datetime import datetime

    stored = "2026-06-01T00:00:00.000000Z"
    expected = (
        datetime.fromisoformat(stored.replace("Z", "+00:00"))
        .astimezone()
        .strftime("%Y-%m-%d %H:%M:%S")
    )
    assert hist.when_text(stored) == expected


def test_when_text_passes_through_what_the_codec_cannot_read():
    """A V1 row predates the codec; blanking the column would lose the row's date."""
    assert hist.when_text("2026-06-01 08:00:00") == "2026-06-01 08:00:00"
    assert hist.when_text("") == ""
    assert hist.when_text(None) == ""


def test_the_detail_text_names_the_task_its_plan_and_its_metrics():
    import json

    row = {
        "id": "t-1", "task_type": "restore", "status": "completed", "duration_ms": 4200,
        "stages": json.dumps([{"stage_name": "global_restore"}]),
        "metrics": json.dumps({"psnr": 24.5, "ssim": 0.91}),
    }
    text = hist.detail_text(row)
    assert "Task: t-1" in text
    assert "Type: restore" in text
    assert "Stages: global_restore" in text
    assert "PSNR=24.5" in text


def test_the_detail_text_explains_a_missing_result():
    """A blank preview with no sentence beside it reads as a broken page."""
    text = hist.detail_text({"id": "t-2", "status": "completed"}, result_present=False)
    assert "reclaimed" in text


def test_the_detail_text_reports_an_error_and_its_code():
    text = hist.detail_text({
        "id": "t-3", "status": "failed",
        "error_code": "PIPELINE_FAILED", "error_message": "boom",
    })
    assert "Error code: PIPELINE_FAILED" in text
    assert "Error: boom" in text


def test_rerun_options_drop_internal_keys():
    """A ground-truth path from the old run must not be forwarded."""
    import json

    row = {"options_json": json.dumps(
        {"hr": False, "_ground_truth_path": "/old/gt.png"})}
    assert hist.rerun_options(row) == {"hr": False}


def test_rerun_options_are_empty_for_a_row_with_none():
    assert hist.rerun_options({}) == {}
    assert hist.rerun_options({"options_json": None}) == {}
    import json

    assert hist.rerun_options({"options_json": json.dumps(["not", "a", "dict"])}) == {}


def test_every_pipeline_mode_has_a_ui_mode_name():
    """A task type the history table can show must be re-runnable from it."""
    from fiximg.application import pipeline_modes

    for mode, cfg in pipeline_modes.PIPELINE_MODES.items():
        assert cfg["task_type"], mode
        assert cfg["label"], mode
        assert tp.task_label(cfg["task_type"]) == cfg["label"]
