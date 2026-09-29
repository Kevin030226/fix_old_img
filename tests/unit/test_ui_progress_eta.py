"""搂3.9.1's progress panel: the ETA, the queue readout, and the option checkboxes.

The panel used to show a bar, a percent and a stage name. The plan asks for four
things in that strip 鈥?current stage, total progress, ETA, queue 鈥?and for three
switches in the input area; neither half existed.
"""
from __future__ import annotations

import pytest

from fiximg.application.pipeline_modes import (
    ANALYSIS_DECIDED_MODES,
    PIPELINE_MODES,
    SWITCHES,
)
from fiximg.infrastructure.db import timestamps
from fiximg.ui import task_progress as tp
from tests.fixtures import (
    components_by_id,
    demo_config,
    deps_writing_to,
    submit_deps as _tab_submit_deps,
)

#: Two analyses for the Auto tab's switches to be measured against: one where the
#: analyzer wants no optional stage, one where it wants both. 搂10's thresholds put
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
    """搂3.9.1: bar + percent, stage, ETA and queue in one line."""
    line = tp.progress_head("Restore (no scratches)", "face_enhancement", 50, 40.0, 3)

    assert line.startswith("[")
    assert "50%" in line
    assert "Restore (no scratches)" in line
    assert "face_enhancement" in line
    assert "ETA  ~40 s" in line
    assert "Queue  3" in line
    # One separator style, so the line cannot silently drop a field into the bar.
    assert line.count("·") == 5, line


def test_the_eta_is_aged_against_the_run_not_against_the_submit():
    """A task queued for ten minutes must not open with "ETA ~10 min".

    `started_at` is the worker's own mark; while the task sat in the queue the run
    had not begun, so the progress-to-time ratio has to be measured from it.
    """
    row_started_30s_ago = timestamps.deadline(-30.0)
    assert tp._elapsed_since_run_start({"started_at": row_started_30s_ago}, 0.0) == pytest.approx(
        30.0, abs=2.0
    )
    # No run start yet: fall back to the poll's own clock.
    assert tp._elapsed_since_run_start({"status": "queued"}, 12.0) == pytest.approx(
        tp.time.monotonic() - 12.0, abs=1.0
    )
    assert tp._elapsed_since_run_start({"started_at": "not-a-timestamp"}, 12.0) >= 0.0


# ------------------------------------------------------------------- the panel
class _PanelService:
    """Queue-backed task service double: one running tick, then completed."""

    def __init__(self, rows, depth):
        self._rows = list(rows)
        self.depth = depth
        self.enqueued = []

    def has_worker(self):
        return True

    def queue_depth(self):
        return self.depth

    def enqueue(self, image, user_state, mode, options=None):
        self.enqueued.append(options)
        return "tid-1"

    def get_task(self, task_id):
        return self._rows.pop(0) if self._rows else self._rows_done

    def planner_decisions(self, result_path):
        return {}

    _rows_done = {
        "status": "completed", "progress": 100, "duration_ms": 4200,
        "result_path": "final.png", "evaluation_text": "PSNR: 20",
        "stages": [{"stage_name": "global_restore", "status": "completed"}],
    }


def _drive(monkeypatch, rows, depth):
    import gradio as gr

    service = _PanelService(rows, depth)
    monkeypatch.setattr(tp, "task_service", service)
    monkeypatch.setattr(tp.time, "sleep", lambda *_: None)
    handler = tp.make_submit_handler("restore")
    frames = list(handler("img", {"username": "u"}, True, False, False))
    return service, frames, gr


def test_a_queued_task_shows_the_backlog_without_a_fabricated_eta(monkeypatch):
    service, frames, _gr = _drive(monkeypatch, [
        {"status": "queued", "progress": 0, "current_stage": None, "stages": []},
    ], depth=4)

    text = frames[0][1]
    assert "Queue  4" in text, text
    assert "ETA  --" in text, text
    assert service.enqueued == [{"face_enhance": True, "auto_colorize": False, "hr": False}]


def test_a_running_task_shows_eta_and_backlog_together(monkeypatch):
    _service, frames, _gr = _drive(monkeypatch, [
        {"status": "running", "progress": 50, "current_stage": "face_enhancement",
         "started_at": timestamps.deadline(-20.0),
         "stages": [{"stage_name": "global_restore", "status": "completed",
                     "duration_ms": 20000}]},
    ], depth=0)

    text = frames[0][1]
    assert "ETA  ~20 s" in text, text
    assert "Queue  0" in text, text
    assert "✓ global_restore (20.0 s)" in text, text


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
    # 搂3.9.1 names these three; a switch nobody can toggle is a layout regression.
    assert offered == {"face_enhance", "auto_colorize", "hr"}, sorted(offered)


def test_every_offered_switch_default_is_a_boolean():
    for mode, names in SWITCHES.items():
        for name, default in names.items():
                    assert isinstance(default, bool), f"{mode}.{name} defaults to {default!r}"


def test_the_face_chain_stays_on_by_default_and_off_when_unchecked(monkeypatch):
    """An unchecked box means `false`, not "whatever this task type defaults to"."""
    service, _frames, _gr = _drive(monkeypatch, [], depth=0)
    handler = tp.make_submit_handler("restore")
    list(handler("img", {"username": "u"}, False, True, False))

    assert service.enqueued[-1] == {"face_enhance": False, "auto_colorize": True, "hr": False}


def test_tabs_without_switches_send_no_options(monkeypatch):
    """Colourisation has no face chain to turn off, so it has nothing to send."""
    assert tp.build_switches("colorize") == []
    assert tp.build_switches("detect") == []

    service = _PanelService([], depth=0)
    monkeypatch.setattr(tp, "task_service", service)
    handler = tp.make_submit_handler("colorize")
    list(handler("img", {"username": "u"}))
    assert service.enqueued == [None]


def test_a_mismatched_control_set_is_refused_not_silently_truncated(monkeypatch):
    """One missing checkbox would drop the last switch from the submitted options.

    `zip` stops at the shorter iterable, so a mis-wired tab submits a plan the user
    did not choose with no error anywhere. The guard makes it loud.
    """
    import gradio as gr

    service = _PanelService([], depth=0)
    monkeypatch.setattr(tp, "task_service", service)
    handler = tp.make_submit_handler("restore")

    with pytest.raises(gr.Error, match="expects 3 option"):
        list(handler("img", {"username": "u"}, True, False))
    assert service.enqueued == [], "the call must not reach the queue"


def _preview_label():
    """The row's caption, taken from the builder rather than retyped here."""
    return tp.build_preview("auto").label


def test_every_submit_tab_wires_exactly_the_switches_its_mode_declares():
    """The layout and the handler are built from one table; assert they agree.

    Read out of Gradio's own dependency config, so a tab that passes the wrong
    `switches_*` list (or forgets to pass any) is caught here rather than by a user
    whose checkbox visibly did nothing.

    The submit callbacks are selected by shape (state in, result image out) rather than by
    their auto-generated `api_name`, because the analysis row's callbacks take names from
    the same counter 鈥?a prefix match would quietly start counting those too.
    """
    cfg = demo_config()
    submit_deps = _tab_submit_deps(cfg)
    assert len(submit_deps) == len(PIPELINE_MODES), [d.get("api_name") for d in submit_deps]

    expected = {2, 2 + len(SWITCHES["restore"])}
    counts = {len(dep["inputs"]) for dep in submit_deps}
    assert counts <= expected, f"unexpected input arity per tab: {sorted(counts)}"
    # Every mode that offers switches must have them wired, and every mode that
    # offers none must be called with exactly image + state.
    with_switches = sum(1 for dep in submit_deps if len(dep["inputs"]) > 2)
    assert with_switches == sum(1 for names in SWITCHES.values() if names)


def test_the_analysis_row_recomputes_on_the_upload_and_on_every_switch_it_reads():
    """搂3.9.1's row is a live answer, not a one-shot caption.

    One dependency per trigger source 鈥?the upload plus each switch 鈥?each reading
    `[image, *switches]` in the submit handler's order and writing only to the row. A
    wiring that missed the switches would keep showing the plan for the previous tick,
    which is a confident wrong answer rather than a missing one.
    """
    cfg = demo_config()
    by_id = components_by_id(cfg)
    preview_deps = deps_writing_to(cfg, _preview_label())

    assert len(preview_deps) == 1 + len(SWITCHES["auto"]), len(preview_deps)
    triggers = [target[1] for dep in preview_deps for target in dep["targets"]]
    assert set(triggers) == {"change"}, triggers

    for dep in preview_deps:
        kinds = [by_id[i]["type"] for i in dep["inputs"]]
        assert kinds == ["image", "checkbox", "checkbox", "checkbox"], kinds
        assert not any(by_id[i]["type"] == "state" for i in dep["inputs"]), \
            "the row reads the picture, not the account"

    # The row exists once, and it is read-only.
    label = _preview_label()
    boxes = [c for c in cfg["components"] if (c.get("props") or {}).get("label") == label]
    assert len(boxes) == 1, len(boxes)
    assert boxes[0]["props"]["interactive"] is False, boxes[0]["props"]


def test_build_switches_returns_one_checkbox_per_declared_switch():
    import gradio as gr

    boxes = tp.build_switches("restore")
    assert len(boxes) == len(SWITCHES["restore"])
    assert all(isinstance(box, gr.Checkbox) for box in boxes)
    # Literals, not `PIPELINE_MODES[...]` again: an expectation read back from the
    # value under test passes whatever the default is set to.
    assert [box.value for box in boxes] == [True, False, False]
    assert [box.label for box in boxes] == [
        "Face enhancement (detect, enhance, warp back)",
        "Auto colorize the result",
        "High-resolution face path (needs the HR weights)",
    ]


def test_the_switches_as_drawn_submit_the_same_plan_as_no_options_at_all():
    """Opening a tab must not silently change what runs.

    The defaults are the UI's claim about the backend's defaults, so the check is
    against the planner with an empty options dict 鈥?an independent source. A default
    flipped without anyone touching a box would submit a different pipeline than
    "no preferences".

    Auto Restore is checked against two analyses, because its switches only *permit*
    what the image decided: a checked box that sent ``true`` would add stages the
    analyzer said were unnecessary, so the equality has to hold whether the analysis
    wants those stages or not.
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


def test_unchecking_a_box_is_the_only_way_to_refuse_a_stage(monkeypatch):
    """Unchecked must send ``false``, on the analysis-derived tab too.

    An option mapping that dropped a refusal (sending nothing instead) would leave the
    analyzer in charge of a stage the user just turned off 鈥?and on Auto, where the
    checked state also sends nothing, both states of the box would then mean the same
    thing, which is a control that does nothing.
    """
    service = _PanelService([], depth=0)
    monkeypatch.setattr(tp, "task_service", service)
    monkeypatch.setattr(tp.time, "sleep", lambda *_: None)

    list(tp.make_submit_handler("auto")("img", {"username": "u"}, False, False, False))
    assert service.enqueued[-1] == {
        "face_enhance": False, "auto_colorize": False, "hr": False
    }

    from fiximg.inference.planner import PipelinePlanner

    declined = service.enqueued[-1]
    refused = [n for n, _ in PipelinePlanner().plan_auto_restore(_ANALYSIS_BUSY, declined).stages]
    assert refused == ["global_restore"], refused


def test_a_checked_auto_box_sends_nothing_because_the_analysis_is_still_the_authority():
    """The Auto tab's two states are "as the image decided" and "never", not "always".

    Sending ``true`` for a checked box would tell the planner to run the face chain on
    a face-less photo 鈥?the opposite of what a tab whose whole premise is automatic
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

    The relational checks above ("as drawn = as if nothing were sent") cannot see a
    planner that reads an absent switch the same way as a refusal: both sides of the
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


def test_the_restoration_tabs_send_every_switch_as_typed():
    names = list(SWITCHES["restore"])
    assert tp.switch_options("restore", names, (False, True, False)) == {
        "face_enhance": False, "auto_colorize": True, "hr": False,
    }


def test_the_auto_tab_offers_the_switches_the_dynamic_plan_now_honours():
    """搂3.9.1's parameter row on the one-click tab, with its own captions.

    The captions have to differ from the restoration tab's: the same key means
    "when the image says so" here and "run it" there, and reusing the wording would
    promise a stage the checkbox does not schedule.
    """
    import gradio as gr

    boxes = tp.build_switches("auto")
    assert all(isinstance(box, gr.Checkbox) for box in boxes)
    assert [box.value for box in boxes] == [True, True, False]
    assert [box.label for box in boxes] == [
        "Face enhancement (only when the analysis finds a face)",
        "Auto colorize (only when the photo is black and white)",
        "High-resolution face path (needs the HR weights)",
    ]
    restore_boxes = tp.build_switches("restore")
    assert [box.label for box in boxes[:2]] != [box.label for box in restore_boxes[:2]]


def test_every_analysis_decided_mode_offers_the_switches_it_relabels():
    """A mode listed as analysis-derived must offer both plan switches and a caption.

    Otherwise the decline-only mapping silently applies to keys that are not on the tab
    (so nothing changes), or a caption is written for a checkbox nobody builds.
    """
    from fiximg.application.pipeline_modes import (
        PLAN_SWITCHES,
        SWITCH_LABELS_BY_MODE,
    )

    assert ANALYSIS_DECIDED_MODES <= set(SWITCHES), sorted(ANALYSIS_DECIDED_MODES)
    for mode in sorted(ANALYSIS_DECIDED_MODES):
        offered = set(SWITCHES[mode])
        assert PLAN_SWITCHES <= offered, f"{mode} is analysis-decided but offers no switch"
        assert set(SWITCH_LABELS_BY_MODE.get(mode, {})) <= offered, f"{mode}: stale caption"


def test_the_decline_only_mapping_is_applied_exactly_where_the_label_says():
    """The table the UI reads and the table the captions live in must agree."""
    from fiximg.application.pipeline_modes import SWITCH_LABELS_BY_MODE

    for mode in sorted(SWITCHES):
        if mode not in ANALYSIS_DECIDED_MODES:
            assert mode not in SWITCH_LABELS_BY_MODE, f"{mode} rewrites captions it does not gate"


# ------------------------------------------------------------- the queue source
def test_the_progress_panel_renders_the_stages_the_repository_actually_returns(
    isolated_db, monkeypatch
):
    """The panel's own double was more generous than production.

    ``get_task`` returns ``stages`` as JSON **text** on SQLite (psycopg decodes
    ``json_agg`` for us, ``json_group_array`` is a string), and the panel treated any
    string as "no stages yet". Every panel test built its rows as lists, so on the
    default engine the 搂24 checklist never appeared in a single frame while the suite
    stayed green 鈥?found only by reading the frames a live run produced.
    """
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    task_repo.create_task("panel-1", "restore", "alice")
    task_repo.start_task("panel-1")
    task_repo.record_stage("panel-1", 0, "global_restore", "running")
    real_row = task_repo.get_task("panel-1")
    assert isinstance(real_row["stages"], str), (
        "this test is only worth writing if SQLite hands back JSON text"
    )

    task_repo.finish_stage("panel-1", 0, "completed", 3200, None)
    done_row = {**task_repo.get_task("panel-1"), "status": "completed",
                "progress": 100, "result_path": "final.png",
                "evaluation_text": "PSNR: 20", "duration_ms": 4200}

    service = _PanelService([real_row, done_row], depth=0)
    monkeypatch.setattr(tp, "task_service", service)
    monkeypatch.setattr(tp.time, "sleep", lambda *_: None)
    frames = list(tp.make_submit_handler("restore")("img", {"username": "u"},
                                                   True, False, False))

    running = frames[0][1]
    assert "▶ global_restore" in running, running
    completed = frames[-1][1]
    assert "✓ global_restore (3.2 s)" in completed, completed


def test_a_stage_list_that_is_already_decoded_still_renders(monkeypatch):
    """The other engine's shape: a list must pass through untouched, not double-decode."""
    row = {"status": "running", "progress": 25, "current_stage": "face_detection",
           "stages": [{"stage_name": "global_restore", "status": "completed",
                       "duration_ms": 1000}]}
    service = _PanelService([row], depth=0)
    monkeypatch.setattr(tp, "task_service", service)
    monkeypatch.setattr(tp.time, "sleep", lambda *_: None)
    frames = list(tp.make_submit_handler("restore")("img", {"username": "u"},
                                                    True, False, False))

    assert "✓ global_restore (1.0 s)" in frames[0][1], frames[0][1]


# ------------------------------------------------------------- the queue source
def test_the_service_reports_the_deployment_backlog(isolated_db):
    """`queue_depth()` reads the queue, so a UI in one process counts another's tasks."""
    from fiximg.application.task_service import task_service
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    task_repo.create_task("q-1", "restore", "alice")
    task_repo.create_task("q-2", "restore", "alice")
    assert task_service.queue_depth() == 2

    task_repo.claim_next_task("w-1")
    assert task_service.queue_depth() == 1


def test_an_unreadable_queue_leaves_the_eta_line_intact(monkeypatch):
    """The panel degrades to `Queue ?` rather than losing the whole progress bar."""
    import fiximg.infrastructure.queue as queue_module

    def boom():
        raise RuntimeError("queue backend is down")

    monkeypatch.setattr(queue_module, "get_queue", boom)
    from fiximg.application.task_service import task_service

    assert task_service.queue_depth() is None
