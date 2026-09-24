"""Regression tests for the two UI defects fixed in this round.

1. Role-based tab visibility must be decided *server-side* (page-load handler),
   not by injecting CSS/JS into the DOM.
2. A stage must be able to report progress *inside* its own run, so the bar
   moves during long single-stage plans (restore / restore_scratch).
"""
from types import SimpleNamespace

import pytest

from app.inference import orchestrator as orch
from app.inference.context import StageContext
from app.inference.stages import global_restore as gr_stage
from app.ui import gradio_app


@pytest.fixture(autouse=True)
def _no_runtime_db(tmp_path, monkeypatch):
    """Keep these tests off the repo's runtime data (<repo>/admin_data).

    They used to rely on the developer's real SQLite file being present (and
    littered the working tree with a stray `fixoldimg.db` when it was not):
    the banner lookup hits the database, so on a fresh clone it failed with
    "no such table: users". Pointing the DB at an empty temp file keeps the
    graceful-degradation path under test without touching real data.
    """
    import app.core.config as config_mod
    import app.db as legacy_db

    db_path = tmp_path / "admin_data" / "fixoldimg.db"
    monkeypatch.setattr(legacy_db, "ADMIN_DATA_DIR", str(db_path.parent))
    monkeypatch.setattr(legacy_db, "DB_PATH", str(db_path))
    monkeypatch.setattr(legacy_db, "_conn", None)
    monkeypatch.setattr(config_mod.settings, "db_path", str(db_path))


# --------------------------------------------------------------- role visibility
class _StubRequest:
    """Minimal stand-in for gr.Request (apply_role only reads .username)."""

    def __init__(self, username):
        self.username = username


@pytest.fixture()
def _users(monkeypatch):
    users = {
        "boss": {"username": "boss", "role": "admin", "must_change_password": False},
        "joe": {"username": "joe", "role": "user", "must_change_password": False},
    }
    monkeypatch.setattr(gradio_app, "get_user", lambda name: users.get(name))
    return users


def _visible_flags(result):
    """Map the (state, *updates, tabs, banner) tuple to a {label: visible} dict."""
    labels = ["tab_auto", "tab_restore", "tab_scratch", "tab_detect",
              "tab_colorize", "admin_panel"]
    return {
        label: (update or {}).get("visible")
        for label, update in zip(labels, result[1:7], strict=True)
    }


def test_admin_sees_only_admin_panel(_users):
    result = gradio_app.apply_role(_StubRequest("boss"))
    flags = _visible_flags(result)
    assert flags["admin_panel"] is True
    # Every restoration/colorization tab must be hidden for an admin.
    assert flags["tab_auto"] is False
    assert flags["tab_restore"] is False
    assert flags["tab_scratch"] is False
    assert flags["tab_detect"] is False
    assert flags["tab_colorize"] is False
    assert result[0] == {"username": "boss", "role": "admin"}
    # ...and the caller must land on a tab that is actually visible.
    assert result[7].get("selected") == gradio_app.TAB_ID_ADMIN


def test_user_sees_function_tabs_but_not_admin_panel(_users):
    result = gradio_app.apply_role(_StubRequest("joe"))
    flags = _visible_flags(result)
    assert flags["admin_panel"] is False
    assert all(flags[name] is True for name in
               ("tab_auto", "tab_restore", "tab_scratch", "tab_detect", "tab_colorize"))
    assert result[0] == {"username": "joe", "role": "user"}
    assert result[7].get("selected") == gradio_app.TAB_ID_AUTO


def test_unknown_user_falls_back_to_user_role(_users):
    result = gradio_app.apply_role(_StubRequest(None))
    flags = _visible_flags(result)
    assert flags["admin_panel"] is False
    assert flags["tab_restore"] is True


def test_tab_visibility_updates_are_independent_objects(_users):
    """Each TabItem needs its own update payload; sharing one dict across
    several outputs is fragile."""
    result = gradio_app.apply_role(_StubRequest("boss"))
    updates = result[1:7]
    assert len({id(u) for u in updates}) == len(updates)


def test_admin_panel_is_hidden_by_default():
    """Build-time default must be invisible so users never see it flash.

    Function tabs stay visible by default, so a failed load handler degrades to
    "everything visible" rather than "nothing visible".
    """
    cfg = gradio_app.build_demo().get_config_file()
    by_elem_id = {
        c.get("props", {}).get("elem_id"): c
        for c in cfg["components"]
        if c.get("props", {}).get("elem_id")
    }
    assert by_elem_id["admin_panel"]["props"]["visible"] is False
    for elem_id in ("tab_auto", "tab_restore", "tab_scratch", "tab_detect", "tab_colorize"):
        assert by_elem_id[elem_id]["props"]["visible"] is True


def test_role_stylesheet_targets_tab_button_ids():
    """The role stylesheet must target Gradio's `{elem_id}-button` elements.

    Gradio does not rebuild the Tabs button list when a TabItem's `visible`
    changes at runtime, so the server-side update alone leaves the buttons on
    screen. The earlier CSS attempt used `#tab_restore` (the *panel* id) and
    therefore matched nothing — a real browser check confirmed the buttons only
    disappear when `#tab_restore-button` is targeted.
    """
    from app.ui import middleware

    assert not hasattr(middleware, "_ADMIN_UI_SCRIPT"), "JS DOM patching must be gone"
    assert not hasattr(middleware, "_ADMIN_UI_CSS"), "stale CSS constants must be gone"
    for elem_id in ("tab_auto", "tab_restore", "tab_scratch", "tab_detect",
                    "tab_colorize"):
        assert f"#{elem_id}-button" in middleware._ADMIN_TAB_CSS
    assert "#admin_panel-button" in middleware._USER_TAB_CSS


def test_every_tab_declares_an_explicit_id():
    """`selected` updates are only honoured for TabItems with an explicit id."""
    cfg = gradio_app.build_demo().get_config_file()
    by_elem_id = {
        c["props"]["elem_id"]: c
        for c in cfg["components"]
        if c.get("type") == "tabitem" and c.get("props", {}).get("elem_id")
    }
    expected = {
        "tab_auto": gradio_app.TAB_ID_AUTO,
        "tab_restore": gradio_app.TAB_ID_RESTORE,
        "tab_scratch": gradio_app.TAB_ID_SCRATCH,
        "tab_detect": gradio_app.TAB_ID_DETECT,
        "tab_colorize": gradio_app.TAB_ID_COLORIZE,
        "admin_panel": gradio_app.TAB_ID_ADMIN,
    }
    assert set(by_elem_id) == set(expected)
    for elem_id, tab_id in expected.items():
        assert by_elem_id[elem_id]["props"]["id"] == tab_id


# ------------------------------------------------------------ intra-stage progress
def test_parse_marker():
    assert gr_stage._parse_marker("@@FIXIMG_PROGRESS 2/4 face detection") == (
        2, 4, "face detection")
    assert gr_stage._parse_marker("@@FIXIMG_PROGRESS 4/4") == (4, 4, "")
    assert gr_stage._parse_marker("@@FIXIMG_PROGRESS bogus") is None
    assert gr_stage._parse_marker("@@FIXIMG_PROGRESS 1/0 x") is None
    assert gr_stage._parse_marker("plain log line") is None


def test_run_py_emits_markers():
    """run.py must expose the marker format the stage parses."""
    import run as run_module

    assert run_module.PROGRESS_PREFIX == gr_stage._PROGRESS_PREFIX


def test_report_progress_is_best_effort():
    from app.inference.context import StageContext

    ctx = StageContext(task_id="t")
    ctx.report_progress(0.5)  # no callback -> must not raise

    seen = []
    ctx.progress_cb = lambda frac, msg=None: seen.append((frac, msg))
    ctx.report_progress(1.5)   # clamped to 1.0
    ctx.report_progress(-1.0)  # clamped to 0.0
    assert seen == [(1.0, None), (0.0, None)]

    def boom(_frac, _msg=None):
        raise RuntimeError("callback must not break the run")

    ctx.progress_cb = boom
    ctx.report_progress(0.5)  # swallowed


class _FakeStage:
    """A stage that reports 0.0 -> 1.0 through the context callback."""

    def __init__(self, name, fractions):
        self.name = name
        self._fractions = fractions

    def run(self, image, context):
        for frac in self._fractions:
            context.report_progress(frac, f"{self.name}@{frac}")
        return SimpleNamespace(image=image, artifacts={}, metadata={}, message=None)


def _run_plan_with(monkeypatch, stages, plan_stages):
    recorded = []
    monkeypatch.setattr(orch.task_repo, "record_stage", lambda *a, **k: None)
    monkeypatch.setattr(orch.task_repo, "finish_stage", lambda *a, **k: None)
    monkeypatch.setattr(orch.task_repo, "add_event", lambda *a, **k: None)
    monkeypatch.setattr(
        orch.task_repo, "update_progress",
        lambda tid, pct, stage=None: recorded.append((pct, stage)),
    )

    planner = SimpleNamespace(
        build_stage=lambda name, kwargs: stages[name],
    )
    orchestrator = orch.PipelineOrchestrator.__new__(orch.PipelineOrchestrator)
    orchestrator.planner = planner
    orchestrator.model_manager = None

    plan = SimpleNamespace(stages=plan_stages)
    ctx = StageContext(task_id="t", gpu=-1)
    orchestrator._run_plan("t", "img", plan, ctx)
    return recorded, ctx


def test_single_stage_plan_reports_intra_stage_progress(monkeypatch):
    """A 1-stage plan (restore) must still move the bar during the stage."""
    stages = {"global_restore": _FakeStage("global_restore", [0.25, 0.5, 0.75, 1.0])}
    recorded, ctx = _run_plan_with(
        monkeypatch, stages, [("global_restore", {"with_scratch": False})]
    )
    percentages = [pct for pct, _ in recorded]
    # 25/50/75/100 -> the stage window is the whole bar, capped at 99.
    assert percentages == [0, 25, 50, 75, 99, 99]
    assert ctx.progress_cb is None  # cleared once no stage is running


def test_multi_stage_plan_maps_each_stage_into_its_own_slice(monkeypatch):
    """With 2 stages each owns 50% of the bar; progress never goes backwards."""
    stages = {
        "global_restore": _FakeStage("global_restore", [0.5, 1.0]),
        "colorization": _FakeStage("colorization", [1.0]),
    }
    recorded, _ = _run_plan_with(
        monkeypatch, stages,
        [("global_restore", {}), ("colorization", {})],
    )
    percentages = [pct for pct, _ in recorded]
    assert percentages == sorted(percentages)          # monotonic
    # stage 0 slice [0,50]: start 0, 0.5 -> 25, 1.0 -> 50, then stage-end 50
    # stage 1 slice [50,100]: start 50, 1.0 -> 99 (capped), then stage-end 99
    assert percentages == [0, 25, 50, 50, 50, 99, 99]
    assert recorded[1][1] == "global_restore"
    assert recorded[5][1] == "colorization"


# ------------------------------------------------------- output wiring contract
def test_every_submit_handler_declares_two_outputs():
    """Each submit handler is a generator yielding (image, status_text).

    Declaring any other number of outputs makes Gradio postprocess the whole
    yielded tuple as an image and raise ComponentProcessingError — exactly what
    broke the Scratch Detection and Colorization tabs (they declared 1 output).
    """
    cfg = gradio_app.build_demo().get_config_file()
    submit_deps = [
        d for d in cfg["dependencies"]
        if str(d.get("api_name") or "").startswith("handler")
    ]
    assert len(submit_deps) == 5, [d.get("api_name") for d in submit_deps]
    for dep in submit_deps:
        assert len(dep["outputs"]) == 2, f"{dep['api_name']} -> {dep['outputs']}"


def test_clear_handlers_match_their_declared_outputs():
    """Clear callbacks must return exactly as many values as they write to."""
    cfg = gradio_app.build_demo().get_config_file()
    clear_deps = [
        d for d in cfg["dependencies"]
        if str(d.get("api_name") or "").startswith("clear_inputs")
    ]
    assert clear_deps, "no clear callbacks found"
    for dep in clear_deps:
        assert len(dep["outputs"]) == 3, f"{dep['api_name']} -> {dep['outputs']}"
    assert len(gradio_app.clear_inputs()) == 3
    assert len(gradio_app.clear_inputs_3()) == 3


def test_colorization_stage_reports_phase_progress():
    """DDColor is lazily loaded; the stage must report load vs. inference phases
    so the bar does not sit at 0% for the whole run."""
    import numpy as np
    from PIL import Image

    from app.inference.stages.colorization import ColorizationStage

    class _FakeSettings:
        ddcolor_model_size = "large"

    class _FakeManager:
        settings = _FakeSettings()

        def get(self, name):
            assert name == "ddcolor"
            return SimpleNamespace(process=lambda bgr: bgr)

    ctx = StageContext(task_id="t", model_manager=_FakeManager())
    seen = []
    ctx.progress_cb = lambda frac, msg=None: seen.append((frac, msg))

    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    result = ColorizationStage().run(image, ctx)

    assert result.image is not None
    assert [frac for frac, _ in seen] == [0.05, 0.6, 1.0]
    assert [msg for _, msg in seen] == [
        "loading DDColor model", "colorizing", "colorized"]
