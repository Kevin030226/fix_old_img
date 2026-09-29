"""Regression tests for the two UI defects fixed in this round.

1. Role-based tab visibility must be decided *server-side* (page-load handler),
   not by injecting CSS/JS into the DOM.
2. A stage must be able to report progress *inside* its own run, so the bar
   moves during long single-stage plans (restore / restore_scratch).
"""
import contextlib
from types import SimpleNamespace

import pytest

from fiximg.inference import runtime as orch
from fiximg.inference.context import StageContext
from fiximg.ui import gradio_app


@pytest.fixture(autouse=True)
def _no_runtime_db(tmp_path, monkeypatch):
    """Keep these tests off the repo's runtime data (<repo>/admin_data).

    They used to rely on the developer's real SQLite file being present (and
    littered the working tree with a stray `fixoldimg.db` when it was not):
    the banner lookup hits the database, so on a fresh clone it failed with
    "no such table: users". Pointing the DB at an empty temp file keeps the
    graceful-degradation path under test without touching real data.
    """
    import fiximg.config as config_mod
    import fiximg.infrastructure.db.engine as legacy_db

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
    """Stub the user lookup.

    The lookup moved into :mod:`fiximg.ui.state` (plan 搂3.2: one module owns
    "who is this caller"), so that is where the stub goes now.
    """
    from fiximg.ui import state as ui_state

    users = {
        "boss": {"username": "boss", "role": "admin", "must_change_password": False},
        "joe": {"username": "joe", "role": "user", "must_change_password": False},
    }
    monkeypatch.setattr(ui_state, "get_user", lambda name: users.get(name))
    return users


def _visible_flags(result):
    """Map the (state, *updates, tabs, banner) tuple to a {label: visible} dict."""
    labels = ["tab_auto", "tab_restore", "tab_scratch", "tab_detect",
              "tab_colorize", "tab_history", "admin_panel"]
    return {
        label: (update or {}).get("visible")
        for label, update in zip(labels, result[1:8], strict=True)
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
    assert flags["tab_history"] is False
    assert result[0] == {"username": "boss", "role": "admin"}
    # ...and the caller must land on a tab that is actually visible.
    assert result[8].get("selected") == gradio_app.TAB_ID_ADMIN


def test_user_sees_function_tabs_but_not_admin_panel(_users):
    result = gradio_app.apply_role(_StubRequest("joe"))
    flags = _visible_flags(result)
    assert flags["admin_panel"] is False
    assert all(flags[name] is True for name in
               ("tab_auto", "tab_restore", "tab_scratch", "tab_detect",
                "tab_colorize", "tab_history"))
    assert result[0] == {"username": "joe", "role": "user"}
    assert result[8].get("selected") == gradio_app.TAB_ID_AUTO


def test_unknown_user_falls_back_to_user_role(_users):
    result = gradio_app.apply_role(_StubRequest(None))
    flags = _visible_flags(result)
    assert flags["admin_panel"] is False
    assert flags["tab_restore"] is True


def test_tab_visibility_updates_are_independent_objects(_users):
    """Each TabItem needs its own update payload; sharing one dict across
    several outputs is fragile."""
    result = gradio_app.apply_role(_StubRequest("boss"))
    updates = result[1:8]
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
    for elem_id in ("tab_auto", "tab_restore", "tab_scratch", "tab_detect",
                    "tab_colorize", "tab_history"):
        assert by_elem_id[elem_id]["props"]["visible"] is True


def test_role_stylesheet_targets_tab_button_ids():
    """The role stylesheet must target Gradio's `{elem_id}-button` elements.

    Gradio does not rebuild the Tabs button list when a TabItem's `visible`
    changes at runtime, so the server-side update alone leaves the buttons on
    screen. The earlier CSS attempt used `#tab_restore` (the *panel* id) and
    therefore matched nothing 鈥?a real browser check confirmed the buttons only
    disappear when `#tab_restore-button` is targeted.
    """
    from fiximg.ui import middleware

    assert not hasattr(middleware, "_ADMIN_UI_SCRIPT"), "JS DOM patching must be gone"
    assert not hasattr(middleware, "_ADMIN_UI_CSS"), "stale CSS constants must be gone"
    for elem_id in ("tab_auto", "tab_restore", "tab_scratch", "tab_detect",
                    "tab_colorize", "tab_history"):
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
        "tab_history": gradio_app.TAB_ID_HISTORY,
        "admin_panel": gradio_app.TAB_ID_ADMIN,
    }
    assert set(by_elem_id) == set(expected)
    for elem_id, tab_id in expected.items():
        assert by_elem_id[elem_id]["props"]["id"] == tab_id


def test_the_admin_only_tab_list_is_the_one_visibility_enforces(_users):
    """`ADMIN_ONLY_TABS` is the tab list `tab_visibility` is checked against.

    The list had no reader, so "which tabs are admin-only" was written down twice
    and could drift. `tab_visibility` keys its result by *group* (`function_tabs`,
    `history_tab`, `admin_panel`) rather than by tab id, so the comparison is
    against the group an ordinary user does not get.

    Resolved through `resolve_user_state(request)`, the way the layout does: it
    takes a Gradio request and looks the role up, so handing it a dict would
    quietly answer "plain user" for both callers and the comparison below would
    compare nothing.
    """
    from fiximg.ui.state import resolve_user_state, tab_visibility

    cfg = gradio_app.build_demo().get_config_file()
    built = {
        c["props"]["id"]
        for c in cfg["components"]
        if c.get("type") == "tabitem" and c.get("props", {}).get("id")
    }
    assert set(gradio_app.ADMIN_ONLY_TABS) <= built, (
        "ADMIN_ONLY_TABS names a tab the demo does not build: "
        f"{sorted(set(gradio_app.ADMIN_ONLY_TABS) - built)}"
    )

    admin = tab_visibility(resolve_user_state(_StubRequest("boss")))
    user = tab_visibility(resolve_user_state(_StubRequest("joe")))

    granted = {group for group, on in admin.items() if on and not user.get(group)}
    assert granted == {"admin_panel"}, granted

    # The two must name the same set. `granted` holds `tab_visibility` keys, the
    # constant holds tab ids, so bridge them through the id -> elem_id map the
    # demo actually builds. Asserting the constant is a subset of `built` and
    # separately that `granted` is right leaves a third copy free to disagree
    # with either — which is the defect this test exists for.
    elem_of = {
        c["props"]["id"]: c["props"]["elem_id"]
        for c in cfg["components"]
        if c.get("type") == "tabitem" and c.get("props", {}).get("elem_id")
    }
    declared = {elem_of[tab] for tab in gradio_app.ADMIN_ONLY_TABS}
    assert declared == granted, (
        f"ADMIN_ONLY_TABS names {sorted(declared)} but tab_visibility grants "
        f"{sorted(granted)}"
    )


# ------------------------------------------------------------ intra-stage progress
def test_parse_marker():
    """V3 moved the marker parser into the legacy CLI backend."""
    from fiximg.inference.backends.legacy_cli import parse_marker

    assert parse_marker("@@FIXIMG_PROGRESS 2/4 face detection") == (2, 4, "face detection")
    assert parse_marker("@@FIXIMG_PROGRESS 4/4") == (4, 4, "")
    assert parse_marker("@@FIXIMG_PROGRESS bogus") is None
    assert parse_marker("@@FIXIMG_PROGRESS 1/0 x") is None
    assert parse_marker("plain log line") is None


def test_cli_emits_markers():
    """The CLI and the backend must agree on the marker format."""
    import run as run_module
    from fiximg.inference.backends.legacy_cli import PROGRESS_PREFIX

    assert run_module.PROGRESS_PREFIX == PROGRESS_PREFIX


def test_report_progress_is_best_effort():
    from fiximg.inference.context import StageContext

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
def test_every_submit_handler_declares_matching_outputs():
    """Each submit handler yields (image, status_text, before/after, download).

    The count must match exactly: declaring a different number makes Gradio
    postprocess the whole yielded tuple as an image and raise
    ComponentProcessingError 鈥?exactly what broke the Scratch Detection and
    Colorization tabs when they declared 1 output.
    """
    from tests.fixtures import demo_config, submit_deps

    cfg = demo_config()
    deps = submit_deps(cfg)
    assert len(deps) == 5, [d.get("api_name") for d in deps]
    for dep in deps:
        assert len(dep["outputs"]) == 4, f"{dep['api_name']} -> {dep['outputs']}"


def test_history_row_selection_writes_every_component_it_returns():
    """The preview callback and its wiring must agree on the component count.

    `load_history_entry` fills seven previews plus the download control. Returning
    a different number of values than is declared is a runtime Gradio error on
    click rather than a test failure, so the contract is read off the built Blocks
    here; the direct-call tests in `test_history_panel.py` are what pin the
    handler's own shape.
    """
    cfg = gradio_app.build_demo().get_config_file()
    components = {c["id"]: c for c in cfg["dependencies"] and cfg["components"]}
    deps = [d for d in cfg["dependencies"] if d.get("api_name") == "load_history_entry"]
    assert deps, "the history table exposes no select handler"

    outputs = deps[0]["outputs"]
    assert len(outputs) == 8, outputs
    # The download control is wired to the selection, and it is the element the
    # panel renders 鈥?not an orphan id Gradio would silently ignore.
    rendered_elem_ids = {
        (components.get(oid) or {}).get("props", {}).get("elem_id") for oid in outputs
    }
    assert "history_download" in rendered_elem_ids, rendered_elem_ids


def test_clear_handlers_match_their_declared_outputs():
    """Clear callbacks must return exactly as many values as they write to."""
    cfg = gradio_app.build_demo().get_config_file()
    clear_deps = [
        d for d in cfg["dependencies"]
        if str(d.get("api_name") or "").startswith("clear_inputs")
    ]
    assert clear_deps, "no clear callbacks found"
    for dep in clear_deps:
        assert len(dep["outputs"]) == 5, f"{dep['api_name']} -> {dep['outputs']}"
    assert len(gradio_app.clear_inputs()) == 5
    assert len(gradio_app.clear_inputs_3()) == 5


class _ColorStubManager:
    """The `ModelManager` surface a colorization run needs, answered honestly.

    The stage used to duck-type its way to `get()`. Since it goes through
    `DDColorBackend` now, the double has to provide `load` and `acquire` as well: a
    double that is more forgiving than the protocol it replaces is how a stage keeps
    passing while nobody can reach the real model.
    """

    class settings:  # noqa: N801 - attribute style, like the real object
        ddcolor_model_size = "large"

    def __init__(self):
        self.seen_devices: list[str] = []
        self.pipeline = self._Pipeline(self.seen_devices)

    class _Model:
        """Accepts what `apply_declared_policy` asks of a real module, too.

        `to(memory_format=...)` is part of what the declared channels_last policy
        calls, so a double that only takes a positional device fails the load path
        for a reason that has nothing to do with what the test is about.
        """

        def __init__(self, sink: list[str]) -> None:
            self.sink = sink

        def to(self, device=None, **kwargs):
            self.sink.append(str(device if device is not None else kwargs))
            return self

    class _Pipeline:
        def __init__(self, sink: list[str]) -> None:
            self.model = _ColorStubManager._Model(sink)
            self.device = "cpu"

        def process(self, bgr):
            return bgr

    def get(self, name):
        assert name == "ddcolor"
        return self.pipeline

    def load(self, name, version=None):
        return self.pipeline

    @contextlib.contextmanager
    def acquire(self, name):
        yield SimpleNamespace(handle=self.pipeline, version="1.0.0")


class _MoveRecorder:
    """A pipeline stand-in that records every device it was asked to move to."""

    class _Model:
        def __init__(self) -> None:
            self.moves: list[str] = []

        def to(self, device):
            self.moves.append(str(device))
            return self

    def __init__(self, device: str = "cpu") -> None:
        self.model = self._Model()
        self.device = device

    def process(self, bgr):
        return bgr


def test_reside_on_moves_the_pipeline_and_reports_where_it_landed():
    """搂4.3 Step 4: the routed device has to reach the weights.

    `context.gpu` was scheduled and charged by the memory sampler, but the colorize
    pipeline ran wherever load had put it 鈥?so the report paired a device number with
    work that never happened on that device.
    """
    from fiximg.inference.backends.ddcolor import reside_on

    pipeline = _MoveRecorder()
    landed = reside_on(pipeline, "1", cuda_available=lambda: True)

    assert landed == "cuda:1"
    assert pipeline.model.moves == ["cuda:1"], pipeline.model.moves
    assert str(pipeline.device) == "cuda:1"


def test_a_device_the_host_cannot_offer_is_declined_loudly():
    """The other half: report where it ran, not what was asked for.

    Honouring a routing setting the host cannot satisfy would have to be a lie
    somewhere; it is now a `device` of "cpu" plus a `device_requested` the operator can
    see, instead of a quiet "cuda:1" in the report.
    """
    from fiximg.inference.backends.ddcolor import reside_on

    pipeline = _MoveRecorder()
    landed = reside_on(pipeline, "1", cuda_available=lambda: False)

    assert landed == "cpu"
    assert pipeline.model.moves == [], "the weights were moved onto a device that is not there"
    assert str(pipeline.device) == "cpu"


def test_the_colorize_stage_reports_the_device_the_model_ran_on(monkeypatch):
    """The stage's `metadata.device` is an observation, not an echo of the plan."""
    import numpy as np
    from PIL import Image

    import fiximg.inference.backends.ddcolor as ddcolor_mod
    from fiximg.inference.stages.colorization import ColorizationStage

    manager = _ColorStubManager()
    monkeypatch.setattr(ddcolor_mod, "reside_on",
                        lambda pipeline, device, cuda_available=None: "cpu")
    ctx = StageContext(task_id="t", model_manager=manager, gpu=1)
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    result = ColorizationStage().run(image, ctx)

    assert result.metadata["device"] == "cpu", result.metadata
    assert result.metadata["device_requested"] == "1", result.metadata
    # Only the backend produces a precision record, so this also pins the fact that
    # the colorize path runs *through* the 搂5.4 contract rather than around it.
    assert "precision" in result.metadata, result.metadata


def test_the_precision_policy_wraps_the_production_colorize_call(monkeypatch):
    """搂3.5.4: a declared policy must apply on the path that actually runs.

    `inference_mode`/autocast were entered inside `DDColorBackend._do_infer` only, and
    the stage called the pipeline directly 鈥?so the manifest's declaration was true for
    the models API and false for every picture a user ever colorized.
    """
    import numpy as np
    from PIL import Image

    import fiximg.inference.backends.ddcolor as ddcolor_mod
    from fiximg.inference.stages.colorization import ColorizationStage

    entered: list[tuple] = []
    real = ddcolor_mod.inference_context

    def _spy(policy, device=None):
        entered.append((policy.inference_mode, device))
        return real(policy, device)

    monkeypatch.setattr(ddcolor_mod, "inference_context", _spy)
    # A realised device the host would not have given us anyway, so the assertion
    # below cannot be satisfied by echoing the request.
    monkeypatch.setattr(
        ddcolor_mod, "reside_on",
        lambda pipeline, device, cuda_available=None: "cuda:7",
    )
    manager = _ColorStubManager()
    ctx = StageContext(task_id="t", model_manager=manager, gpu=-1)
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    ColorizationStage().run(image, ctx)

    assert entered, "the model call ran without any precision context"
    inference_mode, device = entered[-1]
    assert device == "cuda:7", f"the policy was opened for {device}, not for {entered}"
    assert inference_mode is True, "the declared inference_mode did not reach the call"


def test_colorization_stage_reports_phase_progress():
    """DDColor is lazily loaded; the stage must report load vs. inference phases
    so the bar does not sit at 0% for the whole run."""
    import numpy as np
    from PIL import Image

    from fiximg.inference.stages.colorization import ColorizationStage

    class _FakeSettings:
        ddcolor_model_size = "large"

    class _FakeManager(_ColorStubManager):
        """The double the stage's backend drives; see `_ColorStubManager`."""

    ctx = StageContext(task_id="t", model_manager=_FakeManager())
    seen = []
    ctx.progress_cb = lambda frac, msg=None: seen.append((frac, msg))

    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    result = ColorizationStage().run(image, ctx)

    assert result.image is not None
    assert [frac for frac, _ in seen] == [0.05, 0.6, 1.0]
    assert [msg for _, msg in seen] == [
        "loading DDColor model", "colorizing", "colorized"]


def test_the_retry_button_is_wired_to_the_retry_handler():
    """A handler nobody can reach is how "retry" silently means "duplicate task".

    The panel has had Re-run for ages; the 搂3.8 same-id retry is a different
    operation, and the only proof it is reachable from the page is this wiring.
    """
    cfg = gradio_app.build_demo().get_config_file()
    deps = [d for d in cfg["dependencies"] if d.get("api_name") == "retry_history_entry"]
    assert deps, "the history panel exposes no retry callback"
    assert len(deps[0]["outputs"]) == 5, deps[0]["outputs"]


def test_every_result_tab_offers_a_download():
    """Plan 搂3.9.1 asks for a result the user can take away, on every tab.

    `gr.Image` in Gradio 6 renders a preview with no save affordance of its own, so
    the control has to be a component the column builds. Asserting it per tab (not
    once) is what keeps a future tab from being added with the image and slider only.

    The tabs are enumerated by the shared shape-based selector, not by `api_name`: the
    auto-generated names come from one counter across every callback, so a tab that
    gains a second kind of handler would otherwise be counted as a submit tab here.
    """
    from tests.fixtures import components_by_id, demo_config, submit_deps

    cfg = demo_config()
    components = components_by_id(cfg)
    deps = submit_deps(cfg)
    assert len(deps) == 5, [d.get("api_name") for d in deps]

    for dep in deps:
        kinds = {components.get(oid, {}).get("type") for oid in dep["outputs"]}
        assert "downloadbutton" in {k or "" for k in kinds}, (
            f"{dep['api_name']} outputs {sorted(x for x in kinds if x)} "
            "with no download control"
        )
