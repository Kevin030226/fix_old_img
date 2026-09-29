"""The 搂12 option switches reach the *dynamic* plan, not just the static ones.

``plan_auto_restore(analysis, options)`` is where the two meet: the Auto tab's whole
premise is that the image decides the pipeline, so an option the planner ignores there
is a switch the API accepts and the UI can tick with no effect anywhere. These cases
run a real submission through :class:`PipelineOrchestrator` with stubbed stages and read
the stages back off the task row 鈥?the same list ``GET /tasks/{id}`` and the UI checklist
show 鈥?because a planner that honours the option while the runtime drops it would pass a
planner-only test.
"""
import json
import os

import pytest
from PIL import Image

from fiximg.inference.stages.base import BaseStage, StageResult

pytestmark = pytest.mark.integration

_STUBBED_STAGES = (
    "global_restore", "scratch_repair", "scratch_detection",
    "face_detection", "face_enhancement", "warp_back", "colorization",
)

#: What the analyzer is pinned to report: scratched, grayscale, two faces. Every
#: optional stage in 搂10's decision table therefore wants to run, so a switch that
#: declines one has somewhere visibly to land.
_ANALYSIS = {
    "width": 800, "height": 600,
    "is_grayscale": True, "face_count": 2,
    "blur_score": 0.6, "scratch_score": 0.8,
}


class _StubStage(BaseStage):
    """Passes the image through; which stages ran is the whole measurement."""

    name = "global_restore"
    version = "stub-1.0"
    capabilities = frozenset({"restore", "scratch_repair", "face_detection",
                              "face_restore", "face_composite", "warp_back",
                              "colorize", "deblur", "denoise"})

    def __init__(self, **kwargs) -> None:
        # The registry instantiates a stage with its plan kwargs (`with_scratch`), so
        # the stub has to take them; the measurement is which stages ran, not what
        # each was configured to do.
        self.plan_kwargs = kwargs

    def run(self, image, context):
        return StageResult(image=image, metadata={"stub": True})


def _stub_for(name: str) -> type[_StubStage]:
    return type(f"Stub_{name}", (_StubStage,), {"name": name})


@pytest.fixture()
def auto_env(isolated_db, monkeypatch):
    """Stub stages and a pinned analysis, on an isolated database and artifact root."""
    import fiximg.config as config_mod
    from fiximg.inference import analyzer, registry as registry_module

    monkeypatch.setattr(
        config_mod.settings, "tasks_root", str(isolated_db / "storage" / "tasks")
    )
    monkeypatch.setattr(config_mod.settings, "eval_mode", "inline")

    registry = registry_module.build_default_registry()
    for stage_name in _STUBBED_STAGES:
        registry.register(stage_name, _stub_for(stage_name), replace=True)
    monkeypatch.setattr(registry_module, "stage_registry", registry)

    monkeypatch.setattr(analyzer, "analyze_image", lambda image: dict(_ANALYSIS))
    return isolated_db


def _run(options: dict | None) -> tuple[str, list[str], dict]:
    """One synchronous auto_restore submission: (task_id, executed stages, decisions)."""
    from fiximg.inference.runtime import PipelineOrchestrator

    result_path, _text = PipelineOrchestrator().run(
        Image.new("RGB", (32, 24), "gray"), {"username": "alice"},
        "auto_restore", options=options,
    )
    run_dir = os.path.dirname(os.path.dirname(result_path))
    with open(os.path.join(run_dir, "report.json"), encoding="utf-8") as handle:
        decisions = json.load(handle)["planner_decisions"]

    from fiximg.infrastructure.db.repositories import task_repository

    row = task_repository.get_task(os.path.basename(run_dir))
    stages = row["stages"]
    if isinstance(stages, str):
        stages = json.loads(stages)
    return (
        row["id"],
        [stage["stage_name"] for stage in stages],
        decisions,
    )


def test_no_options_leaves_every_decision_to_the_analysis(auto_env):
    """The pre-existing behaviour, pinned so the override cannot change it by accident."""
    _task_id, names, decisions = _run(None)

    assert decisions["analysis"]["face_count"] == 2, "the analysis is what decided this"
    assert names == [
        "global_restore", "face_detection", "face_enhancement", "warp_back", "colorization",
    ], names


def test_face_enhance_false_declines_the_chain_the_analysis_wanted(auto_env):
    """The switch that started this: an explicit refusal the planner could not hear."""
    _task_id, names, decisions = _run({"face_enhance": False})

    assert "face_detection" not in names and "face_enhancement" not in names, names
    assert names == ["global_restore", "colorization"], names
    assert decisions["face_enhancement"] is False
    assert decisions["options"]["face_enhance"] is False


def test_auto_colorize_false_declines_the_colorization(auto_env):
    _task_id, names, decisions = _run({"auto_colorize": False})

    assert names[-1] != "colorization", names
    assert decisions["colorization"] is False


def test_both_switches_off_leaves_only_the_restoration(auto_env):
    _task_id, names, _decisions = _run({"face_enhance": False, "auto_colorize": False})

    assert names == ["global_restore"], names


def test_the_scratch_branch_is_not_a_switch(auto_env):
    """Declining the optional stages must not silently change the required one.

    ``scratch_score`` is above the threshold in the pinned analysis, so the plan is the
    ``with_scratch`` variant whatever the options say 鈥?that branch is the Auto tab's
    reason to exist, and it has no checkbox.
    """
    for options in (None, {"face_enhance": False}, {"face_enhance": True}):
        _task_id, _names, decisions = _run(options)
        assert decisions["used_scratch_repair"] is True


def test_an_override_is_visible_in_the_report_next_to_what_the_image_said(auto_env):
    """The report has to show the refusal, not just the shortened stage list.

    ``decisions["analysis"]`` still says two faces were found while
    ``options.face_enhance`` says false: a reader comparing the two can tell the
    pipeline was declined rather than that the analyzer missed something.
    """
    _task_id, _names, decisions = _run({"face_enhance": False})

    assert decisions["analysis"]["face_count"] == 2
    assert decisions["face_enhancement"] is False
    assert decisions["options"] == {"face_enhance": False, "auto_colorize": True}


def test_the_preview_promises_exactly_the_pipeline_the_run_records(auto_env):
    """The analysis row above the switches and the task row must list the same stages.

    The row is produced by ``TaskService.auto_preview`` (analyzer + planner) while the
    run's list is what the worker recorded per stage, so this catches a preview that
    ignores the switches, drops a stage, or reorders the chain.

    What it deliberately cannot check is the *naming* rule, and the reason is worth
    keeping in the file: the stages here are stubs registered under the plan's own keys,
    and ``auto_preview`` resolves labels through the same registry 鈥?so both sides say
    ``global_restore`` even on the scratch branch, where the real stage reports itself as
    ``scratch_repair``. Asserting the rename against a stub would be measuring the
    stub's imitation of production. The rename is proven in
    ``tests/unit/test_analysis_preview.py`` against the real registry, and live by
    comparing the row to a completed task's stage rows.
    """
    from PIL import Image

    from fiximg.application.task_service import task_service

    baseline = _run(None)[1]
    for options in (None, {"face_enhance": False}, {"auto_colorize": False},
                    {"face_enhance": False, "auto_colorize": False},
                    {"face_enhance": True, "auto_colorize": True}):
        _task_id, names, _decisions = _run(options)
        preview = task_service.auto_preview(Image.new("RGB", (32, 24), "gray"), options)

        assert preview["stages"] == names, (options, preview["stages"], names)
        # The notes are computed from the same difference the rows show, so a preview that
        # lists the right stages for the wrong reason still fails.
        assert preview["declined"] == [n for n in baseline if n not in names], (options, preview)
        assert preview["forced"] == [n for n in names if n not in baseline], (options, preview)


def test_hr_is_not_read_as_a_pipeline_switch(auto_env):
    """``hr`` changes what the face stages do, not whether they are scheduled.

    Treating it as a plan switch would let ``{"hr": false}`` decline the face chain 鈥?    a stage list the user did not ask for 鈥?while the checked box on the Auto tab
    would then send something the planner has an opinion about.
    """
    _task_id, names, decisions = _run({"hr": False})

    assert names == [
        "global_restore", "face_detection", "face_enhancement", "warp_back", "colorization",
    ], names
    assert set(decisions["options"]) == {"face_enhance", "auto_colorize"}
