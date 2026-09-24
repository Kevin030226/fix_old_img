"""Unit tests for the V2 planner (plan section 9)."""
import pytest

from app.core.exceptions import StageNotAvailableError
from app.inference.planner import PipelinePlanner


@pytest.fixture()
def planner():
    return PipelinePlanner()


def test_restore_plan(planner):
    plan = planner.plan("restore")
    assert plan.task_type == "restore"
    assert [name for name, _ in plan.stages] == ["global_restore"]


def test_restore_scratch_plan(planner):
    plan = planner.plan("restore_scratch")
    assert [name for name, _ in plan.stages] == ["global_restore"]


def test_detect_scratch_plan(planner):
    plan = planner.plan("detect_scratch")
    assert [name for name, _ in plan.stages] == ["scratch_detection"]


def test_colorize_plan(planner):
    plan = planner.plan("colorize")
    assert [name for name, _ in plan.stages] == ["colorization"]


def test_unknown_task_type_raises(planner):
    with pytest.raises(StageNotAvailableError):
        planner.plan("super_resolution")


def test_build_stage_registry(planner):
    stage = planner.build_stage("global_restore", {"with_scratch": False})
    assert stage.name == "global_restore"
    scratch = planner.build_stage("global_restore", {"with_scratch": True})
    assert scratch.name == "scratch_repair"
    detect = planner.build_stage("scratch_detection", {})
    assert detect.name == "scratch_detection"


def test_build_unknown_stage_raises(planner):
    with pytest.raises(StageNotAvailableError):
        planner.build_stage("nope", {})
