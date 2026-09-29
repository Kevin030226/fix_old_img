"""Unit tests for the V2 planner (plan section 9)."""
import pytest

from fiximg.domain.errors import StageNotAvailableError
from fiximg.inference.planner import PipelinePlanner


@pytest.fixture()
def planner():
    return PipelinePlanner()


#: The full restoration chain (plan 搂3.6/搂3.7): each stage does exactly one job,
#: so the face work happens once instead of being buried inside global restore.
FACE_CHAIN = ["face_detection", "face_enhancement", "warp_back"]


def test_restore_plan_includes_the_face_chain_by_default(planner):
    """`restore` keeps the V2 behaviour (faces are enhanced) but explicitly."""
    plan = planner.plan("restore")
    assert plan.task_type == "restore"
    assert [name for name, _ in plan.stages] == ["global_restore", *FACE_CHAIN]
    assert plan.decisions["options"]["face_enhance"] is True


def test_restore_plan_can_opt_out_of_face_enhancement(planner):
    plan = planner.plan("restore", options={"face_enhance": False})
    assert [name for name, _ in plan.stages] == ["global_restore"]
    assert plan.decisions["options"]["face_enhance"] is False


def test_restore_scratch_plan(planner):
    plan = planner.plan("restore_scratch")
    assert [name for name, _ in plan.stages] == ["global_restore", *FACE_CHAIN]
    # The scratch variant is expressed as a constructor kwarg, and the stage
    # widens its own capabilities accordingly.
    assert plan.stages[0][1] == {"with_scratch": True}


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


#: What each stage in a plan is known to need, written out here rather than read
#: from the code under test: the point is to catch a plan that grows a stage whose
#: capability the enqueue-time hint does not declare.
STAGE_CAPABILITIES = {
    "global_restore": {"restore"},
    "scratch_repair": {"restore", "scratch_repair"},
    "face_detection": {"face_detection"},
    "face_enhancement": {"face_enhance", "face_restore"},
    "warp_back": {"warp_back", "face_composite"},
    "colorization": {"colorize"},
}

SWITCH_COMBINATIONS = [
    None,
    {},
    {"face_enhance": False},
    {"face_enhance": True},
    {"auto_colorize": True},
    {"face_enhance": False, "auto_colorize": True},
    {"face_enhance": True, "auto_colorize": True},
]


@pytest.mark.parametrize("task_type", ["restore", "restore_scratch", "colorize", "detect_scratch"])
@pytest.mark.parametrize("options", SWITCH_COMBINATIONS)
def test_the_routing_hint_describes_the_plan_that_will_run(planner, task_type, options):
    """搂4.3 Step 4: `required_capabilities` must be the plan's capability set.

    It used to take the task type only, so `plan()` added stages the hint never
    mentioned: a `restore` submitted with `{"auto_colorize": true}` declared no
    `colorize`, and a worker serving only restoration claimed it and colorized on its
    own card 鈥?the routing promise inverted. The same blindness in the other direction
    (`{"face_enhance": false}`) over-declared, which strands a task on a queue where
    every worker looks healthy.
    """
    plan = planner.plan(task_type, options)
    expected: set[str] = set()
    for stage_name, _kwargs in plan.stages:
        expected |= STAGE_CAPABILITIES.get(stage_name, set())

    declared = planner.required_capabilities(task_type, options)
    assert declared >= expected, (
        f"{task_type} {options}: plan runs {sorted(expected - declared)} that the hint hides"
    )
