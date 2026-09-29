"""Unit tests for the ImageAnalyzer (plan 搂11) and auto planning (plan 搂10)."""
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from fiximg.inference import analyzer
from fiximg.inference.planner import PipelinePlanner


def _flat_gray_image(value=140, size=(160, 120)) -> Image.Image:
    arr = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    return Image.fromarray(arr)


def _textured_color_image(seed=3, size=(160, 120)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8))


# ------------------------------------------------------------------ 搂11 analysis
def test_analysis_keys_present():
    result = analyzer.analyze_image(_textured_color_image())
    assert {"width", "height", "is_grayscale", "blur_score", "scratch_score", "face_count"} <= set(result)


def test_grayscale_detection():
    assert analyzer.analyze_image(_flat_gray_image(140))["is_grayscale"] is True
    assert analyzer.analyze_image(_textured_color_image())["is_grayscale"] is False


def test_blur_score_orders_flat_vs_textured():
    flat = analyzer.analyze_image(_flat_gray_image())
    textured = analyzer.analyze_image(_textured_color_image())
    assert flat["blur_score"] > textured["blur_score"]
    assert 0.0 <= flat["blur_score"] <= 1.0
    assert 0.0 <= textured["blur_score"] <= 1.0


def test_scratch_score_detects_thin_lines():
    # Dense white scratches on mid-gray should score much higher than flat gray.
    img = _flat_gray_image(120)
    d = ImageDraw.Draw(img)
    for x in range(0, 160, 6):
        d.line([(x, 0), (x, 120)], fill=255, width=1)
    scratched = analyzer.analyze_image(img)
    clean = analyzer.analyze_image(_flat_gray_image(120))
    assert scratched["scratch_score"] > clean["scratch_score"]
    assert scratched["scratch_score"] > 0.2


def test_scratch_score_bounded():
    result = analyzer.analyze_image(_textured_color_image())
    assert 0.0 <= result["scratch_score"] <= 1.0


def test_resolution_reported():
    result = analyzer.analyze_image(_flat_gray_image(size=(320, 200)))
    assert result["width"] == 320 and result["height"] == 200


def test_format_summary():
    text = analyzer.format_analysis_summary(
        {"width": 10, "height": 5, "is_grayscale": True, "blur_score": 0.5,
         "scratch_score": 0.25, "face_count": 2}
    )
    assert "grayscale=yes" in text and "faces=2" in text


# ------------------------------------------------------------- 搂10 auto planning
@pytest.fixture()
def planner():
    return PipelinePlanner()


def test_auto_plan_full_chain(planner):
    """grayscale + scratched + faces -> repair + faces + colorization (搂10 example)."""
    plan = planner.plan_auto_restore(
        {"width": 800, "height": 600, "is_grayscale": True, "face_count": 2,
         "blur_score": 0.6, "scratch_score": 0.8}
    )
    names = [n for n, _ in plan.stages]
    assert names[0] == "global_restore"
    assert plan.decisions["used_scratch_repair"] is True
    assert "face_detection" in names and "face_enhancement" in names
    assert names[-1] == "colorization"


def test_auto_plan_clean_color_image(planner):
    """color + no scratches + no faces -> plain global restoration only (搂10 example)."""
    plan = planner.plan_auto_restore(
        {"width": 800, "height": 600, "is_grayscale": False, "face_count": 0,
         "blur_score": 0.2, "scratch_score": 0.1}
    )
    names = [n for n, _ in plan.stages]
    assert names == ["global_restore"]
    assert plan.decisions["used_scratch_repair"] is False


def test_auto_plan_no_faces_no_colorization(planner):
    plan = planner.plan_auto_restore(
        {"is_grayscale": False, "face_count": None, "blur_score": 0.1, "scratch_score": 0.9}
    )
    names = [n for n, _ in plan.stages]
    assert "colorization" not in names
    assert "face_enhancement" not in names
    assert plan.decisions["used_scratch_repair"] is True


def test_auto_plan_empty_analysis_raises(planner):
    from fiximg.domain.errors import StageNotAvailableError

    with pytest.raises(StageNotAvailableError):
        planner.plan_auto_restore({})
    with pytest.raises(StageNotAvailableError):
        planner.plan("auto_restore")


def test_available_tasks_includes_auto(planner):
    assert "auto_restore" in planner.available_tasks()


# ------------------------------------- 搂12 switches applied to the dynamic plan
def _analysis(**overrides) -> dict:
    """A clean colour photo: no optional stage wants to run unless told to."""
    base = {"width": 800, "height": 600, "is_grayscale": False, "face_count": 0,
            "blur_score": 0.2, "scratch_score": 0.1}
    return {**base, **overrides}


def _names(plan) -> list[str]:
    return [name for name, _kwargs in plan.stages]


@pytest.mark.parametrize("analysis", [
    _analysis(),
    _analysis(face_count=2),
    _analysis(is_grayscale=True),
    _analysis(face_count=3, is_grayscale=True),
])
def test_omitting_a_switch_leaves_the_decision_with_the_analysis(planner, analysis):
    """Adding ``options`` must not move the default behaviour by one stage."""
    assert _names(planner.plan_auto_restore(analysis, {})) == _names(
        planner.plan_auto_restore(analysis)
    )
    assert _names(planner.plan_auto_restore(analysis, None)) == _names(
        planner.plan_auto_restore(analysis)
    )


def test_a_false_switch_overrules_a_stage_the_analysis_wanted(planner):
    plan = planner.plan_auto_restore(
        _analysis(face_count=2, is_grayscale=True),
        {"face_enhance": False, "auto_colorize": False},
    )
    assert _names(plan) == ["global_restore"]
    assert plan.decisions["face_enhancement"] is False
    assert plan.decisions["colorization"] is False


def test_a_true_switch_forces_a_stage_the_analysis_declined(planner):
    """The reason the override exists: the analysis runs on a 512 px copy.

    Small faces in a large scan are invisible to it, and a faded black-and-white photo
    can carry enough hand-tint to read as colour, so "the analyzer found nothing" is not
    an instruction from the user.
    """
    plan = planner.plan_auto_restore(
        _analysis(face_count=0, is_grayscale=False),
        {"face_enhance": True, "auto_colorize": True},
    )
    assert _names(plan) == [
        "global_restore", "face_detection", "face_enhancement", "warp_back", "colorization",
    ]


def test_a_true_switch_also_overrules_no_detector_installed(planner):
    """`face_count` is None when no face backend exists, which is not "no faces"."""
    plan = planner.plan_auto_restore(_analysis(face_count=None), {"face_enhance": True})

    assert _names(plan) == [
        "global_restore", "face_detection", "face_enhancement", "warp_back",
    ]
    assert planner.plan_auto_restore(_analysis(face_count=None)).stages == [
        ("global_restore", {"with_scratch": False})
    ]


def test_colorization_stays_last_whatever_forced_it(planner):
    """Warp-back composites the enhanced crops, so colouring must come after it.

    Colorizing before the face chain would leave every restored face taken from the
    uncoloured original 鈥?the stage list would look right and the picture would not.
    """
    plan = planner.plan_auto_restore(
        _analysis(face_count=2, is_grayscale=True), {"auto_colorize": True}
    )
    assert _names(plan)[-1] == "colorization"
    assert _names(plan).index("warp_back") < len(plan.stages) - 1


@pytest.mark.parametrize("options", [None, {}, {"face_enhance": False}, {"face_enhance": True}])
def test_the_scratch_branch_is_nobody_elses_decision(planner, options):
    """Only `scratch_score` picks the restoration variant; the switches pick stages."""
    plan = planner.plan_auto_restore(_analysis(scratch_score=0.9), options)
    assert [kwargs for name, kwargs in plan.stages if name == "global_restore"] == [
        {"with_scratch": True}
    ]

    plan = planner.plan_auto_restore(_analysis(scratch_score=0.1), options)
    assert [kwargs for name, kwargs in plan.stages if name == "global_restore"] == [
        {"with_scratch": False}
    ]


def test_hr_is_not_treated_as_a_pipeline_switch(planner):
    """``hr`` selects weights inside the face stages, not whether they are scheduled.

    Reading it here would make ``{"hr": false}`` decline the face chain, a plan change
    nobody asked for.
    """
    analysis = _analysis(face_count=2)
    assert _names(planner.plan_auto_restore(analysis, {"hr": False})) == _names(
        planner.plan_auto_restore(analysis)
    )


def test_both_planners_report_the_applied_switches_under_the_same_keys(planner):
    """``decisions["options"]`` is one block with one shape, whichever plan ran.

    The report shows it as "Pipeline decisions", so a reader comparing an Auto run with
    a Restore run has to be able to read the same keys out of both 鈥?and the UI sends
    exactly those keys when it decides what a checked box means.
    """
    from fiximg.application.pipeline_modes import PLAN_SWITCHES

    static = planner.plan("restore", {"face_enhance": False, "auto_colorize": True})
    dynamic = planner.plan_auto_restore(_analysis(), {"face_enhance": False})

    assert set(static.decisions["options"]) == set(dynamic.decisions["options"])
    assert set(dynamic.decisions["options"]) == set(PLAN_SWITCHES)
    assert set(planner.PLAN_SWITCHES) == set(PLAN_SWITCHES)
    assert dynamic.decisions["options"]["face_enhance"] is False


# ------------------------------------------------------------- 搂12 options wiring
def test_face_enhance_option_appends_the_whole_chain(planner):
    plan = planner.plan("restore", options={"face_enhance": True})
    names = [n for n, _ in plan.stages]
    assert names == ["global_restore", "face_detection", "face_enhancement", "warp_back"]
    assert plan.decisions["options"]["face_enhance"] is True


def test_auto_colorize_option_appends_stage(planner):
    """auto_colorize keeps the default face chain, then colorizes."""
    plan = planner.plan("restore_scratch", options={"auto_colorize": True})
    names = [n for n, _ in plan.stages]
    assert names == [
        "global_restore", "face_detection", "face_enhancement", "warp_back", "colorization",
    ]


def test_auto_colorize_alone_can_skip_face_enhancement(planner):
    plan = planner.plan(
        "restore", options={"auto_colorize": True, "face_enhance": False}
    )
    assert [n for n, _ in plan.stages] == ["global_restore", "colorization"]


def test_both_options(planner):
    plan = planner.plan("restore", options={"face_enhance": True, "auto_colorize": True})
    names = [n for n, _ in plan.stages]
    assert names == [
        "global_restore", "face_detection", "face_enhancement", "warp_back", "colorization",
    ]


def test_options_default_and_ignored_for_other_types(planner):
    # `restore` defaults to the face chain (V2 parity), opt-out is explicit.
    assert [n for n, _ in planner.plan("restore").stages] == [
        "global_restore", "face_detection", "face_enhancement", "warp_back",
    ]
    # detect/colorize flows are unaffected by the switches.
    assert [n for n, _ in planner.plan("colorize", options={"face_enhance": True}).stages] == ["colorization"]
    assert [n for n, _ in planner.plan("detect_scratch").stages] == ["scratch_detection"]


def test_api_options_now_reach_planner():
    """Regression: the API option whitelist must cover what the planner reads.

    V3 moved the whitelist from the route module into the Pydantic schema, which
    is now the single source of truth for accepted switches.
    """
    from fiximg.api.schemas.task import TaskOptionsSchema, _ALLOWED_OPTIONS

    assert {"face_enhance", "auto_colorize"} <= _ALLOWED_OPTIONS
    parsed = TaskOptionsSchema.parse('{"face_enhance": true, "auto_colorize": true}')
    assert parsed.face_enhance is True
    assert parsed.auto_colorize is True

    # Unknown switches are rejected instead of being silently dropped.
    import pytest

    with pytest.raises(ValueError):
        TaskOptionsSchema.parse('{"not_a_switch": true}')


@pytest.mark.parametrize("key", ["face_enhance", "auto_colorize"])
def test_a_refusal_survives_the_serializer_that_strips_defaults(key):
    """``false`` and "not mentioned" have to stay two different requests on the wire.

    The route forwards only the keys that differ from the schema default, so a field
    defaulting to ``False`` would make an explicit refusal indistinguishable from
    silence 鈥?and on ``auto_restore`` silence means "the analysis decides", so a
    grayscale photo would be colorized after being told not to. Both switches therefore
    default to ``None``.
    """
    from fiximg.api.schemas.task import TaskOptionsSchema

    refused = TaskOptionsSchema.parse(json.dumps({key: False}))
    assert refused.model_dump(exclude_defaults=True) == {key: False}

    silent = TaskOptionsSchema.parse(None)
    assert silent.model_dump(exclude_defaults=True) == {}

    forced = TaskOptionsSchema.parse(json.dumps({key: True}))
    assert forced.model_dump(exclude_defaults=True) == {key: True}
