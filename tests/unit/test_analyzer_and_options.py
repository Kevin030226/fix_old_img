"""Unit tests for the ImageAnalyzer (plan §11) and auto planning (plan §10)."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.inference import analyzer
from app.inference.planner import PipelinePlanner


def _flat_gray_image(value=140, size=(160, 120)) -> Image.Image:
    arr = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    return Image.fromarray(arr)


def _textured_color_image(seed=3, size=(160, 120)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8))


# ------------------------------------------------------------------ §11 analysis
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


# ------------------------------------------------------------- §10 auto planning
@pytest.fixture()
def planner():
    return PipelinePlanner()


def test_auto_plan_full_chain(planner):
    """grayscale + scratched + faces -> repair + faces + colorization (§10 example)."""
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
    """color + no scratches + no faces -> plain global restoration only (§10 example)."""
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
    from app.core.exceptions import StageNotAvailableError

    with pytest.raises(StageNotAvailableError):
        planner.plan_auto_restore({})
    with pytest.raises(StageNotAvailableError):
        planner.plan("auto_restore")


def test_available_tasks_includes_auto(planner):
    assert "auto_restore" in planner.available_tasks()


# ------------------------------------------------------------ §12 options wiring
def test_face_enhance_option_appends_stages(planner):
    plan = planner.plan("restore", options={"face_enhance": True})
    names = [n for n, _ in plan.stages]
    assert names == ["global_restore", "face_detection", "face_enhancement"]
    assert plan.decisions["options"] == {"face_enhance": True}


def test_auto_colorize_option_appends_stage(planner):
    plan = planner.plan("restore_scratch", options={"auto_colorize": True})
    names = [n for n, _ in plan.stages]
    assert names == ["global_restore", "colorization"]


def test_both_options(planner):
    plan = planner.plan("restore", options={"face_enhance": True, "auto_colorize": True})
    names = [n for n, _ in plan.stages]
    assert names == ["global_restore", "face_detection", "face_enhancement", "colorization"]


def test_options_off_by_default_and_ignored_for_other_types(planner):
    assert [n for n, _ in planner.plan("restore").stages] == ["global_restore"]
    # detect/colorize flows are unaffected by the switches.
    assert [n for n, _ in planner.plan("colorize", options={"face_enhance": True}).stages] == ["colorization"]


def test_api_options_now_reach_planner():
    """Regression: the API whitelist must cover the options the planner reads."""
    from app.api import tasks as api_tasks

    assert {"face_enhance", "auto_colorize"} <= api_tasks._KNOWN_OPTIONS
