"""搂3.9.1's 鑷姩鍒嗘瀽 row: what the Auto tab shows before the user presses submit.

The plan puts "Auto Analysis" in the input column, above the three switches. The analyzer
already existed and already ran → but only inside ``_execute_core``, so the user met its
verdict *after* the GPU work, and the captions the previous batch added ("only when the
analysis finds a face") referred to a number no screen ever showed. The same numbers are in
the run report's ``planner_decisions.analysis``, and the UI's completed frame filters dict
values out of that block, so they were invisible there too.

These tests keep three properties honest, in order of how badly each breaks the row:

* the preview is computed by the same call the worker makes, so it cannot drift;
* the stage labels are the names the task row will carry (a plan step keyed
  ``global_restore`` reports itself as ``scratch_repair`` on the scratch branch, and a
  preview showing the registry key would be contradicted by history);
* the switches are inputs to the row, because refusing a stage changes the answer.

The analyzer's own numbers are asserted on synthetic images only through *relations*
(dense lines score above flat grey) → the threshold crossings are pinned by monkeypatching
the analysis, so a change in a threshold constant cannot silently retarget these tests.
"""
from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from fiximg.application.task_service import task_service
from fiximg.inference import analyzer
from fiximg.inference.planner import PipelinePlanner
from fiximg.ui import task_progress as tp

SWITCH_NAMES = ["face_enhance", "auto_colorize", "hr"]

#: A busy photo: scratched, grey, two faces → every optional stage wants to run.
BUSY = {"width": 800, "height": 600, "is_grayscale": True, "face_count": 2,
        "blur_score": 0.6, "scratch_score": 0.9}
#: A clean colour photo: nothing optional is wanted.
CLEAN = {"width": 800, "height": 600, "is_grayscale": False, "face_count": 0,
         "blur_score": 0.1, "scratch_score": 0.05}


def _flat_gray(size=(160, 120)) -> Image.Image:
    return Image.new("RGB", size, (140, 140, 140))


def _scratched_gray(size=(160, 120)) -> Image.Image:
    """Flat grey with dense 1 px white lines: what the scratch heuristic looks for."""
    image = Image.new("RGB", size, (120, 120, 120))
    draw = ImageDraw.Draw(image)
    for x in range(0, size[0], 6):
        draw.line([(x, 0), (x, size[1] - 1)], fill=(255, 255, 255), width=1)
    return image


@pytest.fixture()
def pinned(monkeypatch):
    """Pin the analyzer to a given dict, so thresholds are not what is under test."""
    def _pin(analysis: dict):
        monkeypatch.setattr(analyzer, "analyze_image", lambda image: dict(analysis))
        return analysis
    return _pin


# ------------------------------------------------- the analyzer's own observations
def test_the_preview_reports_what_the_analyzer_sees():
    preview = task_service.auto_preview(_flat_gray())

    assert preview["available"] is True
    assert {"width", "height", "is_grayscale", "blur_score", "scratch_score",
            "face_count"} <= set(preview["analysis"])
    assert preview["analysis"]["is_grayscale"] is True, "a flat grey image is grey"
    # The summary is the analyzer's formatter's output, not a second spelling of it.
    assert preview["summary"] == analyzer.format_analysis_summary(preview["analysis"])


def test_the_scratch_signal_the_planner_branches_on_is_visible_in_the_row():
    dense = task_service.auto_preview(_scratched_gray())
    flat = task_service.auto_preview(_flat_gray())

    assert dense["analysis"]["scratch_score"] > flat["analysis"]["scratch_score"], (
        dense["analysis"], flat["analysis"]
    )


# ------------------------------------------------------- what the plan says back
def test_the_labels_are_the_names_the_task_row_will_record(pinned):
    """The preview must not use registry keys where the run reports stage names.

    ``global_restore(with_scratch=True)`` is a plan step keyed ``global_restore`` whose
    stage reports itself as ``scratch_repair``. Showing the key would make the history
    panel and ``GET /tasks/{id}`` contradict a sentence printed seconds earlier.
    """
    pinned(BUSY)
    preview = task_service.auto_preview(_flat_gray())
    plan = PipelinePlanner().plan_auto_restore(BUSY)

    assert [name for name, _kwargs in plan.stages][0] == "global_restore", plan.stages
    assert preview["stages"][0] == "scratch_repair", preview["stages"]
    assert preview["stages"] == [
        "scratch_repair", "face_detection", "face_enhancement", "warp_back", "colorization",
    ], preview["stages"]


def test_a_refusal_removes_the_stages_and_names_them_declined(pinned):
    pinned(BUSY)
    preview = task_service.auto_preview(_flat_gray(), {"face_enhance": False})

    assert "face_detection" not in preview["stages"], preview["stages"]
    assert preview["declined"] == [
        "face_detection", "face_enhancement", "warp_back",
    ], preview
    assert preview["forced"] == [], preview


def test_a_force_names_the_stages_the_image_did_not_ask_for(pinned):
    """The case the switch exists for: faces the downscaled copy could not count.

    ``forced`` is the row's evidence that a tick did something, on a photo where the
    analysis itself would have scheduled no face chain at all.
    """
    pinned(CLEAN)
    preview = task_service.auto_preview(_flat_gray(), {"face_enhance": True,
                                                      "auto_colorize": True})

    assert preview["declined"] == [], preview
    assert preview["forced"] == [
        "face_detection", "face_enhancement", "warp_back", "colorization",
    ], preview
    assert preview["stages"] == [
        "global_restore", "face_detection", "face_enhancement", "warp_back", "colorization",
    ], preview


def test_no_switches_means_no_notes(pinned):
    """An untouched row must not claim the user decided anything.

    ``declined``/``forced`` are differences against the analysis-only plan, so they have to
    be empty when the options are: a row that printed a note for a switch nobody touched
    would be the same kind of lie as a checkbox that does nothing.
    """
    pinned(BUSY)
    for options in (None, {}, {"hr": True}):
        preview = task_service.auto_preview(_flat_gray(), options)
        assert preview["declined"] == [], options
        assert preview["forced"] == [], options
        assert "Declined" not in tp.describe_preview(preview)


def test_an_unreadable_upload_degrades_the_row_instead_of_the_tab(monkeypatch):
    """The preview is a convenience; the submit button must survive its failure.

    The analyzer runs on bytes a stranger chose, so a decoder complaint is a real
    outcome rather than a programming error → and the row then says what happened.
    """
    def boom(_image):
        raise ValueError("decoder said no")

    monkeypatch.setattr(analyzer, "analyze_image", boom)
    preview = task_service.auto_preview(_flat_gray())

    assert preview["available"] is False
    assert "decoder said no" in preview["reason"], preview
    assert "ValueError" in preview["reason"], preview


# ---------------------------------------------------------------------- the row
def _preview(available=True, **over):
    base = {"available": True,
            "summary": "grayscale=yes scratch=0.80 blur=0.00 faces=2 size=800x600",
            "stages": ["scratch_repair", "face_detection", "face_enhancement",
                       "warp_back", "colorization"], "declined": [], "forced": []}
    return {**base, **over}


def test_the_row_shows_the_numbers_and_the_pipeline_they_select():
    text = tp.describe_preview(_preview())

    assert text.startswith("Analysis: ")
    # The service's summary string survives verbatim: no rounding, no reordering, no
    # second formatter that can disagree with the log line and the run report.
    assert "grayscale=yes scratch=0.80 blur=0.00 faces=2 size=800x600" in text
    assert ("scratch_repair → face_detection → face_enhancement → warp_back"
            " → colorization") in text
    assert "Declined" not in text and "Forced" not in text


def test_the_row_only_states_a_switch_effect_when_one_happened():
    forced = tp.describe_preview(_preview(
        stages=["global_restore", "colorization"], forced=["colorization"],
    ))
    declined = tp.describe_preview(_preview(
        stages=["scratch_repair"],
        declined=["face_detection", "face_enhancement", "warp_back"],
    ))

    assert "Forced by your switches, not by the image: colorization" in forced, forced
    assert "Declined by your switches: face_detection, face_enhancement, warp_back" \
        in declined, declined
    assert "Forced" not in declined and "Declined" not in forced


def test_a_failed_analysis_says_so_in_the_row_that_asks_for_it():
    assert tp.describe_preview({"available": False, "reason": "ValueError: decoder said no"}
                               ) == "Analysis unavailable: ValueError: decoder said no"


def test_only_the_tab_whose_plan_comes_from_the_image_gets_a_row():
    """The restoration tabs decide their stages from the task type, not the picture.

    A row there would show numbers that change nothing about what runs, which is the
    decoration this project keeps paying to remove.
    """
    from fiximg.application.pipeline_modes import SWITCHES

    auto = tp.build_preview("auto")
    assert auto is not None and auto.label == "Auto analysis (before you submit)"
    assert auto.interactive is False, "the row is an answer, not an input"
    assert tp.build_preview("restore") is None
    assert tp.build_preview("colorize") is None
    assert set(SWITCHES) - {"auto"} == {"restore", "restore_scratch", "colorize", "detect"}


# ----------------------------------------------------------------- the callback
class _PreviewService:
    def __init__(self, result=None, raises=None):
        self.calls: list[dict | None] = []
        self.result = result if result is not None else _preview()
        self.raises = raises

    def auto_preview(self, image, options=None):
        self.calls.append(options)
        if self.raises:
            raise self.raises
        return self.result


def _drive(monkeypatch, service, values):
    monkeypatch.setattr(tp, "task_service", service)
    return tp.make_preview_handler("auto")(_flat_gray(), *values)


def test_the_row_is_recomputed_from_the_switches_as_they_stand(monkeypatch):
    """Unticking a box must move the row, or the row is decoration beside a live control.

    The submitted-options mapping is reused, so the row is asked about exactly the options
    the submit button would send → a preview that showed the plan for one mapping while
    submitting another would be worse than no preview at all.
    """
    service = _PreviewService()

    _drive(monkeypatch, service, (True, True, False))
    # Checked plan switches send nothing; `hr` is a stage option, so it goes as typed.
    assert service.calls[-1] == {"hr": False}, service.calls
    _drive(monkeypatch, service, (False, False, False))
    assert service.calls[-1] == {"face_enhance": False, "auto_colorize": False,
                                 "hr": False}, service.calls
    _drive(monkeypatch, service, (True, False, True))
    assert service.calls[-1] == {"auto_colorize": False, "hr": True}, service.calls


def test_the_row_asks_for_the_plan_of_the_options_it_forwarded(monkeypatch):
    """The handler's answer is the service's answer, unedited.

    Asserted through the mapping rather than against a literal list, so a switch that
    stops reaching the planner breaks this too.
    """
    from fiximg.application.pipeline_modes import SWITCHES

    service = _PreviewService()
    names = list(SWITCHES["auto"])
    _drive(monkeypatch, service, (False, True, False))

    assert service.calls[-1] == tp.switch_options("auto", names, (False, True, False))


def test_no_upload_clears_the_row_instead_of_inventing_one(monkeypatch):
    service = _PreviewService()
    monkeypatch.setattr(tp, "task_service", service)

    update = tp.make_preview_handler("auto")(None, True, True, False)

    assert update["value"] == "", update
    assert service.calls == [], "the preview ran without an image"


def test_a_mismatched_control_set_is_refused_by_the_row_too(monkeypatch):
    """Same guard as the submit handler: a mis-wired tab must not guess.

    One missing checkbox would preview the plan for switch values the user never set.
    """
    import gradio as gr

    monkeypatch.setattr(tp, "task_service", _PreviewService())
    handler = tp.make_preview_handler("auto")

    with pytest.raises(gr.Error, match="expects 3 option"):
        handler(_flat_gray(), True, False)
