"""Unit tests for the Gradio task-progress panel (plan sections 23/24)."""
import gradio as gr
import pytest

import fiximg.ui.task_progress as tp


def test_bar_rendering():
    assert tp._bar(0) == "[" + "░" * 24 + "]"
    assert tp._bar(50) == "[" + "█" * 12 + "░" * 12 + "]"
    assert tp._bar(100) == "[" + "█" * 24 + "]"


def test_stage_lines_rendering():
    stages = [
        {"stage_name": "global_restore", "status": "completed", "duration_ms": 3200},
        {"stage_name": "face_enhancement", "status": "running", "duration_ms": None},
        {"stage_name": "evaluation", "status": "pending", "duration_ms": None},
    ]
    lines = tp._stage_lines(stages)
    assert "✓ global_restore (3.2 s)" in lines
    assert "▶ face_enhancement" in lines
    assert "○ evaluation" in lines

    failed = [{"stage_name": "global_restore", "status": "failed",
               "duration_ms": 100, "message": "Subprocess failed"}]
    out = tp._stage_lines(failed)
    assert "✗ global_restore (0.1 s) — Subprocess failed" in out


def test_skipped_stage_is_marked_and_explains_itself():
    """`skipped` is not `completed`.

    `StageStatus.SKIPPED` was declared for several batches with no writer, so
    every skip was persisted as `completed`: the progress panel showed a green
    tick for a stage that never ran, the task claimed more than it did, and a
    photo whose face chain was skipped because dlib is missing looked exactly
    like one where enhancement succeeded. The reason has to travel with the
    status, or the reader cannot tell a skip from a success.
    """
    out = tp._stage_lines([
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
    out = tp._stage_lines([
        {"stage_name": "warp_back", "status": "skipped", "duration_ms": None},
    ])
    assert out.strip().startswith("⊘"), ascii(out)
    assert "warp_back" in out


def test_handler_requires_image():
    handler = tp.make_submit_handler("restore")
    with pytest.raises(gr.Error):
        list(handler(None, {"username": "u"}))


def test_handler_sync_fallback_when_no_worker(monkeypatch):
    """Without an inline/external worker the handler keeps the V1 sync path."""

    class FakeService:
        def has_worker(self):
            return False

        def submit(self, mode, image, user_state, options=None):
            return "result.png", "PSNR: 20"

        def queue_depth(self):
            return 0

    monkeypatch.setattr(tp, "task_service", FakeService())
    handler = tp.make_submit_handler("restore")
    # The three switches `restore` declares, in `SWITCHES` order, all left on:
    # a handler wired with the wrong arity raises instead of guessing (below).
    results = list(handler("img", {"username": "u"}, True, False, False))
    # (result_image, progress_text, comparison_slider_update, download_update)
    result_image, text, comparison, download = results[0]
    assert result_image == "result.png"
    assert text == "PSNR: 20"
    assert comparison["value"] == ("img", "result.png"), "the Before/After pair"
    assert download["value"] == "result.png" and download["visible"] is True


def test_a_tab_wired_with_the_wrong_number_of_switches_fails_loudly(monkeypatch):
    """A missing checkbox must not be silently `zip`ped away.

    `zip` truncates, so a tab that wired two of the three switches would submit
    the plan for the two the user could reach and drop the one they last ticked.
    """
    handler = tp.make_submit_handler("restore")
    with pytest.raises(gr.Error, match="expects 3 option"):
        list(handler("img", {"username": "u"}, True, False))


def test_handler_streams_progress_until_completed(monkeypatch):
    """With a worker present the handler polls and yields progressive updates."""
    calls = {"n": 0}

    class FakeService:
        def has_worker(self):
            return True

        def enqueue(self, image, user_state, mode, options=None):
            return "tid-1"

        def queue_depth(self):
            return 0

        def planner_decisions(self, result_path):
            return {}

        def get_task(self, task_id):
            calls["n"] += 1
            if calls["n"] < 3:
                return {
                    "status": "running", "progress": 50,
                    "current_stage": "global_restore",
                    "stages": [{"stage_name": "global_restore", "status": "running"}],
                }
            return {
                "status": "completed", "progress": 100, "duration_ms": 4200,
                "result_path": "final.png",
                "stages": [{"stage_name": "global_restore", "status": "completed",
                            "duration_ms": 4200}],
            }

    monkeypatch.setattr(tp, "task_service", FakeService())
    monkeypatch.setattr(tp.time, "sleep", lambda *_: None)

    handler = tp.make_submit_handler("restore")
    updates = list(handler("img", {"username": "u"}, True, False, False))
    assert len(updates) == 3
    # Each frame is (result_image, progress_text, comparison_slider, download).
    assert "50%" in updates[0][1]
    assert updates[-1][0] == "final.png"
    assert "completed" in updates[-1][1]
