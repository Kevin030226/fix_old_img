"""Unit tests for the Gradio task-progress panel (plan sections 23/24)."""
import gradio as gr
import pytest

from app.ui import task_progress as tp


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

    monkeypatch.setattr(tp, "task_service", FakeService())
    handler = tp.make_submit_handler("restore")
    results = list(handler("img", {"username": "u"}))
    assert results == [("result.png", "PSNR: 20")]


def test_handler_streams_progress_until_completed(monkeypatch):
    """With a worker present the handler polls and yields progressive updates."""
    calls = {"n": 0}

    class FakeService:
        def has_worker(self):
            return True

        def enqueue(self, image, user_state, mode, options=None):
            return "tid-1"

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
    updates = list(handler("img", {"username": "u"}))
    assert len(updates) == 3
    assert updates[0] == (None, "") or "█" in updates[0][1]  # progress frames first
    assert "50%" in updates[0][1]
    assert updates[-1][0] == "final.png"
    assert "completed" in updates[-1][1]
