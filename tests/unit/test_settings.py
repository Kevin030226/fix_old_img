"""Unit tests for Settings and artifact storage (plan sections 15 and 20)."""
from app.core.config import settings
from app.services import artifact_service


def test_settings_defaults():
    assert settings.max_image_side == 4096
    assert settings.ddcolor_input_size == 512
    assert settings.history_max == 2000
    assert settings.base_dir


def test_artifact_run_dir_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "tasks_root", str(tmp_path / "tasks"))
    run_dir = artifact_service.create_run_dir("req-1")
    assert run_dir.endswith("req-1")
    for sub in ("input", "stages", "output"):
        assert (run_dir / sub if False else f"{run_dir}/{sub}") and __import__("os").path.isdir(
            f"{run_dir}/{sub}"
        )


def test_save_input_and_report(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(settings, "tasks_root", str(tmp_path / "tasks"))
    from PIL import Image

    run_dir = artifact_service.create_run_dir("req-2")
    img = Image.new("RGB", (8, 8), (0, 255, 0))
    input_path = artifact_service.save_input_image(run_dir, "req-2", img)
    assert os.path.exists(input_path)

    report_path = artifact_service.write_report(run_dir, {"task_id": "req-2"})
    assert os.path.exists(report_path)
    import json

    with open(report_path, encoding="utf-8") as f:
        assert json.load(f)["task_id"] == "req-2"
