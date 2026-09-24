"""Unit tests for the V2 task repository (plan section 14)."""
import pytest

from app.repositories import task_repository as repo


@pytest.fixture()
def db_env(tmp_path, monkeypatch):
    """Point the shared SQLite file at a temp directory for test isolation."""
    import app.db as legacy_db
    import app.core.config as config_mod

    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(legacy_db, "DB_PATH", db_path)
    monkeypatch.setattr(legacy_db, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(legacy_db, "_conn", None)
    monkeypatch.setattr(config_mod.settings, "db_path", db_path)
    repo._DDL_DONE = False
    return db_path


def test_task_lifecycle(db_env):
    repo.create_task("t1", "restore", "alice")
    row = repo.get_task("t1")
    assert row["status"] == "queued"
    assert row["progress"] == 0

    repo.start_task("t1")
    repo.update_progress("t1", 40, "global_restore")
    row = repo.get_task("t1")
    assert row["status"] == "running"
    assert row["progress"] == 40
    assert row["current_stage"] == "global_restore"

    repo.record_stage("t1", 0, "global_restore", "running")
    repo.finish_stage("t1", 0, "completed", 1234, None)
    repo.finish_task("t1", "/tmp/out.png", "PSNR: 20", 5000)
    row = repo.get_task("t1")
    assert row["status"] == "completed"
    assert row["result_path"] == "/tmp/out.png"
    assert row["duration_ms"] == 5000
    stages = row["stages"]
    import json

    stages = json.loads(stages) if isinstance(stages, str) else stages
    assert stages[0]["stage_name"] == "global_restore"
    assert stages[0]["duration_ms"] == 1234


def test_failed_task(db_env):
    repo.create_task("t2", "colorize", "bob")
    repo.fail_task("t2", "boom", 100)
    row = repo.get_task("t2")
    assert row["status"] == "failed"
    assert row["error_message"] == "boom"


def test_cancel_only_queued(db_env):
    repo.create_task("t3", "restore", "carol")
    assert repo.cancel_task("t3") is True
    # Already cancelled -> not cancellable again.
    assert repo.cancel_task("t3") is False

    repo.create_task("t4", "restore", "carol")
    repo.start_task("t4")
    assert repo.cancel_task("t4") is False


def test_metrics_and_artifacts(db_env, tmp_path):
    repo.create_task("t5", "restore", "dave")
    repo.add_metric("t5", "psnr", 22.5)
    repo.add_metric("t5", "ssim", 0.8)
    metrics = repo.get_metrics("t5")
    assert metrics["psnr"] == "22.5"
    assert metrics["ssim"] == "0.8"

    img_path = str(tmp_path / "out.png")
    from PIL import Image

    Image.new("RGB", (10, 10), (255, 0, 0)).save(img_path)
    repo.add_artifact("t5", "output", img_path, "image/png")
    arts = repo.get_artifacts("t5")
    assert len(arts) == 1
    assert arts[0]["width"] == 10 and arts[0]["height"] == 10
    assert arts[0]["size_bytes"] > 0


def test_list_tasks_by_user(db_env):
    repo.create_task("t6", "restore", "eve")
    repo.create_task("t7", "colorize", "eve")
    repo.create_task("t8", "restore", "frank")
    eve = repo.list_tasks(user_id="eve")
    assert {r["id"] for r in eve} == {"t6", "t7"}
    assert len(repo.list_tasks(limit=10)) == 3
