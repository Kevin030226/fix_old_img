"""Unit tests for V2 round-3 additions: options passthrough, sha256 artifacts,
system_events and the §30 stats aggregation."""
import pytest

from app.repositories import task_repository as repo


@pytest.fixture()
def db_env(tmp_path, monkeypatch):
    import app.db as legacy_db
    import app.core.config as config_mod

    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(legacy_db, "DB_PATH", db_path)
    monkeypatch.setattr(legacy_db, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(legacy_db, "_conn", None)
    monkeypatch.setattr(config_mod.settings, "db_path", db_path)
    repo._DDL_DONE = False
    return db_path


# ------------------------------------------------------------- options_json
def test_create_task_with_options(db_env):
    repo.create_task("o1", "restore", "u", options={"hr": True})
    row = repo.get_task("o1")
    assert row["options_json"] == '{"hr": true}'

    repo.create_task("o2", "colorize", "u")
    assert repo.get_task("o2")["options_json"] is None


# -------------------------------------------------------------------- sha256
def test_add_artifact_computes_sha256(db_env, tmp_path):
    from PIL import Image

    img_path = str(tmp_path / "out.png")
    Image.new("RGB", (8, 8), (10, 200, 30)).save(img_path)
    repo.create_task("a1", "restore", "u")
    repo.add_artifact("a1", "output", img_path, "image/png")
    art = repo.get_artifacts("a1")[0]
    assert art["sha256"] and len(art["sha256"]) == 64

    import hashlib

    expected = hashlib.sha256(open(img_path, "rb").read()).hexdigest()
    assert art["sha256"] == expected


# -------------------------------------------------------------- system_events
def test_add_and_list_events(db_env):
    repo.create_task("e1", "restore", "alice")
    repo.add_event("task.enqueued", task_id="e1", user_id="alice", message="queued")
    repo.add_event("task.failed", level="error", task_id="e1", message="boom")
    repo.add_event("worker.started")

    events = repo.list_events()
    assert [e["event_type"] for e in events] == ["worker.started", "task.failed", "task.enqueued"]
    errs = repo.list_events(level="error")
    assert len(errs) == 1 and errs[0]["message"] == "boom"


# ----------------------------------------------------------------- stats §30
def test_task_stats_aggregation(db_env):
    # Two completed + one failed restore tasks, two stages each.
    for i, (status, dur) in enumerate([("completed", 100), ("completed", 300), ("failed", 200)]):
        tid = f"s{i}"
        repo.create_task(tid, "restore", "u", input_path=None)
        repo.start_task(tid)
        repo.record_stage(tid, 0, "global_restore", "running")
        repo.finish_stage(tid, 0, "completed" if status == "completed" else "failed", dur)
        if status == "completed":
            repo.finish_task(tid, f"/tmp/{tid}.png", None, dur)
        else:
            repo.fail_task(tid, "boom", dur)

    stats = repo.task_stats()
    tasks = stats["tasks"]
    assert tasks["total"] == 3
    assert tasks["completed"] == 2
    assert tasks["failed"] == 1
    assert tasks["success_rate"] == 0.6667

    gr = stats["stages"]["global_restore"]
    assert gr["runs"] == 3
    assert gr["p50_ms"] == 200
    assert gr["p95_ms"] == 300  # ceil(3*0.95)=3rd element
    assert gr["avg_ms"] == 200


def test_task_stats_window_filters(db_env):
    import sqlite3

    repo.create_task("w1", "restore", "u")
    repo.start_task("w1")
    repo.record_stage("w1", 0, "global_restore", "running")
    repo.finish_stage("w1", 0, "completed", 500)
    repo.finish_task("w1", "/tmp/w1.png", None, 500)

    # Backdate created_at beyond the window.
    conn = sqlite3.connect(db_env)
    conn.execute(
        "UPDATE task_stages SET created_at=datetime('now','localtime','-30 days')"
    )
    conn.commit()
    conn.close()

    assert "global_restore" not in repo.task_stats(window_days=7)["stages"]
    assert "global_restore" in repo.task_stats(window_days=60)["stages"]
    assert "global_restore" in repo.task_stats()["stages"]
