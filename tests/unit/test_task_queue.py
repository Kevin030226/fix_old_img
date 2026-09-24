"""Unit tests for the DB-backed task queue primitives (plan sections 12/28)."""

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


def test_create_task_with_input_path(db_env):
    repo.create_task("q1", "restore", "alice", input_path="/data/in.png")
    row = repo.get_task("q1")
    assert row["input_path"] == "/data/in.png"
    assert row["status"] == "queued"


def test_claim_returns_oldest_queued(db_env):
    repo.create_task("q2", "restore", "u")
    repo.create_task("q3", "colorize", "u")
    # ids sort after created_at; make timestamps deterministic
    import sqlite3

    conn = sqlite3.connect(db_env)
    conn.execute("UPDATE tasks SET created_at='2026-01-01 00:00:01' WHERE id='q3'")
    conn.execute("UPDATE tasks SET created_at='2026-01-01 00:00:00' WHERE id='q2'")
    conn.commit()
    conn.close()

    claimed = repo.claim_next_task("w1")
    assert claimed is not None
    assert claimed["id"] == "q2"
    assert claimed["status"] == "running"
    assert claimed["claimed_by"] == "w1"
    # Second claim gets the other one.
    claimed2 = repo.claim_next_task("w1")
    assert claimed2["id"] == "q3"
    # Queue empty now.
    assert repo.claim_next_task("w1") is None
    assert repo.queued_count() == 0


def test_claim_skips_non_queued(db_env):
    repo.create_task("q4", "restore", "u")
    repo.start_task("q4")
    repo.create_task("q5", "restore", "u")
    repo.fail_task("q5", "boom")
    repo.create_task("q6", "restore", "u")
    repo.cancel_task("q6")
    assert repo.claim_next_task("w") is None


def test_queued_count(db_env):
    assert repo.queued_count() == 0
    repo.create_task("q7", "restore", "u")
    repo.create_task("q8", "restore", "u")
    assert repo.queued_count() == 2
    repo.start_task("q7")
    assert repo.queued_count() == 1


def test_reset_stale_running(db_env):
    import sqlite3

    repo.create_task("q9", "restore", "u")
    repo.start_task("q9")
    # Backdate started_at beyond the stale timeout.
    conn = sqlite3.connect(db_env)
    conn.execute(
        "UPDATE tasks SET started_at=datetime('now','localtime','-2 hours') WHERE id='q9'"
    )
    conn.commit()
    conn.close()

    assert repo.reset_stale_running(timeout_seconds=3600) == 1
    row = repo.get_task("q9")
    assert row["status"] == "queued"
    assert row["progress"] == 0
    # Fresh running tasks are not touched.
    repo.start_task("q9")
    assert repo.reset_stale_running(timeout_seconds=3600) == 0
