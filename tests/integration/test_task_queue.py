"""Unit tests for the DB-backed task queue primitives (plan sections 12/28)."""

import pytest

from fiximg.infrastructure.db.repositories import task_repository as repo

# Cross-layer: repository + DB queue primitives over a real temp database.
pytestmark = pytest.mark.integration


@pytest.fixture()
def db_env(tmp_path, monkeypatch):
    """Point the shared SQLite file at a temp directory for test isolation."""
    import fiximg.infrastructure.db.engine as legacy_db
    import fiximg.config as config_mod

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
    repo.start_task("q5")  # a terminal write only lands on a running row
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

    from fiximg.infrastructure.db import timestamps

    repo.create_task("q9", "restore", "u")
    repo.start_task("q9")
    # Backdate the start two hours, in the form the code writes. The previous
    # version of this line used SQLite's `datetime('now','localtime','-2 hours')`,
    # which is neither canonical nor in UTC: on this UTC+8 host the stored value
    # read as six hours *in the future*, and the sweep only "passed" while the
    # local calendar date happened to equal the UTC one, because a space sorts
    # before the `T` of a canonical instant.
    two_hours_ago = timestamps.deadline(-2 * 3600)
    conn = sqlite3.connect(db_env)
    conn.execute("UPDATE tasks SET started_at=? WHERE id='q9'", (two_hours_ago,))
    conn.commit()
    conn.close()

    assert repo.reset_stale_running(timeout_seconds=3600) == 1
    row = repo.get_task("q9")
    assert row["status"] == "queued"
    assert row["progress"] == 0
    # Fresh running tasks are not touched.
    repo.start_task("q9")
    assert repo.reset_stale_running(timeout_seconds=3600) == 0


def test_each_clock_is_aged_against_a_bound_of_its_own_shape(db_env, monkeypatch):
    """A pre-codec `started_at` has to be dated the way its writer spelled it (§2.7).

    V2 stored local wall-clock seconds - `2026-03-14 17:59:00` on a UTC+8 host, no
    `T`, no zone. Compared against a canonical UTC bound the verdict depends on the
    *shape* rather than the time, because a space sorts before `T`: a task started a
    minute ago reads as older than any cutoff and is requeued from under its own
    worker, while a task from two days ago reads as newer whenever the local date has
    rolled past the UTC one and is never recovered.

    The clock is injected instead of sampled, so both directions are pinned whatever
    the host's offset and time of day happen to be. Before this, the sweep used one
    bound for both shapes and the test above passed only while the local and UTC
    calendar dates agreed.
    """
    import sqlite3

    from fiximg.infrastructure.db import timestamps

    # One fictional instant, spelled both ways: 10:00 UTC is 18:00 here.
    monkeypatch.setattr(timestamps, "now", lambda: "2026-03-14T10:00:00.000000Z")
    monkeypatch.setattr(timestamps, "cutoff",
                        lambda seconds: "2026-03-14T09:00:00.000000Z")
    monkeypatch.setattr(timestamps, "legacy_cutoff",
                        lambda seconds: "2026-03-14 17:00:00")

    def restate(task_id: str, started_at, lease_until) -> None:
        conn = sqlite3.connect(db_env)
        conn.execute(
            "UPDATE tasks SET status='running', started_at=?, lease_until=? WHERE id=?",
            (started_at, lease_until, task_id),
        )
        conn.commit()
        conn.close()

    repo.create_task("q-fresh", "restore", "u")
    repo.start_task("q-fresh")
    restate("q-fresh", "2026-03-14 17:59:00", None)      # one minute ago, legacy shape
    assert repo.reset_stale_running(timeout_seconds=3600) == 0, (
        "a task started a minute ago was requeued because ' ' sorts before 'T'"
    )
    assert repo.get_task("q-fresh")["status"] == "running"
    assert repo.running_rows_without_a_clock() == 0, "a legacy instant is still datable"

    repo.create_task("q-ancient", "restore", "u")
    repo.start_task("q-ancient")
    restate("q-ancient", "2026-03-12 09:00:00", None)     # two days ago, legacy shape
    assert repo.reset_stale_running(timeout_seconds=3600) == 1, (
        "a two-day-old row was never recovered - the legacy bound is not being used"
    )
    assert repo.get_task("q-ancient")["status"] == "queued"
    assert repo.get_task("q-fresh")["status"] == "running", "swept by age, not by shape"

    # A lease is canonical, so a past one ages the row through the first branch.
    restate("q-fresh", "2026-03-14 17:59:00", "2026-03-14T09:59:00.000000Z")
    assert repo.reset_stale_running(timeout_seconds=3600) == 1
    assert repo.get_task("q-fresh")["status"] == "queued"

    repo.create_task("q-noclock", "restore", "u")
    restate("q-noclock", None, None)
    assert repo.running_rows_without_a_clock() == 1
    assert repo.reset_stale_running(timeout_seconds=3600) == 0, "nothing to age"
