"""Unit tests for the V1 -> V2 history migration script (plan section 31 Phase 5)."""
import sqlite3

import pytest

from app.core.config import settings


@pytest.fixture()
def db_env(tmp_path, monkeypatch):
    """Fresh temp database with a legacy history table seeded."""
    import app.db as legacy_db
    import app.repositories.task_repository as task_repo

    db_path = str(tmp_path / "migrate.db")
    monkeypatch.setattr(legacy_db, "DB_PATH", db_path)
    monkeypatch.setattr(legacy_db, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(legacy_db, "_conn", None)
    monkeypatch.setattr(settings, "db_path", db_path)
    monkeypatch.setattr(task_repo, "_DDL_DONE", False)

    legacy_db.init_db()
    conn = legacy_db.get_conn()
    now = "2026-01-02 03:04:05"
    conn.executemany(
        "INSERT INTO history(id, timestamp, user, type, input_path, output_path, psnr, ssim, mae) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        [
            ("h1", now, "alice", "restore", "in1.png", "out1.png", "20.5", "0.8", "0.02"),
            ("h2", now, "bob", "colorize", "in2.png", "out2.png", "N/A", "N/A", "N/A"),
            ("h3", now, "carol", "detect", "in3.png", "out3.png", "", "", ""),
            ("h4", now, "dave", "weird_type", "in4.png", "out4.png", "", "", ""),
        ],
    )
    conn.commit()
    return db_path


def _run_migrate(monkeypatch, apply: bool):
    """Run the migration main() against the temp db; returns (stats, conn)."""
    import app.db as legacy_db
    import scripts.migrate_v1 as mig

    argv_backup = __import__("sys").argv
    monkeypatch.setattr(
        __import__("sys"), "argv",
        ["migrate_v1.py", "--db", settings.db_path] + (["--apply"] if apply else []),
    )
    rc = mig.main()
    monkeypatch.setattr(__import__("sys"), "argv", argv_backup)
    assert rc == 0
    conn = legacy_db.get_conn()
    return conn


def test_dry_run_writes_nothing(db_env, monkeypatch):
    _run_migrate(monkeypatch, apply=False)
    conn = sqlite3.connect(db_env)
    n = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    conn.close()
    assert n == 0  # dry-run rolled back


def test_apply_migrates_history(db_env, monkeypatch):
    _run_migrate(monkeypatch, apply=True)
    conn = sqlite3.connect(db_env)
    conn.row_factory = sqlite3.Row
    rows = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM tasks")}
    metrics = {
        (r["task_id"], r["metric_name"]): r["metric_value"]
        for r in conn.execute("SELECT task_id, metric_name, metric_value FROM metrics")
    }
    conn.close()

    # h4 has an unknown type -> skipped; the other three migrated.
    assert set(rows) == {"h1", "h2", "h3"}
    assert rows["h1"]["task_type"] == "restore"
    assert rows["h2"]["task_type"] == "colorize"
    assert rows["h3"]["task_type"] == "detect_scratch"  # legacy 'detect' renamed
    assert rows["h1"]["status"] == "completed"
    assert rows["h1"]["user_id"] == "alice"
    assert rows["h1"]["input_path"] == "in1.png"
    # Metrics copied with the V2 reference type; N/A/empty skipped.
    assert metrics[("h1", "psnr")] == "20.5"
    assert ("h2", "psnr") not in metrics


def test_rerun_is_idempotent(db_env, monkeypatch):
    _run_migrate(monkeypatch, apply=True)
    _run_migrate(monkeypatch, apply=True)  # second pass: all skipped
    conn = sqlite3.connect(db_env)
    n = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    conn.close()
    assert n == 3
