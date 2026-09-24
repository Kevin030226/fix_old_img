"""V1 -> V2 task data migration (plan sections 31 Phase 5 / 27 scripts.migrate_v1).

Converts legacy `history` rows (id, timestamp, user, type, input_path,
output_path, psnr/ssim/mae) into the V2 `tasks` table, optionally registering
the existing input/output images as `artifacts` when the files still exist.

Usage:
    python -m scripts.migrate_v1            # dry-run: report only, no writes
    python -m scripts.migrate_v1 --apply    # perform the migration
    python -m scripts.migrate_v1 --apply --register-artifacts
"""
import argparse
import os
import sqlite3
import sys

# Project root on sys.path for direct execution.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings  # noqa: E402

# history.type -> V2 task_type
_TYPE_MAP = {
    "restore": "restore",
    "restore_scratch": "restore_scratch",
    "detect": "detect_scratch",
    "detect_scratch": "detect_scratch",
    "colorize": "colorize",
}

# Migrated V1 rows are finished work; they land as completed.
_STATUS_COMPLETED = "completed"
_REFERENCE_TYPE = "input_output_difference"  # plan section 16 semantics


def _ensure_v2_schema(conn: sqlite3.Connection) -> None:
    """Run the V2 DDL (tasks/task_stages/artifacts/metrics) idempotently."""
    from app.repositories.task_repository import _SCHEMA

    conn.executescript(_SCHEMA)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
    if "input_path" not in cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN input_path TEXT")
    conn.commit()


def _image_dims(path: str):
    """Best-effort (width, height) for artifact registration."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:  # noqa: BLE001
        return None, None


def _insert_artifact(conn, task_id, kind, path) -> None:
    """Raw artifacts INSERT without commit (transaction controlled by caller)."""
    width = height = size = None
    try:
        size = os.path.getsize(path)
        width, height = _image_dims(path)
    except OSError:
        pass
    conn.execute(
        "INSERT INTO artifacts(task_id, kind, path, mime_type, width, height, "
        "size_bytes, created_at) VALUES (?,?,?,?,?,?,?,datetime('now','localtime'))",
        (task_id, kind, path, "image/png", width, height, size),
    )


def _insert_metric(conn, task_id, name, value) -> None:
    conn.execute(
        "INSERT INTO metrics(task_id, metric_name, metric_value, reference_type, "
        "created_at) VALUES (?,?,?,?,datetime('now','localtime'))",
        (task_id, name, str(value), _REFERENCE_TYPE),
    )


def migrate_history_to_tasks(conn: sqlite3.Connection, register_artifacts: bool = False) -> dict:
    """Copy legacy history rows into tasks; returns a stats dict.

    Idempotent: rows whose id already exists in tasks are skipped, so the
    script can be re-run safely. All writes happen on the caller's transaction
    — apply commits, dry-run rolls back.
    """
    stats = {"scanned": 0, "migrated": 0, "skipped": 0, "invalid_type": 0, "artifacts": 0}

    if not conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
    ).fetchone():
        print("[migrate_v1] no legacy `history` table found - nothing to do")
        return stats

    rows = conn.execute("SELECT * FROM history ORDER BY timestamp, id").fetchall()
    stats["scanned"] = len(rows)

    for row in rows:
        rec = dict(row)
        hid = rec.get("id")
        task_type = _TYPE_MAP.get((rec.get("type") or "").strip())
        if not task_type:
            stats["invalid_type"] += 1
            continue
        if conn.execute("SELECT 1 FROM tasks WHERE id=?", (hid,)).fetchone():
            stats["skipped"] += 1
            continue

        timestamp = rec.get("timestamp") or ""
        user = rec.get("user") or "unknown"
        input_path = rec.get("input_path") or None
        output_path = rec.get("output_path") or None
        conn.execute(
            "INSERT INTO tasks(id, user_id, task_type, status, progress, input_path, "
            "result_path, evaluation_text, created_at, started_at, finished_at) "
            "VALUES (?,?,?,?,100,?,?,?,?,?,?)",
            (
                hid,
                user,
                task_type,
                _STATUS_COMPLETED,
                input_path,
                output_path,
                None,
                timestamp,
                timestamp,
                timestamp,
            ),
        )
        if register_artifacts:
            for kind, path in (("input", input_path), ("output", output_path)):
                if path and os.path.exists(path):
                    _insert_artifact(conn, hid, kind, path)
                    stats["artifacts"] += 1
        for metric in ("psnr", "ssim", "mae"):
            value = rec.get(metric)
            if value not in (None, "", "N/A"):
                _insert_metric(conn, hid, metric, value)
        stats["migrated"] += 1
    return stats


def _report(stats: dict, mode: str) -> None:
    print(
        f"[migrate_v1] {mode}: scanned={stats['scanned']} migrated={stats['migrated']} "
        f"skipped={stats['skipped']} invalid_type={stats['invalid_type']} "
        f"artifacts={stats['artifacts']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate V1 history rows to V2 tasks")
    parser.add_argument(
        "--apply", action="store_true",
        help="actually write changes (default: dry-run report only)",
    )
    parser.add_argument(
        "--register-artifacts", action="store_true",
        help="register existing input/output files in the artifacts table",
    )
    parser.add_argument("--db", default=None, help="override database path (for tests)")
    args = parser.parse_args()

    if args.db:
        settings.db_path = args.db

    # app.db owns the connection (WAL + row factory); point it at settings.
    from app import db as legacy_db

    legacy_db.DB_PATH = settings.db_path
    legacy_db.ADMIN_DATA_DIR = os.path.dirname(settings.db_path)
    legacy_db._conn = None
    conn = legacy_db.get_conn()
    legacy_db.init_db()
    _ensure_v2_schema(conn)

    if not args.apply:
        # Same reads and inserts inside a transaction, then roll back.
        conn.execute("BEGIN")
        try:
            stats = migrate_history_to_tasks(conn, register_artifacts=args.register_artifacts)
        finally:
            conn.execute("ROLLBACK")
        _report(stats, "DRY-RUN")
        print("[migrate_v1] re-run with --apply to write changes")
        return 0

    stats = migrate_history_to_tasks(conn, register_artifacts=args.register_artifacts)
    conn.commit()
    _report(stats, "applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
