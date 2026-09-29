"""V1 -> V2 task data migration (plan §31 Phase 5 / §27 `fiximg.cli.migrate_v1`).

Converts legacy `history` rows (id, timestamp, user, type, input_path,
output_path, psnr/ssim/mae) into the V2 `tasks` table, optionally registering
the existing input/output images as `artifacts` when the files still exist.

Usage:
    python -m fiximg.cli.migrate_v1            # dry-run: report only, no writes
    python -m fiximg.cli.migrate_v1 --apply    # perform the migration
    python -m fiximg.cli.migrate_v1 --apply --register-artifacts
"""
import argparse
import os
import sys

# Repository root on sys.path so the vendored model packages stay importable.
from fiximg.paths import ensure_legacy_importable

ensure_legacy_importable()

from fiximg.config import settings  # noqa: E402
from fiximg.infrastructure.db import timestamps  # noqa: E402
from fiximg.infrastructure.db.connection import Connection  # noqa: E402
from fiximg.infrastructure.db.repositories.task_repository import (  # noqa: E402
    split_metric,
)

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


def _instant(value) -> str:
    """A legacy ``history.timestamp`` in the form the repositories store.

    V1 wrote naive local-time strings, which :func:`timestamps.parse` reads and
    re-emits as the same instant in canonical UTC. An unparseable value becomes
    "now" rather than being stored verbatim: a row whose date cannot be read
    cannot be windowed or ordered either, and a wrong-but-sortable date is the
    only option that keeps the rest of the table usable.
    """
    try:
        return timestamps.canonical(timestamps.parse(value))
    except (TypeError, ValueError):
        return timestamps.now()


def _ensure_v2_schema(conn: Connection) -> None:
    """Make sure the task tables exist and carry every column the insert needs.

    Delegates to the repository rather than repeating its DDL: this script and
    ``ensure_schema`` used to keep two lists of "columns added later", and the
    copy had already fallen behind (no ``metric_text``).
    """
    from fiximg.infrastructure.db.repositories import task_repository

    task_repository.ensure_schema()


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
        "size_bytes, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (task_id, kind, path, "image/png", width, height, size, timestamps.now()),
    )


def _insert_metric(conn, task_id, name, value) -> None:
    number, text = split_metric(value)
    conn.execute(
        "INSERT INTO metrics(task_id, metric_name, metric_value, metric_text, "
        "reference_type, created_at) VALUES (?,?,?,?,?,?)",
        (task_id, name, number, text, _REFERENCE_TYPE, timestamps.now()),
    )


def migrate_history_to_tasks(conn: Connection, register_artifacts: bool = False) -> dict:
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

        # The row's own instant, in the form the rest of the data layer stores
        # (plan §2.7): V1 wrote naive local strings, and copying them verbatim
        # would leave migrated tasks unorderable next to native ones.
        created = _instant(rec.get("timestamp"))
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
                created,
                created,
                created,
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
    from fiximg.infrastructure.db import engine as legacy_db

    legacy_db.DB_PATH = settings.db_path
    legacy_db.ADMIN_DATA_DIR = os.path.dirname(settings.db_path)
    # Any handle opened against the previous path has to go, including the
    # thread-local ones — see engine.close_connections().
    legacy_db.close_connections()
    conn = legacy_db.get_conn()
    legacy_db.init_db()
    _ensure_v2_schema(conn)

    if not args.apply:
        # Same reads and inserts, then undo them. No explicit BEGIN: both engines
        # open a transaction at the first statement (sqlite3's implicit handling,
        # psycopg with autocommit off), so rollback is the whole contract.
        try:
            stats = migrate_history_to_tasks(conn, register_artifacts=args.register_artifacts)
        finally:
            conn.rollback()
        _report(stats, "DRY-RUN")
        print("[migrate_v1] re-run with --apply to write changes")
        return 0

    stats = migrate_history_to_tasks(conn, register_artifacts=args.register_artifacts)
    conn.commit()
    _report(stats, "applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
