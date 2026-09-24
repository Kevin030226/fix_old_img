"""History service: legacy `history` table kept for the V1 admin panel (plan section 14).

New task-level records go to the V2 tables via task_repository; this module keeps
the existing admin-panel statistics working during the migration window.
"""
from app import db as _db


def list_history(limit: int = 50):
    return _db.list_history(limit)


def get_history_record(history_id: str):
    return _db.get_history_record(history_id)


def clear_history():
    _db.clear_history()


def history_stats() -> str:
    return _db.history_stats()


ARCHIVE_INPUT_DIR = _db.ARCHIVE_INPUT_DIR
ARCHIVE_OUTPUT_DIR = _db.ARCHIVE_OUTPUT_DIR
purge_stale_archives = _db.purge_stale_archives
