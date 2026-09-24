"""Task repository: tasks / task_stages / artifacts / metrics tables (plan section 14).

Uses the same SQLite connection as app.db (single database file) but manages the
new V2 tables and their schema independently.
"""
import json
import os
import threading
import time
from datetime import datetime

from app.core.config import settings

_DDL_LOCK = threading.Lock()
_DDL_DONE = False
_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id             TEXT PRIMARY KEY,
    user_id        TEXT,
    task_type      TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'queued',
    progress       INTEGER NOT NULL DEFAULT 0,
    current_stage  TEXT,
    input_path     TEXT,
    error_message  TEXT,
    result_path    TEXT,
    evaluation_text TEXT,
    created_at     TEXT NOT NULL,
    started_at     TEXT,
    finished_at    TEXT,
    duration_ms    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

CREATE TABLE IF NOT EXISTS task_stages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      TEXT NOT NULL,
    stage_name   TEXT NOT NULL,
    stage_order  INTEGER NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    duration_ms  INTEGER,
    message      TEXT,
    created_at   TEXT NOT NULL,
    finished_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_task_stages_task ON task_stages(task_id);

CREATE TABLE IF NOT EXISTS artifacts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    path       TEXT NOT NULL,
    mime_type  TEXT,
    width      INTEGER,
    height     INTEGER,
    size_bytes INTEGER,
    sha256     TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_task ON artifacts(task_id);

CREATE TABLE IF NOT EXISTS metrics (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id        TEXT NOT NULL,
    metric_name    TEXT NOT NULL,
    metric_value   TEXT,
    reference_type TEXT NOT NULL DEFAULT 'input_output_difference',
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_task ON metrics(task_id);

CREATE TABLE IF NOT EXISTS system_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    level      TEXT NOT NULL DEFAULT 'info',
    event_type TEXT NOT NULL,
    task_id    TEXT,
    user_id    TEXT,
    message    TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_system_events_task ON system_events(task_id);
CREATE INDEX IF NOT EXISTS idx_system_events_time ON system_events(created_at);
"""


def _get_conn():
    """Reuse the shared app.db connection (same file, WAL, busy timeout)."""
    from app.db import get_conn

    return get_conn()


def ensure_schema() -> None:
    """Create V2 task tables once per process; add missing columns if upgrading."""
    global _DDL_DONE
    if _DDL_DONE:
        return
    with _DDL_LOCK:
        if _DDL_DONE:
            return
        os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
        conn = _get_conn()
        conn.executescript(_SCHEMA)
        # Lightweight migration for databases created before V2 Phase 4:
        # tasks.input_path stores where the queued input image was saved.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        if "input_path" not in cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN input_path TEXT")
        if "options_json" not in cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN options_json TEXT")
        conn.commit()
        _DDL_DONE = True


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _cutoff_iso(seconds: float) -> str:
    """Local-time ISO string `seconds` ago (matches _now() storage format)."""
    return datetime.fromtimestamp(time.time() - seconds).strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------- tasks
def create_task(
    task_id: str, task_type: str, user_id: str = "unknown", input_path: str | None = None,
    options: dict | None = None,
) -> None:
    ensure_schema()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO tasks(id, user_id, task_type, status, progress, input_path, options_json, created_at) "
        "VALUES (?,?,?,?,0,?,?,?)",
        (
            task_id, user_id, task_type, "queued", input_path,
            json.dumps(options) if options else None, _now(),
        ),
    )
    conn.commit()


def start_task(task_id: str) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE tasks SET status='running', started_at=? WHERE id=?", (_now(), task_id)
    )
    conn.commit()


def update_progress(task_id: str, progress: int, current_stage: str | None = None) -> None:
    conn = _get_conn()
    if current_stage is None:
        conn.execute("UPDATE tasks SET progress=? WHERE id=?", (progress, task_id))
    else:
        conn.execute(
            "UPDATE tasks SET progress=?, current_stage=? WHERE id=?",
            (progress, current_stage, task_id),
        )
    conn.commit()


def finish_task(task_id: str, result_path: str, evaluation_text: str | None, duration_ms: int) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE tasks SET status='completed', progress=100, result_path=?, "
        "evaluation_text=?, finished_at=?, duration_ms=? WHERE id=?",
        (result_path, evaluation_text, _now(), duration_ms, task_id),
    )
    conn.commit()


def fail_task(task_id: str, error_message: str, duration_ms: int | None = None) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE tasks SET status='failed', error_message=?, finished_at=?, duration_ms=? WHERE id=?",
        (error_message, _now(), duration_ms, task_id),
    )
    conn.commit()


def get_task(task_id: str):
    ensure_schema()
    row = _get_conn().execute(
        """
        SELECT t.*,
               (SELECT json_group_array(json_object(
                    'stage_name', stage_name, 'status', status,
                    'duration_ms', duration_ms, 'message', message))
                FROM (SELECT * FROM task_stages WHERE task_id=t.id ORDER BY stage_order)
               ) AS stages,
               (SELECT json_group_object(metric_name, metric_value) FROM metrics WHERE task_id=t.id)
               AS metrics
        FROM tasks t WHERE t.id=?
        """,
        (task_id,),
    ).fetchone()
    return dict(row) if row else None


def list_tasks(limit: int = 50, user_id: str | None = None):
    ensure_schema()
    conn = _get_conn()
    if user_id:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE user_id=? ORDER BY created_at DESC, id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def cancel_task(task_id: str) -> bool:
    """Mark a queued (not yet running) task cancelled. Returns False if not cancellable."""
    ensure_schema()
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE tasks SET status='cancelled', finished_at=? WHERE id=? AND status='queued'",
        (_now(), task_id),
    )
    conn.commit()
    return cur.rowcount > 0


def claim_next_task(worker_id: str = "worker") -> dict | None:
    """Atomically claim the oldest queued task (DB-backed queue primitive).

    The WHERE status='queued' guard inside the UPDATE makes this safe across
    processes (the future Redis/Celery backend replaces only this transport).
    Returns the claimed row dict or None when the queue is empty.
    """
    ensure_schema()
    conn = _get_conn()
    with _DDL_LOCK:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT * FROM tasks WHERE status='queued' "
                "ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            conn.execute(
                "UPDATE tasks SET status='running', started_at=?, current_stage=NULL "
                "WHERE id=? AND status='queued'",
                (_now(), row["id"]),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    claimed = dict(row)
    claimed["status"] = "running"
    claimed["claimed_by"] = worker_id
    return claimed


def reset_stale_running(timeout_seconds: float = 3600) -> int:
    """Requeue 'running' tasks whose started_at is older than the timeout.

    Covers worker crashes between claim and finish; returns the number of
    tasks requeued. Inputs stay on disk so a requeued task can run again.
    """
    ensure_schema()
    cutoff = _cutoff_iso(timeout_seconds)
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE tasks SET status='queued', started_at=NULL, progress=0, current_stage=NULL "
        "WHERE status='running' AND started_at IS NOT NULL AND started_at < ?",
        (cutoff,),
    )
    conn.commit()
    return cur.rowcount


def queued_count() -> int:
    ensure_schema()
    row = _get_conn().execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE status='queued'"
    ).fetchone()
    return int(row["n"])


# --------------------------------------------------------------- task stages
def record_stage(task_id: str, order: int, stage_name: str, status: str = "pending") -> None:
    ensure_schema()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO task_stages(task_id, stage_name, stage_order, status, created_at) "
        "VALUES (?,?,?,?,?)",
        (task_id, stage_name, order, status, _now()),
    )
    conn.commit()


def finish_stage(task_id: str, order: int, status: str, duration_ms: int, message: str | None = None) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE task_stages SET status=?, duration_ms=?, message=?, finished_at=? "
        "WHERE task_id=? AND stage_order=?",
        (status, duration_ms, message, _now(), task_id, order),
    )
    conn.commit()


def get_task_stages(task_id: str):
    ensure_schema()
    rows = _get_conn().execute(
        "SELECT * FROM task_stages WHERE task_id=? ORDER BY stage_order", (task_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# ----------------------------------------------------------------- artifacts
def _sha256_of(path: str) -> str | None:
    """Best-effort SHA-256 of an artifact file (plan section 14/19)."""
    import hashlib

    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def add_artifact(task_id: str, kind: str, path: str, mime_type: str | None = None) -> None:
    ensure_schema()
    width = height = size = None
    sha256 = None
    try:
        size = os.path.getsize(path)
        sha256 = _sha256_of(path)
        from PIL import Image

        with Image.open(path) as im:
            width, height = im.size
    except Exception:  # noqa: BLE001
        pass
    conn = _get_conn()
    conn.execute(
        "INSERT INTO artifacts(task_id, kind, path, mime_type, width, height, size_bytes, sha256, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (task_id, kind, path, mime_type, width, height, size, sha256, _now()),
    )
    conn.commit()


def get_artifacts(task_id: str):
    ensure_schema()
    rows = _get_conn().execute(
        "SELECT * FROM artifacts WHERE task_id=? ORDER BY id", (task_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------- metrics
def add_metric(task_id: str, name: str, value, reference_type: str = "input_output_difference") -> None:
    ensure_schema()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO metrics(task_id, metric_name, metric_value, reference_type, created_at) "
        "VALUES (?,?,?,?,?)",
        (task_id, name, str(value), reference_type, _now()),
    )
    conn.commit()


def get_metrics(task_id: str) -> dict:
    ensure_schema()
    rows = _get_conn().execute(
        "SELECT metric_name, metric_value FROM metrics WHERE task_id=?", (task_id,)
    ).fetchall()
    return {r["metric_name"]: r["metric_value"] for r in rows}


# ------------------------------------------------------------- system_events
def add_event(
    event_type: str,
    level: str = "info",
    task_id: str | None = None,
    user_id: str | None = None,
    message: str | None = None,
) -> None:
    """Append a system event (plan section 14 system_events table)."""
    ensure_schema()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO system_events(level, event_type, task_id, user_id, message, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (level, event_type, task_id, user_id, message, _now()),
    )
    conn.commit()


def list_events(limit: int = 100, level: str | None = None) -> list:
    ensure_schema()
    conn = _get_conn()
    if level:
        rows = conn.execute(
            "SELECT * FROM system_events WHERE level=? ORDER BY id DESC LIMIT ?",
            (level, limit),
    ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM system_events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------- stats (§30)
def task_stats(window_days: int | None = None) -> dict:
    """Aggregate task success rate and per-stage P50/P95 timings (plan section 30)."""
    ensure_schema()
    conn = _get_conn()
    where = ""
    params: list = []
    if window_days:
        where = "WHERE created_at >= ?"
        params.append(_cutoff_iso(window_days * 86400))
    row = conn.execute(
        f"SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed, "
        "SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed, "
        "SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END) AS cancelled, "
        "SUM(CASE WHEN status IN ('queued','running') THEN 1 ELSE 0 END) AS pending, "
        "AVG(duration_ms) AS avg_ms "
        f"FROM tasks {where}",
        params,
    ).fetchone()
    summary = dict(row)
    for key in ("completed", "failed", "cancelled", "pending"):
        summary[key] = int(summary[key] or 0)
    summary["avg_ms"] = round(summary["avg_ms"]) if summary["avg_ms"] is not None else None
    summary["success_rate"] = (
        round(summary["completed"] / summary["total"], 4) if summary["total"] else None
    )

    stage_where = " WHERE duration_ms IS NOT NULL"
    stage_params: list = []
    if window_days:
        stage_where += " AND created_at >= ?"
        stage_params.append(_cutoff_iso(window_days * 86400))
    durations: dict = {}
    for name, dur in conn.execute(
        "SELECT stage_name, duration_ms FROM task_stages" + stage_where,
        stage_params,
    ):
        durations.setdefault(name, []).append(int(dur))
    stages = {}
    for name, durs in durations.items():
        durs.sort()
        stages[name] = {
            "runs": len(durs),
            "avg_ms": int(sum(durs) / len(durs)),
            "p50_ms": durs[len(durs) // 2],
            "p95_ms": durs[min(len(durs) - 1, int(len(durs) * 0.95))],
        }
    return {"window_days": window_days, "tasks": summary, "stages": stages}
