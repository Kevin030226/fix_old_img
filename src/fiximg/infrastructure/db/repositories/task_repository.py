"""Task repository: tasks / task_stages / artifacts / metrics tables (plan section 14).

Uses the same SQLite connection as app.db (single database file) but manages the
new V2 tables and their schema independently.
"""
import json
import math
import os
import threading
from collections.abc import Iterable

from fiximg.config import settings
from fiximg.domain.tasks import decode_json_column
from fiximg.infrastructure.db import timestamps
from fiximg.infrastructure.db.dialect import default_dialect
from fiximg.infrastructure.observability.logging import get_logger, log_event

logger = get_logger("fiximg.db.tasks")

_DDL_LOCK = threading.Lock()
#: The database this process has built the task schema *in* — ``ensure_schema``
#: returns early while it matches the configured target. It is a per-database
#: latch, not a per-process one: a test fixture (or an embedder) that repoints
#: the database must get a fresh ``CREATE TABLE``, otherwise the first insert
#: after the switch fails with "no such table" in code that looks initialised.
#: Any falsy value means "not built", which is what the fixtures set.
_DDL_DONE = None
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
    -- V3 (plan §6): machine-readable failure code so clients can branch on the
    -- failure kind and history can be aggregated by cause.
    error_code     TEXT,
    result_path    TEXT,
    evaluation_text TEXT,
    -- Every *_at column below is an instant in the canonical form defined by
    -- fiximg.infrastructure.db.timestamps: fixed-width UTC ISO-8601 with
    -- microseconds ("2026-09-27T12:34:56.789012Z"). TEXT is deliberate on
    -- SQLite, which has no datetime type and compares such a string
    -- lexicographically in index order — the value's precision and zone were the
    -- bug, not the declared type.
    created_at     TEXT NOT NULL,
    started_at     TEXT,
    finished_at    TEXT,
    duration_ms    INTEGER,
    -- V3 queue reliability (plan §2.6) --
    priority          INTEGER NOT NULL DEFAULT 0,
    attempt_count     INTEGER NOT NULL DEFAULT 0,
    max_attempts      INTEGER NOT NULL DEFAULT 3,
    worker_id         TEXT,
    lease_until       TEXT,
    last_heartbeat    TEXT,
    retry_at          TEXT,
    idempotency_key   TEXT,
    -- V3 multi-GPU routing (plan §4.3 Step 4): JSON array of capabilities
    -- needed to complete the task, so a specialised worker can skip it.
    required_capabilities TEXT
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
    stage_version TEXT,
    -- V3 (plan §5.3): numeric measurements the stage reported itself.
    metrics_json TEXT,
    created_at   TEXT NOT NULL,
    finished_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_task_stages_task ON task_stages(task_id);

CREATE TABLE IF NOT EXISTS artifacts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    path       TEXT NOT NULL,
    -- V3 (plan §2.8): the object-store key this artifact was published under, set
    -- only when the configured store is remote. NULL means "the path above *is*
    -- the location", which is what keeps a local deployment reading exactly as
    -- before instead of resolving a key that duplicates the path.
    uri        TEXT,
    mime_type  TEXT,
    width      INTEGER,
    height     INTEGER,
    size_bytes INTEGER,
    sha256     TEXT,
    model_version TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_task ON artifacts(task_id);

CREATE TABLE IF NOT EXISTS metrics (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id        TEXT NOT NULL,
    metric_name    TEXT NOT NULL,
    -- V3 (plan §2.7): numbers are stored as numbers so they can be aggregated in
    -- SQL. `metric_text` carries what a REAL cannot: PSNR after a bit-identical
    -- image is +inf, and any label a future metric decides to report. Exactly
    -- one of the two is set, and readers COALESCE them.
    metric_value   REAL,
    metric_text    TEXT,
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
    data_json  TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_system_events_task ON system_events(task_id);
CREATE INDEX IF NOT EXISTS idx_system_events_time ON system_events(created_at);
"""

#: Columns added after the first V2 release; applied with ALTER TABLE on upgrade.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("tasks", "input_path", "TEXT"),
    ("tasks", "options_json", "TEXT"),
    ("tasks", "priority", "INTEGER NOT NULL DEFAULT 0"),
    ("tasks", "attempt_count", "INTEGER NOT NULL DEFAULT 0"),
    ("tasks", "max_attempts", "INTEGER NOT NULL DEFAULT 3"),
    ("tasks", "worker_id", "TEXT"),
    ("tasks", "lease_until", "TEXT"),
    ("tasks", "last_heartbeat", "TEXT"),
    ("tasks", "retry_at", "TEXT"),
    ("tasks", "idempotency_key", "TEXT"),
    ("tasks", "required_capabilities", "TEXT"),
    ("task_stages", "metrics_json", "TEXT"),
    ("tasks", "error_code", "TEXT"),
    ("task_stages", "stage_version", "TEXT"),
    ("artifacts", "model_version", "TEXT"),
    ("system_events", "data_json", "TEXT"),
    # The non-numeric half of a metric value (plan §2.7). Adding a column is safe
    # here; changing `metric_value`'s *type* is not, which is why that part lives
    # only in migration 0003.
    ("metrics", "metric_text", "TEXT"),
    # Store key for a published artifact (plan §2.8). NULL for a local store,
    # whose `path` already is the location — see add_artifact's docstring.
    ("artifacts", "uri", "TEXT"),
)


def _get_conn():
    """Reuse the shared app.db connection (same file, WAL, busy timeout)."""
    from fiximg.infrastructure.db.engine import get_conn

    return get_conn()


def _database_key() -> tuple[str, str]:
    """The ``(kind, target)`` the schema latch is remembered against."""
    from fiximg.infrastructure.db.engine import resolve_database

    return resolve_database()


def ensure_schema() -> None:
    """Create V2/V3 task tables once per database; add missing columns if upgrading."""
    global _DDL_DONE
    wanted = _database_key()
    if wanted == _DDL_DONE:
        return
    with _DDL_LOCK:
        if wanted == _DDL_DONE:
            return
        os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
        conn = _get_conn()
        conn.executescript(_SCHEMA)
        # Lightweight forward migration: add any column introduced after the
        # database was first created (plan §2.6 queue bookkeeping).
        dialect = default_dialect()
        for table, column, ddl in _ADDED_COLUMNS:
            statement, params = dialect.table_columns_sql(table)
            cols = {r["name"] for r in conn.execute(statement, params)}
            if column not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        # Indexes that depend on the migrated columns must be created afterwards.
        conn.executescript(_POST_MIGRATION_SCHEMA)
        conn.commit()
        _DDL_DONE = wanted


#: Indexes created only after :data:`_ADDED_COLUMNS` have been applied.
_POST_MIGRATION_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_tasks_claim ON tasks(status, priority DESC, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_idem
    ON tasks(idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tasks_lease ON tasks(status, lease_until);
"""

#: How many queued rows a capability-filtered claim inspects before giving up.
#: Bounded so a queue full of tasks for other workers cannot stall a claim.
_CLAIM_SCAN_WINDOW = 64

#: Every column that holds an instant, canonical or not. ``users`` is listed
#: although the engine owns that table: the point of the list is one place that
#: says where time is stored, and the read-only check below is the only user.
#: Revision ``0003`` keeps its own frozen copy on purpose — a migration must not
#: change because the application's schema did.
INSTANT_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tasks", ("created_at", "started_at", "finished_at",
               "lease_until", "last_heartbeat", "retry_at")),
    ("task_stages", ("created_at", "finished_at")),
    ("artifacts", ("created_at",)),
    ("metrics", ("created_at",)),
    ("system_events", ("created_at",)),
    ("users", ("created_at", "updated_at")),
)

#: ``len("YYYY-MM-DD HH:MM:SS")`` — what V2 wrote. The canonical form is 27.
LEGACY_INSTANT_LENGTH = 19


def legacy_instant_rows() -> dict[str, int]:
    """Count rows whose instants are still in the pre-§2.7 format.

    Nothing in this repository writes that format any more, so a non-empty result
    means the code has moved on and revision ``0003`` has not been applied. That
    combination is not merely untidy: a string comparison between a canonical
    ``2026-09-27T02:00:00.000000Z`` and a legacy ``2026-09-27 23:00:00`` (which on
    a UTC+8 host is 15:00Z) says the legacy one is older, so an unexpired lease
    reads as expired and a running task gets requeued underneath its worker.

    A deliberate one-shot check rather than a startup scan: counting requires a
    full scan per column, and "is this database migrated?" is an operator's
    question, not a per-boot one. See ``make db-check``.
    """
    ensure_schema()
    conn = _get_conn()
    counts: dict[str, int] = {}
    for table, columns in INSTANT_COLUMNS:
        for column in columns:
            try:
                row = conn.execute(
                    f"SELECT COUNT(*) AS n FROM {table} WHERE length({column}) = ?",
                    (LEGACY_INSTANT_LENGTH,),
                ).fetchone()
            except Exception:  # noqa: BLE001
                # The table (or the column) is not there yet — `engine.init_db`
                # owns users/history and may not have run. That is a database
                # that has nothing to migrate, not one that is behind, so it must
                # not turn `make db-check` red on a fresh install.
                continue
            if int(row["n"]):
                counts[f"{table}.{column}"] = int(row["n"])
    return counts


# --------------------------------------------------------------------- tasks
def create_task(
    task_id: str, task_type: str, user_id: str = "unknown", input_path: str | None = None,
    options: dict | None = None, priority: int = 0, max_attempts: int | None = None,
    idempotency_key: str | None = None, required_capabilities: Iterable[str] | None = None,
) -> None:
    """Insert a queued task row (plan §2.6).

    ``idempotency_key`` is backed by a partial unique index: re-submitting the
    same key raises the driver's integrity error (:func:`integrity_errors`
    lists them for both engines), which the service layer turns into "return the
    existing task" instead of running the work twice.

    ``required_capabilities`` records what the task needs so a worker pinned to
    a GPU (or specialised in a subset of capabilities) can skip it (§4.3 Step 4).
    """
    ensure_schema()
    from fiximg.config import settings

    caps = sorted(set(required_capabilities or ()))
    conn = _get_conn()
    conn.execute(
        "INSERT INTO tasks(id, user_id, task_type, status, progress, input_path, options_json, "
        "created_at, priority, attempt_count, max_attempts, idempotency_key, required_capabilities) "
        "VALUES (?,?,?,?,0,?,?,?,?,0,?,?,?)",
        (
            task_id, user_id, task_type, "queued", input_path,
            json.dumps(options) if options else None, timestamps.now(),
            int(priority),
            int(max_attempts if max_attempts is not None else settings.task_max_attempts),
            idempotency_key,
            json.dumps(caps) if caps else None,
        ),
    )
    conn.commit()


def find_by_idempotency_key(idempotency_key: str):
    """Return the task previously created with this key, or None."""
    if not idempotency_key:
        return None
    ensure_schema()
    row = _get_conn().execute(
        "SELECT * FROM tasks WHERE idempotency_key=?", (idempotency_key,)
    ).fetchone()
    return dict(row) if row else None


def start_task(task_id: str) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE tasks SET status='running', started_at=? WHERE id=?", (timestamps.now(), task_id)
    )
    conn.commit()


def update_progress(task_id: str, progress: int, current_stage: str | None = None) -> None:
    """Advance the progress bar, while the task is still ours to advance.

    Fenced at `running` like the heartbeat and the terminal writes: a task cancelled
    mid-run keeps reporting the progress it reached when it was cancelled, rather than
    climbing after the user was told it stopped.
    """
    conn = _get_conn()
    if current_stage is None:
        conn.execute(
            "UPDATE tasks SET progress=? WHERE id=? AND status='running'", (progress, task_id)
        )
    else:
        conn.execute(
            "UPDATE tasks SET progress=?, current_stage=? WHERE id=? AND status='running'",
            (progress, current_stage, task_id),
        )
    conn.commit()


def finish_task(task_id: str, result_path: str, evaluation_text: str | None,
                duration_ms: int, worker_id: str | None = None) -> bool:
    """Close a task as completed; False when this worker no longer owns the row.

    The fence is what makes the lease mean anything. A worker that stalls past
    `lease_until` gets its task requeued and possibly re-run elsewhere; without
    `status='running'` (and, on the queued path, the claiming `worker_id`) the
    stalled original could come back and overwrite the newer attempt's row —
    or mark a task the user cancelled as completed.
    """
    conn = _get_conn()
    sql = ("UPDATE tasks SET status='completed', progress=100, result_path=?, "
           "evaluation_text=?, finished_at=?, duration_ms=? "
           "WHERE id=? AND status='running'")
    params: list = [result_path, evaluation_text, timestamps.now(), duration_ms, task_id]
    if worker_id:
        sql += " AND worker_id=?"
        params.append(worker_id)
    cur = conn.execute(sql, params)
    conn.commit()
    return cur.rowcount > 0


def fail_task(task_id: str, error_message: str, duration_ms: int | None = None,
              error_code: str | None = None, worker_id: str | None = None) -> bool:
    """Mark a task failed, recording both the message and the machine code.

    ``error_code`` comes from the unified error taxonomy (plan §3.4.1) so a
    client can branch on the failure kind instead of parsing prose. Returns False
    when the row is no longer running under this worker — see :func:`finish_task`
    for why the write is fenced at all.
    """
    conn = _get_conn()
    sql = ("UPDATE tasks SET status='failed', error_message=?, error_code=?, "
           "finished_at=?, duration_ms=? WHERE id=? AND status='running'")
    params: list = [error_message, error_code, timestamps.now(), duration_ms, task_id]
    if worker_id:
        sql += " AND worker_id=?"
        params.append(worker_id)
    cur = conn.execute(sql, params)
    conn.commit()
    return cur.rowcount > 0


def _json_fragments() -> dict:
    """Dialect-specific JSON function names for the stage/metric subqueries.

    Two different jobs, and the engines do not agree on either name: the
    *aggregates* that fold rows into an array/object are
    ``json_group_array``/``json_group_object`` on SQLite and
    ``json_agg``/``json_object_agg`` on PostgreSQL, while the *scalar* that builds
    one object per stage row is ``json_object`` vs ``json_build_object``. Both
    aggregates return NULL for zero rows, which the callers rely on.
    """
    dialect = default_dialect()
    return {
        "json_array": dialect.json_array_fn,
        "json_object": dialect.json_object_fn,
        "row_object": dialect.row_object_fn,
        "metric_value": dialect.json_metric_value("metric_value", "metric_text"),
    }


def get_task(task_id: str):
    ensure_schema()
    row = _get_conn().execute(
        """
        SELECT t.*,
               (SELECT {json_array}({row_object}(
                    'stage_name', stage_name, 'stage_order', stage_order,
                    'status', status,
                    'duration_ms', duration_ms, 'message', message,
                    'stage_version', stage_version, 'metrics_json', metrics_json))
                FROM (SELECT * FROM task_stages WHERE task_id=t.id ORDER BY stage_order)
               ) AS stages,
               (SELECT {json_object}(metric_name, {metric_value}) FROM metrics WHERE task_id=t.id)
               AS metrics
        FROM tasks t WHERE t.id=?
        """.format(**_json_fragments()),
        (task_id,),
    ).fetchone()
    return dict(row) if row else None


def list_tasks(limit: int = 50, user_id: str | None = None):
    """List tasks newest-first, each with its stage checklist and metrics.

    The aggregates mirror :func:`get_task` so callers (history panel, list API)
    can render stages/metrics without an N+1 fetch per row.
    """
    ensure_schema()
    conn = _get_conn()
    columns = """
        t.*,
        (SELECT {json_array}({row_object}(
             'stage_name', stage_name, 'stage_order', stage_order,
             'status', status,
             'duration_ms', duration_ms, 'message', message,
             'stage_version', stage_version, 'metrics_json', metrics_json))
         FROM (SELECT * FROM task_stages WHERE task_id=t.id ORDER BY stage_order)
        ) AS stages,
        (SELECT {json_object}(metric_name, {metric_value}) FROM metrics WHERE task_id=t.id)
        AS metrics
    """.format(**_json_fragments())
    if user_id:
        rows = conn.execute(
            f"SELECT {columns} FROM tasks t WHERE t.user_id=? "
            "ORDER BY t.created_at DESC, t.id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {columns} FROM tasks t ORDER BY t.created_at DESC, t.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def cancel_task(task_id: str) -> bool:
    """Cancel a task that has not finished. False when it has, or does not exist.

    Both states before completion are cancellable. A *running* task is cancelled
    cooperatively: the row flips immediately so the API, the SSE stream and the user
    agree at once, and the worker that owns it stops at the next stage boundary
    (`_run_plan` checks before claiming each stage). It is not killed mid-stage —
    interrupting a running model call would leave the CUDA context and the run
    directory in a state no one has specified — so a single long stage still finishes
    before the stop takes effect.

    The row keeps its `worker_id` and `lease_until`. That is deliberate: every
    terminal write that worker makes is fenced at `status='running'`, so keeping the
    owner stamped proves the fence holds, and the stale sweep ignores the row because
    it is no longer running.
    """
    ensure_schema()
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE tasks SET status='cancelled', finished_at=? "
        "WHERE id=? AND status IN ('queued','running')",
        (timestamps.now(), task_id),
    )
    conn.commit()
    return cur.rowcount > 0


def interruption(task_id: str, owner_worker_id: str | None = None) -> str | None:
    """Why this run should stop now, or None to keep going.

    Two real decisions stop a run, and neither of them is an error:

    * the user cancelled it — the row says `cancelled`;
    * the lease expired and the sweep gave the task to another worker — the row says
      `running` under a **different** `worker_id`, so continuing would execute the same
      task twice on two devices (plan §2.6's duplicate-execution failure).

    Anything else keeps the run going, deliberately:

    * no row at all: an embedder or harness running a plan without persisting it is not
      an interruption, and the terminal writes are fenced anyway;
    * `queued` with no known owner: same case, seen before any claim existed;
    * `completed`/`failed`: those states can only have been written by this run or by a
      fenced-out one, and the fence — not this check — is what decides the outcome.

    Reading the status *and* the owner is the point: the earlier version of this
    predicate compared the status alone, and a task legitimately handed over reads
    `running`, so the losing worker never noticed and ran the whole plan a second time.
    """
    ensure_schema()
    row = _get_conn().execute(
        "SELECT status, worker_id FROM tasks WHERE id=?", (task_id,)
    ).fetchone()
    if row is None:
        return None
    status = row["status"]
    if status == 'cancelled':
        return "cancelled by request"
    if (owner_worker_id and status == 'running'
            and row["worker_id"] and row["worker_id"] != owner_worker_id):
        return f"lease handed to worker {row['worker_id']}"
    if owner_worker_id and status == 'queued':
        return "requeued: the lease expired"
    return None


def claim_next_task(
    worker_id: str = "worker", lease_seconds: float | None = None,
    capabilities: Iterable[str] | None = None,
) -> dict | None:
    """Atomically claim the next eligible task (DB-backed queue primitive).

    V3 semantics (plan §2.6):

    * eligibility = ``status='queued'`` **and** the retry backoff has elapsed
      (``retry_at IS NULL OR retry_at <= now``),
    * ordering is ``priority DESC, created_at`` so urgent work jumps the line,
    * claiming increments ``attempt_count``, stamps ``worker_id`` and starts a
      lease (``lease_until``) plus a heartbeat timestamp — a worker that dies
      mid-run is detected by lease expiry rather than by a wall-clock guess.

    ``capabilities`` restricts the claim to tasks this worker can serve
    (plan §4.3 Step 4): a task whose ``required_capabilities`` are not a subset
    is skipped. Passing nothing keeps the original "take the head of the queue"
    behaviour. Candidate filtering happens inside the same ``BEGIN IMMEDIATE``
    transaction as the claim, so it stays atomic.
    """
    ensure_schema()
    from fiximg.config import settings

    lease = float(lease_seconds if lease_seconds is not None else settings.worker_lease_seconds)
    served = set(capabilities or ())
    now = timestamps.now()
    conn = _get_conn()
    dialect = default_dialect()
    with _DDL_LOCK:
        # SQLite needs BEGIN IMMEDIATE so the read-then-update cannot be upgraded
        # mid-transaction (SQLITE_BUSY under contention). PostgreSQL takes a row
        # lock instead and returns None here.
        begin = dialect.begin_write_sql()
        if begin:
            conn.execute(begin)
        try:
            row = _select_claimable(conn, now, served, dialect)
            if row is None:
                conn.commit()
                return None
            conn.execute(
                "UPDATE tasks SET status='running', started_at=?, current_stage=NULL, "
                "worker_id=?, lease_until=?, last_heartbeat=?, attempt_count=attempt_count+1 "
                "WHERE id=? AND status='queued'",
                (now, worker_id, timestamps.deadline(lease), now, row["id"]),
            )
            # Re-read inside the transaction so the caller sees the claimed
            # state (worker_id/lease/attempt) rather than the pre-update row.
            claimed_row = conn.execute(
                "SELECT * FROM tasks WHERE id=?", (row["id"],)
            ).fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    claimed = dict(claimed_row if claimed_row is not None else row)
    claimed["status"] = "running"
    claimed["claimed_by"] = worker_id
    return claimed


def claim_statement(dialect=None, *, lock: bool = True) -> str:
    """The claim's SELECT, spelled for this dialect, with the placeholders as ``?``.

    Exposed rather than kept inline because three things have to agree with one
    statement: the ordering that makes a submitted ``priority`` mean anything, the row
    lock that keeps two workers off the same row, and the dialect test that compiles it.
    A copy of the SQL inside a test keeps passing after the real statement changes —
    which is exactly how ``ORDER BY priority DESC`` went unnoticed when it was removed.
    """
    dialect = dialect or default_dialect()
    suffix = f" {dialect.claim_row_lock()}" if lock and dialect.supports_skip_locked else ""
    return dialect.adapt(
        "SELECT * FROM tasks WHERE status='queued' "
        "AND (retry_at IS NULL OR retry_at <= ?) "
        f"ORDER BY priority DESC, created_at, id LIMIT ?{suffix}"
    )


def _select_claimable(conn, now: str, served: set[str], dialect=None):
    """Pick the highest-priority eligible row the worker can actually serve.

    Without a capability filter the database picks the head of the queue (one
    row). With a filter we scan a bounded window and apply the subset test in
    Python: neither SQLite nor PostgreSQL has a portable JSON-containment
    operator for a plain TEXT column, and the window is small enough that the
    extra rows are cheap.

    On a dialect that supports it, the row is locked as it is selected
    (``FOR UPDATE SKIP LOCKED``), so two workers racing for the head of the queue
    do not block each other — one gets the row, the other skips to the next.
    """
    dialect = dialect or default_dialect()
    base = claim_statement(dialect)
    params = dialect.params((now, 1))
    if not served:
        return conn.execute(base, params).fetchone()

    window = dialect.params((now, _CLAIM_SCAN_WINDOW))
    for candidate in conn.execute(base, window).fetchall():
        required = _decode_capabilities(candidate["required_capabilities"])
        if required <= served:
            return candidate
    return None


def _decode_capabilities(raw) -> set[str]:
    """The routing hint as a set of names, in whichever shape this engine returned it.

    The column is JSON text on SQLite and a real list on PostgreSQL (``json_agg`` is
    decoded by psycopg). Reading it with ``json.loads`` alone was not an inefficiency
    but a silent behaviour change: on PostgreSQL ``json.loads(list)`` raises
    ``TypeError``, this returned the empty set, and every task then looked like it
    needed no capability — so a colour-only worker claimed restoration tasks.
    """
    decoded = decode_json_column(raw, [])
    return {str(item) for item in decoded} if isinstance(decoded, list) else set()


def heartbeat_task(task_id: str, lease_seconds: float | None = None) -> None:
    """Refresh a running task's lease (plan §2.6 worker heartbeat)."""
    from fiximg.config import settings

    lease = float(lease_seconds if lease_seconds is not None else settings.worker_lease_seconds)
    conn = _get_conn()
    conn.execute(
        "UPDATE tasks SET last_heartbeat=?, lease_until=? WHERE id=? AND status='running'",
        (timestamps.now(), timestamps.deadline(lease), task_id),
    )
    conn.commit()


def retry_task(task_id: str, delay_seconds: float | None = None) -> bool:
    """Return a claimed/failed task to the queue.

    Returns False when the task has exhausted ``max_attempts`` — the caller then
    leaves it failed. ``delay_seconds`` implements the retry backoff.
    """
    ensure_schema()
    from fiximg.config import settings

    delay = float(delay_seconds if delay_seconds is not None else settings.task_retry_backoff_seconds)
    conn = _get_conn()
    row = conn.execute(
        "SELECT status, attempt_count, max_attempts FROM tasks WHERE id=?", (task_id,)
    ).fetchone()
    if row is None:
        return False
    if int(row["attempt_count"] or 0) >= int(row["max_attempts"] or 1):
        return False
    conn.execute(
        "UPDATE tasks SET status='queued', started_at=NULL, finished_at=NULL, progress=0, "
        "current_stage=NULL, worker_id=NULL, lease_until=NULL, retry_at=?, "
        "error_message=NULL, error_code=NULL WHERE id=?",
        (timestamps.deadline(delay) if delay > 0 else None, task_id),
    )
    conn.commit()
    return True


def requeue_for_manual_retry(task_id: str) -> bool:
    """Operator-initiated retry (plan §3.8 ``POST /tasks/{id}/retry``).

    Unlike :func:`retry_task` this ignores ``max_attempts`` and resets the
    attempt counter — a human explicitly asked for another run. Returns False
    when the task is missing or still active.
    """
    ensure_schema()
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE tasks SET status='queued', started_at=NULL, finished_at=NULL, progress=0, "
        "current_stage=NULL, worker_id=NULL, lease_until=NULL, retry_at=NULL, "
        "attempt_count=0, error_message=NULL, error_code=NULL "
        "WHERE id=? AND status IN ('failed','cancelled')",
        (task_id,),
    )
    conn.commit()
    return cur.rowcount > 0


#: A canonical instant ends in ``Z``; a pre-codec one is 19 characters long and
#: carries the host's local wall clock. Which bound each may be compared against
#: depends on that, so the discriminator lives next to the query.
_CANONICAL_INSTANT_LIKE = "%Z"


def running_rows_without_a_clock() -> int:
    """Running tasks the stale sweep cannot date at all (plan §2.6).

    No lease and no ``started_at`` - there is nothing to compare, so the row stays
    ``running`` and is counted here instead of being guessed about. A *legacy-shaped*
    instant is not in this set: it is aged against a legacy-shaped bound, see
    :func:`reset_stale_running`.
    """
    ensure_schema()
    row = _get_conn().execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE status='running' "
        "AND lease_until IS NULL AND started_at IS NULL"
    ).fetchone()
    return int(row["n"])


def reset_stale_running(timeout_seconds: float = 3600) -> int:
    """Requeue 'running' tasks whose lease expired (plan §2.6).

    Covers worker crashes between claim and finish. Inputs stay on disk, so a
    requeued task can run again.

    Three clocks, each compared only against a bound written in its own convention:

    * ``lease_until`` - canonical, written by this code, the normal path;
    * ``started_at`` in canonical form - a row the 0003 migration already touched;
    * ``started_at`` in the pre-codec form (19 characters, local wall clock) - aged
      against :func:`timestamps.legacy_cutoff`.

    The split is not pedantry: string comparison is exact for fixed-width UTC and
    meaningless across conventions, because a space sorts before the ``T`` of a
    canonical instant. With one bound for both shapes, a task started one second ago
    was "older than any cutoff" and got requeued from under its own worker, while a
    genuinely ancient row survived whenever the local date had rolled past the UTC
    one. Both directions are pinned by tests in
    ``tests/integration/test_queue_reliability.py``.
    """
    ensure_schema()
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE tasks SET status='queued', started_at=NULL, progress=0, current_stage=NULL, "
        "worker_id=NULL, lease_until=NULL, retry_at=NULL "
        "WHERE status='running' AND ("
        "  (lease_until IS NOT NULL AND lease_until < ?) "
        "  OR (lease_until IS NULL AND started_at IS NOT NULL "
        "     AND started_at LIKE ? AND started_at < ?)"
        "  OR (lease_until IS NULL AND started_at IS NOT NULL "
        "     AND started_at NOT LIKE ? AND started_at < ?)"
        ")",
        (timestamps.now(),
         _CANONICAL_INSTANT_LIKE, timestamps.cutoff(timeout_seconds),
         _CANONICAL_INSTANT_LIKE, timestamps.legacy_cutoff(timeout_seconds)),
    )
    conn.commit()
    blocked = running_rows_without_a_clock()
    if blocked:
        log_event(logger, "WARNING",
                  "running tasks have neither a lease nor a start instant",
                  rows=blocked)
    return cur.rowcount


def queued_count() -> int:
    ensure_schema()
    row = _get_conn().execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE status='queued'"
    ).fetchone()
    return int(row["n"])


# --------------------------------------------------------------- task stages
def record_stage(
    task_id: str, order: int, stage_name: str, status: str = "pending",
    stage_version: str | None = None,
) -> None:
    ensure_schema()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO task_stages(task_id, stage_name, stage_order, status, stage_version, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (task_id, stage_name, order, status, stage_version, timestamps.now()),
    )
    conn.commit()


def finish_stage(task_id: str, order: int, status: str, duration_ms: int,
                 message: str | None = None, metrics: dict | None = None) -> None:
    """Close a stage row, recording the metrics the stage reported (plan §5.3)."""
    conn = _get_conn()
    conn.execute(
        "UPDATE task_stages SET status=?, duration_ms=?, message=?, metrics_json=?, "
        "finished_at=? WHERE task_id=? AND stage_order=?",
        (
            status, duration_ms, message,
            json.dumps(metrics) if metrics else None,
            timestamps.now(), task_id, order,
        ),
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


def add_artifact(
    task_id: str, kind: str, path: str, mime_type: str | None = None,
    model_version: str | None = None, uri: str | None = None,
) -> None:
    """Record an artifact, replacing any earlier row of the same ``kind``.

    One task has one row per artifact kind, and that is what the callers already
    assume: the runtime collects a run's artifacts in a **dict keyed by kind**, so
    two stages emitting the same name would collapse to one row anyway. Inserting
    instead of replacing was not "keeping history" either — a retried task re-runs
    into the *same* run directory, so the previous attempt's row kept a `sha256`,
    size and dimensions for bytes that had already been overwritten by the new one.
    A client reading the first `output` row would then hash the file it downloaded
    and get a different digest.

    ``path`` is where the run left the bytes. ``uri`` is the object-store key the
    artifact was **published** under, set only when the configured store is remote:
    with the local store the path already is the location, and mirroring it into
    ``uri`` would hand every reader (``row["uri"] or row["path"]``) a relative key
    to resolve against whatever the process's working directory happens to be.
    """
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
    conn.execute("DELETE FROM artifacts WHERE task_id=? AND kind=?", (task_id, kind))
    conn.execute(
        "INSERT INTO artifacts(task_id, kind, path, uri, mime_type, width, height, "
        "size_bytes, sha256, model_version, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            task_id, kind, path, uri, mime_type, width, height, size, sha256,
            model_version, timestamps.now(),
        ),
    )
    conn.commit()


def set_artifact_uri(task_id: str, kind: str, uri: str) -> bool:
    """Attach a store key to the newest row of ``kind``; False when there is none.

    Publishing happens after the stage finished — the file has to exist before it
    can be uploaded — so the key is written here rather than at insert time.
    """
    ensure_schema()
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE artifacts SET uri=? WHERE id=(SELECT id FROM artifacts "
        "WHERE task_id=? AND kind=? ORDER BY id DESC LIMIT 1)",
        (uri, task_id, kind),
    )
    conn.commit()
    return cur.rowcount > 0


def expired_published_artifacts(cutoff: str, limit: int = 200) -> list[dict]:
    """Rows whose store object passed the retention cutoff (oldest first).

    Bounded by ``limit`` because the sweeper runs on a worker's idle loop: a
    database with years of history should be reclaimed over several passes rather
    than inside one long transaction.
    """
    ensure_schema()
    rows = _get_conn().execute(
        "SELECT id, task_id, uri FROM artifacts "
        "WHERE uri IS NOT NULL AND created_at < ? ORDER BY id LIMIT ?",
        (cutoff, int(limit)),
    ).fetchall()
    return [dict(r) for r in rows]


def clear_artifact_uri(row_id: int) -> None:
    """Forget a store key once its object is gone, keeping the row as history."""
    ensure_schema()
    conn = _get_conn()
    conn.execute("UPDATE artifacts SET uri=NULL WHERE id=?", (int(row_id),))
    conn.commit()


def get_artifacts(task_id: str):
    ensure_schema()
    rows = _get_conn().execute(
        "SELECT * FROM artifacts WHERE task_id=? ORDER BY id", (task_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def artifact_exists(task_id: str, kind: str) -> bool:
    """True when an artifact of ``kind`` is already recorded for the task.

    Used to keep registration idempotent: the input image is registered at
    enqueue time, and the runtime must not insert a second row for it.
    """
    ensure_schema()
    row = _get_conn().execute(
        "SELECT 1 FROM artifacts WHERE task_id=? AND kind=? LIMIT 1", (task_id, kind)
    ).fetchone()
    return row is not None


# ------------------------------------------------------------------- metrics
def split_metric(value) -> tuple[float | None, str | None]:
    """Split a metric value into its numeric and textual halves (plan §2.7).

    A measurement that is a number — or a string that means one, as the V1
    history rows store them — becomes a REAL, so SQL can average and rank it.
    Anything else, including the infinities a REAL cannot hold and the JSON
    serializers cannot emit, keeps its text.
    """
    if value is None:
        return None, None
    if isinstance(value, bool):
        return (1.0 if value else 0.0), None
    if isinstance(value, (int, float)):
        number = float(value)
        return (number, None) if math.isfinite(number) else (None, str(value))
    text = str(value).strip()
    if not text:
        # An empty value is a missing measurement, not a measurement of "".
        return None, None
    try:
        number = float(text)
    except ValueError:
        return None, text
    return (number, None) if math.isfinite(number) else (None, text)


def add_metric(task_id: str, name: str, value, reference_type: str = "input_output_difference") -> None:
    ensure_schema()
    number, text = split_metric(value)
    conn = _get_conn()
    conn.execute(
        "INSERT INTO metrics(task_id, metric_name, metric_value, metric_text, "
        "reference_type, created_at) VALUES (?,?,?,?,?,?)",
        (task_id, name, number, text, reference_type, timestamps.now()),
    )
    conn.commit()


def read_metrics(task_id: str) -> dict:
    """One task's metrics, as the API returns them.

    The production answer is the ``metrics`` subquery of :func:`get_task` /
    :func:`list_tasks`; this reads the same rows in Python for the places that
    want one task's metrics without a task row (admin views, tests, migration
    checks). Both collapse the ``metric_value``/``metric_text`` pair through
    :meth:`Dialect.json_metric_value`, because a SQL ``COALESCE(double
    precision, text)`` is a ``DatatypeMismatch`` on PostgreSQL and a cast to text
    would turn a measured 23.5 into the string "23.5" for every API consumer.

    ``tests/unit/test_metric_readers_agree.py`` asserts this and the aggregate
    return the same mapping on both engines, so the pair cannot drift apart.
    """
    ensure_schema()
    rows = _get_conn().execute(
        "SELECT metric_name, metric_value, metric_text FROM metrics WHERE task_id=?",
        (task_id,),
    ).fetchall()
    return {
        r["metric_name"]: r["metric_value"] if r["metric_value"] is not None
        else r["metric_text"]
        for r in rows
    }


def _age_or_none(value) -> float | None:
    """Age in seconds, or None when the stored instant cannot be dated.

    `timestamps.age_seconds` raises on a value it cannot parse — the right answer for
    its own callers, and the wrong one here: a pre-codec row has no zone, and a
    liveness probe that guesses "0 s" or "ancient" would call a wedged worker alive or
    an idle deployment dead.
    """
    if not value:
        return None
    try:
        return round(timestamps.age_seconds(value), 1)
    except (TypeError, ValueError):
        return None


def worker_heartbeats() -> list:
    """One entry per worker id currently holding a running task, with its beat.

    This is the only liveness signal a *separate* worker process leaves in the
    database: it heartbeats every task it holds. A worker that is alive but idle
    writes nothing, which is why the readiness probe answers "unknown" when the
    queue is empty instead of "dead".
    """
    ensure_schema()
    rows = _get_conn().execute(
        "SELECT worker_id, COUNT(*) AS tasks, MAX(last_heartbeat) AS beat "
        "FROM tasks WHERE status='running' AND worker_id IS NOT NULL "
        "GROUP BY worker_id ORDER BY worker_id"
    ).fetchall()
    out = []
    for row in rows:
        out.append({"worker_id": row["worker_id"], "tasks": int(row["tasks"]),
                    "heartbeat_age_seconds": _age_or_none(row["beat"])})
    return out


def oldest_queued_age_seconds() -> float | None:
    """How long the longest-waiting queued task has been waiting; None when empty."""
    ensure_schema()
    row = _get_conn().execute(
        "SELECT MIN(created_at) AS oldest FROM tasks WHERE status='queued'"
    ).fetchone()
    return _age_or_none(None if row is None else row["oldest"])


def max_metric(name: str) -> float | None:
    """The largest numeric value ever recorded for one task metric.

    Deliberately not windowed by the caller's ``days``: a peak a worker reached last
    month is still the capacity figure an operator plans against, and the row it came
    from is never re-measured. Only the typed numeric column participates — a metric
    stored as text (``"inf"``) has no ordering to take a maximum of.
    """
    ensure_schema()
    row = _get_conn().execute(
        "SELECT MAX(metric_value) AS peak FROM metrics WHERE metric_name=?", (name,)
    ).fetchone()
    return None if row is None or row["peak"] is None else float(row["peak"])


def set_evaluation(task_id: str, evaluation_text: str | None) -> None:
    """Attach the report text to an already-completed task.

    Used by deferred evaluation (plan §3.16): the task is marked ``completed``
    and the result is downloadable before the metrics exist; this fills the text
    in afterwards. Metric *rows* are written by the evaluator itself, so this
    deliberately does not touch the ``metrics`` table (that would duplicate).
    """
    if evaluation_text is None:
        return
    ensure_schema()
    conn = _get_conn()
    conn.execute(
        "UPDATE tasks SET evaluation_text=? WHERE id=?", (evaluation_text, task_id)
    )
    conn.commit()


# ------------------------------------------------------------- system_events
def add_event(
    event_type: str,
    level: str = "info",
    task_id: str | None = None,
    user_id: str | None = None,
    message: str | None = None,
    data: dict | None = None,
) -> None:
    """Append a system event (plan §14; ``data`` feeds the SSE stream)."""
    ensure_schema()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO system_events(level, event_type, task_id, user_id, message, data_json, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            level, event_type, task_id, user_id, message,
            json.dumps(data) if data else None, timestamps.now(),
        ),
    )
    conn.commit()


def list_task_events(task_id: str, after_id: int = 0, limit: int = 200) -> list:
    """Events for one task, ordered oldest → newest (SSE replay + tailing)."""
    ensure_schema()
    rows = _get_conn().execute(
        "SELECT * FROM system_events WHERE task_id=? AND id>? ORDER BY id LIMIT ?",
        (task_id, int(after_id), int(limit)),
    ).fetchall()
    return [dict(r) for r in rows]


def list_events(limit: int = 100, level: str | None = None) -> list:
    """Deployment-wide events, newest first, optionally filtered by level.

    Distinct from :func:`list_task_events`, which is the per-task replay the SSE
    route tails. This one answers "what has this deployment been doing lately",
    which is what the ops endpoints and the operator's own triage need — the
    events that carry no ``task_id`` (``task.rejected`` on the saturated-queue
    path, worker lifecycle) are invisible to a per-task read.
    """
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
def _ceil_rank(count: str, frac: float) -> str:
    """SQL for the nearest-rank position of fraction ``frac`` over ``count`` rows.

    Nearest-rank (``ceil(p * n)``, clamped to the row range) rather than
    PostgreSQL's interpolating ``percentile_cont``, because the interpolation has
    no SQLite counterpart and the two engines would then report different numbers
    for identical rows. Written with ``CASE`` because ``ceil``/``max``/``min`` are
    not spelled the same everywhere — and SQLite's two-argument ``max`` is an
    aggregate on PostgreSQL.
    """
    raw = f"({count} * {frac})"
    floor = f"CAST({raw} AS INTEGER)"
    ceil = f"({floor} + CASE WHEN {raw} > {floor} THEN 1 ELSE 0 END)"
    return (
        f"CASE WHEN {ceil} < 1 THEN 1 WHEN {ceil} > {count} "
        f"THEN {count} ELSE {ceil} END"
    )


#: Per-stage run count, mean and P50/P95 in one portable statement.
_STAGE_STATS_SQL = """
WITH timed AS (
    SELECT stage_name, duration_ms FROM task_stages
    WHERE duration_ms IS NOT NULL {window}
),
agg AS (
    SELECT stage_name, COUNT(*) AS runs, AVG(duration_ms) AS avg_ms
    FROM timed GROUP BY stage_name
),
ranked AS (
    SELECT stage_name, duration_ms,
           ROW_NUMBER() OVER (PARTITION BY stage_name ORDER BY duration_ms) AS rn
    FROM timed
)
SELECT a.stage_name AS stage_name, a.runs AS runs, a.avg_ms AS avg_ms,
       (SELECT duration_ms FROM ranked r
         WHERE r.stage_name = a.stage_name
           AND r.rn = """ + _ceil_rank("a.runs", 0.5) + """) AS p50_ms,
       (SELECT duration_ms FROM ranked r
         WHERE r.stage_name = a.stage_name
           AND r.rn = """ + _ceil_rank("a.runs", 0.95) + """) AS p95_ms
FROM agg a ORDER BY a.stage_name
"""


def task_stats(window_days: int | None = None) -> dict:
    """Task totals, per-type and metric aggregates, per-stage P50/P95 (plan section 30).

    The single source both ``GET /api/v1/stats`` and the admin panel's statistics
    box read, so the two cannot disagree about how many tasks ran.
    """
    ensure_schema()
    conn = _get_conn()
    where = ""
    params: list = []
    if window_days:
        where = "WHERE created_at >= ?"
        params.append(timestamps.cutoff(window_days * 86400))
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

    # Who and what, counted in SQL for the same window. The admin panel used to build
    # these two from `list_history()` in Python, which could only ever describe the
    # rows it had fetched (capped at HISTORY_MAX) and silently presented a partial
    # count as the total.
    summary["users"] = int(
        conn.execute(
            f"SELECT COUNT(DISTINCT user_id) AS users FROM tasks {where}", params
        ).fetchone()["users"]
    )
    by_type = {
        row["task_type"]: int(row["n"])
        for row in conn.execute(
            f"SELECT task_type, COUNT(*) AS n FROM tasks {where} "
            "GROUP BY task_type ORDER BY n DESC, task_type",
            params,
        )
    }

    # Average of each measured quality metric over the window's tasks. `AVG()` skips
    # NULL, and a metric stored as text (PSNR "+inf" after a bit-identical pair, or
    # any label a metric decides to report) has a NULL number half, so it contributes
    # no row rather than skewing the mean with a guess.
    metric_window = "WHERE t.created_at >= ?" if window_days else ""
    metric_params: list = [timestamps.cutoff(window_days * 86400)] if window_days else []
    metric_averages = {
        row["metric_name"]: round(float(row["avg_value"]), 4)
        for row in conn.execute(
            "SELECT m.metric_name AS metric_name, AVG(m.metric_value) AS avg_value "
            "FROM metrics m JOIN tasks t ON t.id = m.task_id "
            f"{metric_window} GROUP BY m.metric_name ORDER BY m.metric_name",
            metric_params,
        )
        if row["avg_value"] is not None
    }

    # One statement, not one row per stage run (plan §2.7): the old version
    # selected every duration in the window and sorted them in Python, which
    # fetches the whole table to produce four numbers per stage.
    window = "AND created_at >= ?" if window_days else ""
    stage_params: list = [timestamps.cutoff(window_days * 86400)] if window_days else []
    stages = {}
    for row in conn.execute(_STAGE_STATS_SQL.format(window=window), stage_params):
        stages[row["stage_name"]] = {
            "runs": int(row["runs"]),
            # PostgreSQL answers AVG() of integers with a Decimal.
            "avg_ms": int(round(row["avg_ms"])),
            "p50_ms": None if row["p50_ms"] is None else int(row["p50_ms"]),
            "p95_ms": None if row["p95_ms"] is None else int(row["p95_ms"]),
        }
    return {
        "window_days": window_days,
        "tasks": summary,
        "by_type": by_type,
        "metric_averages": metric_averages,
        "stages": stages,
    }


def _nearest_rank(ordered: list[float], fraction: float) -> float | None:
    """The same nearest-rank definition `task_stats` uses in SQL.

    `None` for an empty sample, because that is what the shared implementation
    returns — the wrapper names the SQL-side spelling of one statistic, it does not
    quietly change its behaviour. Callers guard on the sample being non-empty.
    """
    from fiximg.infrastructure.observability.metrics import nearest_rank

    return nearest_rank(ordered, fraction)


def deployment_metrics(window_days: int | None = None) -> dict:
    """Deployment-wide series for the Prometheus exposition (plan §2.10).

    The in-process registry in :mod:`fiximg.infrastructure.observability.metrics`
    only knows what *this* process did. On the documented split topology —
    ``api`` plus a separate ``worker`` — that is two of the eleven declared names:
    ``task_submit_total`` and ``queue_depth``, both recorded on the submission
    path. The other nine are recorded by the worker, in a process the API cannot
    read, so ``/api/v1/stats/metrics`` exported nothing for stage durations, model
    load or inference time, VRAM, utilisation, retries or queue wait — while
    ``docs/deployment.md`` names it the Prometheus scrape target and lists
    ``gpu_utilization_percent{stage,device}`` among its series.

    Everything here is derived from the rows the worker already writes, so no new
    hot-path write is added. Percentiles come from :func:`task_stats`, the same
    function ``/api/v1/stats`` and the admin panel read: a second implementation
    would be a third answer to "how long is a stage".

    Two declared names stay process-local and are not in this mapping:
    ``model_inference_seconds`` and ``artifact_io_seconds``. Neither is persisted
    (the second is measured around publish/recover, not around a stage), so a
    deployment-wide figure would have to be invented. They are reported by the
    worker that measured them, and :func:`render_prometheus` says so in the output
    rather than letting a reader assume they are absent.
    """
    stats = task_stats(window_days)
    conn = _get_conn()

    def seconds(ms):
        return None if ms is None else round(int(ms) / 1000.0, 4)

    series: dict = {}

    totals = stats["tasks"]
    series["task_duration_seconds"] = {
        "runs": totals["total"],
        "avg": seconds(totals["avg_ms"]),
    }

    series["stage_duration_seconds"] = {
        stage: {
            "runs": s["runs"],
            "avg": seconds(s["avg_ms"]),
            "p50": seconds(s["p50_ms"]),
            "p95": seconds(s["p95_ms"]),
        }
        for stage, s in stats["stages"].items()
    }

    # Queue wait is not a column: it is the gap the worker closed. Read from the
    # canonical instants, the same pair the worker's own `_queue_wait_seconds`
    # uses, so the two cannot disagree about one task.
    waits: list[float] = []
    where = "WHERE started_at IS NOT NULL"
    params: list = []
    if window_days:
        where += " AND created_at >= ?"
        params.append(timestamps.cutoff(window_days * 86400))
    for row in conn.execute(f"SELECT created_at, started_at FROM tasks {where}", params):
        started = timestamps.parse(row["started_at"])
        created = timestamps.parse(row["created_at"])
        if started is not None and created is not None:
            waits.append(max(0.0, (started - created).total_seconds()))
    waits.sort()
    # Resolved through locals rather than inline conditionals: `nearest_rank` is
    # `float | None`, and an empty window has to report "no samples" rather than a
    # percentile of nothing. Reading it as two steps also keeps mypy honest about
    # the None it is entitled to return.
    p50 = _nearest_rank(waits, 0.5)
    p95 = _nearest_rank(waits, 0.95)
    series["queue_wait_seconds"] = {
        "runs": len(waits),
        "avg": round(sum(waits) / len(waits), 4) if waits else None,
        "p50": None if p50 is None else round(p50, 4),
        "p95": None if p95 is None else round(p95, 4),
    }

    # Retries are recoverable from the rows: a task that took N attempts was
    # re-claimed N-1 times, whichever worker process did it.
    retried = conn.execute(
        "SELECT COALESCE(SUM(CASE WHEN attempt_count > 1 THEN attempt_count - 1 "
        "ELSE 0 END), 0) AS retries FROM tasks"
    ).fetchone()["retries"]
    series["worker_retry_total"] = {"value": int(retried or 0)}

    # Per-stage VRAM and utilisation, as each stage recorded them. `gpu_peak_mb`
    # is megabytes, so the series is named for what it holds rather than quietly
    # changing the unit underneath the declared `gpu_memory_bytes` name.
    stage_gpu: dict = {}
    stage_util: dict = {}
    for row in conn.execute(
        "SELECT stage_name, metrics_json FROM task_stages WHERE metrics_json IS NOT NULL"
    ):
        blob = decode_json_column(row["metrics_json"], {})
        if not isinstance(blob, dict):
            continue
        name = row["stage_name"]
        peak = blob.get("gpu_peak_mb")
        if peak is not None:
            entry = stage_gpu.setdefault(
                name, {"device": blob.get("gpu_device"), "peaks": []}
            )
            entry["peaks"].append(float(peak) * 1024 * 1024)
        if blob.get("gpu_util_pct") is not None:
            entry = stage_util.setdefault(
                name, {"device": blob.get("gpu_device"), "samples": []}
            )
            entry["samples"].append(float(blob["gpu_util_pct"]))
    series["gpu_memory_bytes"] = {
        name: {"max": max(e["peaks"]), "avg": sum(e["peaks"]) / len(e["peaks"]),
               "device": e["device"]}
        for name, e in stage_gpu.items() if e["peaks"]
    }
    series["gpu_utilization_percent"] = {
        name: {"avg": sum(e["samples"]) / len(e["samples"]),
               "max": max(e["samples"]), "samples": len(e["samples"]),
               "device": e["device"]}
        for name, e in stage_util.items() if e["samples"]
    }

    # Model load time is persisted per (name, version) by whichever process loaded
    # it, so unlike inference time it *is* readable deployment-wide.
    loads: dict = {}
    try:
        from fiximg.infrastructure.db.repositories import model_repository

        for row in model_repository.list_versions():
            if row.get("load_ms") is None:
                continue
            loads[f"{row['name']}@{row.get('version')}"] = int(row["load_ms"]) / 1000.0
    except Exception:  # noqa: BLE001 - observability must not fail on a missing table
        loads = {}
    if loads:
        series["model_load_seconds"] = {"per_model": loads}

    return series
