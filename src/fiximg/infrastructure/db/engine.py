"""The application's connection owner (replaces YAML/JSONL file storage).

- users: user table (passwords remain pbkdf2 hashes, compatible with the old users.yaml)
- history: processing history table (compatible with the old processing_history.json)
- On first startup, legacy rows are imported from config/users.yaml and
  admin_data/processing_history.json

V2 note: this module remains the low-level connection owner; the repositories
layer (fiximg.infrastructure.db.repositories) is the only sanctioned access path
for new code. The V2/V3 tasks/task_stages/artifacts/metrics tables live in
``repositories/task_repository.py`` using the same database.

Database URL (plan §4.3 Step 2)
-------------------------------
``FIXIMG_DATABASE_URL`` is parsed here so a misconfiguration is reported at
startup instead of being silently ignored::

    ""                              -> the default admin_data/fixoldimg.db
    sqlite:///relative/path.db      -> relative to the project root
    sqlite:////absolute/path.db     -> absolute
    sqlite+pysqlite:///path.db      -> accepted (SQLAlchemy spelling)
    postgresql://user@host/db       -> opened through psycopg (fiximg[postgres])

Only the environment variable relocates :data:`DB_PATH`; a profile's YAML value
does not, so a ``DB_PATH`` set by a test or an embedder stays authoritative. An
unknown scheme raises :class:`UnsupportedDatabaseError`: a silent fallback would
look like a successful switch while writing to the wrong database. The SQL that
differs between the two supported engines lives in
:mod:`fiximg.infrastructure.db.dialect`, the connection that runs it in
:mod:`fiximg.infrastructure.db.connection`.
"""
import json
import os
import threading
import time
from urllib.parse import urlparse

from fiximg.infrastructure.db import timestamps

from fiximg.infrastructure.db.connection import integrity_errors
from fiximg.paths import CONFIG_DIR, DATA_DIR, PROJECT_ROOT

BASE_DIR = PROJECT_ROOT
ADMIN_DATA_DIR = DATA_DIR
DB_PATH = os.path.join(ADMIN_DATA_DIR, "fixoldimg.db")
LEGACY_USERS_YAML = os.path.join(CONFIG_DIR, "users.yaml")
LEGACY_HISTORY_FILE = os.path.join(ADMIN_DATA_DIR, "processing_history.json")


def _settings():
    """The one place this module reaches configuration.

    Read per call rather than captured at import: `config.Settings` resolves the
    environment *and* the YAML profile, and a module-level snapshot let the engine
    and `settings` disagree — which is how the archive paths ended up declared
    twice, once derived from `ADMIN_DATA_DIR` and once as
    `settings.archive_*`, with only the first pair ever read.
    """
    from fiximg.config import settings

    return settings


def archive_input_dir() -> str:
    return _settings().archive_input_dir


def archive_output_dir() -> str:
    return _settings().archive_output_dir

#: URL schemes this engine can actually serve today.
SQLITE_SCHEMES = ("sqlite", "sqlite3", "sqlite+pysqlite")
POSTGRES_SCHEMES = ("postgres", "postgresql", "postgresql+psycopg", "postgresql+psycopg2")


class UnsupportedDatabaseError(RuntimeError):
    """Raised when ``FIXIMG_DATABASE_URL`` names a database V3 cannot serve."""


def resolve_database(url: str | None = None) -> tuple[str, str]:
    """Resolve the configured URL to ``(kind, target)`` (plan §4.3 Step 2).

    ``kind`` is ``"sqlite"`` (``target`` is a file path) or ``"postgresql"``
    (``target`` is the connection URL). An unknown scheme raises
    :class:`UnsupportedDatabaseError` rather than silently falling back to
    SQLite, which would look like a successful switch while writing to the wrong
    database.
    """
    raw = (url if url is not None else _configured_database_url()) or ""
    raw = raw.strip()
    if not raw:
        return "sqlite", DB_PATH

    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()

    if not scheme:
        # A bare path is accepted for convenience.
        return "sqlite", (raw if os.path.isabs(raw) else os.path.join(PROJECT_ROOT, raw))

    if scheme in POSTGRES_SCHEMES:
        return "postgresql", raw

    if scheme not in SQLITE_SCHEMES:
        raise UnsupportedDatabaseError(
            f"FIXIMG_DATABASE_URL uses the '{scheme}' scheme, which this build "
            "cannot serve. Supported: SQLite and PostgreSQL. Leave the variable "
            "empty to use the default SQLite file."
        )

    # sqlite:///relative.db, sqlite:////abs/path.db, sqlite:///:memory:
    path = parsed.path or ""
    if parsed.netloc and parsed.netloc not in ("", "localhost"):
        # sqlite://host/path is not a thing; treat the netloc as part of a path.
        path = f"/{parsed.netloc}{path}"
    if path in ("", "/"):
        return "sqlite", DB_PATH
    if path == "/:memory:" or path.startswith(":memory:"):
        return "sqlite", ":memory:"
    path = path.lstrip("/") if not path.startswith("//") else path[1:]
    if os.path.isabs(path):
        return "sqlite", path
    return "sqlite", os.path.join(PROJECT_ROOT, path)


def resolve_database_path(url: str | None = None) -> str:
    """The SQLite file path for ``url``.

    Raises for a non-SQLite URL: this helper exists for callers that specifically
    need a file (Alembic's offline mode, backup scripts). Use
    :func:`resolve_database` when the engine may be PostgreSQL.
    """
    kind, target = resolve_database(url)
    if kind != "sqlite":
        raise UnsupportedDatabaseError(
            f"FIXIMG_DATABASE_URL points at {kind}, which has no filesystem path. "
            "Use resolve_database() instead."
        )
    return target

HISTORY_MAX = int(os.environ.get("FIXIMG_HISTORY_MAX", "2000"))

_write_lock = threading.Lock()
_conn = None
#: The ``(kind, target)`` ``_conn`` and the thread-local handles were opened for.
#: ``get_conn`` reopens when the configured database stops matching it.
_conn_target: tuple[str, str] | None = None
#: Connections opened for a non-SQLite driver, one per thread (see get_conn).
_thread_connections = threading.local()
#: Every thread-local connection still open, so they can be closed together.
_extra_connections: list = []
_conn_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username   TEXT PRIMARY KEY,
    password   TEXT NOT NULL,
    role       TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'active',
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS history (
    id          TEXT PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    "user"      TEXT NOT NULL,
    type        TEXT NOT NULL,
    input_path  TEXT NOT NULL,
    output_path TEXT NOT NULL,
    psnr        TEXT,
    ssim        TEXT,
    mae         TEXT
);
CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history(timestamp);
-- ModelVersion as plan §6 spells it: one row per (name, version) that some process
-- has actually tried to load. The registry answers "what is resident *here*", which
-- in a split topology is half the question — an API node that never loads a model
-- used to report no versions at all while its workers served them. `status` and
-- `loaded_at` are therefore observations recorded at load time, not the claims a
-- manifest makes about a version nobody ran.
CREATE TABLE IF NOT EXISTS model_versions (
    name          TEXT NOT NULL,
    version       TEXT NOT NULL,
    framework     TEXT,
    weight_uri    TEXT,
    sha256        TEXT,
    status        TEXT NOT NULL,
    loaded_at     TEXT,
    load_ms       INTEGER,
    load_count    INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT,
    writer        TEXT,
    metadata_json TEXT,
    PRIMARY KEY (name, version)
);
"""


def get_conn():
    """The connection for this thread and the configured database.

    Returns a :class:`~fiximg.infrastructure.db.connection.Connection`, which
    adapts placeholders and row access for the dialect. For SQLite it is a thin
    wrapper over ``sqlite3`` with the same PRAGMAs the engine always used, and one
    handle is shared on purpose — the inline worker and the request threads are the
    same deployment (plan §3.2), and a pool of SQLite handles would just be several
    writers fighting over one file.

    What that sharing requires is the wrapper's own lock, not ``check_same_thread``:
    the flag only disables the ownership check, while a unit of work here is
    *execute, fetch, commit* — three driver calls another thread can cut in half.
    Without the serialisation Python 3.14 answered with ``SystemError: error return
    without exception set`` under exactly that interleaving. See
    :class:`~fiximg.infrastructure.db.connection.Connection`.

    Other drivers are not so forgiving: psycopg documents a connection as
    single-thread-at-a-time, while the API serves requests from a thread pool and
    the worker adds its own heartbeat thread. Sharing one of those would corrupt
    transactions or raise ``InterfaceError`` under load, so each thread gets its
    own and the server-side pool (if any) sits behind it.

    A cached handle is only reused while it points at the database that is
    configured *now*. That matters because the configuration can move under a
    running process — a test fixture repointing ``DB_PATH``, an embedder
    switching profiles — and a stale handle keeps writing to the previous
    database in perfect silence. Roughly 700 tests once shared one development
    database exactly this way, so the target is part of the cache key here
    instead of something every caller has to remember to reset.
    """
    global _conn, _conn_target
    from fiximg.infrastructure.db.connection import connect

    kind, target = resolve_database()
    wanted = (kind, target)
    if _conn_target is not None and _conn_target != wanted:
        # The database moved. Everything cached points at the old one; the next
        # call reopens against `wanted`.
        close_connections()

    if kind == "sqlite":
        with _conn_lock:
            if _conn is None:
                os.makedirs(ADMIN_DATA_DIR, exist_ok=True)
                _conn = connect(target, kind)
                _conn.database_target = wanted
                _conn_target = wanted
            return _conn

    cached = getattr(_thread_connections, "conn", None)
    if (
        cached is not None
        and not cached.closed
        and getattr(cached, "database_target", None) == wanted
    ):
        return cached
    if cached is not None:
        _forget_thread_connection(cached)

    conn = connect(target, kind)
    conn.database_target = wanted
    _thread_connections.conn = conn
    with _conn_lock:
        _extra_connections.append(conn)
    return conn


def _forget_thread_connection(conn) -> None:
    """Drop a thread-local handle that no longer describes the target database."""
    global _extra_connections
    conn.close()
    with _conn_lock:
        _extra_connections = [c for c in _extra_connections if c is not conn]


def reset_thread_connection_cache() -> None:
    """Forget *this* thread's cached connection without touching the others.

    Fixtures that repoint the database use this instead of assigning
    ``engine._conn = None``: that attribute is only the SQLite half of the cache,
    and clearing it leaves a thread-local handle behind, which then keeps serving
    statements to the previous database file.
    """
    cached = getattr(_thread_connections, "conn", None)
    if cached is not None:
        _forget_thread_connection(cached)
    _thread_connections.conn = None


def close_connections() -> None:
    """Close every connection this process opened through :func:`get_conn`.

    The thread-local ones are closed and unregistered; the calling thread's cache
    is cleared, and the others are left with a closed handle that a fresh
    :func:`get_conn` call replaces. Used by test fixtures and by shutdown.
    """
    global _conn, _conn_target, _extra_connections
    with _conn_lock:
        if _conn is not None:
            _conn.close()
            _conn = None
        _conn_target = None
        for conn in _extra_connections:
            conn.close()
        _extra_connections = []
    if getattr(_thread_connections, "conn", None) is not None:
        _thread_connections.conn = None


def validate_database_url(url: str | None = None) -> str:
    """Raise for an unsupported scheme; return the raw URL otherwise.

    This is the fail-fast half of :func:`apply_database_url` and is safe to call
    from anywhere: it has no side effects. ``init_db`` calls it so a
    misconfigured deployment stops at startup instead of writing to the wrong
    database.
    """
    source = url if url is not None else _configured_database_url()
    resolve_database(source)               # raises UnsupportedDatabaseError
    return source


def _configured_database_url() -> str:
    """The explicitly configured URL — the environment variable, and only it.

    ``DB_PATH`` is what the engine opens; :func:`apply_database_url` moves it when
    a deployment states a URL. Reading ``settings.database_url`` here as well
    would invert that: a profile value (or a default derived from ``db_path``)
    outranked the ``DB_PATH`` an isolation fixture or an embedder sets, so every
    test silently shared the developer database — 176 ``database is locked`` and
    ``UNIQUE constraint failed`` failures at once.
    """
    return os.environ.get("FIXIMG_DATABASE_URL", "")


def apply_database_url(url: str | None = None) -> str:
    """Relocate :data:`DB_PATH` from an **explicitly configured** URL.

    Only the environment variable triggers relocation. The shipped YAML files
    also declare ``database_url``, but treating that as authoritative would
    override a ``DB_PATH`` set directly by a test or an embedder — and the
    resolved default is the same file anyway. The env var is the deployment's
    explicit statement of intent, so that is what moves the path.

    Returns the effective path (or the URL, for PostgreSQL, which has no path).
    Raises :class:`UnsupportedDatabaseError` for an unknown scheme, before
    anything is opened.
    """
    global DB_PATH, ADMIN_DATA_DIR
    source = url if url is not None else os.environ.get("FIXIMG_DATABASE_URL", "")
    if not (source or "").strip():
        return DB_PATH

    kind, target = resolve_database(source)
    if kind == "postgresql":
        # Nothing to relocate: the connection is opened from the URL itself.
        return target
    DB_PATH = target
    if target != ":memory:":
        ADMIN_DATA_DIR = os.path.dirname(target) or ADMIN_DATA_DIR
    return target


def init_db():
    """Create tables and run one-time migrations."""
    with _write_lock:
        # Fail fast on an unsupported database URL before opening anything.
        validate_database_url()
        conn = get_conn()
        conn.executescript(_SCHEMA)
        _remember_core_schema()
        _migrate_users_schema(conn)
        conn.commit()
        _migrate_users(conn)
        _migrate_history(conn)


#: Databases the app-level tables are known to exist in (see ensure_core_schema).
_core_schema_targets: set[tuple[str, str]] = set()
_core_schema_lock = threading.Lock()


def _remember_core_schema() -> None:
    """Mark the current database as holding the app-level tables."""
    with _core_schema_lock:
        _core_schema_targets.add(resolve_database())


def ensure_core_schema() -> None:
    """Create the app-level tables (users, history, model_versions) if they are absent.

    The lazy path for a process that writes one of these tables without booting the
    application. :func:`init_db` also migrates the V1 users and history tables, which
    a model-load audit write has no business triggering, and no write path should
    depend on the boot order having happened to be right: an absent table here would
    otherwise be swallowed by the callers' best-effort handling and the audit trail
    would stay quietly empty.

    Keyed by the resolved database rather than a process-wide flag, so a test (or a
    reload) that repoints `DB_PATH` gets the schema applied again instead of inheriting
    the previous database's "already done".
    """
    wanted = resolve_database()
    with _core_schema_lock:
        if wanted in _core_schema_targets:
            return
    from fiximg.config import settings

    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    conn = get_conn()
    conn.executescript(_SCHEMA)
    conn.commit()
    _remember_core_schema()


def _migrate_users_schema(conn):
    """V2 plan section 14: add status/updated_at to pre-existing users tables."""
    statement, params = conn.dialect.table_columns_sql("users")
    cols = {r["name"] for r in conn.execute(statement, params)}
    if "status" not in cols:
        conn.execute(
            "ALTER TABLE users ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
        )
    if "updated_at" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN updated_at TEXT")
    if "must_change_password" not in cols:
        conn.execute(
            "ALTER TABLE users ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0"
        )


def set_must_change_password(username: str, flag: bool) -> None:
    """Plan section 21: force (or clear) the change-password requirement."""
    with _write_lock:
        get_conn().execute(
            "UPDATE users SET must_change_password=?, updated_at=? WHERE username=?",
            (1 if flag else 0, timestamps.now(), username),
        )
        get_conn().commit()


def get_user(username):
    row = get_conn().execute(
        "SELECT * FROM users WHERE username=?", (username,)
    ).fetchone()
    return dict(row) if row else None


def list_users():
    rows = get_conn().execute(
        "SELECT username, role, created_at FROM users ORDER BY username"
    ).fetchall()
    return [dict(r) for r in rows]


def add_user(username, password_hash, role="user"):
    """Create a user; returns False if the username already exists (avoids a race on concurrent registration)."""
    with _write_lock:
        conn = get_conn()
        try:
            conn.execute(
                "INSERT INTO users(username, password, role, created_at) VALUES (?,?,?,?)",
                (username, password_hash, role, timestamps.now()),
            )
            conn.commit()
        except integrity_errors():
            conn.rollback()
            return False
    return True


def update_user(username, password_hash=None, role=None):
    with _write_lock:
        conn = get_conn()
        if password_hash is not None:
            conn.execute(
                "UPDATE users SET password=?, must_change_password=0, updated_at=? WHERE username=?",
                (password_hash, timestamps.now(), username),
            )
        if role is not None:
            conn.execute(
                "UPDATE users SET role=?, updated_at=? WHERE username=?",
                (role, timestamps.now(), username),
            )
        conn.commit()


def delete_user(username):
    with _write_lock:
        conn = get_conn()
        conn.execute("DELETE FROM users WHERE username=?", (username,))
        conn.commit()


#: The `history` table is the V1 record and has **no writer** in V2/V3 — every
#: result is a `tasks` row. It is kept because `ui/admin_panel.py` still reads it
#: (the V1 admin History tab) and because `cli/migrate_v1.py` imports rows out of
#: it, so both directions matter. The writer that used to live here was called
#: from nowhere in the tree, which meant the tab could only ever show whatever a
#: V1 install had already written; it is gone rather than left as a second way to
#: record a result.


#: The `history` table is the V1 record. V2/V3 write `tasks`, not `history`, so
#: nothing can add a row here: the cap that used to run on every append has no
#: writer left to run for, and `FIXIMG_HISTORY_MAX` / the `history_max` YAML key
#: configured a limit on a table that cannot grow. The cap is gone rather than
#: left as documentation of a mechanism that cannot fire. `HISTORY_MAX` survives
#: below as the default row cap for `list_history` — a read-side limit, which
#: still does something.
def list_history(limit=HISTORY_MAX):
    rows = get_conn().execute(
        "SELECT * FROM history ORDER BY timestamp DESC, id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_history_record(history_id):
    row = get_conn().execute(
        "SELECT * FROM history WHERE id=?", (history_id,)
    ).fetchone()
    return dict(row) if row else None


def clear_history():
    with _write_lock:
        conn = get_conn()
        conn.execute("DELETE FROM history")
        conn.commit()


def history_stats():
    rows = list_history()
    total = len(rows)
    users = len({r["user"] for r in rows})
    type_counts: dict[str, int] = {}
    psnr_vals, ssim_vals = [], []
    for r in rows:
        tp = r.get("type") or "unknown"
        type_counts[tp] = type_counts.get(tp, 0) + 1
        try:
            psnr_raw = r.get("psnr")
            if psnr_raw not in (None, "", "∞", "N/A"):
                psnr_vals.append(float(psnr_raw))
        except (TypeError, ValueError):
            pass
        try:
            ssim_raw = r.get("ssim")
            if ssim_raw not in (None, "", "N/A"):
                ssim_vals.append(float(ssim_raw))
        except (TypeError, ValueError):
            pass
    lines = [f"Total tasks: {total}", f"Users: {users}"]
    for tp, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  · {tp}: {cnt} times")
    if psnr_vals:
        lines.append(f"Average PSNR: {sum(psnr_vals) / len(psnr_vals):.2f}")
    if ssim_vals:
        lines.append(f"Average SSIM: {sum(ssim_vals) / len(ssim_vals):.4f}")
    return "\n".join(lines)


def purge_stale_archives(ttl_seconds=None):
    """Delete archived copies older than the configured TTL.

    The TTL is `settings.archive_ttl`, resolved per call; the parameter exists
    for a caller with its own budget and defaults to the deployment's. Read from
    `settings` rather than a second environment lookup, so the YAML profile and
    the environment cannot disagree about it.
    """
    if ttl_seconds is None:
        ttl_seconds = _settings().archive_ttl
    now = time.time()
    for d in (archive_input_dir(), archive_output_dir()):
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            p = os.path.join(d, name)
            try:
                if now - os.path.getmtime(p) > ttl_seconds:
                    os.remove(p)
            except OSError:
                pass


def _migrate_users(conn):
    # The count is read by name on purpose: psycopg is configured with `dict_row`,
    # so an unaliased `SELECT COUNT(*)` has no key and `row[0]` raises KeyError —
    # `sqlite3.Row` forgives both, which is why this only failed on a real server.
    if conn.execute("SELECT COUNT(*) AS total FROM users").fetchone()["total"] > 0:
        return
    if not os.path.exists(LEGACY_USERS_YAML):
        return
    try:
        import yaml

        with open(LEGACY_USERS_YAML, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        users = data.get("users") or {}
        created = timestamps.now()
        statement = conn.dialect.insert_ignore(
            "INSERT INTO users(username,password,role,created_at) VALUES (?,?,?,?)",
            ("username",),
        )
        for username, info in users.items():
            conn.execute(
                statement,
                (username, info.get("password", ""), info.get("role", "user"), created),
            )
        conn.commit()
        print(f"[Migration] imported {len(users)} users into the database")
    except Exception as exc:  # noqa: BLE001
        print("[Migration] user import failed (skipped):", exc)


def _parse_legacy_history(content):
    """Handle legacy array / JSONL / mixed formats, deduplicating by id."""
    records: list[dict] = []
    text = content.strip()
    try:
        obj, end = json.JSONDecoder().raw_decode(text)
        if isinstance(obj, list):
            records.extend(r for r in obj if isinstance(r, dict))
            tail = text[end:].strip()
        else:
            tail = text
    except json.JSONDecodeError:
        tail = text
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                records.append(rec)
        except json.JSONDecodeError:
            continue
    seen: dict[str, dict] = {}
    for r in records:
        seen[r.get("id") or f"__no_id_{len(seen)}__"] = r
    return list(seen.values())


def _migrate_history(conn):
    if conn.execute("SELECT COUNT(*) AS total FROM history").fetchone()["total"] > 0:
        return
    if not os.path.exists(LEGACY_HISTORY_FILE):
        return
    try:
        with open(LEGACY_HISTORY_FILE, encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return
        records = _parse_legacy_history(content)
        statement = conn.dialect.insert_ignore(
            'INSERT INTO history(id,timestamp,"user",type,input_path,output_path,psnr,ssim,mae) '
            "VALUES (?,?,?,?,?,?,?,?,?)",
            ("id",),
        )
        for r in records:
            conn.execute(
                statement,
                (
                    r.get("id"),
                    r.get("timestamp", ""),
                    r.get("user", ""),
                    r.get("type", ""),
                    r.get("input_path", ""),
                    r.get("output_path", ""),
                    str(r.get("psnr", "")),
                    str(r.get("ssim", "")),
                    str(r.get("mae", "")),
                ),
            )
        conn.commit()
        print(f"[Migration] imported {len(records)} history records into the database")
    except Exception as exc:  # noqa: BLE001
        print("[Migration] history import failed (skipped):", exc)
