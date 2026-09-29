"""The PostgreSQL data path, exercised without a server (plan §4.3 Step 2).

`test_sql_dialect.py` proves the SQL *text* compiles; `test_db_connection.py`
proves the wrapper adapts what it is handed. This file closes the gap between
them: it runs the real startup and queue primitives against a fake DB-API driver
speaking the PostgreSQL dialect, so a statement that only SQLite understands is
caught here rather than on a deployment's first boot.

The fake answers exactly enough queries for the code to reach the next one; the
assertions are about **what was sent**, which is the part that differs per engine.
"""
import pytest

from fiximg.infrastructure.db import connection as connection_module
from fiximg.infrastructure.db.connection import Connection
from fiximg.infrastructure.db.dialect import POSTGRESQL, dialect_for

PG_URL = "postgresql://user:pw@localhost:5432/fiximg"


class Row:
    """A driver row that reads the way psycopg's ``dict_row`` actually reads.

    ``row["col"]`` and ``dict(row)`` are what the repositories use. Positional
    access is deliberately *not* supported: on a real server an unaliased
    ``SELECT COUNT(*)`` has no key, so ``row[0]`` raises ``KeyError`` there. The
    fake used to answer it anyway, and the two legacy ``COUNT(*)`` guards that
    leaned on that passed here while crashing on the first PostgreSQL boot
    (``engine._migrate_users``). A stand-in must not be more forgiving than the
    driver it replaces.
    """

    def __init__(self, values: dict):
        self._values = dict(values)

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise KeyError(
                f"dict_row has no positional access; asked for {key!r}. "
                "Alias the column and read it by name."
            )
        return self._values[key]

    def keys(self):
        return self._values.keys()

    def __iter__(self):
        return iter(self._values)

    def __repr__(self):
        return f"Row({self._values!r})"


#: One seven-member sequence per column; see PEP 249. The production wrapper
#: reads `description` to decide whether a statement has a result set, so a
#: cursor double that omits it is a less capable cursor, not a smaller fake.
_SELECT_DESCRIPTION = [("column", None, None, None, None, None, None)]


class FakeCursor:
    def __init__(self, server):
        self._server = server
        self._rows = []
        self.rowcount = 0
        self.description = None

    def execute(self, sql, params=()):
        self._rows = self._server.answer(sql, params)
        self.rowcount = len(self._rows)
        self.description = _SELECT_DESCRIPTION if self._rows else None
        return self

    def executemany(self, sql, seq):
        for values in seq:
            self.execute(sql, values)
        return self

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def __iter__(self):
        # Real DB-API cursors are iterable, and the introspection helpers use
        # ``for row in conn.execute(...)`` rather than fetchall().
        return iter(self._rows)

    def close(self):
        pass


class FakeServer:
    """A stand-in PostgreSQL session: records statements, answers by pattern."""

    def __init__(self, columns=()):
        self.log = []
        self.columns = list(columns)
        self.commits = 0
        self.rollbacks = 0

    def answer(self, sql, params):
        self.log.append((sql, tuple(params)))
        upper = sql.strip().upper()
        if "INFORMATION_SCHEMA.COLUMNS" in upper:
            return [Row({"name": name}) for name in self.columns]
        if upper.startswith("SELECT COUNT(*)"):
            # Name the column the way a real driver would: after its alias, and
            # `count` when there is none. Answering with a key the statement never
            # asked for (or with positional access) is what let the legacy
            # ``fetchone()[0]`` guards look fine here and fail on a real server.
            alias = upper.rfind(" AS ")
            name = sql[alias + 4:].split()[0] if alias != -1 else "count"
            return [Row({name: 0})]
        if upper.startswith("SELECT * FROM TASKS"):
            return []
        return []

    def cursor(self):
        cursor = FakeCursor(self)
        return cursor

    def execute(self, sql, params=()):
        return self.cursor().execute(sql, params)

    def executemany(self, sql, seq):
        return self.cursor().executemany(sql, seq)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        pass


@pytest.fixture()
def pg_session(monkeypatch, tmp_path):
    """Boot the application's data layer against PostgreSQL (fake)."""
    import fiximg.config as config_mod
    import fiximg.infrastructure.db.engine as engine
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    # Every column the forward migration may look for, so no ALTER is emitted and
    # the statement log contains only what the caller asked for.
    server = FakeServer(columns=[
        "id", "status", "task_type", "user_id", "input_path", "result_path",
        "options_json", "progress", "current_stage", "created_at", "started_at",
        "finished_at", "duration_ms", "error_message", "attempt_count",
        "max_attempts", "worker_id", "lease_until", "last_heartbeat", "retry_at",
        "priority", "idempotency_key", "required_capabilities", "error_code",
        "metrics_json", "stage_version", "model_version", "evaluation_text",
    ])
    conn = Connection(server, dialect_for(POSTGRESQL), driver=POSTGRESQL)

    monkeypatch.setenv("FIXIMG_DATABASE_URL", PG_URL)
    monkeypatch.setattr(engine, "DB_PATH", str(tmp_path / "unused.db"))
    monkeypatch.setattr(engine, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config_mod.settings, "db_path", str(tmp_path / "unused.db"))
    monkeypatch.setattr(connection_module, "connect", lambda target, kind="sqlite", dialect=None: conn)
    monkeypatch.setattr(engine, "_conn", None)
    monkeypatch.setattr(task_repo, "_DDL_DONE", False)
    engine.close_connections()

    yield server
    engine.close_connections()


def statements(server):
    return [sql for sql, _ in server.log]


# ================================ startup =======================================
def test_booting_on_postgres_sends_no_sqlite_only_statement(pg_session):
    import fiximg.infrastructure.db.engine as engine

    engine.init_db()
    text = "\n".join(statements(pg_session))
    assert "PRAGMA" not in text
    assert "INSERT OR IGNORE" not in text
    assert "BEGIN IMMEDIATE" not in text


def test_the_schema_reaches_postgres_translated(pg_session):
    """The shipped DDL is SQLite's; the wrapper must translate before the wire."""
    import fiximg.infrastructure.db.engine as engine

    engine.init_db()
    ddl = [sql for sql in statements(pg_session) if sql.strip().upper().startswith("CREATE TABLE")]
    assert ddl, "the users/history tables must be created"
    joined = "\n".join(ddl)
    assert "AUTOINCREMENT" not in joined.upper()
    assert " REAL" not in joined


def test_the_task_schema_reaches_postgres_translated(pg_session):
    """The counter tables are the ones with ``INTEGER PRIMARY KEY AUTOINCREMENT``."""
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    task_repo.ensure_schema()
    ddl = "\n".join(statements(pg_session))
    assert "BIGSERIAL PRIMARY KEY" in ddl, "the auto-increment counters must be translated"
    assert "AUTOINCREMENT" not in ddl.upper()
    assert "CREATE UNIQUE INDEX" in ddl, "the idempotency index is a partial unique index"


def test_column_introspection_uses_information_schema(pg_session):
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    task_repo.ensure_schema()
    introspection = [
        sql for sql in statements(pg_session)
        if "information_schema" in sql.lower() or "PRAGMA" in sql.upper()
    ]
    assert introspection, "the forward migration must inspect the columns"
    assert all("information_schema" in sql.lower() for sql in introspection)


def test_a_legacy_user_import_uses_on_conflict(pg_session, monkeypatch, tmp_path):
    """``INSERT OR IGNORE`` is SQLite spelling; the portable form is ON CONFLICT."""
    import fiximg.infrastructure.db.engine as engine

    users_yaml = tmp_path / "users.yaml"
    users_yaml.write_text("users:\n  alice:\n    password: hash\n    role: admin\n", encoding="utf-8")
    monkeypatch.setattr(engine, "LEGACY_USERS_YAML", str(users_yaml))

    engine.init_db()
    inserts = [sql for sql in statements(pg_session) if "into users" in sql.lower()]
    assert inserts, "the legacy user must be imported"
    assert "ON CONFLICT (username) DO NOTHING" in inserts[0]
    assert "INSERT OR IGNORE" not in inserts[0]


# ================================== queue =======================================
def test_creating_a_task_uses_dollar_free_placeholders(pg_session):
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    task_repo.create_task("t-1", "restore", "alice")
    insert = next(sql for sql in statements(pg_session) if "INSERT INTO tasks" in sql)
    assert "%s" in insert and "?" not in insert


def test_claiming_locks_the_row_instead_of_the_database(pg_session):
    """PostgreSQL gets ``FOR UPDATE SKIP LOCKED``; ``BEGIN IMMEDIATE`` is SQLite's tool."""
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    task_repo.claim_next_task("worker-1")
    text = "\n".join(statements(pg_session))
    assert "FOR UPDATE SKIP LOCKED" in text
    assert "BEGIN IMMEDIATE" not in text


def test_an_empty_queue_claim_sends_no_sqlite_transaction_control(pg_session):
    """On PostgreSQL the driver owns the transaction; only the row lock is ours."""
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    assert task_repo.claim_next_task("worker-1") is None
    sent = [sql.strip().upper() for sql, _ in pg_session.log]
    assert not any(sql.startswith("BEGIN") for sql in sent)
