"""Tests for the dialect-aware DB-API connection (plan 搂4.3 Step 2).

:mod:`fiximg.infrastructure.db.dialect` is tested by compiling its SQL; this
module tests the *transport* that runs it. Nothing here needs a PostgreSQL
server: a fake DB-API connection records exactly what reaches the wire, which is
what proves placeholder adaptation, DDL translation and statement splitting
actually happen. The SQLite half is proven against real ``sqlite3``.

The reason this file exists at all: before it, ``connection.py`` was imported by
one call site (``engine.get_conn``) and had never been executed with a non-SQLite
driver 鈥?so the entire PostgreSQL path was unverified code.
"""
import sqlite3
import sys

import pytest

from fiximg.infrastructure.db import connection as connection_module
from fiximg.infrastructure.db.connection import (
    Connection,
    ConnectionError,
    connect,
    open_postgres,
    open_sqlite,
)
from fiximg.infrastructure.db.dialect import POSTGRESQL, SQLITE, dialect_for


#: One seven-member sequence per column, which is what PEP 249 describes.
_SELECT_DESCRIPTION = [("column", None, None, None, None, None, None)]


class FakeCursor:
    """Minimal DB-API cursor that appends every statement to a shared log.

    `description` is modelled rather than omitted: it is how a real driver says
    "this statement has a result set", and the production wrapper reads it to
    decide whether to fetch. A cursor without it would hide the difference
    between a SELECT and a DELETE.
    """

    def __init__(self, log, rows=()):
        self.log = log
        self.rows = list(rows)
        self.closed = False
        self.rowcount = len(self.rows)
        # A cursor built with rows has a result set, whatever the statement was:
        # real drivers report `description` for RETURNING and PRAGMA too, so
        # keying it on a SQL prefix would be a fiction the tests then depend on.
        self.description = _SELECT_DESCRIPTION if self.rows else None

    def execute(self, sql, params=()):
        self.log.append((sql, tuple(params)))
        return self

    def executemany(self, sql, seq):
        for values in seq:
            self.log.append((sql, tuple(values)))
        return self

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def close(self):
        self.closed = True


class FakeConnection:
    """A stand-in for ``psycopg.Connection``.

    Deliberately has **no** ``executescript`` 鈥?that is a sqlite3 extension, and
    the point of the wrapper is that callers never depend on it.
    """

    def __init__(self, rows=()):
        self.log = []
        self.cursors = []
        self.rows = list(rows)
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def execute(self, sql, params=()):
        cursor = self.cursor()
        return cursor.execute(sql, params)

    def executemany(self, sql, seq):
        cursor = self.cursor()
        return cursor.executemany(sql, seq)

    def cursor(self):
        cursor = FakeCursor(self.log, self.rows)
        self.cursors.append(cursor)
        return cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


@pytest.fixture()
def pg():
    return dialect_for(POSTGRESQL)


@pytest.fixture()
def fake_pg_connection(pg):
    """A Connection over the fake driver, speaking PostgreSQL."""
    raw = FakeConnection(rows=[{"name": "id"}, {"name": "status"}])
    return Connection(raw, pg, driver=POSTGRESQL), raw


# =============================== what hits the wire ===============================
def test_postgres_placeholders_are_rewritten_on_the_wire(fake_pg_connection):
    conn, raw = fake_pg_connection
    conn.execute("SELECT * FROM tasks WHERE id=? AND status=?", ("t-1", "queued"))
    assert raw.log == [("SELECT * FROM tasks WHERE id=%s AND status=%s", ("t-1", "queued"))]


def test_sqlite_placeholders_pass_through_untouched(tmp_path):
    conn = open_sqlite(str(tmp_path / "wire.db"))
    try:
        conn.execute("CREATE TABLE t (a TEXT)")
        conn.execute("INSERT INTO t VALUES (?)", ("x",))
        conn.commit()
        row = conn.execute("SELECT a FROM t WHERE a=?", ("x",)).fetchone()
        assert row["a"] == "x"
    finally:
        conn.close()


def test_parameters_are_adapted_to_the_driver_shape(fake_pg_connection):
    conn, raw = fake_pg_connection
    conn.execute("DELETE FROM tasks WHERE id=?", ["t-1"])
    assert raw.log[0][1] == ("t-1",)


def test_executemany_adapts_the_statement_and_each_row(fake_pg_connection):
    conn, raw = fake_pg_connection
    conn.executemany(
        "INSERT INTO metrics(task_id, name) VALUES (?, ?)",
        [("t-1", "psnr"), ("t-2", "ssim")],
    )
    assert raw.log == [
        ("INSERT INTO metrics(task_id, name) VALUES (%s, %s)", ("t-1", "psnr")),
        ("INSERT INTO metrics(task_id, name) VALUES (%s, %s)", ("t-2", "ssim")),
    ]


def test_a_statement_without_placeholders_is_unchanged(fake_pg_connection):
    conn, raw = fake_pg_connection
    conn.execute("SELECT COUNT(*) FROM tasks")
    assert raw.log == [("SELECT COUNT(*) FROM tasks", ())]


def test_every_cursor_double_in_the_tests_models_the_columns_attribute():
    """A fake cursor missing `description` makes the wrapper's contract a lie.

    `Connection.execute` reads `description` to decide whether to fetch, so a double
    without it is *less capable than the protocol* 鈥?the production code then looks
    broken when the fake is thin. Two such doubles existed at once (this module's and
    `test_postgres_data_path`'s), which is why the rule is a scan rather than a fix in
    one place.
    """
    import ast
    import pathlib

    tests_dir = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in tests_dir.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {
                item.name for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if not {"execute", "fetchone"} <= methods:
                continue
            body = ast.unparse(node)
            if "description" not in body:
                offenders.append(f"{path.name}::{node.name}")
    assert not offenders, f"cursor doubles without `description`: {offenders}"


def test_rows_keep_the_mapping_access_the_repositories_use(fake_pg_connection):
    """``row["col"]`` and ``dict(row)`` are used all over the data layer."""
    conn, _ = fake_pg_connection
    rows = conn.execute("PRAGMA-equivalent introspection").fetchall()
    assert {r["name"] for r in rows} == {"id", "status"}
    assert dict(rows[0]) == {"name": "id"}


# ================================ schema scripts =================================
#: A slice of the shipped schema using the two constructs that differ.
SCRIPT = """
CREATE TABLE IF NOT EXISTS tasks (
    id      TEXT PRIMARY KEY,
    score   REAL,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    value REAL
);
"""


def test_postgres_executescript_translates_and_splits_the_ddl(fake_pg_connection):
    conn, raw = fake_pg_connection
    conn.executescript(SCRIPT)

    assert len(raw.log) == 2, "one statement per script entry"
    first, second = raw.log[0][0], raw.log[1][0]
    assert "DOUBLE PRECISION" in first and "REAL" not in first.replace("DOUBLE PRECISION", "")
    assert "BIGSERIAL PRIMARY KEY" in second and "AUTOINCREMENT" not in second
    assert all(c.closed for c in raw.cursors), "the script cursor must not leak"


def test_postgres_executescript_strips_comment_lines(fake_pg_connection):
    conn, raw = fake_pg_connection
    conn.executescript("-- a comment\nCREATE TABLE a (x TEXT);\n")
    assert raw.log[0][0] == "CREATE TABLE a (x TEXT)"


def test_sqlite_executescript_uses_the_native_single_call_path(tmp_path):
    conn = open_sqlite(str(tmp_path / "script.db"))
    try:
        conn.executescript(SCRIPT)
        conn.commit()
        assert conn.execute("SELECT id FROM metrics").fetchall() == []
    finally:
        conn.close()


def test_executescript_on_a_driver_without_the_method_never_calls_it(pg):
    """The fake has no ``executescript``: reaching it would raise AttributeError."""
    raw = FakeConnection()
    Connection(raw, pg, driver=POSTGRESQL).executescript("CREATE TABLE a (x TEXT);")
    assert raw.log and not hasattr(raw, "executescript")


# ================================= transactions ==================================
def test_commit_rollback_close_reach_the_driver(fake_pg_connection):
    conn, raw = fake_pg_connection
    conn.commit()
    conn.rollback()
    conn.close()
    assert (raw.commits, raw.rollbacks, raw.closed) == (1, 1, 1)


def test_context_manager_commits_a_clean_block(fake_pg_connection):
    conn, raw = fake_pg_connection
    with conn:
        conn.execute("UPDATE tasks SET status=? WHERE id=?", ("running", "t-1"))
    assert raw.commits == 1 and raw.rollbacks == 0


def test_context_manager_rolls_back_on_exception(fake_pg_connection):
    conn, raw = fake_pg_connection
    with pytest.raises(RuntimeError):
        with conn:
            conn.execute("UPDATE tasks SET status=?", ("running",))
            raise RuntimeError("stage failed")
    assert raw.rollbacks == 1 and raw.commits == 0


# ================================== open helpers =================================
def test_open_sqlite_returns_indexable_rows(tmp_path):
    conn = open_sqlite(str(tmp_path / "rows.db"))
    try:
        conn.execute("CREATE TABLE u (username TEXT)")
        conn.execute("INSERT INTO u VALUES (?)", ("kevin",))
        conn.commit()
        row = conn.execute("SELECT username FROM u").fetchone()
        assert row["username"] == "kevin"
        assert dict(row) == {"username": "kevin"}
    finally:
        conn.close()


def test_open_sqlite_tunes_wal_and_busy_timeout(tmp_path):
    conn = open_sqlite(str(tmp_path / "pragma.db"))
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        conn.close()


def test_open_sqlite_keeps_the_sqlite_dialect_when_postgres_is_configured(
    tmp_path, monkeypatch
):
    """``open_sqlite`` must not inherit the deployment dialect.

    It opens a file, so it always speaks SQLite. Reading ``default_dialect()``
    here would hand a ``sqlite3`` connection a PostgreSQL dialect, whose ``%s``
    placeholders and no-PRAGMA behaviour silently break the connection.
    """
    monkeypatch.setenv("FIXIMG_DATABASE_URL", "postgresql://user:pw@localhost/fiximg")
    conn = open_sqlite(str(tmp_path / "forced.db"))
    try:
        assert conn.dialect.name == SQLITE
        conn.execute("CREATE TABLE t (a TEXT)")
        conn.execute("INSERT INTO t VALUES (?)", ("x",))
        conn.commit()
        assert conn.execute("SELECT a FROM t WHERE a=?", ("x",)).fetchone()[0] == "x"
    finally:
        conn.close()


def test_open_sqlite_creates_the_parent_directory(tmp_path):
    path = str(tmp_path / "nested" / "deeper" / "db.sqlite")
    conn = open_sqlite(path)
    try:
        assert conn.dialect.name == SQLITE
    finally:
        conn.close()


def test_connect_dispatches_on_the_resolved_kind(tmp_path):
    conn = connect(str(tmp_path / "dispatch.db"), "sqlite")
    try:
        assert conn.driver == SQLITE
    finally:
        conn.close()


def test_connect_dispatches_on_a_postgres_url_without_an_explicit_kind(monkeypatch):
    """A ``postgres://`` target must not be opened with sqlite3."""
    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(ConnectionError, match="psycopg"):
        connect("postgresql://user:pw@localhost/fiximg")


def test_postgres_without_the_driver_reports_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(ConnectionError) as excinfo:
        open_postgres("postgresql://user:pw@localhost/fiximg")
    message = str(excinfo.value)
    assert "FIXIMG_DATABASE_URL" in message
    assert "fiximg[postgres]" in message, "the fix must be actionable"


def test_postgres_opens_a_dict_row_connection(monkeypatch, pg):
    """psycopg is configured the way the repositories need: dict rows, no autocommit."""
    captured = {}

    class FakeRows:
        @staticmethod
        def dict_row(cursor):
            return []

    class FakePsycopg:
        rows = FakeRows

        @staticmethod
        def connect(url, row_factory=None, autocommit=None, connect_timeout=None):
            captured.update(url=url, row_factory=row_factory, autocommit=autocommit,
                            connect_timeout=connect_timeout)
            return FakeConnection()

    # `from psycopg.rows import dict_row` needs the submodule registered too 鈥?    # a bare sys.modules entry for "psycopg" is not a package.
    monkeypatch.setitem(sys.modules, "psycopg", FakePsycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", FakeRows)
    conn = open_postgres("postgresql://h/db", pg)
    assert conn.driver == POSTGRESQL
    assert captured["url"] == "postgresql://h/db"
    assert captured["autocommit"] is False, "the wrapper drives commit() explicitly"
    assert captured["row_factory"] is FakeRows.dict_row


def test_postgres_connection_attempt_is_bounded(monkeypatch, pg):
    """A database that is down must be an error, never a hang.

    Measured on a Windows host with nothing listening on 5432: a raw socket is
    refused in 2 s, ``psycopg.connect`` without ``connect_timeout`` never returned
    (killed after 70 s), and with ``connect_timeout=3`` it raised
    ``ConnectionTimeout`` in 3.01 s. ``0`` means "wait forever" to libpq, so the
    keyword has to be a positive number by default rather than psycopg's own.
    """
    captured = {}

    class FakeRows:
        @staticmethod
        def dict_row(cursor):
            return []

    class FakePsycopg:
        rows = FakeRows

        @staticmethod
        def connect(url, row_factory=None, autocommit=None, connect_timeout=None):
            captured.update(connect_timeout=connect_timeout)
            return FakeConnection()

    monkeypatch.setitem(sys.modules, "psycopg", FakePsycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", FakeRows)

    open_postgres("postgresql://h/db", pg)
    default = captured["connect_timeout"]
    assert default == connection_module.DEFAULT_CONNECT_TIMEOUT
    assert isinstance(default, int) and default > 0, f"unbounded: {default!r}"

    open_postgres("postgresql://h/db", pg, connect_timeout=2)
    assert captured["connect_timeout"] == 2, "an operator must be able to shorten it"


def test_connection_exposes_the_raw_handle_for_migrations(fake_pg_connection):
    conn, raw = fake_pg_connection
    assert conn.raw is raw
    assert conn.driver == POSTGRESQL


def test_the_default_driver_is_sqlite(pg):
    """A Connection built without ``driver`` behaves like the SQLite path."""
    raw = FakeConnection()
    conn = Connection(raw, dialect_for(SQLITE))
    assert conn.driver == SQLITE


# ============================ conflict bookkeeping ============================
def test_a_conflict_rolls_back_only_off_sqlite(fake_pg_connection):
    """PostgreSQL aborts the whole transaction; SQLite aborts just the statement.

    The idempotency race handler runs a SELECT after catching the conflict, and
    on PostgreSQL that SELECT fails until something rolls back.
    """
    conn, raw = fake_pg_connection
    conn.rollback_after_conflict()
    assert raw.rollbacks == 1


def test_a_conflict_leaves_a_sqlite_transaction_alone(tmp_path):
    conn = open_sqlite(str(tmp_path / "conflict.db"))
    try:
        conn.execute("CREATE TABLE t (a TEXT UNIQUE)")
        conn.execute("INSERT INTO t VALUES (?)", ("x",))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO t VALUES (?)", ("x",))
        # The uncommitted first insert must still be there to read and commit.
        conn.rollback_after_conflict()
        conn.commit()
        assert conn.execute("SELECT COUNT(*) AS n FROM t").fetchone()["n"] == 1
    finally:
        conn.close()


def test_integrity_errors_covers_every_driver_that_is_installed():
    errors = connection_module.integrity_errors()
    assert sqlite3.IntegrityError in errors
    assert isinstance(errors, tuple) and errors, "an except clause needs a tuple"


def test_integrity_errors_grows_when_psycopg_is_importable(monkeypatch):
    """A PostgreSQL deployment must not let a duplicate key become a 500."""

    class PsycopgIntegrityError(Exception):
        pass

    class FakePsycopg:
        IntegrityError = PsycopgIntegrityError

    monkeypatch.setitem(sys.modules, "psycopg", FakePsycopg)
    errors = connection_module.integrity_errors()
    assert PsycopgIntegrityError in errors
    assert sqlite3.IntegrityError in errors


def test_integrity_errors_survives_a_missing_psycopg(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", None)
    assert connection_module.integrity_errors() == (sqlite3.IntegrityError,)


# ====================== one connection per thread (搂4.3 Step 2) ===============
class RecordingConnector:
    """Stands in for ``connection.connect``, handing out a distinct object each call."""

    def __init__(self):
        self.calls = []

    def __call__(self, target, kind="sqlite", dialect=None):
        conn = FakeConnection()
        self.calls.append((target, kind, conn))
        return Connection(conn, dialect_for(kind), driver=kind)


@pytest.fixture()
def engine_with_fake_connect(monkeypatch):
    """Point the engine at a fake driver, with a clean connection cache."""
    import fiximg.infrastructure.db.engine as engine

    connector = RecordingConnector()
    monkeypatch.setattr(connection_module, "connect", connector)
    monkeypatch.setattr(engine, "_conn", None)
    engine.close_connections()
    yield engine, connector
    engine.close_connections()


def _set_url(engine, monkeypatch, url, kind, target):
    """Resolve every connection request to ``kind``/``target``."""
    monkeypatch.setenv("FIXIMG_DATABASE_URL", url)
    monkeypatch.setattr(engine, "resolve_database", lambda _url=None: (kind, target))


def test_sqlite_shares_one_connection_across_threads(engine_with_fake_connect, monkeypatch):
    """The documented SQLite behaviour: one handle, safe by construction."""
    import threading

    engine, connector = engine_with_fake_connect
    _set_url(engine, monkeypatch, "", "sqlite", ":memory:")

    seen: list = []
    thread = threading.Thread(target=lambda: seen.append(engine.get_conn()))
    thread.start()
    thread.join()

    main_conn = engine.get_conn()
    assert seen and seen[0] is main_conn
    assert len(connector.calls) == 1


def test_postgres_gives_each_thread_its_own_connection(engine_with_fake_connect, monkeypatch):
    """psycopg connections are not thread-safe, and the API is threaded."""
    import threading

    engine, connector = engine_with_fake_connect
    _set_url(engine, monkeypatch, "postgresql://h/db", "postgresql", "postgresql://h/db")

    seen: list = []
    threads = [threading.Thread(target=lambda: seen.append(engine.get_conn())) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    main_conn = engine.get_conn()
    assert len(seen) == 2
    assert seen[0] is not seen[1], "two worker threads must not share one psycopg connection"
    assert main_conn not in seen
    assert len(connector.calls) == 3, "one open per thread, not per call"


def test_postgres_reuses_the_connection_within_a_thread(engine_with_fake_connect, monkeypatch):
    engine, connector = engine_with_fake_connect
    _set_url(engine, monkeypatch, "postgresql://h/db", "postgresql", "postgresql://h/db")
    assert engine.get_conn() is engine.get_conn()
    assert len(connector.calls) == 1


def test_close_connections_closes_the_thread_handles(engine_with_fake_connect, monkeypatch):
    import threading

    engine, connector = engine_with_fake_connect
    _set_url(engine, monkeypatch, "postgresql://h/db", "postgresql", "postgresql://h/db")

    seen: list = []
    thread = threading.Thread(target=lambda: seen.append(engine.get_conn()))
    thread.start()
    thread.join()
    main_conn = engine.get_conn()

    engine.close_connections()
    assert seen[0].raw.closed is True
    assert main_conn.raw.closed is True
    assert engine._conn is None

