"""Dialect-aware DB-API connection (plan §4.3 Step 2).

:mod:`fiximg.infrastructure.db.dialect` decides *what* SQL to write; this module
is the *transport* that runs it. It exists because the repositories were written
for ``sqlite3`` and use four of its conveniences:

===========================  ====================================================
``?`` placeholders           PostgreSQL wants ``%s``
``row["col"]``               needs ``sqlite3.Row`` / ``dict_row``
``executescript()``          a sqlite3 method with no DB-API equivalent
``conn.execute()`` returning  psycopg also does this, but the row type differs
a cursor
===========================  ====================================================

Rather than rewrite 700 lines of repository SQL, the connection adapts at the
seam: statements keep their ``?`` placeholders and their ``row["col"]`` access,
and the wrapper translates.

For SQLite the wrapper is deliberately transparent — it delegates to the real
``sqlite3.Connection`` with the same ``row_factory`` the code already used, so
behaviour is provably unchanged. The full suite passing on SQLite is the proof.

Two more seams the repositories need to be engine-neutral about: constraint
violations arrive as different exception classes per driver
(:func:`integrity_errors`), and a failed statement destroys more of the
transaction on PostgreSQL than on SQLite
(:meth:`Connection.rollback_after_conflict`). How many connections exist, and
which thread owns them, is the engine's business — see
:func:`fiximg.infrastructure.db.engine.get_conn`.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from typing import Any

from fiximg.infrastructure.db.dialect import SQLITE, SqlDialect, default_dialect, dialect_for


class ConnectionError(RuntimeError):
    """Raised when a database driver is missing or the URL cannot be served."""


class _Result:
    """One statement's outcome, read while the connection was still locked.

    Deliberately not a cursor: the point is that nothing here can be invalidated by
    another thread executing on the same handle afterwards. `rows` is already a list
    of the row objects the caller's ``row_factory`` produced, so a repository's
    ``dict(row)`` / ``row["n"]`` work unchanged.
    """

    __slots__ = ("_rows", "rowcount", "lastrowid")

    def __init__(self, rows: list, rowcount: int, lastrowid) -> None:
        self._rows = rows
        self.rowcount = rowcount
        self.lastrowid = lastrowid

    @staticmethod
    def of(cursor) -> _Result:
        # `description` is None for statements that return no result set, and
        # fetching from such a cursor is an error on some drivers.
        rows = cursor.fetchall() if cursor.description is not None else []
        return _Result(list(rows), cursor.rowcount, getattr(cursor, "lastrowid", None))

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list:
        return list(self._rows)

    def __iter__(self):
        return iter(self._rows)


class Connection:
    """A DB-API connection that speaks the configured dialect.

    Deliberately small: ``execute`` / ``executescript`` / ``commit`` /
    ``rollback`` / ``close``, which is exactly what the repositories use.

    Every method is serialised by an instance lock, because the SQLite path shares
    **one** connection between the API's request threads and the inline worker.
    ``check_same_thread=False`` only turns off the ownership check; it does not make
    a *transaction* safe to interleave, and a transaction here is several C-level
    calls (execute…execute…commit). Without the lock, 3.14's sqlite3 driver answers
    the race with ``SystemError: error return without exception set`` or
    ``InterfaceError: bad parameter or other API misuse`` out of ``commit()`` — seen
    as one failing test per full suite run, and a different test each time.

    What the lock does *not* buy: transaction boundaries. Two threads' statements
    can still land in one transaction, and whichever commits first publishes both.
    The claim path is the exception — it opens a real write transaction itself
    (``begin_write_sql``), which is the only sequence in this codebase where an
    interleaved commit would change which worker owns a task.
    """

    def __init__(self, raw: Any, dialect: SqlDialect, *, driver: str = SQLITE) -> None:
        self._raw = raw
        self._dialect = dialect
        self._driver = driver
        self._closed = False
        self._lock = threading.RLock()
        #: ``(kind, target)`` this handle was opened for, recorded by
        #: :func:`fiximg.infrastructure.db.engine.get_conn`. A cached handle is
        #: only reusable while it still matches what is configured now.
        self.database_target: tuple[str, str] | None = None

    # ------------------------------------------------------------- properties
    @property
    def dialect(self) -> SqlDialect:
        return self._dialect

    @property
    def driver(self) -> str:
        return self._driver

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has run.

        Cached handles are looked up before use, and a closed handle must not be
        handed out again — see :func:`fiximg.infrastructure.db.engine.get_conn`.
        """
        return self._closed

    @property
    def raw(self) -> Any:
        """The underlying driver connection (escape hatch for migrations)."""
        return self._raw

    # --------------------------------------------------------------- execution
    def execute(self, sql: str, params=()):
        """Run one statement and return its result, both under the lock.

        The result is materialised here rather than handed back as a live cursor:
        ``sqlite3.Connection.execute()`` returns the connection's shared cursor, so a
        second thread that executes before the first one fetches leaves it fetching
        *someone else's* statement — which surfaced as ``COUNT(*)`` answering
        ``None``. Fetching inside the lock is what makes "execute then read" one
        operation instead of two that another thread can cut in half.
        """
        with self._lock:
            cursor = self._raw.execute(self._dialect.adapt(sql), self._dialect.params(params))
            return _Result.of(cursor)

    def executemany(self, sql: str, seq_of_params):
        statement = self._dialect.adapt(sql)
        with self._lock:
            cursor = self._raw.executemany(
                statement, [self._dialect.params(p) for p in seq_of_params]
            )
            return _Result.of(cursor)

    def executescript(self, script: str) -> None:
        """Apply a multi-statement script.

        ``sqlite3`` has ``executescript``; other drivers need the script split and
        run statement by statement. The DDL is translated first, because the
        shipped schema is written in SQLite's dialect.
        """
        translated = self._dialect.adapt_ddl(script)
        if self._driver == SQLITE:
            with self._lock:
                self._raw.executescript(translated)
            return
        cursor = self._raw.cursor()
        try:
            for statement in self._dialect.split_statements(translated):
                cursor.execute(statement)
        finally:
            cursor.close()

    # ------------------------------------------------------------ transaction
    def commit(self) -> None:
        with self._lock:
            self._raw.commit()

    def rollback(self) -> None:
        with self._lock:
            self._raw.rollback()

    def rollback_after_conflict(self) -> None:
        """Clear the transaction state after a constraint violation.

        The engines differ in how much a failed statement destroys: SQLite
        aborts only that statement, so a follow-up query (the idempotency lookup
        that turns a lost race into "return the first submission") just works.
        PostgreSQL aborts the *whole* transaction and refuses every statement
        after it until something rolls back.

        The SQLite path deliberately does nothing here, so behaviour is exactly
        what it was before PostgreSQL was reachable.
        """
        if self._driver != SQLITE:
            self._raw.rollback()

    def close(self) -> None:
        self._closed = True
        self._raw.close()

    def __enter__(self) -> Connection:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()


def integrity_errors() -> tuple[type[BaseException], ...]:
    """The exception classes a driver raises for a constraint violation.

    The data layer has always caught ``sqlite3.IntegrityError`` — that is how a
    duplicate username and a lost idempotency race become a normal return value.
    psycopg has its own hierarchy, so catching only the sqlite3 class lets the
    same collision escape as a 500 on PostgreSQL. This returns whichever classes
    are importable here, for use as the tuple in an ``except`` clause.
    """
    errors: list[type[BaseException]] = [sqlite3.IntegrityError]
    try:
        import psycopg

        errors.append(psycopg.IntegrityError)
    except ImportError:  # pragma: no cover - depends on the environment
        pass
    return tuple(errors)


def _sqlite_row_factory():
    """``sqlite3.Row`` — supports both ``row["col"]`` and ``dict(row)``."""
    return sqlite3.Row


def open_sqlite(path: str, *, timeout: float = 5.0) -> Connection:
    """Open (and tune) a SQLite connection.

    The PRAGMAs are the ones the engine used before: WAL for concurrent readers,
    a busy timeout so a concurrent writer retries instead of failing, and
    ``synchronous=NORMAL`` as the durability/speed trade-off WAL makes safe.
    """
    if path != ":memory:":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    raw = sqlite3.connect(path, check_same_thread=False, timeout=timeout)
    raw.row_factory = _sqlite_row_factory()

    # This function opens a *file*, so it always speaks SQLite — even when the
    # deployment URL is PostgreSQL and some other call site opened this path
    # explicitly. Inheriting default_dialect() here would hand sqlite3 a
    # dialect whose %s placeholders it cannot parse.
    dialect = dialect_for(SQLITE)
    for name, value in (
        ("journal_mode", "WAL"),
        ("busy_timeout", "5000"),
        ("synchronous", "NORMAL"),
    ):
        statement = dialect.pragma(name, value)
        if statement:
            try:
                raw.execute(statement)
            except sqlite3.OperationalError:
                # `:memory:` cannot use WAL; the rest still applies.
                continue
    return Connection(raw, dialect, driver=SQLITE)


#: Seconds psycopg waits for a TCP connection before it raises. Measured on a
#: Windows host with nothing listening on 5432: a raw socket gets
#: ``ConnectionRefusedError`` in 2.0 s, while ``psycopg.connect`` without this
#: keyword never returned (killed after 70 s), and ``connect_timeout=3`` returned
#: ``ConnectionTimeout`` in 3.0 s. Without a bound, every caller that opens a
#: PostgreSQL connection — API boot, repositories, worker, the smoke script —
#: hangs on a database that is simply down. ``0`` is libpq's "wait forever", so
#: the protection comes from the default, not from a later clamp.
DEFAULT_CONNECT_TIMEOUT = 10


def open_postgres(url: str, dialect: SqlDialect | None = None, *,
                  connect_timeout: int = DEFAULT_CONNECT_TIMEOUT) -> Connection:
    """Open a PostgreSQL connection through psycopg 3.

    ``row_factory=dict_row`` gives the ``row["col"]`` / ``dict(row)`` access the
    repositories expect, matching ``sqlite3.Row``.
    """
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConnectionError(
            "psycopg is required for a PostgreSQL FIXIMG_DATABASE_URL "
            "(pip install 'fiximg[postgres]')"
        ) from exc

    raw = psycopg.connect(url, row_factory=dict_row, autocommit=False,
                          connect_timeout=connect_timeout)
    return Connection(raw, dialect or default_dialect(), driver="postgresql")


def connect(target: str, kind: str = "sqlite", dialect: SqlDialect | None = None) -> Connection:
    """Open a connection to an already-resolved target.

    ``target`` is a file path for SQLite or a URL for PostgreSQL; the caller
    resolves it (see :func:`fiximg.infrastructure.db.engine.resolve_database`)
    so this function never has to re-read the configuration — which is what keeps
    a ``DB_PATH`` set by a test or an embedder authoritative.
    """
    if kind == "postgresql" or str(target).startswith("postgres"):
        return open_postgres(target, dialect)
    return open_sqlite(target)


__all__ = [
    "Connection",
    "ConnectionError",
    "connect",
    "integrity_errors",
    "open_postgres",
    "open_sqlite",
]
