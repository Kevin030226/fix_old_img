"""SQL dialect abstraction (plan §4.3 Step 2).

The repositories were written against SQLite and use four constructs that do not
port::

    ?  placeholders            ->  %s
    json_group_array/object    ->  json_agg / json_object_agg
    PRAGMA table_info(x)       ->  information_schema.columns
    BEGIN IMMEDIATE            ->  SELECT ... FOR UPDATE SKIP LOCKED

Each of those is a *fragment*, not a rewrite: the SQLite implementation returns
exactly the strings the repository used before this module existed, so
introducing the seam cannot have changed behaviour, and the PostgreSQL
implementation emits SQL that is verified to compile under the real PostgreSQL
dialect. The connection that runs either one is in
:mod:`fiximg.infrastructure.db.connection`; ``test_postgres_data_path.py`` boots
the data layer over a fake PostgreSQL driver, which is what catches a stray
``PRAGMA`` or ``INSERT OR IGNORE`` before a deployment does.

What is still not claimed: no run of the suite against a live PostgreSQL server.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Dialect names this module understands.
SQLITE = "sqlite"
POSTGRESQL = "postgresql"


@dataclass(frozen=True)
class SqlDialect:
    """The SQLite implementation — the strings the repository already used.

    Subclasses override only what differs, so the SQLite path is provably
    unchanged.
    """

    name: str = SQLITE
    #: DB-API parameter placeholder.
    placeholder: str = "?"
    #: Whether ``CREATE UNIQUE INDEX ... WHERE`` is supported.
    supports_partial_index: bool = True
    #: Whether a row-level claim can skip locked rows (avoids write contention).
    supports_skip_locked: bool = False

    # ------------------------------------------------------------ adaptation
    def adapt(self, sql: str) -> str:
        """Rewrite a ``?``-style statement for this dialect.

        The repository keeps writing ``?`` because that is what ``sqlite3`` wants;
        other dialects get the statement rewritten here, at one place.
        """
        return sql

    def params(self, values) -> tuple:
        """Adapt a parameter sequence (identity for SQLite)."""
        return tuple(values)

    # ---------------------------------------------------------- SQL fragments
    @property
    def json_array_fn(self) -> str:
        """Name of the aggregate producing a JSON array from rows."""
        return "json_group_array"

    @property
    def json_object_fn(self) -> str:
        """Name of the aggregate producing a JSON object from rows."""
        return "json_group_object"

    @property
    def row_object_fn(self) -> str:
        """Name of the *scalar* that builds one JSON object from key/value pairs.

        A different job from :attr:`json_object_fn`, and the two engines do not
        share a name for it: SQLite's ``json_object('a', a, 'b', b)`` builds a
        value per row, while PostgreSQL spells the same thing
        ``json_build_object(...)`` and reserves ``json_object_agg`` for the
        aggregate — whose two-argument-only signature makes a copied name a
        ``UndefinedFunction`` error on the first real server.
        """
        return "json_object"

    def json_array(self, element_expr: str) -> str:
        """Aggregate rows into a JSON array; NULL when there are no rows."""
        return f"{self.json_array_fn}({element_expr})"

    def json_object(self, key_expr: str, value_expr: str) -> str:
        """Aggregate rows into a JSON object."""
        return f"{self.json_object_fn}({key_expr}, {value_expr})"

    def json_metric_value(self, numeric: str, text: str) -> str:
        """SQL for "this metric's value": a JSON number when it has one, else text.

        ``metrics`` keeps the numeric and the textual half in two columns (plan
        §2.7), so every reader collapses them. On SQLite ``COALESCE`` is enough,
        because a column there carries the type of the value it holds and the JSON
        functions follow it. PostgreSQL types the expression first, so
        ``COALESCE(double precision, text)`` is a ``DatatypeMismatch`` and a cast to
        text would silently turn ``23.5`` into ``"23.5"`` in the API payload; each
        branch therefore goes through ``to_json`` and keeps its own JSON type.
        """
        return f"COALESCE({numeric}, {text})"

    def table_columns_sql(self, table: str) -> tuple[str, tuple]:
        """(statement, params) returning one row per column with ``name``."""
        return f"PRAGMA table_info({table})", ()

    def begin_write_sql(self) -> str | None:
        """Statement taking the database's write lock, or None.

        SQLite needs ``BEGIN IMMEDIATE`` to avoid a deferred transaction
        upgrading mid-way (which raises ``SQLITE_BUSY`` under contention).
        PostgreSQL takes row locks instead and needs nothing here.
        """
        return "BEGIN IMMEDIATE"

    def claim_row_lock(self) -> str:
        """Suffix that locks the claimed row against a concurrent claimant."""
        return ""

    # ------------------------------------------------------------------ writes
    def insert_ignore(self, sql: str, conflict_columns=()) -> str:
        """Make an ``INSERT`` skip a row whose key already exists.

        The two engines spell this on opposite ends of the statement: SQLite
        takes a prefix (``INSERT OR IGNORE INTO``), PostgreSQL an action suffix
        naming the conflict target. Callers write the plain ``INSERT INTO`` form
        they already know and get the right spelling for the engine in use.
        """
        return sql.replace("INSERT INTO", "INSERT OR IGNORE INTO", 1)

    # ------------------------------------------------------------------- DDL
    def adapt_ddl(self, sql: str) -> str:
        """Translate a schema script for this dialect."""
        return sql

    def pragma(self, name: str, value: str | None = None) -> str | None:
        """A connection-tuning statement, or None when the dialect has no such knob.

        SQLite needs WAL + a busy timeout to behave under concurrent writers;
        PostgreSQL has equivalents configured on the server, not per connection.
        """
        return f"PRAGMA {name}={value}" if value is not None else f"PRAGMA {name}"

    def split_statements(self, script: str) -> list[str]:
        """Split a multi-statement script into individual statements.

        ``executescript`` is a sqlite3 convenience; other drivers need the script
        split. The shipped scripts contain no embedded semicolons.
        """
        statements: list[str] = []
        for chunk in script.split(";"):
            lines = [ln for ln in chunk.splitlines() if not ln.strip().startswith("--")]
            cleaned = "\n".join(lines).strip()
            if cleaned:
                statements.append(cleaned)
        return statements


@dataclass(frozen=True)
class PostgresDialect(SqlDialect):
    """PostgreSQL: different placeholders, JSON functions and locking."""

    name: str = POSTGRESQL
    placeholder: str = "%s"
    supports_partial_index: bool = True
    supports_skip_locked: bool = True

    def adapt(self, sql: str) -> str:
        # `?` is not a PostgreSQL placeholder. The repository's statements never
        # contain a literal `?` (no JSON operators), so a plain substitution is
        # faithful.
        return sql.replace("?", "%s")

    @property
    def json_array_fn(self) -> str:
        return "json_agg"

    @property
    def json_object_fn(self) -> str:
        return "json_object_agg"

    @property
    def row_object_fn(self) -> str:
        return "json_build_object"

    def json_array(self, element_expr: str) -> str:
        # `json_agg` returns NULL for zero rows, matching json_group_array's
        # behaviour that the callers rely on.
        return f"json_agg({element_expr})"

    def json_object(self, key_expr: str, value_expr: str) -> str:
        return f"json_object_agg({key_expr}, {value_expr})"

    def json_metric_value(self, numeric: str, text: str) -> str:
        return (
            f"CASE WHEN {numeric} IS NOT NULL "
            f"THEN to_json({numeric}) ELSE to_json({text}) END"
        )

    def table_columns_sql(self, table: str) -> tuple[str, tuple]:
        return (
            "SELECT column_name AS name FROM information_schema.columns "
            "WHERE table_name = %s",
            (table,),
        )

    def begin_write_sql(self) -> str | None:
        # PostgreSQL has no equivalent and does not need one: the claim uses a
        # row lock with SKIP LOCKED so concurrent workers never block.
        return None

    def claim_row_lock(self) -> str:
        return "FOR UPDATE SKIP LOCKED"

    def insert_ignore(self, sql: str, conflict_columns=()) -> str:
        """``ON CONFLICT ... DO NOTHING`` — PostgreSQL has no ``INSERT OR IGNORE``.

        ``conflict_columns`` names the unique key the caller expects to collide
        on. Omitting it is legal (`ON CONFLICT DO NOTHING` needs no target), but
        naming it is what makes a collision on *that* key the skipped case.
        """
        target = f" ({', '.join(conflict_columns)})" if tuple(conflict_columns) else ""
        return f"{sql} ON CONFLICT{target} DO NOTHING"

    # ------------------------------------------------------------------- DDL
    def adapt_ddl(self, sql: str) -> str:
        """Translate SQLite DDL to PostgreSQL.

        Only two constructs in the shipped schema differ:

        * ``INTEGER PRIMARY KEY AUTOINCREMENT`` is SQLite's rowid alias;
          PostgreSQL spells it ``BIGSERIAL PRIMARY KEY``.
        * ``REAL`` is SQLite's 8-byte float; PostgreSQL calls it
          ``DOUBLE PRECISION``.

        Everything else the schema uses (``TEXT``, ``INTEGER``, partial unique
        indexes) is already valid PostgreSQL.
        """
        return (
            sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")
            .replace("AUTOINCREMENT", "")
            .replace(" REAL", " DOUBLE PRECISION")
        )

    def pragma(self, name: str, value: str | None = None) -> str | None:
        """PostgreSQL has no per-connection PRAGMA; these are server settings."""
        return None

    def split_statements(self, script: str) -> list[str]:
        """PostgreSQL DDL is applied statement by statement."""
        return super().split_statements(script)


#: The registered dialects.
_DIALECTS = {
    SQLITE: SqlDialect(),
    POSTGRESQL: PostgresDialect(),
}

#: URL schemes (and their SQLAlchemy spellings) mapped to a dialect name.
_SCHEME_ALIASES = {
    "sqlite": SQLITE,
    "sqlite3": SQLITE,
    "sqlite+pysqlite": SQLITE,
    "postgres": POSTGRESQL,
    "postgresql": POSTGRESQL,
    "postgresql+psycopg": POSTGRESQL,
    "postgresql+psycopg2": POSTGRESQL,
}


def dialect_for(name: str) -> SqlDialect:
    """Look up a dialect by name; defaults to SQLite for an unknown name."""
    return _DIALECTS.get((name or "").strip().lower(), _DIALECTS[SQLITE])


def dialect_for_url(url: str | None) -> SqlDialect:
    """Resolve the dialect from a ``FIXIMG_DATABASE_URL``."""
    text = (url or "").strip()
    if not text or "://" not in text:
        return _DIALECTS[SQLITE]
    scheme = text.split("://", 1)[0].lower()
    return dialect_for(_SCHEME_ALIASES.get(scheme, SQLITE))


def default_dialect() -> SqlDialect:
    """The dialect for the configured database URL."""
    import os

    return dialect_for_url(os.environ.get("FIXIMG_DATABASE_URL", ""))


__all__ = [
    "POSTGRESQL",
    "SQLITE",
    "PostgresDialect",
    "SqlDialect",
    "default_dialect",
    "dialect_for",
    "dialect_for_url",
]
