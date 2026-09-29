"""0001 — V3 schema baseline.

Creates the whole schema from the DDL the application already ships, so
``alembic upgrade head`` on an empty database produces a working database. The
statements are **imported rather than copied**: one definition of the schema
means a migration cannot silently drift from what the runtime creates.

Covers:

* ``users`` / ``history`` — :data:`fiximg.infrastructure.db.engine._SCHEMA`
* ``tasks`` / ``task_stages`` / ``artifacts`` / ``metrics`` / ``system_events``
  — :data:`fiximg.infrastructure.db.repositories.task_repository._SCHEMA`

Revision ID: 0001
Revises: None
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

#: Tables this revision owns, in drop order (children before parents).
_TABLES = (
    "metrics",
    "artifacts",
    "task_stages",
    "system_events",
    "tasks",
    "history",
    "users",
)


def _shipped_ddl() -> list[str]:
    """The DDL constants the runtime uses, as ``CREATE TABLE IF NOT EXISTS``."""
    from fiximg.infrastructure.db.engine import _SCHEMA as app_schema
    from fiximg.infrastructure.db.repositories.task_repository import _SCHEMA as task_schema

    return [app_schema, task_schema]


def upgrade() -> None:
    """Create every table the application expects (idempotent).

    The shipped DDL constants are written in SQLite spelling, because that is the
    engine this project started on, so each statement has to pass through the
    target dialect's translation before reaching the bind: ``INTEGER PRIMARY KEY
    AUTOINCREMENT`` is a syntax error on PostgreSQL, as is an unquoted ``user``
    column. Importing the constants rather than copying them only stays correct if
    the copy is adapted — on a fresh PostgreSQL database this revision aborted at
    its first statement, which is how "PostgreSQL supported" turned out never to
    have been tested.
    """
    bind = op.get_bind()
    from fiximg.infrastructure.db.dialect import dialect_for

    adapt = dialect_for(bind.dialect.name).adapt_ddl
    for script in _shipped_ddl():
        for statement in _statements(script):
            bind.execute(sa.text(adapt(statement)))


def downgrade() -> None:
    """Drop everything this revision created, newest first."""
    bind = op.get_bind()
    for table in _TABLES:
        bind.execute(sa.text(f"DROP TABLE IF EXISTS {table}"))


def _statements(script: str) -> list[str]:
    """Split a DDL script into individual statements.

    ``executescript`` is a sqlite3 convenience Alembic does not offer, and the
    shipped constants are plain ``CREATE ...;`` lists with no embedded
    semicolons, so a simple split is faithful.
    """
    statements: list[str] = []
    for chunk in script.split(";"):
        text = chunk.strip()
        if not text or text.startswith("--"):
            continue
        # Drop trailing comment lines that belong to the next statement.
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("--")]
        cleaned = "\n".join(lines).strip()
        if cleaned:
            statements.append(cleaned)
    return statements
