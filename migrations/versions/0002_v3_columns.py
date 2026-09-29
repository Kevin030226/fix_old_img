"""0002 — columns added after the V2 schema (upgrade path).

Before this revision those columns were created by the ad-hoc ``_ADDED_COLUMNS``
guard in :mod:`fiximg.infrastructure.db.repositories.task_repository`, which runs
at every startup. That guard stays (it makes a cold start work without Alembic),
but the change is now also a *versioned* one, so:

* ``alembic history`` documents when each column appeared,
* a downgrade removes them again,
* a future schema change has a place to live instead of growing the guard list.

The columns:

===========================  ==============================================
``tasks.error_code``         machine-readable failure code (plan §6)
``tasks.required_capabilities``  capability routing hint (plan §4.3 Step 4)
``task_stages.metrics_json``  metrics a stage reported itself (plan §5.3)
===========================  ==============================================

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

#: (table, column, DDL type) — mirrors task_repository._ADDED_COLUMNS.
_COLUMNS = (
    ("tasks", "error_code", "TEXT"),
    ("tasks", "required_capabilities", "TEXT"),
    ("task_stages", "metrics_json", "TEXT"),
)


def _offline() -> bool:
    """True when Alembic is emitting SQL instead of connecting (``--sql``)."""
    return bool(op.get_context().as_sql)


def _columns_of(bind, table: str) -> set[str]:
    """Existing column names, via SQLAlchemy's inspector (dialect-agnostic).

    Returns an empty set in offline mode: there is no connection to inspect, and
    an offline script is meant to be reviewed rather than conditionally skipped.
    """
    if _offline():
        return set()
    inspector = sa.inspect(bind)
    if not inspector.has_table(table):
        return set()
    return {column["name"] for column in inspector.get_columns(table)}


def upgrade() -> None:
    """Add the post-V2 columns; a no-op when they already exist."""
    bind = op.get_bind()
    offline = _offline()
    for table, column, ddl_type in _COLUMNS:
        if not offline:
            existing = _columns_of(bind, table)
            if not existing:
                # The table itself does not exist yet — 0001 creates it with
                # these columns already present, so there is nothing to do.
                continue
            if column in existing:
                continue
        op.execute(sa.text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))


def downgrade() -> None:
    """Drop the post-V2 columns.

    SQLite cannot ``ALTER TABLE ... DROP COLUMN`` before 3.35, so this uses
    Alembic's batch mode, which rebuilds the table. Data in the dropped columns
    is lost — that is the point of a downgrade.
    """
    with op.batch_alter_table("tasks") as batch:
        batch.drop_column("error_code")
        batch.drop_column("required_capabilities")
    with op.batch_alter_table("task_stages") as batch:
        batch.drop_column("metrics_json")
