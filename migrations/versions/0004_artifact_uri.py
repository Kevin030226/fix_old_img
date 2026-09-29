"""0004 — where a published artifact lives (plan §2.8).

``artifacts.path`` recorded a *filesystem* location, which is the coupling V3's
store protocol exists to remove: with the API and the worker on different nodes,
a path written by one is meaningless to the other, and ``FIXIMG_STORAGE_BACKEND=s3``
changed nothing about where bytes landed because the pipeline wrote through the
local helpers regardless.

The new column is the object-store **key** the runtime uploaded the file under,
and it stays NULL for a local store on purpose. Every reader already prefers
``row["uri"] or row["path"]``, so filling ``uri`` with a *relative* key in local
mode would turn those reads into a path resolved against the process's working
directory — a regression for the common single-node deployment, which is exactly
what this column must never do.

Revision ID: 0004
Revises: 0003
"""
import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

_TABLE = "artifacts"
_COLUMN = "uri"


def _columns() -> set[str]:
    """Existing column names, or an empty set in offline mode."""
    bind = op.get_bind()
    if op.get_context().as_sql:
        return set()
    inspector = sa.inspect(bind)
    if not inspector.has_table(_TABLE):
        return set()
    return {column["name"] for column in inspector.get_columns(_TABLE)}


def upgrade() -> None:
    """Add the store-key column; a no-op when 0001's shipped DDL already has it.

    The guard is "add it only if the table exists and lacks the column". A database
    seeded without the artifacts table at all (a partial legacy file) gets nothing,
    exactly as 0003 decided: there is no shape to correct until 0001 creates it.
    """
    existing = _columns()
    if existing and _COLUMN not in existing:
        op.execute(sa.text(f"ALTER TABLE {_TABLE} ADD COLUMN {_COLUMN} TEXT"))


def downgrade() -> None:
    """Drop it. Published keys are lost, so a later upgrade re-publishes nothing.

    Nothing is deleted from the object store here: a migration that removed the
    last pointer to an object would make it unreclaimable *and* unrecoverable,
    which is a worse outcome than an orphan the sweeper can still be taught about.
    """
    if _COLUMN not in _columns():
        return
    with op.batch_alter_table(_TABLE) as batch:
        batch.drop_column(_COLUMN)
