"""0005 — the ModelVersion table (plan §6).

§6 lists five data models; four were persisted and one was not. ``ModelVersion``
existed only as a domain class parsed out of ``models/manifest.yaml``, so
``status`` and ``loaded_at`` described what a deployment *claimed* rather than what
had happened, and ``GET /api/v1/models/{name}/versions`` could only answer for the
process serving the request — an API node beside a working pool of GPU workers
reported "no versions" for models that were loading images a second later.

The row is keyed ``(name, version)`` and **upserted**: reloads update the
observation and bump ``load_count``. Nothing here records residency — one process
dropping a version says nothing about whether another still serves it, which is
why the endpoint keeps residency in the process-local half of its answer.

Revision ID: 0005
Revises: 0004
"""
import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

_TABLE = "model_versions"

_DDL = """
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
)
"""


def _has_table() -> bool:
    if op.get_context().as_sql:
        return False
    return sa.inspect(op.get_bind()).has_table(_TABLE)


def upgrade() -> None:
    """Create the table; a no-op when 0001's shipped DDL already made it.

    ``_shipped_ddl()`` imports the schema constant rather than copying it, so a
    database seeded *after* this change already has the table from 0001 — the same
    relationship 0004 has to the ``artifacts.uri`` column.
    """
    if _has_table():
        return
    bind = op.get_bind()
    from fiximg.infrastructure.db.dialect import dialect_for

    statement = dialect_for(bind.dialect.name).adapt_ddl(_DDL).strip().rstrip(";")
    bind.execute(sa.text(statement))


def downgrade() -> None:
    """Drop it. The audit trail goes with it; the models themselves are unaffected.

    Weights, manifests and residency live outside this table, so a downgrade that
    removes it loses history rather than capability — and re-applying 0005 gives an
    empty catalog that the next load of each version fills back in.
    """
    bind = op.get_bind()
    bind.execute(sa.text(f"DROP TABLE IF EXISTS {_TABLE}"))
