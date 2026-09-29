"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

Schema changes here are hand-written SQL: the repositories use SQLite-specific
DDL (partial indexes, ``json_group_array``) that autogenerate cannot reproduce
faithfully. See ``migrations/env.py``.

Keep every revision idempotent — ``upgrade()`` must tolerate a database that
already has the change, because ``ensure_schema()`` runs at startup and may have
created it first.
"""
from alembic import op  # noqa: F401
import sqlalchemy as sa  # noqa: F401
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
