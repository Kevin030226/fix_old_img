"""0003 — typed instants and numeric metrics (plan §2.7).

Two of the three data-layer smells §2.7 lists are *value* problems, and one is a
*type* problem, so the revision does three things:

1. **Instants** are rewritten from the V2 form (``2026-06-01 08:00:00`` — naive,
   local, one-second granularity) to the canonical form
   (``2026-06-01T00:00:00.000000Z``). The column type stays TEXT; what changes is
   that the string carries a zone and enough precision to order two submissions
   in the same second. The legacy values were written with the host's local
   clock, so that is the zone assumed — SQLite's ``'utc'`` modifier and
   PostgreSQL's ``AT TIME ZONE`` both do exactly that conversion.
2. **``metrics.metric_value``** becomes a REAL, so a PSNR can be averaged in SQL.
   Non-numeric values (``inf`` after a bit-identical image, a label) move to
   ``metric_text`` first, so nothing is silently dropped by the cast.
3. The added column is also created here, so a database that never ran the
   runtime's ``ensure_schema`` still ends up with the shape the repositories
   write into.

Everything is guarded to be a no-op on a fresh database: 0001 imports the
runtime's own DDL, which already declares the new shape, and re-running the
conversion would rewrite values that are already canonical.

``history.timestamp`` is deliberately *not* converted. It is V1 display data the
UI renders as the user's local time; rewriting it would change what old rows say
without anything depending on the answer.

The downgrade is a genuine inverse for the instants (UTC renders back through the
host zone), and lossy for metrics in one way worth knowing about: a number
re-rendered as text can gain a trailing zero (``'0'`` → ``'0.0'``). It parses to
the same value, which is all the pre-0003 code promised.

Revision ID: 0003
Revises: 0002
"""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

#: (table, columns) holding an instant. Every one is TEXT in 0001.
_INSTANT_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tasks", ("created_at", "started_at", "finished_at",
               "lease_until", "last_heartbeat", "retry_at")),
    ("task_stages", ("created_at", "finished_at")),
    ("artifacts", ("created_at",)),
    ("metrics", ("created_at",)),
    ("system_events", ("created_at",)),
    ("users", ("created_at", "updated_at")),
)

#: ``len("YYYY-MM-DD HH:MM:SS")`` — what the legacy writer produced.
LEGACY_LENGTH = 19
#: ``len("YYYY-MM-DDTHH:MM:SS.ffffffZ")`` — the canonical form.
CANONICAL_LENGTH = 27

#: A stored metric that is a number rather than a label. GLOB is used instead of
#: a regex because SQLite ships without ``REGEXP`` unless the app registers one.
_SQLITE_NUMERIC = (
    "NOT (metric_value GLOB '[0-9+-]*' AND metric_value GLOB '*[0-9]')"
)
_PG_NON_NUMERIC = (
    "metric_value !~ '^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'"
)


def _dialect() -> str:
    return op.get_bind().dialect.name


def _to_canonical(table: str, column: str) -> str:
    """SQL rewriting one legacy instant into the canonical form."""
    if _dialect() == "postgresql":
        return (
            f"UPDATE {table} SET {column} = to_char("
            f"({column}::timestamp AT TIME ZONE current_setting('TimeZone')) "
            f"AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS\".000000Z\"') "
            f"WHERE {column} IS NOT NULL AND length({column}) = {LEGACY_LENGTH} "
            f"AND {column} ~ '^[0-9]{{4}}-'"
        )
    return (
        f"UPDATE {table} SET {column} = "
        f"strftime('%Y-%m-%dT%H:%M:%S', datetime({column}, 'utc')) || '.000000Z' "
        f"WHERE {column} IS NOT NULL AND length({column}) = {LEGACY_LENGTH} "
        f"AND datetime({column}) IS NOT NULL"
    )


def _to_legacy(table: str, column: str) -> str:
    """SQL rewriting the canonical form back to naive local second precision."""
    if _dialect() == "postgresql":
        return (
            f"UPDATE {table} SET {column} = to_char("
            f"({column}::timestamptz) AT TIME ZONE current_setting('TimeZone'), "
            f"'YYYY-MM-DD HH24:MI:SS') "
            f"WHERE {column} IS NOT NULL AND length({column}) = {CANONICAL_LENGTH}"
        )
    return (
        f"UPDATE {table} SET {column} = "
        # 'localtime' is what makes the round trip reversible: the canonical
        # string is UTC, and the form this rewrites back to is naive *local*.
        f"strftime('%Y-%m-%d %H:%M:%S', datetime({column}, 'localtime')) "
        f"WHERE {column} IS NOT NULL AND length({column}) = {CANONICAL_LENGTH}"
    )


def _column_type(table: str, column: str):
    """The reflected type of a column, or None when the table is absent."""
    bind = op.get_bind()
    if op.get_context().as_sql:
        return None
    inspector = sa.inspect(bind)
    if not inspector.has_table(table):
        return None
    for existing in inspector.get_columns(table):
        if existing["name"] == column:
            return existing["type"]
    return None


def _has_column(table: str, column: str) -> bool:
    return _column_type(table, column) is not None


def upgrade() -> None:
    bind = op.get_bind()
    postgres = _dialect() == "postgresql"

    for table, columns in _INSTANT_COLUMNS:
        for column in columns:
            if not _has_column(table, column):
                continue
            bind.execute(sa.text(_to_canonical(table, column)))

    # --- metrics.metric_value TEXT -> REAL
    if not _has_column("metrics", "metric_text"):
        op.execute(sa.text("ALTER TABLE metrics ADD COLUMN metric_text TEXT"))
    if _has_column("metrics", "metric_value"):
        reflected = _column_type("metrics", "metric_value")
        numeric_test = _PG_NON_NUMERIC if postgres else _SQLITE_NUMERIC
        if not isinstance(reflected, sa.Double) and not (
            isinstance(reflected, sa.Float) and not postgres
        ):
            # Move what a REAL cannot hold into the text column *before* the
            # cast, so `inf` survives the upgrade instead of becoming NULL.
            bind.execute(sa.text(
                f"UPDATE metrics SET metric_text = metric_value "
                f"WHERE metric_value IS NOT NULL AND {numeric_test}"
            ))
            bind.execute(sa.text(
                "UPDATE metrics SET metric_value = NULL WHERE metric_text IS NOT NULL"
            ))
            with op.batch_alter_table("metrics") as batch:
                batch.alter_column(
                    "metric_value",
                    existing_type=sa.Text(),
                    type_=sa.Double(),
                    existing_nullable=True,
                    postgresql_using=(
                        "CASE WHEN metric_value ~ "
                        "'^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$' "
                        "THEN metric_value::double precision END"
                    ),
                )


def downgrade() -> None:
    bind = op.get_bind()

    if _has_column("metrics", "metric_value"):
        reflected = _column_type("metrics", "metric_value")
        if isinstance(reflected, (sa.Float, sa.Double)):
            # Order matters, and only PostgreSQL says so: assigning the text half
            # back into ``metric_value`` is fine in SQLite, whose columns do not
            # enforce a type, and a straight type error in PostgreSQL ("You will
            # need to rewrite or cast the expression") while the column is still
            # DOUBLE PRECISION. So the column is widened to TEXT first, and only
            # then do the values move back. `downgrade base` had never been run
            # against a real server before this was found.
            with op.batch_alter_table("metrics") as batch:
                batch.alter_column(
                    "metric_value",
                    existing_type=sa.Double(),
                    type_=sa.Text(),
                    existing_nullable=True,
                    postgresql_using="metric_value::text",
                )
            bind.execute(sa.text(
                "UPDATE metrics SET metric_value = COALESCE("
                "metric_text, CAST(metric_value AS TEXT)), metric_text = NULL "
                "WHERE metric_text IS NOT NULL OR metric_value IS NOT NULL"
            ))
            if _has_column("metrics", "metric_text"):
                with op.batch_alter_table("metrics") as batch:
                    batch.drop_column("metric_text")

    for table, columns in _INSTANT_COLUMNS:
        for column in columns:
            if not _has_column(table, column):
                continue
            bind.execute(sa.text(_to_legacy(table, column)))
