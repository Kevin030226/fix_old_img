"""Migration runner — Alembic (plan §4.3 Step 2).

V3 originally shipped a hand-rolled runner (ordered revisions, an
``schema_migrations`` table, up/down). It worked, but it was ours to maintain and
had no ecosystem tooling: no ``history``, no ``stamp``, no ``--sql`` review mode,
no way to adopt a database that predates the runner.

This module replaces it with Alembic, keeping the same entry points so nothing
downstream changes:

    python -m fiximg.infrastructure.db.migrations.runner status
    python -m fiximg.infrastructure.db.migrations.runner upgrade
    python -m fiximg.infrastructure.db.migrations.runner downgrade

Programmatic use::

    from fiximg.infrastructure.db.migrations.runner import upgrade, current
    upgrade()                    # alembic upgrade head
    applied_revision()           # the applied revision

The database URL is never read from ``alembic.ini``: it comes from
:func:`fiximg.infrastructure.db.engine.resolve_database_path`, so migrations and
the service always agree on which database they mean — and an unsupported scheme
fails here for the same reason it fails at startup.
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from fiximg.paths import PROJECT_ROOT

#: Alembic's config file and script directory.
INI_PATH = os.path.join(PROJECT_ROOT, "alembic.ini")
SCRIPT_LOCATION = os.path.join(PROJECT_ROOT, "migrations")

#: The revision a fresh database should be brought to.
HEAD = "head"

#: Where Alembic records what has been applied.
VERSION_TABLE = "alembic_version"


class MigrationUnavailableError(RuntimeError):
    """Alembic (or SQLAlchemy) is not installed."""


def _require_alembic():
    try:
        from alembic import command
        from alembic.config import Config
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise MigrationUnavailableError(
            "alembic and SQLAlchemy are required for migrations "
            "(pip install -r requirements/runtime.txt)"
        ) from exc
    return command, Config


def alembic_config():
    """Build the Alembic config used by every entry point here."""
    _command, Config = _require_alembic()
    config = Config(INI_PATH)
    config.set_main_option("script_location", SCRIPT_LOCATION)
    # The URL is resolved in migrations/env.py; setting it here as well keeps
    # offline (`--sql`) output deterministic.
    from fiximg.infrastructure.db.engine import resolve_database

    kind, target = resolve_database(os.environ.get("FIXIMG_DATABASE_URL", "") or None)
    if kind == "postgresql":
        url = (
            target.replace("postgresql://", "postgresql+psycopg://", 1)
            if target.startswith("postgresql://") else target
        )
    else:
        url = "sqlite://" if target == ":memory:" else "sqlite:///" + target.replace("\\", "/")
    config.set_main_option("sqlalchemy.url", url)
    return config


@contextmanager
def _quiet_alembic_logging():
    """Silence Alembic's per-statement logging during programmatic use."""
    import logging

    logger = logging.getLogger("alembic")
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(previous)


# ------------------------------------------------------------------ commands
def upgrade(revision: str = HEAD) -> None:
    """Apply every pending revision up to ``revision``."""
    command, _Config = _require_alembic()
    with _quiet_alembic_logging():
        command.upgrade(alembic_config(), revision)


def downgrade(revision: str = "-1") -> None:
    """Roll back to ``revision`` (``-1`` for one step, ``base`` for everything)."""
    command, _Config = _require_alembic()
    with _quiet_alembic_logging():
        command.downgrade(alembic_config(), revision)


def stamp(revision: str = HEAD) -> None:
    """Record ``revision`` as applied without running it.

    The escape hatch for adopting Alembic on a database that already has the
    schema (created by ``ensure_schema`` at startup).
    """
    command, _Config = _require_alembic()
    with _quiet_alembic_logging():
        command.stamp(alembic_config(), revision)


def heads() -> list[str]:
    """The current head revision(s)."""
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(alembic_config())
    return sorted({rev.revision for rev in script.get_revisions(HEAD)})


def applied_revision() -> str | None:
    """The revision recorded in the database, or None when never migrated.

    Read directly rather than parsed from ``alembic current`` output, so the
    answer is a value rather than a console string.

    ``apply_database_url()`` runs first: without it this would read the default
    SQLite file even when ``FIXIMG_DATABASE_URL`` points somewhere else — and a
    CLI invocation would then disagree with the migration it just ran.
    """
    from fiximg.infrastructure.db.engine import apply_database_url, get_conn

    apply_database_url()
    try:
        row = get_conn().execute(f"SELECT version_num FROM {VERSION_TABLE}").fetchone()
    except Exception:  # noqa: BLE001 — the table does not exist yet
        return None
    if row is None:
        return None
    return row[0] if not hasattr(row, "keys") else row["version_num"]


def history() -> list[dict]:
    """The revision chain, newest first."""
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(alembic_config())
    out: list[dict] = []
    for revision in script.walk_revisions():
        out.append(
            {
                "revision": revision.revision,
                "down_revision": revision.down_revision,
                "doc": (revision.doc or "").splitlines()[0] if revision.doc else "",
                "is_head": revision.is_head,
            }
        )
    return out


@dataclass
class MigrationStatus:
    """What :func:`status` reports."""

    applied: str | None = None
    heads: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)

    @property
    def is_current(self) -> bool:
        return bool(self.applied) and self.applied in self.heads

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "heads": self.heads,
            "pending": self.pending,
            "current": self.is_current,
        }


def status() -> MigrationStatus:
    """Applied revision, head revision(s) and what is still pending."""
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(alembic_config())
    head_revisions = heads()
    applied = applied_revision()

    if applied is None:
        pending = [rev.revision for rev in script.walk_revisions()]
    else:
        # Walk from the head *down to* the applied revision; everything on that
        # path except the applied revision itself is still pending.
        pending = [
            rev.revision
            for rev in script.walk_revisions(base=applied, head=HEAD)
            if rev.revision != applied
        ]
    return MigrationStatus(applied=applied, heads=head_revisions, pending=pending)


# --------------------------------------------------------- compatibility shim
class MigrationRunner:
    """Deprecated alias kept so existing callers keep working.

    V3's hand-rolled runner had this name. It now delegates to Alembic, so the
    ``conn`` argument is accepted and ignored — Alembic opens its own connection
    from the configured URL.
    """

    def __init__(self, conn: Any = None, versions_dir: str | None = None) -> None:
        self.conn = conn
        self.versions_dir = versions_dir or SCRIPT_LOCATION

    def run(self, dry_run: bool = False) -> list[str]:
        """Apply pending revisions; returns the revisions that were pending."""
        pending = status().pending
        if dry_run:
            return pending
        upgrade()
        return pending

    def status(self) -> dict:
        return status().to_dict()

    def applied(self) -> list[str]:
        current = applied_revision()
        return [current] if current else []


# ----------------------------------------------------------------------- CLI
def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Database migrations (Alembic)")
    parser.add_argument(
        "action",
        choices=("status", "upgrade", "downgrade", "history", "current", "stamp",
                 "check"),
        help="what to do",
    )
    parser.add_argument(
        "revision", nargs="?", default=None,
        help="target revision (default: head for upgrade, -1 for downgrade)",
    )
    parser.add_argument("--sql", action="store_true",
                        help="offline mode: print SQL instead of applying it")
    args = parser.parse_args(argv)

    if args.sql:
        command, _Config = _require_alembic()
        config = alembic_config()
        config.attributes["configure_logger"] = False
        command.upgrade(config, args.revision or HEAD, sql=True)
        return 0

    if args.action == "status":
        report = status()
        print(f"applied: {report.applied or '(none)'}")
        print(f"head:    {', '.join(report.heads) or '(none)'}")
        print(f"pending: {', '.join(report.pending) or '(none)'}")
        return 0

    if args.action == "upgrade":
        upgrade(args.revision or HEAD)
        print(f"upgraded to {args.revision or HEAD}")
        return 0

    if args.action == "downgrade":
        downgrade(args.revision or "-1")
        print(f"downgraded to {args.revision or '-1'}")
        return 0

    if args.action == "history":
        for entry in history():
            marker = " (head)" if entry["is_head"] else ""
            print(f"{entry['revision']}  <- {entry['down_revision'] or 'base'}{marker}")
            if entry["doc"]:
                print(f"          {entry['doc']}")
        return 0

    if args.action == "current":
        print(applied_revision() or "(none)")
        return 0

    if args.action == "check":
        return check_data_layer()

    stamp(args.revision or HEAD)
    print(f"stamped {args.revision or HEAD}")
    return 0


def check_data_layer() -> int:
    """Report rows still stored in a format the current code no longer writes.

    Exit code 1 means "this database is behind the code", which is what makes
    ``make db-check`` useful in a deploy script: an in-place upgrade that skips
    ``upgrade`` keeps working quietly for a while (the readers are tolerant) and
    then mis-detects an expired lease. See
    :func:`fiximg.infrastructure.db.repositories.task_repository.legacy_instant_rows`.
    """
    from fiximg.infrastructure.db.repositories import task_repository

    report = status()
    print(f"applied: {report.applied or '(none)'}")
    print(f"head:    {', '.join(report.heads) or '(none)'}")
    if report.pending:
        print(f"pending: {', '.join(report.pending)}")

    offenders = task_repository.legacy_instant_rows()
    if offenders:
        print("legacy-format rows (run `fiximg-db upgrade`):")
        for key, count in sorted(offenders.items()):
            print(f"  {key}: {count} row(s)")
        return 1
    print("instants: all canonical")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
