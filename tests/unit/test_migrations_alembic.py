"""Alembic migration tests (plan 搂4.3 Step 2).

What is pinned:

* ``upgrade head`` on an empty database produces the full schema,
* ``downgrade base`` removes it and ``upgrade head`` works again (the round trip
  that makes a migration set trustworthy),
* the URL comes from ``FIXIMG_DATABASE_URL``, not from ``alembic.ini`` 鈥?so a
  migration can never run against a different database than the service,
* an unsupported scheme fails here exactly as it fails at startup,
* ``stamp`` adopts a database whose schema was created by ``ensure_schema``.

Skipped when alembic/SQLAlchemy are not installed, so a minimal deployment can
still run the rest of the suite.
"""
import os
import subprocess
import sys

import pytest

from fiximg.infrastructure.db.migrations import runner as migration_runner


def _has_alembic() -> bool:
    try:
        import alembic  # noqa: F401
        import sqlalchemy  # noqa: F401

        return True
    except ImportError:
        return False


requires_alembic = pytest.mark.skipif(
    not _has_alembic(), reason="alembic / SQLAlchemy not installed"
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Tables the migrations must create 鈥?derived from the DDL the runtime ships.
#:
#: Listing them by hand could only ever record what someone remembered, which is
#: the failure this is meant to catch: a table added to a `_SCHEMA` constant
#: without a revision that creates it leaves every *fresh* database correct and
#: every existing one missing, silently, until a query hits it.
def _shipped_tables() -> set[str]:
    import re

    from fiximg.infrastructure.db.engine import _SCHEMA as app_schema
    from fiximg.infrastructure.db.repositories.task_repository import (
        _SCHEMA as task_schema,
    )

    return set(
        re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", app_schema + task_schema)
    )


EXPECTED_TABLES = _shipped_tables()

#: The floor the derivation must clear. If a parser change or a renamed constant
#: ever yields an empty set, `EXPECTED_TABLES <= tables` would pass vacuously.
_MINIMUM_TABLES = {
    "users", "history",
    "tasks", "task_stages", "artifacts", "metrics", "system_events", "model_versions",
}


@pytest.fixture()
def migrated_db(tmp_path, monkeypatch):
    """A temp SQLite database wired into the engine and the migration runner."""
    import fiximg.infrastructure.db.engine as engine
    from fiximg.infrastructure.db.repositories import task_repository

    db_path = tmp_path / "migrated.db"
    monkeypatch.setenv("FIXIMG_DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setattr(engine, "DB_PATH", str(db_path))
    monkeypatch.setattr(engine, "ADMIN_DATA_DIR", str(tmp_path))
    # Every cached handle, not just engine._conn: a thread-local one left over
    # from an earlier test keeps pointing at that test's database, and the
    # migration then runs against a file this test is not looking at.
    engine.close_connections()
    # Same reasoning for the repository's schema latch 鈥?it is keyed per
    # database, but a test that never built the schema here must not inherit
    # another test's "already done".
    monkeypatch.setattr(task_repository, "_DDL_DONE", None)
    yield db_path
    engine.close_connections()


def _tables(db_path) -> set[str]:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


def _columns(db_path, table: str) -> set[str]:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


# -------------------------------------------------------------- the round trip
def test_the_expected_table_set_is_derived_and_non_trivial():
    """The derivation is not allowed to quietly return nothing."""
    assert _MINIMUM_TABLES <= EXPECTED_TABLES, _MINIMUM_TABLES - EXPECTED_TABLES


@requires_alembic
def test_upgrade_head_creates_the_whole_schema(migrated_db):
    migration_runner.upgrade()
    tables = _tables(migrated_db)
    assert EXPECTED_TABLES <= tables, EXPECTED_TABLES - tables
    assert migration_runner.VERSION_TABLE in tables


@requires_alembic
def test_upgrade_adds_the_post_v2_columns(migrated_db):
    migration_runner.upgrade()
    assert "error_code" in _columns(migrated_db, "tasks")
    assert "required_capabilities" in _columns(migrated_db, "tasks")
    assert "metrics_json" in _columns(migrated_db, "task_stages")


@requires_alembic
def test_downgrade_to_base_removes_everything(migrated_db):
    migration_runner.upgrade()
    migration_runner.downgrade("base")

    tables = _tables(migrated_db)
    for table in EXPECTED_TABLES:
        assert table not in tables, f"{table} survived a downgrade"


@requires_alembic
def test_upgrade_downgrade_upgrade_is_repeatable(migrated_db):
    """The round trip must be clean, or a rollback leaves a broken database."""
    migration_runner.upgrade()
    migration_runner.downgrade("base")
    migration_runner.upgrade()
    assert EXPECTED_TABLES <= _tables(migrated_db)


@requires_alembic
def test_downgrade_one_step_reverts_only_the_tip(migrated_db):
    migration_runner.upgrade()
    before = _tables(migrated_db)
    migration_runner.downgrade("-1")

    tables = _tables(migrated_db)
    assert "tasks" in tables                     # the baseline is still applied
    assert "error_code" in _columns(migrated_db, "tasks")   # 0002 survives
    assert "metric_text" in _columns(migrated_db, "metrics")  # 0003 survives
    assert _is_float_type(_types_of(migrated_db, "metrics")["metric_value"])
    # Only the tip's own object goes away. Read from the tip's file rather than
    # named here, so this stays a test of "one step reverts one step" instead of a
    # note about which revision happens to be on top today.
    assert migration_runner.applied_revision() == PREVIOUS_REVISION
    tip_table = _revision_objects(HEAD_REVISION).get("table")
    assert tip_table, f"revision {HEAD_REVISION} owns no table; update this test"
    assert tip_table not in tables, before - tables
    assert before - tables == {tip_table}, before - tables
    assert "uri" in _columns(migrated_db, "artifacts")


@requires_alembic
def test_upgrade_is_idempotent(migrated_db):
    migration_runner.upgrade()
    first = _tables(migrated_db)
    migration_runner.upgrade()
    assert _tables(migrated_db) == first


@requires_alembic
def test_a_database_created_before_the_tip_revision_gains_its_object(migrated_db):
    """The tip's own DDL is only ever *run* on an existing database.

    A fresh database gets the table from 0001's shipped DDL, so every other test in
    this file would pass even if the new revision created nothing at all. This is the
    upgrade an operator in place actually experiences: schema from yesterday, revision
    from today.
    """
    migration_runner.upgrade()
    import sqlite3

    conn = sqlite3.connect(str(migrated_db))
    try:
        tip_table = _revision_objects(HEAD_REVISION).get("table")
        assert tip_table, f"revision {HEAD_REVISION} owns no table"
        conn.execute(f"DROP TABLE {tip_table}")
        conn.commit()
    finally:
        conn.close()
    migration_runner.stamp(PREVIOUS_REVISION)
    assert tip_table not in _tables(migrated_db)

    migration_runner.upgrade()

    assert tip_table in _tables(migrated_db), "the tip revision created nothing"
    assert migration_runner.applied_revision() == HEAD_REVISION


# ------------------------------------------------------------------- status
@requires_alembic
def test_status_before_any_migration(migrated_db):
    report = migration_runner.status()
    assert report.applied is None
    assert report.is_current is False
    assert report.pending, "a fresh database has every revision pending"


@requires_alembic
def test_status_after_migration_is_current(migrated_db):
    migration_runner.upgrade()
    report = migration_runner.status()
    assert report.applied in report.heads
    assert report.is_current is True
    assert report.pending == []


@requires_alembic
def test_status_is_json_friendly(migrated_db):
    migration_runner.upgrade()
    payload = migration_runner.status().to_dict()
    assert set(payload) == {"applied", "heads", "pending", "current"}
    assert payload["current"] is True


@requires_alembic
def test_applied_revision_is_none_without_the_version_table(migrated_db):
    assert migration_runner.applied_revision() is None


def _revision_sources() -> dict[str, str]:
    """``{revision id: file text}`` for every file in `migrations/versions/`."""
    import pathlib
    import re

    directory = pathlib.Path(PROJECT_ROOT) / "migrations" / "versions"
    found: dict[str, str] = {}
    for path in directory.glob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        revision = re.search(r'^revision = "([^"]+)"', text, re.M)
        if revision is not None:
            found[revision.group(1)] = text
    return found


def _revision_objects(revision: str) -> dict[str, str]:
    """What one revision owns, read from its own `_TABLE` / `_COLUMN` constants."""
    import re

    text = _revision_sources().get(revision, "")
    out: dict[str, str] = {}
    for key, pattern in (("table", r'^_TABLE = "([^"]+)"'),
                         ("column", r'^_COLUMN = "([^"]+)"')):
        match = re.search(pattern, text, re.M)
        if match is not None:
            out[key] = match.group(1)
    return out


def _chain_from_files() -> list[str]:
    """The revision ids read out of `migrations/versions/`, oldest first.

    A separate source on purpose. Six assertions in this file used to name the tip
    literally, so adding 0005 broke all of them at once 鈥?which is the same drift
    the migrations themselves are for: the expectations had to be remembered
    instead of derived.
    """
    import re

    parents: dict[str, str | None] = {}
    for revision, text in _revision_sources().items():
        down = re.search(r'^down_revision = (?:None|"([^"]+)"|\'([^\']+)\')', text, re.M)
        parents[revision] = (down.group(1) or down.group(2)) if down else None

    root = [name for name, down in parents.items() if down is None]
    assert len(root) == 1, f"expected one migration root, found {root}"
    ordered = [root[0]]
    while len(ordered) < len(parents):
        nxt = [n for n, d in parents.items() if d == ordered[-1] and n not in ordered]
        assert len(nxt) == 1, f"branch or gap after {ordered[-1]}: {nxt}"
        ordered.append(nxt[0])
    return ordered


HEAD_REVISION = _chain_from_files()[-1]
PREVIOUS_REVISION = _chain_from_files()[-2]

# ------------------------------------------------------------------ history
@requires_alembic
def test_history_lists_the_chain_oldest_to_newest_heads_first(migrated_db):
    chain = migration_runner.history()
    revisions = [entry["revision"] for entry in chain]

    assert revisions == list(reversed(_chain_from_files()))  # newest-first
    by_revision = {entry["revision"]: entry for entry in chain}
    ordered = _chain_from_files()
    for index, revision in enumerate(ordered):
        expected_previous = ordered[index - 1] if index else None
        assert by_revision[revision]["down_revision"] == expected_previous
    assert by_revision[HEAD_REVISION]["is_head"] is True
    assert by_revision[ordered[0]]["is_head"] is False


@requires_alembic
def test_heads_reports_the_tip(migrated_db):
    assert migration_runner.heads() == [HEAD_REVISION]


@requires_alembic
def test_history_documents_each_revision(migrated_db):
    for entry in migration_runner.history():
        assert entry["doc"], f"revision {entry['revision']} has no summary line"


# --------------------------------------------------------------------- stamp
@requires_alembic
def test_stamp_adopts_an_existing_database(migrated_db):
    """A database created by `ensure_schema` predates Alembic; stamp adopts it."""
    import fiximg.infrastructure.db.engine as engine
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    engine.init_db()
    task_repo.ensure_schema()
    assert "tasks" in _tables(migrated_db)

    migration_runner.stamp()
    report = migration_runner.status()
    assert report.applied == HEAD_REVISION
    assert report.is_current is True


# --------------------------------------------------------------- URL handling
@requires_alembic
def test_the_url_comes_from_the_environment_not_the_ini(migrated_db):
    config = migration_runner.alembic_config()
    configured = config.get_main_option("sqlalchemy.url")
    assert str(migrated_db).replace("\\", "/") in configured


@requires_alembic
def test_an_unservable_url_fails_at_migration_time(monkeypatch):
    """Same fail-fast as startup: no silent write to the wrong database."""
    from fiximg.infrastructure.db.engine import UnsupportedDatabaseError

    monkeypatch.setenv("FIXIMG_DATABASE_URL", "mysql://user:pw@localhost/fiximg")
    with pytest.raises(UnsupportedDatabaseError):
        migration_runner.alembic_config()


@requires_alembic
def test_a_postgres_url_reaches_alembic_with_the_driver_spelling(monkeypatch):
    """Migrations run against the same database the service uses (plan 搂4.3).

    PostgreSQL is a served target now, so the assertion is not "it refuses" but
    "it names the same database in the spelling SQLAlchemy's engine wants" 鈥?
    otherwise ``alembic upgrade`` would migrate one database while the app opens
    another.
    """
    monkeypatch.setenv("FIXIMG_DATABASE_URL", "postgresql://user:pw@localhost/fiximg")
    url = migration_runner.alembic_config().get_main_option("sqlalchemy.url")
    assert url == "postgresql+psycopg://user:pw@localhost/fiximg"


@requires_alembic
def test_memory_database_url_is_supported(monkeypatch):
    monkeypatch.setenv("FIXIMG_DATABASE_URL", "sqlite:///:memory:")
    assert migration_runner.alembic_config().get_main_option("sqlalchemy.url") == "sqlite://"


# ------------------------------------------------------------- compatibility
@requires_alembic
def test_the_legacy_runner_api_still_works(migrated_db):
    """`MigrationRunner` was the V3 API; it now delegates to Alembic."""
    legacy = migration_runner.MigrationRunner(conn=None)
    pending = legacy.run(dry_run=True)
    assert pending, "a fresh database reports pending revisions"
    legacy.run()
    assert legacy.status()["current"] is True
    assert legacy.applied() == [HEAD_REVISION]


@requires_alembic
def test_dry_run_does_not_touch_the_database(migrated_db):
    migration_runner.MigrationRunner().run(dry_run=True)
    assert migration_runner.applied_revision() is None


# ------------------------------------------------------------------- CLI
@requires_alembic
def test_cli_status_reports_pending(migrated_db):
    result = subprocess.run(
        [sys.executable, "-m", "fiximg.infrastructure.db.migrations.runner", "status"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
        env={**os.environ, "FIXIMG_DATABASE_URL": f"sqlite:///{migrated_db.as_posix()}",
             "PYTHONPATH": os.path.join(PROJECT_ROOT, "src")},
    )
    assert result.returncode == 0, result.stderr
    assert "applied: (none)" in result.stdout
    assert "pending:" in result.stdout


@requires_alembic
def test_cli_upgrade_then_current(migrated_db):
    env = {**os.environ, "FIXIMG_DATABASE_URL": f"sqlite:///{migrated_db.as_posix()}",
           "PYTHONPATH": os.path.join(PROJECT_ROOT, "src")}

    upgraded = subprocess.run(
        [sys.executable, "-m", "fiximg.infrastructure.db.migrations.runner", "upgrade"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, env=env,
    )
    assert upgraded.returncode == 0, upgraded.stderr

    current = subprocess.run(
        [sys.executable, "-m", "fiximg.infrastructure.db.migrations.runner", "current"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, env=env,
    )
    assert current.stdout.strip() == HEAD_REVISION


@requires_alembic
def test_cli_history_prints_the_chain(migrated_db):
    result = subprocess.run(
        [sys.executable, "-m", "fiximg.infrastructure.db.migrations.runner", "history"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
        env={**os.environ, "FIXIMG_DATABASE_URL": f"sqlite:///{migrated_db.as_posix()}",
             "PYTHONPATH": os.path.join(PROJECT_ROOT, "src")},
    )
    assert result.returncode == 0, result.stderr
    assert "0001" in result.stdout
    assert "0002" in result.stdout
    assert "(head)" in result.stdout


@requires_alembic
def test_offline_sql_mode_emits_ddl_without_connecting(migrated_db):
    """`--sql` is what makes a migration reviewable before it runs."""
    result = subprocess.run(
        [sys.executable, "-m", "fiximg.infrastructure.db.migrations.runner",
         "upgrade", "--sql"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
        env={**os.environ, "FIXIMG_DATABASE_URL": f"sqlite:///{migrated_db.as_posix()}",
             "PYTHONPATH": os.path.join(PROJECT_ROOT, "src")},
    )
    assert result.returncode == 0, result.stderr
    assert "CREATE TABLE" in result.stdout.upper()
    # Nothing was applied.
    assert migration_runner.applied_revision() is None


# --------------------------------------------------------- shipped artefacts
def test_alembic_ini_and_env_are_committed():
    assert os.path.exists(os.path.join(PROJECT_ROOT, "alembic.ini"))
    assert os.path.exists(os.path.join(PROJECT_ROOT, "migrations", "env.py"))
    assert os.path.exists(os.path.join(PROJECT_ROOT, "migrations", "script.py.mako"))


def test_env_does_not_autogenerate():
    """Autogenerate cannot reproduce the SQLite-specific DDL; it is disabled."""
    with open(os.path.join(PROJECT_ROOT, "migrations", "env.py"), encoding="utf-8") as handle:
        source = handle.read()
    assert "target_metadata = None" in source


def test_revisions_are_numbered_and_ordered():
    versions = os.path.join(PROJECT_ROOT, "migrations", "versions")
    files = sorted(f for f in os.listdir(versions) if f.endswith(".py"))
    assert files, "no revisions found"
    assert all(f[:4].isdigit() for f in files), files


# ------------------------------------------------------------ revision 0003
#: The pre-0003 shape: instants as naive local strings, metrics as text. Only the
#: columns 0003 touches are here 鈥?the point is the *values*, not a second copy
#: of the schema.
_LEGACY_DDL = """
CREATE TABLE tasks (
    id TEXT PRIMARY KEY, user_id TEXT, task_type TEXT, status TEXT,
    progress INTEGER, created_at TEXT NOT NULL, started_at TEXT,
    finished_at TEXT, lease_until TEXT, last_heartbeat TEXT, retry_at TEXT
);
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, metric_name TEXT,
    metric_value TEXT, reference_type TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE users (
    username TEXT PRIMARY KEY, password TEXT NOT NULL, role TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT
);
"""

#: What V2's `datetime.now().strftime("%Y-%m-%d %H:%M:%S")` produced.
LEGACY_INSTANT = "2026-06-01 08:00:00"


def _seed_legacy(db_path) -> None:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.executescript(_LEGACY_DDL)
    conn.execute(
        "INSERT INTO tasks(id, user_id, task_type, status, progress, created_at,"
        " started_at, lease_until, retry_at) VALUES (?,?,?,?,?,?,?,?,?)",
        ("t1", "u", "restore", "running", 40, LEGACY_INSTANT, LEGACY_INSTANT,
         "2026-06-01 08:05:00", "2026-06-01 08:01:00"),
    )
    conn.executemany(
        "INSERT INTO metrics(task_id, metric_name, metric_value, reference_type,"
        " created_at) VALUES (?,?,?,?,?)",
        [
            ("t1", "psnr", "22.5", "x", LEGACY_INSTANT),
            ("t1", "psnr_inf", "inf", "x", LEGACY_INSTANT),
            ("t1", "mode", "quality", "x", LEGACY_INSTANT),
            ("t1", "nothing", None, "x", LEGACY_INSTANT),
        ],
    )
    conn.execute(
        "INSERT INTO users(username,password,role,created_at,updated_at)"
        " VALUES ('u','h','user',?,?)", (LEGACY_INSTANT, None)
    )
    conn.commit()
    conn.close()


def _rows(db_path, sql: str):
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql)]
    finally:
        conn.close()


@requires_alembic
def test_0003_rewrites_legacy_instants_to_utc(migrated_db):
    """Same instant, now expressible: zone-carrying, microsecond, sortable.

    The expected value is built with the stdlib and a literal format string, not
    with the codec the application uses 鈥?otherwise this test could only ever
    agree with itself.
    """
    from datetime import UTC, datetime

    _seed_legacy(migrated_db)
    migration_runner.stamp("0002")
    migration_runner.upgrade()

    row = _rows(migrated_db, "SELECT * FROM tasks")[0]
    expected = datetime.fromisoformat(LEGACY_INSTANT).astimezone(UTC).strftime(
        "%Y-%m-%dT%H:%M:%S.000000Z"
    )
    assert row["created_at"] == expected
    assert len(row["created_at"]) == 27 and row["created_at"].endswith("Z")
    assert row["lease_until"].endswith("Z") and row["retry_at"].endswith("Z")
    # A NULL stays NULL: nothing invented a deadline.
    assert _rows(migrated_db, "SELECT updated_at FROM users")[0]["updated_at"] is None
    assert _rows(migrated_db, "SELECT created_at FROM metrics LIMIT 1")[0][
        "created_at"].endswith("Z")


@requires_alembic
def test_0003_types_metrics_and_keeps_what_a_real_cannot_hold(migrated_db):
    _seed_legacy(migrated_db)
    migration_runner.stamp("0002")
    migration_runner.upgrade()

    by_name = {r["metric_name"]: r for r in _rows(
        migrated_db, "SELECT metric_name, metric_value, typeof(metric_value) AS vtype,"
        " metric_text FROM metrics"
    )}
    assert by_name["psnr"]["metric_value"] == 22.5
    assert by_name["psnr"]["vtype"] == "real"
    # +inf and labels move to metric_text *before* the cast, so they survive.
    assert by_name["psnr_inf"]["metric_value"] is None
    assert by_name["psnr_inf"]["metric_text"] == "inf"
    assert by_name["mode"]["metric_text"] == "quality"
    assert by_name["nothing"]["metric_value"] is None
    assert by_name["nothing"]["metric_text"] is None
    assert "metric_value" in _columns(migrated_db, "metrics")
    assert _is_float_type(_types_of(migrated_db, "metrics")["metric_value"])


def _types_of(db_path, table: str) -> dict:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        return {r[1]: r[2] for r in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


#: SQLite spells a float column REAL or DOUBLE depending on whether it came from
#: DDL or from a batch rebuild; PostgreSQL says DOUBLE PRECISION. All of them are
#: the numeric affinity the repositories now write into.
FLOAT_TYPE_NAMES = ("REAL", "DOUBLE", "FLOAT")


def _is_float_type(name: str) -> bool:
    upper = (name or "").upper()
    return any(token in upper for token in FLOAT_TYPE_NAMES)


@requires_alembic
def test_0003_downgrade_is_the_inverse(migrated_db):
    """Roll back, then forward again: the values must land where they started."""
    _seed_legacy(migrated_db)
    migration_runner.stamp("0002")
    migration_runner.upgrade()
    upgraded = _rows(migrated_db, "SELECT created_at FROM tasks")
    assert upgraded[0]["created_at"].endswith("Z")

    migration_runner.downgrade("0002")   # "-1" would only undo the tip (0004)
    back = _rows(migrated_db, "SELECT * FROM tasks")[0]
    assert back["created_at"] == LEGACY_INSTANT
    assert back["lease_until"] == "2026-06-01 08:05:00"
    names = {r["metric_name"]: r["metric_value"] for r in _rows(
        migrated_db, "SELECT metric_name, metric_value FROM metrics")}
    assert names["psnr"] == "22.5"
    assert names["psnr_inf"] == "inf"
    assert "metric_text" not in _columns(migrated_db, "metrics")
    assert migration_runner.applied_revision() == "0002"

    migration_runner.upgrade()
    again = _rows(migrated_db, "SELECT created_at FROM tasks")[0]
    assert again == upgraded[0], "a second upgrade must not shift the instant again"


@requires_alembic
def test_0003_is_a_no_op_on_a_database_that_never_had_legacy_rows(migrated_db):
    """0001 already ships the new shape, so head must not disturb a fresh DB."""
    migration_runner.upgrade()
    assert migration_runner.applied_revision() == HEAD_REVISION
    assert _is_float_type(_types_of(migrated_db, "metrics")["metric_value"])
    # An already-canonical value is left exactly as it was.
    from fiximg.infrastructure.db import timestamps

    import sqlite3

    conn = sqlite3.connect(str(migrated_db))
    stamp = timestamps.now()
    conn.execute(
        "INSERT INTO tasks(id,user_id,task_type,status,progress,created_at)"
        " VALUES ('t9','u','restore','queued',0,?)", (stamp,)
    )
    conn.commit()
    conn.close()
    migration_runner.upgrade()  # already at head; nothing to run
    assert _rows(migrated_db, "SELECT created_at FROM tasks WHERE id='t9'")[0][
        "created_at"] == stamp


@requires_alembic
def test_check_data_layer_passes_a_current_database(migrated_db):
    migration_runner.upgrade()
    assert migration_runner.check_data_layer() == 0


@requires_alembic
def test_check_data_layer_fails_a_database_behind_the_code(migrated_db, capsys):
    """The exit code is the point: a deploy script can gate on it."""
    _seed_legacy(migrated_db)
    migration_runner.stamp("0002")
    assert migration_runner.check_data_layer() == 1
    printed = capsys.readouterr().out
    assert "tasks.created_at" in printed
    assert "fiximg-db upgrade" in printed

    migration_runner.upgrade()
    assert migration_runner.check_data_layer() == 0


def test_running_migrations_does_not_silence_the_application(isolated_db, tmp_path):
    """The alembic env used to call `fileConfig` with its default arguments.

    That disables every logger created beforehand, so a process which ran a
    migration programmatically kept working and served requests 鈥?with no logs. It
    also poisoned the test session: any later test that asserted on an emitted log
    line saw nothing, and only when the migration tests happened to run first.
    """
    import logging

    from fiximg.infrastructure.db.migrations import runner

    # A logger created *before* the migration runs, like every application logger is.
    probe = logging.getLogger("fiximg.probe.for-this-test")
    runner.upgrade()
    assert not probe.disabled, "running migrations disabled pre-existing application loggers"
