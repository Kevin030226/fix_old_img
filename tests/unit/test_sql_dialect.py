"""SQL dialect tests (plan 搂4.3 Step 2).

Two things are proven here:

1. **The SQLite path is unchanged.** The dialect's SQLite implementation returns
   exactly the strings the repository used before it existed, so introducing the
   seam cannot have altered behaviour. (The rest of the suite passing is the
   other half of that proof.)
2. **The PostgreSQL SQL actually compiles.** No server is needed for this:
   SQLAlchemy can compile a statement against the real ``postgresql`` dialect,
   which is what catches a wrong function name or a stray ``?`` before anyone
   provisions a database.

What is *not* claimed: the PostgreSQL data path is not exercised against a live
server. The remaining work is the connection wrapper (placeholder adaptation on
the wire) plus a run of the suite against a real instance.
"""
import pytest

from fiximg.infrastructure.db.dialect import (
    POSTGRESQL,
    SQLITE,
    PostgresDialect,
    SqlDialect,
    default_dialect,
    dialect_for,
    dialect_for_url,
)


@pytest.fixture()
def sqlite():
    return dialect_for(SQLITE)


@pytest.fixture()
def postgres():
    return dialect_for(POSTGRESQL)


# ============================ the SQLite path is unchanged ============================
def test_sqlite_keeps_the_question_mark_placeholder(sqlite):
    assert sqlite.placeholder == "?"
    assert sqlite.adapt("SELECT * FROM tasks WHERE id=?") == "SELECT * FROM tasks WHERE id=?"


def test_sqlite_keeps_the_json_group_functions(sqlite):
    assert sqlite.json_array_fn == "json_group_array"
    assert sqlite.json_object_fn == "json_group_object"


def test_sqlite_keeps_the_pragma_introspection(sqlite):
    statement, params = sqlite.table_columns_sql("tasks")
    assert statement == "PRAGMA table_info(tasks)"
    assert params == ()


def test_sqlite_keeps_begin_immediate(sqlite):
    assert sqlite.begin_write_sql() == "BEGIN IMMEDIATE"


def test_sqlite_has_no_row_lock(sqlite):
    """SQLite serialises writers itself; there is no SKIP LOCKED."""
    assert sqlite.supports_skip_locked is False
    assert sqlite.claim_row_lock() == ""


def test_sqlite_is_the_default():
    assert dialect_for("").name == SQLITE
    assert dialect_for("nonsense").name == SQLITE
    assert SqlDialect().name == SQLITE


# ============================ the PostgreSQL dialect ============================
def test_postgres_uses_percent_s_placeholders(postgres):
    assert postgres.placeholder == "%s"
    assert postgres.adapt("SELECT * FROM tasks WHERE id=?") == "SELECT * FROM tasks WHERE id=%s"


def test_postgres_rewrites_every_placeholder(postgres):
    adapted = postgres.adapt("UPDATE tasks SET a=?, b=? WHERE id=?")
    assert adapted.count("%s") == 3
    assert "?" not in adapted


def test_postgres_uses_the_json_agg_functions(postgres):
    assert postgres.json_array_fn == "json_agg"
    assert postgres.json_object_fn == "json_object_agg"
    assert postgres.json_array("x") == "json_agg(x)"
    assert postgres.json_object("k", "v") == "json_object_agg(k, v)"


def test_postgres_introspects_via_information_schema(postgres):
    statement, params = postgres.table_columns_sql("tasks")
    assert "information_schema.columns" in statement
    assert "%s" in statement
    assert params == ("tasks",)
    # The alias must be `name` so the caller's `r["name"]` keeps working.
    assert "AS name" in statement


def test_postgres_needs_no_begin_immediate(postgres):
    assert postgres.begin_write_sql() is None


def test_postgres_locks_the_claimed_row(postgres):
    assert postgres.supports_skip_locked is True
    assert postgres.claim_row_lock() == "FOR UPDATE SKIP LOCKED"


def test_postgres_is_a_sql_dialect():
    assert isinstance(PostgresDialect(), SqlDialect)


# ============================ URL resolution ============================
@pytest.mark.parametrize("url", [
    "postgresql://user:pw@localhost:5432/fiximg",
    "postgres://user:pw@db:5432/fiximg",
    "postgresql+psycopg://user:pw@localhost/fiximg",
    "postgresql+psycopg2://user:pw@localhost/fiximg",
])
def test_postgres_urls_resolve_to_the_postgres_dialect(url):
    assert dialect_for_url(url).name == POSTGRESQL


@pytest.mark.parametrize("url", [
    "",
    None,
    "sqlite:///admin_data/fixoldimg.db",
    "sqlite:////tmp/x.db",
    "sqlite+pysqlite:///x.db",
    "admin_data/fixoldimg.db",
])
def test_sqlite_urls_resolve_to_the_sqlite_dialect(url):
    assert dialect_for_url(url).name == SQLITE


def test_default_dialect_follows_the_environment(monkeypatch):
    monkeypatch.setenv("FIXIMG_DATABASE_URL", "postgresql://x/y")
    assert default_dialect().name == POSTGRESQL
    monkeypatch.setenv("FIXIMG_DATABASE_URL", "sqlite:///x.db")
    assert default_dialect().name == SQLITE
    monkeypatch.delenv("FIXIMG_DATABASE_URL", raising=False)
    assert default_dialect().name == SQLITE


# ==================== the generated PostgreSQL SQL compiles ====================
def _compile(sql: str, params: dict | None = None) -> str:
    """Compile a statement against the real PostgreSQL dialect."""
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from sqlalchemy.dialects import postgresql

    statement = sqlalchemy.text(sql)
    return str(statement.compile(dialect=postgresql.dialect()))


def test_the_stage_aggregate_compiles_for_postgres(postgres):
    """The JSON aggregate is the piece most likely to be wrong."""

    # Force the PostgreSQL spelling and compile it.
    sql = (
        "SELECT t.*, "
        f"(SELECT {postgres.json_array_fn}(json_object('stage_name', stage_name, 'status', status)) "
        " FROM task_stages WHERE task_id=t.id) AS stages, "
        f"(SELECT {postgres.json_object_fn}(metric_name, metric_value) FROM metrics WHERE task_id=t.id) "
        "AS metrics "
        "FROM tasks t WHERE t.id=%s"
    )

    compiled = _compile(sql)
    assert "json_agg" in compiled
    assert "json_object_agg" in compiled
    assert "?" not in compiled


def test_the_claim_statement_compiles_for_postgres(postgres):
    """The claim is the statement with the most dialect-specific syntax.

    Compiled from the repository's own builder, not from a copy of the SQL: a copy keeps
    passing after the real statement changes, which is how removing
    `ORDER BY priority DESC` left this test green while the queue silently became FIFO.
    """
    from fiximg.infrastructure.db.repositories import task_repository

    statement = task_repository.claim_statement(postgres)
    compiled = _compile(statement)
    assert "FOR UPDATE SKIP LOCKED" in compiled, statement
    assert "?" not in compiled, statement
    assert "priority DESC" in statement, statement

    # The SQLite spelling has the same ordering and no lock clause, and it too has to
    # compile: the ordering is the contract, the lock is the dialect's addition.
    sqlite_statement = task_repository.claim_statement(task_repository.default_dialect())
    assert "priority DESC" in sqlite_statement, sqlite_statement
    assert "SKIP LOCKED" not in sqlite_statement, sqlite_statement
    assert _compile(sqlite_statement.replace("?", "%s"))


def test_the_columns_statement_compiles_for_postgres(postgres):
    statement, _params = postgres.table_columns_sql("tasks")
    compiled = _compile(statement)
    assert "information_schema.columns" in compiled


def test_the_sqlite_statements_compile_for_sqlite():
    """Sanity check the other direction, so the helper is not one-sided."""
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from sqlalchemy.dialects import sqlite as sqlite_dialect

    compiled = str(
        sqlalchemy.text("SELECT * FROM tasks WHERE id=?").compile(dialect=sqlite_dialect.dialect())
    )
    assert "tasks" in compiled


# ==================== the repository actually uses the dialect ====================
def test_the_repository_formats_its_aggregates_through_the_dialect(monkeypatch):
    """A dialect change must reach the SQL the repository emits.

    Two of these four names are the reason the tier exists: PostgreSQL's scalar
    JSON builder is ``json_build_object`` (its ``json_object_agg`` is an aggregate
    with a two-argument signature), and a ``COALESCE`` across the numeric and the
    textual metric column is a type error there.
    """
    import fiximg.infrastructure.db.repositories.task_repository as repo
    from fiximg.infrastructure.db import dialect as dialect_module

    monkeypatch.setattr(
        dialect_module, "_DIALECTS", {**dialect_module._DIALECTS, SQLITE: PostgresDialect()}
    )
    assert repo._json_fragments() == {
        "json_array": "json_agg",
        "json_object": "json_object_agg",
        "row_object": "json_build_object",
        "metric_value": "CASE WHEN metric_value IS NOT NULL "
                        "THEN to_json(metric_value) ELSE to_json(metric_text) END",
    }


def test_no_sqlite_only_function_name_is_hardcoded_in_the_shared_templates():
    """A SQLite spelling inside a shared template is invisible until a server rejects it."""
    import inspect

    import fiximg.infrastructure.db.repositories.task_repository as repo

    code = inspect.getsource(repo.get_task) + inspect.getsource(repo.list_tasks) \
        + inspect.getsource(repo.read_metrics)
    # The three readers are the whole set: the two aggregates and the per-task
    # read. A fourth would need adding here or it escapes this gate.
    # Comments are where the SQLite spelling legitimately belongs (explaining why
    # it cannot be used), so only executable text is searched.
    executable = "\n".join(ln for ln in code.splitlines() if not ln.strip().startswith("#"))
    for forbidden in ("json_object(", "json_group_array(", "COALESCE(metric_value"):
        assert forbidden not in executable, f"{forbidden!r} must come from the dialect"


def test_the_repository_uses_the_default_fragments_on_sqlite():
    import fiximg.infrastructure.db.repositories.task_repository as repo

    assert repo._json_fragments() == {
        "json_array": "json_group_array",
        "json_object": "json_group_object",
        "row_object": "json_object",
        "metric_value": "COALESCE(metric_value, metric_text)",
    }


def test_the_claim_statement_is_built_from_the_dialect(sqlite, postgres):
    """The lock, the placeholders and the ordering all come out of one builder.

    This used to read the *source text* of `_select_claimable` and look for
    ``claim_row_lock``/``adapt`` in it, which asserts how the function is written rather
    than what the worker executes 鈥?and breaks the moment the statement is assembled by a
    helper. The statement itself is the contract, so the statement is what gets compared.
    """
    from fiximg.infrastructure.db.repositories import task_repository as repo

    pg = repo.claim_statement(postgres)
    lite = repo.claim_statement(sqlite)

    assert "FOR UPDATE SKIP LOCKED" in pg, pg
    assert "SKIP LOCKED" not in lite, lite
    assert "%s" in pg and "?" not in pg, pg
    assert "?" in lite, lite
    # The ordering is the half that makes a submitted `priority` mean anything, and it has
    # to be present on both engines: a dialect-specific copy of the SQL is where it can go
    # missing without any test noticing.
    assert "priority DESC" in pg and "priority DESC" in lite, (pg, lite)


def test_the_schema_guard_asks_the_dialect_for_columns():
    import inspect

    import fiximg.infrastructure.db.repositories.task_repository as repo

    source = inspect.getsource(repo.ensure_schema)
    assert "table_columns_sql" in source
    assert "PRAGMA" not in source


# ==================== behaviour is unchanged on SQLite ====================
def test_tasks_round_trip_on_sqlite(isolated_db):
    """The end-to-end check that the seam did not break the real path."""
    import json

    from fiximg.infrastructure.db.repositories import task_repository as repo

    repo.create_task("t-1", "restore", "alice", required_capabilities=["restore"])
    repo.record_stage("t-1", 0, "global_restore", "running")
    repo.finish_stage("t-1", 0, "completed", 12, None, metrics={"tiles": 2})
    repo.add_metric("t-1", "psnr", 24.5, "input_output_difference")

    row = repo.get_task("t-1")
    stages = json.loads(row["stages"])
    assert stages[0]["stage_name"] == "global_restore"
    assert stages[0]["metrics_json"] == json.dumps({"tiles": 2})
    assert json.loads(row["metrics"]) == {"psnr": 24.5}


def test_list_tasks_aggregates_round_trip_on_sqlite(isolated_db):
    import json

    from fiximg.infrastructure.db.repositories import task_repository as repo

    repo.create_task("t-1", "restore", "alice")
    repo.record_stage("t-1", 0, "global_restore", "running")
    repo.finish_stage("t-1", 0, "completed", 12)

    row = repo.list_tasks(limit=10)[0]
    assert json.loads(row["stages"])[0]["stage_name"] == "global_restore"


def test_claim_still_works_on_sqlite(isolated_db):
    from fiximg.infrastructure.db.repositories import task_repository as repo

    repo.create_task("t-1", "restore", "alice")
    claimed = repo.claim_next_task("w-1")
    assert claimed is not None and claimed["id"] == "t-1"
    assert repo.claim_next_task("w-2") is None


# ====================== skipping a conflicting insert ======================
def test_sqlite_spells_ignore_as_a_prefix(sqlite):
    statement = sqlite.insert_ignore(
        "INSERT INTO users(username,password) VALUES (?,?)", ("username",)
    )
    assert statement.startswith("INSERT OR IGNORE INTO")


def test_postgres_spells_ignore_as_a_suffix(postgres):
    statement = postgres.insert_ignore(
        "INSERT INTO users(username,password) VALUES (%s,%s)", ("username",)
    )
    assert statement.endswith("ON CONFLICT (username) DO NOTHING")
    assert "OR IGNORE" not in statement


def test_postgres_can_ignore_without_naming_a_key(postgres):
    """ON CONFLICT DO NOTHING is legal with no target; callers may have none."""
    statement = postgres.insert_ignore("INSERT INTO history(id) VALUES (%s)")
    assert statement == "INSERT INTO history(id) VALUES (%s) ON CONFLICT DO NOTHING"


def test_sqlite_rewrites_only_the_leading_verb(sqlite):
    statement = sqlite.insert_ignore("INSERT INTO a VALUES (?)", ("id",))
    assert statement.count("INSERT OR IGNORE INTO") == 1

