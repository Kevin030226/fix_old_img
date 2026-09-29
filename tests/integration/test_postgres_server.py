"""The PostgreSQL data path against a real server (plan 搂4.3 Step 2, 搂3.10.1).

The other PostgreSQL tests cannot answer the question this one asks.
``test_sql_dialect.py`` compiles SQL text; ``test_postgres_data_path.py`` drives a
fake driver that answers exactly the statements the code sends 鈥?which by
construction cannot reject a statement the server dislikes, a type the server
enforces, or a keyword the server reserves. Running against PostgreSQL 18.6 found
four defects none of them could see:

* migration 0001 executed the shipped DDL verbatim, so ``INTEGER PRIMARY KEY
  AUTOINCREMENT`` was a syntax error and a fresh PostgreSQL database had no schema
  at all;
* the legacy ``history`` table's ``user`` column is a reserved word in PostgreSQL;
* ``SELECT COUNT(*)`` read with ``row[0]`` raises ``KeyError`` under psycopg's
  ``dict_row`` (and the fake supported positional access, so it never showed);
* ``COALESCE(metric_value, metric_text)`` is a ``DatatypeMismatch`` once
  ``metric_value`` is DOUBLE PRECISION, and the stage subquery called the aggregate
  ``json_object_agg`` with SQLite's variadic ``json_object`` arity.

Point it at a **scratch** database 鈥?it writes rows and does not clean them up:

    docker compose -f docker/compose.yaml --profile postgres up -d
    FIXIMG_TEST_POSTGRES_URL=postgresql://fiximg:fiximg@localhost:5432/fiximg_test \
        python -m pytest tests/integration/test_postgres_server.py

Without the variable every test skips, and the reason names the variable, because a
silently-skipped server tier is what let those four defects through.
"""
import json
import os
import threading
import uuid

import pytest

from fiximg.domain.tasks import decode_json_column
from fiximg.infrastructure.db import engine, timestamps
from fiximg.infrastructure.db.repositories import task_repository as repo

PG_URL = (os.environ.get("FIXIMG_TEST_POSTGRES_URL") or "").strip()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not PG_URL,
        reason="set FIXIMG_TEST_POSTGRES_URL to a scratch PostgreSQL database "
               "(the fake-driver tests cannot replace this tier)",
    ),
    pytest.mark.skipif(
        not PG_URL.startswith(("postgresql://", "postgresql+psycopg://")),
        reason=f"FIXIMG_TEST_POSTGRES_URL is not a PostgreSQL URL: {PG_URL!r}",
    ),
]


@pytest.fixture(autouse=True)
def pg_database(monkeypatch):
    """Point the engine at the scratch server for one test, then close handles."""
    monkeypatch.setenv("FIXIMG_DATABASE_URL", PG_URL)
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "database_url", PG_URL)
    engine.close_connections()
    engine.reset_thread_connection_cache()
    repo._DDL_DONE = False
    engine.init_db()
    yield PG_URL
    engine.close_connections()
    repo._DDL_DONE = False


@pytest.fixture()
def tag():
    """One run's namespace, so concurrent or repeated runs cannot collide."""
    return uuid.uuid4().hex[:8]


def _user(tag: str) -> str:
    username = f"pg-{tag}"
    engine.add_user(username, "hash-1", "user")
    return username


# ------------------------------------------------------- the legacy tables
def test_reserved_word_column_and_unique_username(tag):
    """`history.user` is a PostgreSQL keyword, and the username race is an error."""
    username = f"alice-{tag}"
    assert engine.add_user(username, "hash-1", "user") is True
    assert engine.add_user(username, "hash-2", "user") is False, (
        "a duplicate must come back as False via IntegrityError, not as an exception"
    )
    assert (engine.get_user(username) or {}).get("role") == "user"

    record_id = f"h-{tag}"
    # The V1 `history` table has no writer in V2/V3 (every result is a `tasks`
    # row), so the row is inserted here rather than through the engine. That is
    # also the point: the read path must survive a row that only a V1 install
    # could have produced.
    engine.get_conn().execute(
        'INSERT INTO history(id, timestamp, "user", type, input_path, output_path,'
        " psnr, ssim, mae) VALUES (?,?,?,?,?,?,?,?,?)",
        (record_id, timestamps.now(), username, "restore", "/in.png", "/out.png",
         "23.5", "0.9", "1.2"),
    )
    engine.get_conn().commit()
    rows = [r for r in engine.list_history(200) if r["id"] == record_id]
    assert len(rows) == 1, rows
    assert rows[0]["user"] == username, "the quoted `user` column must read back"


def test_legacy_import_guards_count_rows_by_name(tmp_path, monkeypatch, tag):
    """`SELECT COUNT(*)` has no key under dict_row, so the guards must alias it.

    The legacy importers check "is the table empty?" before parsing files. Written
    as ``fetchone()[0]`` that raises ``KeyError`` on the first boot against
    PostgreSQL, and the fake driver used to answer positional access so the tests
    passed. An empty legacy directory keeps the guards on their fast path.
    """
    monkeypatch.setattr(engine, "LEGACY_USERS_YAML", str(tmp_path / "absent.yaml"))
    monkeypatch.setattr(engine, "LEGACY_HISTORY_FILE", str(tmp_path / "absent.txt"))
    engine._migrate_users(engine.get_conn())
    engine._migrate_history(engine.get_conn())


# ------------------------------------------------- typed rows and aggregates
def test_metrics_keep_their_type_through_the_aggregate(tag):
    """23.5 must come back as a number and "inf" as a string, on both engines."""
    username = _user(tag)
    task_id = f"m-{tag}"
    repo.create_task(task_id, "restore", username)
    repo.add_metric(task_id, "psnr", 23.5)
    repo.add_metric(task_id, "note", "inf")

    assert repo.read_metrics(task_id) == {"psnr": 23.5, "note": "inf"}
    row = repo.get_task(task_id)
    metrics = row["metrics"]
    if isinstance(metrics, str):  # SQLite hands back the JSON text
        metrics = json.loads(metrics)
    assert metrics == {"psnr": 23.5, "note": "inf"}, metrics
    assert isinstance(metrics["psnr"], float), "cast to text would make this a string"


def test_stage_checklist_aggregates_into_objects(tag):
    username = _user(tag)
    task_id = f"s-{tag}"
    repo.create_task(task_id, "restore", username)
    repo.start_task(task_id)
    repo.record_stage(task_id, 0, "global_restore", "running")
    repo.finish_stage(task_id, 0, "completed", 900, None)
    repo.add_metric(task_id, "ssim", 0.91)
    repo.add_artifact(task_id, "output", "/out.png")
    repo.finish_task(task_id, "/out.png", "PSNR 23.5", 1500)

    row = repo.get_task(task_id)
    stages = decode_json_column(row["stages"], [])
    assert [s["stage_name"] for s in stages] == ["global_restore"], stages
    assert stages[0]["duration_ms"] == 900, stages
    assert [a["kind"] for a in repo.get_artifacts(task_id)] == ["output"]

    listed = [r for r in repo.list_tasks(limit=200) if r["id"] == task_id]
    assert len(listed) == 1
    same = decode_json_column(listed[0]["stages"], [])
    assert same == stages, "list_tasks must not disagree with get_task"


def test_the_capability_filter_holds_on_the_shape_this_engine_returns(tag):
    """搂4.3 Step 4 measured where only a server can measure it.

    `required_capabilities` is JSON text on SQLite and an already-decoded list here, and
    the repository read it with ``json.loads`` inside a ``try``. On this engine that call
    raised ``TypeError``, the handler returned an empty set, and an empty set is a subset
    of everything 鈥?so a colour-only worker claimed restoration tasks it could not
    finish. No SQLite test could see it.
    """
    username = _user(tag)
    restore_id = f"cap-{tag}-restore"
    colorize_id = f"cap-{tag}-colorize"
    #: Both above everything else this module queued (priority 0), and the refused one
    #: first: with equal priority the claim order is `created_at`, so a filter that
    #: ignored the hint would hand back `restore_id` and the assertion is unambiguous
    #: even though other tests left rows in the same queue.
    repo.create_task(restore_id, "restore", username,
                     required_capabilities=["restore"], priority=50)
    repo.create_task(colorize_id, "colorize", username,
                     required_capabilities=["colorize"], priority=50)

    claimed = repo.claim_next_task(f"color-only-{tag}", capabilities={"colorize"})
    assert claimed is not None, "the colour worker must find its own task"
    assert claimed["id"] == colorize_id, (
        f"claimed {claimed['id']}: the capability hint was read as empty"
    )
    assert repo.get_task(restore_id)["status"] == "queued", (
        "a task needing a capability this worker does not serve stays in the queue"
    )


def test_stats_percentiles_run_on_the_server(tag):
    """The server-computed percentiles must equal the same ranks computed here.

    The scratch database accumulates rows across runs, so an absolute expectation
    would be wrong; the durable property is that the SQL
    (``_ceil_rank`` + ``ROW_NUMBER``) and an independent nearest-rank calculation
    over the same rows agree. That is what "the statistics were computed in the
    database" has to mean before it can be believed.
    """
    username = _user(tag)
    for order, duration in enumerate((100, 200, 300)):
        task_id = f"p-{tag}-{order}"
        repo.create_task(task_id, "restore", username)
        repo.start_task(task_id)
        repo.record_stage(task_id, 0, "global_restore", "running")
        repo.finish_stage(task_id, 0, "completed", duration, None)
        repo.finish_task(task_id, "/o.png", None, duration, worker_id=None)

    stats = repo.task_stats(window_days=0)
    stage = stats["stages"].get("global_restore")
    assert stage and {"p50_ms", "p95_ms", "runs", "avg_ms"} <= set(stage), stats["stages"]

    conn = engine.get_conn()
    rows = conn.execute(
        "SELECT duration_ms FROM task_stages WHERE stage_name=? AND status='completed' "
        "AND duration_ms IS NOT NULL ORDER BY duration_ms",
        ("global_restore",),
    ).fetchall()
    durations = sorted(int(r["duration_ms"]) for r in rows)
    assert len(durations) == stage["runs"], (len(durations), stage["runs"])

    def nearest_rank(frac: float) -> int:
        import math

        index = max(1, min(len(durations), math.ceil(frac * len(durations))))
        return durations[index - 1]

    assert stage["p50_ms"] == nearest_rank(0.50), (stage, durations[:5], len(durations))
    assert stage["p95_ms"] == nearest_rank(0.95), (stage, durations[:5], len(durations))
    assert stage["p50_ms"] <= stage["p95_ms"]
    assert stats["tasks"]["total"] >= 1


# ----------------------------------------------------------- the queue
def test_claim_lease_and_fence(tag):
    username = _user(tag)
    ids = [f"q-{tag}-{i}" for i in range(3)]
    for task_id in ids:
        repo.create_task(task_id, "restore", username)

    worker_a, worker_b = f"a-{tag}", f"b-{tag}"
    first = repo.claim_next_task(worker_a)
    second = repo.claim_next_task(worker_b)
    assert first and second and first["id"] != second["id"], "claims must be distinct"
    assert first["worker_id"] == worker_a, "the claim stamps its owner"

    before = repo.get_task(first["id"])["lease_until"]
    repo.heartbeat_task(first["id"], 36000)
    after = repo.get_task(first["id"])["lease_until"]
    assert after > before, f"heartbeat did not move the lease: {before} -> {after}"

    assert repo.finish_task(first["id"], "/o.png", None, 10, worker_id=worker_b) is False, (
        "a worker that does not own the row cannot close it"
    )
    assert repo.finish_task(first["id"], "/o.png", None, 10, worker_id=worker_a) is True
    assert repo.fail_task(first["id"], "late", 10, worker_id=worker_a) is False, (
        "a terminal row stays terminal"
    )


def test_concurrent_workers_never_claim_the_same_task(tag):
    """The point of `FOR UPDATE SKIP LOCKED`, which only a server can answer."""
    username = _user(tag)
    ids = [f"c-{tag}-{i}" for i in range(20)]
    for task_id in ids:
        repo.create_task(task_id, "restore", username)

    claimed: list[str] = []
    lock = threading.Lock()

    def grab(worker: str) -> None:
        engine.reset_thread_connection_cache()
        while True:
            row = repo.claim_next_task(worker)
            if row is None:
                return
            with lock:
                claimed.append(row["id"])

    threads = [threading.Thread(target=grab, args=(f"{tag}-{n}",)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    mine = [c for c in claimed if c in ids]
    assert sorted(mine) == sorted(ids), (
        f"{len(mine)} claims / {len(set(mine))} distinct for {len(ids)} tasks"
    )


def test_events_store_their_payload(tag):
    username = _user(tag)
    task_id = f"e-{tag}"
    repo.create_task(task_id, "restore", username)
    repo.add_event("task.probe", "info", task_id=task_id,
                   message="probe event", data={"k": 1})
    events = repo.list_task_events(task_id)
    assert any(e["message"] == "probe event" for e in events), events


def test_the_new_statistics_blocks_survive_the_server(tag):
    """`users`, `by_type` and `metric_averages` are what the admin panel and `/stats` now show.

    Every expected value here is isolated by a name unique to this run -- a task type and
    metric names -- because the scratch database accumulates rows across runs and an
    absolute count over a shared table would pass by accident. The three properties that a
    plausible wrong implementation must violate:

    * `by_type` groups, and does not merely count (a missing GROUP BY yields one row);
    * `users` counts *distinct* accounts, checked against an independent count;
    * a metric that has no number half contributes no average -- `+inf` PSNR after a
      bit-identical pair is stored as text, and reading that text as a number would
      report an infinite mean instead of no mean.
    """
    first, second = _user(tag), _user(tag)
    task_type = f"stats-{tag}"
    numeric, textual = f"psnr-{tag}", f"infinite-{tag}"
    repo.create_task(f"s-{tag}-1", task_type, first)
    repo.add_metric(f"s-{tag}-1", numeric, 20.0)
    repo.create_task(f"s-{tag}-2", task_type, second)
    repo.add_metric(f"s-{tag}-2", numeric, 30.0)
    repo.add_metric(f"s-{tag}-2", textual, "inf")

    stats = repo.task_stats(window_days=0)

    assert stats["by_type"][task_type] == 2, stats["by_type"]
    # 20.0 and 30.0 are the only rows carrying this metric name.
    assert stats["metric_averages"][numeric] == 25.0, stats["metric_averages"]
    assert isinstance(stats["metric_averages"][numeric], float)
    assert repo.read_metrics(f"s-{tag}-2")[textual] == "inf"
    assert textual not in stats["metric_averages"], stats["metric_averages"]

    counted = engine.get_conn().execute(
        "SELECT COUNT(DISTINCT user_id) AS users FROM tasks"
    ).fetchone()["users"]
    assert stats["tasks"]["users"] == int(counted), (stats["tasks"]["users"], counted)
    # The window's task total must be the same population `by_type` partitioned.
    assert sum(stats["by_type"].values()) == stats["tasks"]["total"], stats

    import json

    json.dumps(stats)
