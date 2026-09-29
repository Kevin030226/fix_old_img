"""Unit tests for the V2 task repository (plan section 14)."""
import pytest

from fiximg.infrastructure.db.repositories import task_repository as repo


@pytest.fixture()
def db_env(tmp_path, monkeypatch):
    """Point the shared SQLite file at a temp directory for test isolation."""
    import fiximg.infrastructure.db.engine as legacy_db
    import fiximg.config as config_mod

    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(legacy_db, "DB_PATH", db_path)
    monkeypatch.setattr(legacy_db, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(legacy_db, "_conn", None)
    monkeypatch.setattr(config_mod.settings, "db_path", db_path)
    repo._DDL_DONE = False
    return db_path


def test_task_lifecycle(db_env):
    repo.create_task("t1", "restore", "alice")
    row = repo.get_task("t1")
    assert row["status"] == "queued"
    assert row["progress"] == 0

    repo.start_task("t1")
    repo.update_progress("t1", 40, "global_restore")
    row = repo.get_task("t1")
    assert row["status"] == "running"
    assert row["progress"] == 40
    assert row["current_stage"] == "global_restore"

    repo.record_stage("t1", 0, "global_restore", "running")
    repo.finish_stage("t1", 0, "completed", 1234, None)
    repo.start_task("t1")  # a terminal write only lands on a running row
    repo.finish_task("t1", "/tmp/out.png", "PSNR: 20", 5000)
    row = repo.get_task("t1")
    assert row["status"] == "completed"
    assert row["result_path"] == "/tmp/out.png"
    assert row["duration_ms"] == 5000
    stages = row["stages"]
    import json

    stages = json.loads(stages) if isinstance(stages, str) else stages
    assert stages[0]["stage_name"] == "global_restore"
    assert stages[0]["duration_ms"] == 1234


def test_failed_task(db_env):
    repo.create_task("t2", "colorize", "bob")
    repo.start_task("t2")  # a terminal write only lands on a running row
    repo.fail_task("t2", "boom", 100)
    row = repo.get_task("t2")
    assert row["status"] == "failed"
    assert row["error_message"] == "boom"


def test_cancel_reaches_anything_that_has_not_finished(db_env):
    """搂3.8: `running` is cancellable too, cooperatively.

    The stop is the worker's decision to make at its next stage boundary, but the row
    records the user's intent immediately, so the status view, the SSE stream and a
    later `finish_task` fence all agree about what this task is.
    """
    repo.create_task("t3", "restore", "carol")
    assert repo.cancel_task("t3") is True
    # Already cancelled -> not cancellable again.
    assert repo.cancel_task("t3") is False

    repo.create_task("t4", "restore", "carol")
    repo.start_task("t4")
    assert repo.cancel_task("t4") is True
    assert repo.get_task("t4")["status"] == "cancelled"

    repo.finish_task("t4", "/tmp/x.png", None, 5)  # fenced out, must not resurrect
    assert repo.get_task("t4")["status"] == "cancelled"
    assert repo.cancel_task("t4") is False

    repo.create_task("t5", "restore", "carol")
    assert repo.claim_next_task("w-5") is not None
    assert repo.fail_task("t5", "boom", 5, worker_id="w-5") is True
    assert repo.cancel_task("t5") is False, "a failed task is not cancellable"


def test_metrics_and_artifacts(db_env, tmp_path):
    repo.create_task("t5", "restore", "dave")
    repo.add_metric("t5", "psnr", 22.5)
    repo.add_metric("t5", "ssim", 0.8)
    metrics = repo.read_metrics("t5")
    # Numbers come back as numbers (plan 搂2.7) 鈥?the column is a REAL, so SQL can
    # average these without a Python round trip through the table.
    assert metrics["psnr"] == 22.5
    assert metrics["ssim"] == 0.8

    img_path = str(tmp_path / "out.png")
    from PIL import Image

    Image.new("RGB", (10, 10), (255, 0, 0)).save(img_path)
    repo.add_artifact("t5", "output", img_path, "image/png")
    arts = repo.get_artifacts("t5")
    assert len(arts) == 1
    assert arts[0]["width"] == 10 and arts[0]["height"] == 10
    assert arts[0]["size_bytes"] > 0


def test_a_retried_run_leaves_one_row_per_artifact_kind(db_env, tmp_path):
    """The run dir is keyed by task id, so a retry overwrites the previous bytes.

    Two `output` rows used to survive: the older one kept the *first* attempt's
    sha256 and dimensions for a path now holding the second attempt's file, so
    `GET /tasks/{id}/artifacts` could report a digest that did not match the image
    the client downloaded.
    """
    import hashlib

    from PIL import Image

    repo.create_task("t-retry", "restore", "dave")
    live = str(tmp_path / "final.png")
    Image.new("RGB", (8, 8), (255, 0, 0)).save(live)
    repo.add_artifact("t-retry", "output", live, "image/png")

    Image.new("RGB", (16, 12), (0, 0, 255)).save(live)
    repo.add_artifact("t-retry", "output", live, "image/png")

    rows = [r for r in repo.get_artifacts("t-retry") if r["kind"] == "output"]
    assert len(rows) == 1, f"{len(rows)} output rows for one artifact"
    with open(live, "rb") as handle:
        assert rows[0]["sha256"] == hashlib.sha256(handle.read()).hexdigest()
    assert (rows[0]["width"], rows[0]["height"]) == (16, 12)


def test_replacement_is_per_kind_so_the_registered_input_survives(db_env, tmp_path):
    """The input image is registered at submit and must not be re-added to work."""
    from PIL import Image

    repo.create_task("t-input", "restore", "dave")
    source = str(tmp_path / "in.png")
    Image.new("RGB", (4, 4), (0, 255, 0)).save(source)
    repo.add_artifact("t-input", "input", source, "image/png")
    repo.add_artifact("t-input", "input", source, "image/png")
    repo.add_artifact("t-input", "output", source, "image/png")

    assert sorted(r["kind"] for r in repo.get_artifacts("t-input")) == ["input", "output"]


def test_list_tasks_by_user(db_env):
    repo.create_task("t6", "restore", "eve")
    repo.create_task("t7", "colorize", "eve")
    repo.create_task("t8", "restore", "frank")
    eve = repo.list_tasks(user_id="eve")
    assert {r["id"] for r in eve} == {"t6", "t7"}
    assert len(repo.list_tasks(limit=10)) == 3


# ------------------------------------------------------- typed storage (搂2.7)
def test_claims_follow_enqueues_within_one_second(db_env):
    """FIFO is a property of the stored value, not of rowid luck.

    With second-granularity timestamps five tasks enqueued in a loop all
    carried the same ``created_at``, so ``ORDER BY priority, created_at`` could
    return them in any order the planner liked.
    """
    ids = [f"fifo-{i}" for i in range(5)]
    for task_id in ids:
        repo.create_task(task_id, "restore", "u")

    claimed = []
    for _ in ids:
        row = repo.claim_next_task("worker-a")
        assert row is not None, "every enqueued task must be claimable"
        claimed.append(row["id"])
        repo.fail_task(row["id"], "forced", 1)
    assert claimed == ids


def test_created_at_values_are_canonical_and_order_is_stable(db_env, monkeypatch):
    """What the instant format guarantees 鈥?and what it does not (plan 搂2.7).

    The format is microsecond *precision*, not microsecond *uniqueness*: measured on
    this machine, CPython 3.11 gives one stamp shared by four inserts in a tight loop
    (Windows `datetime.now()` advances in ~0.5 ms steps) while 3.14 gives four distinct
    ones. Asserting uniqueness therefore passed on the CPU-only dev interpreter and
    failed the first time the suite ran in the environment the project deploys to.

    The two claims that hold everywhere are the canonical shape (fixed width, `Z`,
    no local-zone dependency) and a deterministic listing order. The second needs the
    `id` tiebreaker in `list_tasks` *because* equal stamps happen 鈥?so the collision is
    produced here by freezing the clock instead of being left to whichever clock the
    test machine happens to have.
    """
    moments = []
    for i in range(4):
        task_id = f"ord-{i}"
        repo.create_task(task_id, "restore", "u")
        moments.append(repo.get_task(task_id)["created_at"])

    for moment in moments:
        assert len(moment) == 27 and moment.endswith("Z"), moment
    assert moments == sorted(moments), "a later insert must never sort earlier"

    from fiximg.infrastructure.db import timestamps

    monkeypatch.setattr(timestamps, "now", lambda: "2026-01-01T00:00:00.000000Z")
    for i in range(4):
        repo.create_task(f"tie-{i}", "restore", "u")

    listed = [row["id"] for row in repo.list_tasks(limit=50)]
    ties = [t for t in listed if t.startswith("tie-")]
    assert ties == ["tie-3", "tie-2", "tie-1", "tie-0"], (
        f"four rows with the same created_at listed without a stable order: {ties}"
    )


def test_metric_values_keep_their_type(db_env):
    repo.create_task("typed", "restore", "u")
    repo.add_metric("typed", "psnr", 22.5)
    repo.add_metric("typed", "face_count", 3)
    repo.add_metric("typed", "note", "skipped")
    repo.add_metric("typed", "psnr_inf", float("inf"))

    metrics = repo.read_metrics("typed")
    assert metrics["psnr"] == 22.5
    assert metrics["face_count"] == 3.0  # a REAL column: ints come back as floats
    assert metrics["note"] == "skipped"
    # +inf cannot be a REAL (and cannot be emitted as JSON), so it is text.
    assert metrics["psnr_inf"] == "inf"


def test_get_task_serialises_metrics_as_numbers(db_env):
    import json

    repo.create_task("jsonm", "restore", "u")
    repo.add_metric("jsonm", "psnr", 31.25)
    repo.add_metric("jsonm", "mode", "quality")
    row = repo.get_task("jsonm")
    assert json.loads(row["metrics"]) == {"psnr": 31.25, "mode": "quality"}


@pytest.mark.parametrize(
    ("value", "number", "text"),
    [
        (22.5, 22.5, None),
        (3, 3.0, None),
        ("22.5", 22.5, None),          # V1 history stored numbers as text
        ("N/A", None, "N/A"),
        ("", None, None),
        (None, None, None),
        (float("inf"), None, "inf"),
        (float("nan"), None, "nan"),
        (True, 1.0, None),
        ({"nested": 1}, None, "{'nested': 1}"),
    ],
)
def test_split_metric(value, number, text):
    assert repo.split_metric(value) == (number, text)


def test_legacy_instant_rows_flags_an_unmigrated_database(db_env):
    """The readers are tolerant, so "behind" is invisible without this count.

    A database that never got revision 0003 mixes naive-local and canonical UTC
    values; the check exists so an operator can see that instead of discovering it
    as a task requeued while its worker was still running it.
    """
    import sqlite3

    repo.create_task("fresh", "restore", "u")
    assert repo.legacy_instant_rows() == {}

    conn = sqlite3.connect(db_env)
    conn.execute(
        "UPDATE tasks SET created_at='2026-06-01 08:00:00', "
        "lease_until='2026-06-01 08:05:00' WHERE id='fresh'"
    )
    conn.commit()
    conn.close()

    assert repo.legacy_instant_rows() == {
        "tasks.created_at": 1,
        "tasks.lease_until": 1,
    }


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 7, 10, 21])
def test_stage_percentiles_are_nearest_rank(db_env, count):
    """The percentile moved into SQL, so its *definition* needs pinning.

    `rank = ceil(p * n)` clamped to the rows that exist, over the ascending
    durations 鈥?which is what the replaced Python code did for n=3 and does not
    for even n, where it took the upper of the two middle values.
    """
    import math

    durations = [(i * 37) % 500 + 1 for i in range(count)]
    for i, duration in enumerate(durations):
        task_id = f"p{i}"
        repo.create_task(task_id, "restore", "u")
        repo.record_stage(task_id, 0, "global_restore", "running")
        repo.finish_stage(task_id, 0, "completed", duration)

    ordered = sorted(durations)
    stats = repo.task_stats()["stages"]["global_restore"]
    assert stats["runs"] == count
    assert stats["avg_ms"] == int(round(sum(ordered) / count))
    for fraction, key in ((0.5, "p50_ms"), (0.95, "p95_ms")):
        rank = min(count, max(1, math.ceil(count * fraction)))
        assert stats[key] == ordered[rank - 1], f"n={count} p={fraction}"


def test_repository_and_benchmark_report_the_same_percentile(db_env):
    """The SQL percentile and the benchmark helper must pick the same row.

    Two tools in this repo compute p50/p95 鈥?the repository in SQL for the stats
    endpoint, `benchmark.common` for latency tables. They are compared here
    because a "p95" that means one thing in the dashboard and another in the
    benchmark is worse than no p95 at all.
    """
    from benchmark.common import Samples

    durations = [(i * 53) % 900 + 1 for i in range(17)]
    for i, duration in enumerate(durations):
        task_id = f"agree-{i}"
        repo.create_task(task_id, "restore", "u")
        repo.record_stage(task_id, 0, "global_restore", "running")
        repo.finish_stage(task_id, 0, "completed", duration)

    # `Samples` is unit-agnostic, so both sides get milliseconds and the numbers
    # must be equal 鈥?not merely close.
    stats = repo.task_stats()["stages"]["global_restore"]
    samples = Samples(name="x", values=[float(d) for d in durations]).summary()
    assert stats["runs"] == samples["count"] == 17
    for key in ("p50", "p95"):
        assert stats[f"{key}_ms"] == round(samples[key]), (key, stats, samples)
