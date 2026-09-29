"""Queue reliability tests (plan 搂2.6).

Pins the V3 additions: atomic claim with attempt counting, lease expiry
recovery, retry with backoff, idempotency keys, priority ordering and the
`QueueBackend` protocol boundary.
"""
import json
import sqlite3
import time

import pytest

from fiximg.infrastructure.db.repositories import task_repository as repo
from fiximg.infrastructure.queue import SqliteQueueBackend, get_queue, reset_queue

# Cross-layer: the queue semantics run against a real SQLite database.
pytestmark = pytest.mark.integration


@pytest.fixture()
def queue(isolated_db):
    return SqliteQueueBackend()


def _create(task_id="t-1", **kwargs):
    repo.create_task(task_id, "restore", "alice", **kwargs)


# --------------------------------------------------------------------- claim
def test_claim_sets_lease_worker_and_attempt(isolated_db, queue):
    _create("t-1")
    task = queue.claim("w-1", lease_seconds=60)

    assert task is not None
    assert task.id == "t-1"
    assert task.status == "running"
    assert task.worker_id == "w-1"
    assert task.attempt_count == 1
    assert task.lease_until is not None

    row = repo.get_task("t-1")
    assert row["worker_id"] == "w-1"
    assert row["attempt_count"] == 1
    assert row["last_heartbeat"] is not None


def test_claim_returns_none_when_queue_is_empty(isolated_db, queue):
    assert queue.claim("w-1") is None


def test_claim_increments_attempt_on_each_attempt(isolated_db, queue):
    _create("t-1")
    assert queue.claim("w-1").attempt_count == 1
    assert repo.retry_task("t-1", delay_seconds=0) is True
    assert queue.claim("w-1").attempt_count == 2


def test_claim_honours_priority_then_age(isolated_db, queue):
    """Ids are chosen so the `id` tie-break cannot imitate the priority order.

    Rows inserted within one clock tick share a `created_at`, so the claim's
    `ORDER BY priority DESC, created_at, id` falls through to `id`; ids whose alphabetical
    order matches the intended claim order would pass even with `priority DESC` removed.
    """
    _create("a-low", priority=0)
    _create("z-high", priority=10)
    _create("m-mid", priority=5)

    assert queue.claim("w-1").id == "z-high"
    assert queue.claim("w-1").id == "m-mid"
    assert queue.claim("w-1").id == "a-low"


def test_a_submission_can_actually_set_the_priority_the_queue_orders_by(
    isolated_db, monkeypatch, queue
):
    """The column the queue orders by has to be reachable from a submission.

    `tasks.priority` existed with its index and its `ORDER BY`, and every test that
    exercised it wrote rows through `repo.create_task(priority=鈥?` 鈥?a producer no client
    can reach. The production path (`TaskService.enqueue`) had no parameter for it, so
    every task the API or the UI ever submitted said 0 and the ordering sorted a column
    nothing varied, while `docs/architecture.md` advertised priority ordering as something
    consumers get.

    Ids and submission order are chosen so that *neither* insertion order nor the `id`
    tie-break can produce the asserted result: rows inserted in one clock tick share
    `created_at` on this machine, and the urgent task is submitted second with the id that
    sorts last. Only `priority DESC` can put it first.
    """
    from PIL import Image

    from fiximg.application.task_service import task_service
    from fiximg.config import settings
    from fiximg.domain.tasks import decode_json_column

    monkeypatch.setattr(settings, "tasks_root", str(isolated_db / "tasks"))
    monkeypatch.setattr(settings, "inline_worker", False, raising=False)
    image = Image.new("RGB", (8, 8), (10, 20, 30))

    assert task_service.submit_queued(
        "a-ordinary", image, {"username": "alice"}, "restore"
    ) is True
    assert task_service.submit_queued(
        "z-urgent", image, {"username": "bob"}, "restore", priority=7
    ) is True

    assert repo.get_task("a-ordinary")["priority"] == 0, "the default stays FIFO"
    assert repo.get_task("z-urgent")["priority"] == 7, "the submission did not reach the row"

    claimed = queue.claim("w-1")
    assert claimed.id == "z-urgent", f"the submitted priority was ignored: {claimed.id}"
    assert claimed.priority == 7, "the worker's own view keeps the position it was claimed on"
    assert queue.claim("w-1").id == "a-ordinary"

    # The event the operator reads carries the number too, or a queue jump is invisible in
    # the audit trail whose job is to explain queue order.
    enqueued = [
        decode_json_column(event["data_json"], {})
        for event in repo.list_events(limit=200) if event["task_id"] == "z-urgent"
    ]
    assert any(data and data.get("priority") == 7 for data in enqueued), enqueued


def test_the_service_reports_no_position_for_a_row_it_cannot_read(isolated_db):
    """`task_priority()` answers None rather than inventing a 0.

    `POST /tasks` replies with this value, and the reply is documented as "the position
    actually stored". A submission path that could not read the row back (a service
    replaced by a recorder, a task written to another node's database) must say
    `null`, because "FIFO" is a claim about the queue and a guess is not that.
    """
    from fiximg.application.task_service import task_service

    assert task_service.task_priority("no-such-task") is None

    _create("plain")
    _create("ranked", priority=4)
    assert task_service.task_priority("plain") == 0
    assert task_service.task_priority("ranked") == 4


def test_claim_skips_tasks_inside_retry_backoff(isolated_db, queue):
    _create("t-1")
    queue.claim("w-1")
    assert repo.retry_task("t-1", delay_seconds=300) is True

    # Still inside the backoff window -> not eligible yet.
    assert queue.claim("w-1") is None
    row = repo.get_task("t-1")
    assert row["status"] == "queued"
    assert row["retry_at"] is not None

    # A zero-delay retry is immediately eligible again.
    repo.retry_task("t-1", delay_seconds=0)
    assert queue.claim("w-1") is not None


def test_claim_is_atomic_across_two_consumers(isolated_db, queue):
    _create("t-1")
    first = queue.claim("w-1")
    second = queue.claim("w-2")
    assert first is not None
    assert second is None


# ------------------------------------------------------------------- retry
def test_retry_stops_at_max_attempts(isolated_db):
    _create("t-1", max_attempts=2)
    assert repo.retry_task("t-1", delay_seconds=0) is True   # attempt 0 -> can retry
    repo.claim_next_task("w-1")                              # attempt 1
    assert repo.retry_task("t-1", delay_seconds=0) is True
    repo.claim_next_task("w-1")                              # attempt 2 == max
    assert repo.retry_task("t-1", delay_seconds=0) is False


def test_retry_clears_running_state(isolated_db, queue):
    _create("t-1")
    queue.claim("w-1")
    repo.retry_task("t-1", delay_seconds=0)

    row = repo.get_task("t-1")
    assert row["status"] == "queued"
    assert row["worker_id"] is None
    assert row["lease_until"] is None
    assert row["started_at"] is None
    assert row["finished_at"] is None
    assert row["progress"] == 0


def test_retry_unknown_task_returns_false(isolated_db):
    assert repo.retry_task("missing") is False


# ---------------------------------------------------------------- heartbeat
def test_heartbeat_extends_the_lease(isolated_db, queue):
    _create("t-1")
    queue.claim("w-1", lease_seconds=1)
    before = repo.get_task("t-1")["lease_until"]

    time.sleep(1.1)
    queue.heartbeat("t-1")
    after = repo.get_task("t-1")["lease_until"]
    assert after > before


# ------------------------------------------------------------- lease expiry
def test_expired_lease_is_requeued(isolated_db, queue):
    _create("t-1")
    queue.claim("w-1", lease_seconds=0)
    time.sleep(1.1)

    requeued = repo.reset_stale_running(timeout_seconds=0)
    assert requeued == 1
    row = repo.get_task("t-1")
    assert row["status"] == "queued"
    assert row["worker_id"] is None


def test_reset_stale_falls_back_to_started_at_for_legacy_rows(isolated_db):
    """A row claimed before the lease columns existed must still be recovered."""
    _create("t-1")
    conn = repo._get_conn()
    conn.execute(
        "UPDATE tasks SET status='running', started_at=?, lease_until=NULL WHERE id=?",
        ("2000-01-01 00:00:00", "t-1"),
    )
    conn.commit()

    assert repo.reset_stale_running(timeout_seconds=60) == 1
    assert repo.get_task("t-1")["status"] == "queued"


def test_reset_stale_ignores_live_tasks(isolated_db, queue):
    _create("t-1")
    queue.claim("w-1", lease_seconds=3600)
    assert repo.reset_stale_running(timeout_seconds=3600) == 0


# ------------------------------------------------------------- idempotency
def test_idempotency_key_rejects_a_second_insert(isolated_db):
    _create("t-1", idempotency_key="key-1")
    with pytest.raises(sqlite3.IntegrityError):
        _create("t-2", idempotency_key="key-1")


def test_find_by_idempotency_key(isolated_db):
    assert repo.find_by_idempotency_key("key-1") is None
    _create("t-1", idempotency_key="key-1")
    found = repo.find_by_idempotency_key("key-1")
    assert found["id"] == "t-1"
    # An empty key must not match unrelated rows.
    assert repo.find_by_idempotency_key("") is None


def test_null_idempotency_keys_do_not_collide(isolated_db):
    _create("t-1")
    _create("t-2")  # must not raise: the unique index is partial
    assert repo.queued_count() == 2


# ------------------------------------------------------------------ manual retry
def test_manual_retry_resets_attempts_and_state(isolated_db):
    _create("t-1", max_attempts=1)
    repo.claim_next_task("w-1")
    repo.fail_task("t-1", "boom", 10)
    assert repo.retry_task("t-1") is False          # exhausted

    assert repo.requeue_for_manual_retry("t-1") is True
    row = repo.get_task("t-1")
    assert row["status"] == "queued"
    assert row["attempt_count"] == 0
    assert row["error_message"] is None


def test_manual_retry_refuses_active_tasks(isolated_db, queue):
    _create("t-1")
    queue.claim("w-1")
    assert repo.requeue_for_manual_retry("t-1") is False


# ------------------------------------------------------------------ protocol
def test_sqlite_backend_satisfies_the_queue_protocol(isolated_db):
    from fiximg.infrastructure.queue.base import QueueBackend

    assert isinstance(SqliteQueueBackend(), QueueBackend)


def test_get_queue_defaults_to_sqlite_and_caches(isolated_db):
    reset_queue()
    first = get_queue()
    second = get_queue()
    assert isinstance(first, SqliteQueueBackend)
    assert first is second
    assert first.kind == "sqlite"
    reset_queue()


def test_queue_depth_counts_queued_rows(isolated_db, queue):
    _create("t-1")
    _create("t-2")
    assert queue.depth() == 2
    queue.claim("w-1")
    assert queue.depth() == 1


def test_cancelling_reaches_every_task_that_has_not_finished(isolated_db, queue):
    """The transport follows the row: queued *and* running are cancellable (搂3.8).

    A running task is cancelled cooperatively 鈥?the worker stops at the next stage
    boundary 鈥?but the user's intent is recorded immediately, so `queue.cancel` must
    not answer "no" just because a worker already holds the row. Only a finished task
    refuses.
    """
    _create("t-1")
    assert queue.cancel("t-1") is True
    assert repo.get_task("t-1")["status"] == "cancelled"

    _create("t-2")
    queue.claim("w-1")
    assert queue.cancel("t-2") is True
    assert repo.get_task("t-2")["status"] == "cancelled"

    _create("t-3")
    queue.claim("w-2")
    assert repo.finish_task("t-3", "/tmp/out.png", None, 5, worker_id="w-2") is True
    assert queue.cancel("t-3") is False


# ------------------------------------------------------------------- events
def test_task_events_are_ordered_and_filterable(isolated_db):
    _create("t-1")
    repo.add_event("task.enqueued", task_id="t-1", message="queued")
    repo.add_event("stage.started", task_id="t-1", message="global_restore",
                   data={"stage": "global_restore"})
    repo.add_event("other", task_id="t-2", message="noise")

    events = repo.list_task_events("t-1")
    assert [e["event_type"] for e in events] == ["task.enqueued", "stage.started"]
    assert json.loads(events[1]["data_json"]) == {"stage": "global_restore"}

    # Incremental replay (SSE cursor semantics).
    assert len(repo.list_task_events("t-1", after_id=events[0]["id"])) == 1


# ------------------------------------------------- terminal writes are fenced
def test_a_stalled_worker_cannot_complete_a_task_another_took_over(isolated_db):
    """The lease is only worth what its write fence enforces (搂2.6).

    Sequence this reproduces: w-1 claims, stalls past `lease_until`, the sweep
    requeues, w-2 claims and is mid-run 鈥?then w-1 wakes up and finishes. Without
    the `status='running' AND worker_id=?` fence, w-1's write would mark *w-2's*
    attempt completed with w-1's (stale) result path.
    """
    repo.create_task("fence-1", "restore", "u", max_attempts=3)
    # A lease that is already expired: that is the state a stalled worker left
    # behind, and the sweep decides it from the stored `lease_until`.
    first = repo.claim_next_task("w-1", lease_seconds=-1)
    assert first["worker_id"] == "w-1"
    assert repo.reset_stale_running() == 1
    assert repo.claim_next_task("w-2")["worker_id"] == "w-2"

    assert repo.finish_task("fence-1", "/tmp/stale.png", None, 10,
                            worker_id="w-1") is False, "the stale write landed"
    row = repo.get_task("fence-1")
    assert row["status"] == "running" and row["worker_id"] == "w-2", row

    assert repo.finish_task("fence-1", "/tmp/fresh.png", "PSNR: 20", 20,
                            worker_id="w-2") is True
    assert repo.get_task("fence-1")["result_path"] == "/tmp/fresh.png"


def test_a_cancelled_task_cannot_be_completed_by_its_worker(isolated_db):
    """Cancel is user intent; a worker finishing later must not resurrect it."""
    repo.create_task("fence-2", "restore", "u")
    assert repo.cancel_task("fence-2") is True

    assert repo.finish_task("fence-2", "/tmp/out.png", None, 10) is False
    assert repo.fail_task("fence-2", "late failure", 10) is False
    assert repo.get_task("fence-2")["status"] == "cancelled"


# ------------------------------------------------------------- saturation (搂2.6)
def _submit(task_id, tmp_path, monkeypatch):
    """Enqueue through the real service with an inline worker switched off."""
    from PIL import Image

    from fiximg.application.task_service import task_service
    from fiximg.config import settings

    monkeypatch.setattr(settings, "tasks_root", str(tmp_path / "tasks"))
    monkeypatch.setattr(settings, "inline_worker", False, raising=False)
    monkeypatch.setattr(settings, "worker_max_queue", 1, raising=False)
    image = Image.new("RGB", (8, 8), (10, 20, 30))
    return task_service.submit_queued(task_id, image, {"username": "alice"}, "restore")


def test_a_saturated_queue_refuses_without_leaving_a_half_written_task(
    isolated_db, tmp_path, monkeypatch
):
    """Refusal has to be refusal all the way down, not "row exists, nobody runs it".

    `FIXIMG_WORKER_MAX_QUEUE` had no coverage at all, so any of these held while the
    suite stayed green: the row written before the check (a task queued forever with
    no artifact behind it), the artifact directory created for a task that was
    rejected (disk growth with no owner), or the error swallowed into a silent True.
    """
    from fiximg.domain.errors import QueueFullError

    assert _submit("sat-1", tmp_path, monkeypatch) is True
    assert _submit("sat-2", tmp_path, monkeypatch) is False

    assert repo.get_task("sat-1") is not None
    assert repo.get_task("sat-2") is None, "the refusal wrote a row"
    # Layout-independent on purpose: `create_run_dir` files the id under
    # tasks_root/<year>/<month>/, so a hand-written path here stayed green while the
    # refusal was still creating the directory.
    tasks_root = tmp_path / "tasks"
    stranded = [p for p in tasks_root.rglob("*") if "sat-2" in p.name] if tasks_root.exists() else []
    assert not stranded, f"the refusal left artifact directories: {stranded}"
    assert repo.queued_count() == 1

    # The client-facing half: `enqueue` turns the False into a 503 and does not
    # hand back an id for a task that was never created.
    from PIL import Image

    from fiximg.application.task_service import task_service

    with pytest.raises(QueueFullError) as excinfo:
        task_service.enqueue(Image.new("RGB", (8, 8), "teal"), {"username": "alice"},
                             "restore")
    assert excinfo.value.code == "QUEUE_FULL"
    assert excinfo.value.status_code == 503
    assert repo.queued_count() == 1, "the refused submission still took a queue slot"


# ------------------------------------------------- cancelling a running task (搂3.8)
def test_a_running_task_can_be_cancelled_and_keeps_what_it_reached(isolated_db):
    """搂3.8's cancel was reachable only while the row still said `queued`.

    The writes that keep a task alive after a cancel are the interesting ones: the
    progress bar must freeze where the user stopped it, the heartbeat must stop
    extending a lease nobody owns, and the worker's eventual `finish_task` must be
    refused rather than resurrect the task.
    """
    repo.create_task("cx-1", "restore", "u", max_attempts=3)
    assert repo.claim_next_task("w-1") is not None
    repo.update_progress("cx-1", 40, "global_restore")

    assert repo.cancel_task("cx-1") is True
    row = repo.get_task("cx-1")
    assert row["status"] == "cancelled" and row["finished_at"]

    repo.update_progress("cx-1", 90, "face_detection")
    lease_before = row["lease_until"]
    repo.heartbeat_task("cx-1", lease_seconds=60)

    after = repo.get_task("cx-1")
    assert after["progress"] == 40, "a cancelled task's progress kept advancing"
    assert after["lease_until"] == lease_before, "the heartbeat renewed a cancelled lease"
    assert repo.finish_task("cx-1", "/tmp/out.png", None, 10,
                            worker_id="w-1") is False
    assert repo.get_task("cx-1")["status"] == "cancelled"


def test_a_finished_task_still_refuses_cancellation(isolated_db):
    repo.create_task("cx-2", "restore", "u")
    repo.claim_next_task("w-1")
    assert repo.finish_task("cx-2", "/tmp/out.png", None, 10, worker_id="w-1") is True
    assert repo.cancel_task("cx-2") is False


def test_the_plan_stops_at_the_stage_boundary_after_a_cancel(isolated_db, tmp_path):
    """The worker cannot be killed mid-stage, so it must *notice* between stages.

    Stage 1 cancels the task the way an HTTP request would 鈥?in another thread, against
    the same database. What is under test is that stage 2 is never started, never
    recorded, and that the stop surfaces as a cancellation rather than a failure.
    """
    import fiximg.inference.runtime as orch
    from types import SimpleNamespace

    from fiximg.domain.errors import TaskCancelledError
    from fiximg.inference.context import StageContext
    from fiximg.inference.gpu import GpuTopology

    started: list[str] = []

    class _Stage:
        def __init__(self, name, cancel=False):
            self.name = name
            self.version = "test-1"
            self.capabilities = frozenset({"restore"})
            self._cancel = cancel

        def run(self, image, context):
            started.append(self.name)
            if self._cancel:
                assert repo.cancel_task("cx-3") is True, "the cancel did not take"
            return SimpleNamespace(image=image, artifacts={}, metadata={},
                                   metrics={}, message=None)

        def __call__(self, *args, **kwargs):  # pragma: no cover - protocol shim
            raise AssertionError("unused")

    repo.create_task("cx-3", "restore", "u", input_path=str(tmp_path / "in.png"))
    assert repo.claim_next_task("w-1") is not None

    stages = {"one": _Stage("one", cancel=True), "two": _Stage("two")}
    orchestrator = orch.PipelineOrchestrator.__new__(orch.PipelineOrchestrator)
    orchestrator.planner = SimpleNamespace(build_stage=lambda name, kwargs: stages[name])
    orchestrator.model_manager = None
    orchestrator.topology = GpuTopology.single_device(-1)
    context = StageContext(task_id="cx-3", run_dir=str(tmp_path), gpu=-1)

    with pytest.raises(TaskCancelledError):
        orchestrator._run_plan("cx-3", "img", SimpleNamespace(stages=[("one", {}), ("two", {})]),
                              context)

    assert started == ["one"], f"the run did not stop: {started}"
    rows = repo.get_task_stages("cx-3")
    assert [r["stage_name"] for r in rows] == ["one"], [dict(r) for r in rows]
    assert repo.get_task("cx-3")["status"] == "cancelled"
    events = [e["event_type"] for e in repo.list_task_events("cx-3")]
    assert "task.failed" not in events, events


def test_a_cancelled_task_never_leaves_a_stage_row_running(isolated_db, tmp_path):
    """A stage that fails while the task is being cancelled is still closed out.

    The task row flips to `cancelled` the instant the request lands. A stage already
    in flight is not interruptible, so it runs to its own end — and if it *fails*
    on the way there, the exception handler is the only thing that will ever close
    its row. `finish_task` is correctly refused for a cancelled task, so nothing
    downstream repairs the stage list either.

    The outcome that must not happen is a stage row left at `running` with a NULL
    duration on a task that is over: no later pass clears it, and a client reading
    the stage list sees a spinner on a finished task.

    Found by running the service, not by reading it: a cancel that landed during
    `global_restore` on the real GPU worker showed `status='running',
    duration_ms=NULL` in the first read, and the same row read `completed` with a
    5145 ms duration 3 s later — correct, but only because that stage returned
    normally. The failing-stage path was untested, and it is the one that depends
    on the handler running.
    """
    import fiximg.inference.runtime as orch
    from types import SimpleNamespace

    from fiximg.inference.context import StageContext
    from fiximg.inference.gpu import GpuTopology

    class _Stage:
        version = "test-1"
        capabilities = frozenset({"restore"})

        def __init__(self, name, cancel_then_raise=False):
            self.name = name
            self._cancel_then_raise = cancel_then_raise

        def run(self, image, context):
            if self._cancel_then_raise:
                # The cancel lands while this stage is in flight, exactly as an
                # HTTP request from another thread would place it, and the stage
                # then fails on its own. This is the path where closing the stage
                # row is a separate act from returning: the exception handler is
                # the only thing that will ever close it.
                assert repo.cancel_task("cx-open") is True, "the cancel did not take"
                raise RuntimeError("the model produced a zero-size output")
            return SimpleNamespace(image=image, artifacts={}, metadata={},
                                   metrics={}, message=None)

    repo.create_task("cx-open", "restore", "u", input_path=str(tmp_path / "in.png"))
    assert repo.claim_next_task("w-1") is not None

    stages = {"one": _Stage("one", cancel_then_raise=True), "two": _Stage("two")}
    orchestrator = orch.PipelineOrchestrator.__new__(orch.PipelineOrchestrator)
    orchestrator.planner = SimpleNamespace(build_stage=lambda name, kwargs: stages[name])
    orchestrator.model_manager = None
    orchestrator.topology = GpuTopology.single_device(-1)
    context = StageContext(task_id="cx-open", run_dir=str(tmp_path), gpu=-1)

    with pytest.raises(RuntimeError):
        orchestrator._run_plan("cx-open", "img",
                              SimpleNamespace(stages=[("one", {}), ("two", {})]),
                              context)

    rows = repo.get_task_stages("cx-open")
    assert len(rows) == 1, [dict(r) for r in rows]
    row = rows[0]
    assert row["status"] not in ("running", "pending", "queued"), (
        f"a terminal task left stage {row['stage_name']} at {row['status']!r}, and "
        "no later pass will ever close it: a stage list on a finished task would "
        "show a spinner forever"
    )
    assert row["status"] == "failed", dict(row)
    assert row["duration_ms"] is not None, (
        "the stage was closed without recording how long it took"
    )
    assert repo.get_task("cx-open")["status"] == "cancelled"


def test_the_queued_entry_point_does_not_report_a_cancellation_as_a_failure(
    isolated_db, tmp_path, monkeypatch
):
    """The stop is raised from inside the plan, and `execute_queued` catches everything.

    Its generic handler writes `failed` and puts `task.failed` on the stream, so without
    a branch for the cancellation a user who pressed cancel would see the task turn into
    a failure 鈥?and the retry path would hand it back to a worker.
    """
    import fiximg.inference.runtime as orch
    from fiximg.domain.errors import TaskCancelledError
    from PIL import Image

    input_path = tmp_path / "in.png"
    Image.new("RGB", (8, 8), "grey").save(input_path)
    repo.create_task("cx-4", "restore", "u", input_path=str(input_path))
    assert repo.claim_next_task("w-1") is not None
    assert repo.cancel_task("cx-4") is True

    orchestrator = orch.PipelineOrchestrator()
    monkeypatch.setattr(
        orchestrator, "_execute_core",
        lambda *a, **k: (_ for _ in ()).throw(TaskCancelledError("cx-4 is no longer running")),
    )
    with pytest.raises(TaskCancelledError):
        orchestrator.execute_queued("cx-4", "restore")

    row = repo.get_task("cx-4")
    assert row["status"] == "cancelled", row
    assert row["error_code"] is None, f"the cancellation was recorded as a failure: {row}"
    events = [e["event_type"] for e in repo.list_task_events("cx-4")]
    assert "task.failed" not in events, events


def test_a_worker_that_loses_its_lease_stops_running_the_task(isolated_db, tmp_path):
    """The same check that honours a cancel also ends the duplicate run.

    The handover happens *while stage one is in flight*, the way it happens in
    production: the lease expires, the sweep requeues the row, another worker claims it.
    If the original process kept going, the task would be executed twice on two devices
    鈥?the failure 搂2.6 names and asks to control.
    """
    from types import SimpleNamespace

    import fiximg.inference.runtime as orch
    from fiximg.domain.errors import TaskCancelledError
    from fiximg.infrastructure.db import timestamps
    from fiximg.inference.context import StageContext
    from fiximg.inference.gpu import GpuTopology

    handed_over: list[str] = []

    class _Stage:
        def __init__(self, name, hand_over=False):
            self.name = name
            self.version = "test-1"
            self.capabilities = frozenset({"restore"})
            self._hand_over = hand_over

        def run(self, image, context):
            if self._hand_over:
                # Expire this worker's lease, let the sweep requeue it, and hand it to
                # another worker 鈥?all while this stage is still executing.
                task_repo = repo
                task_repo._get_conn().execute(
                    "UPDATE tasks SET lease_until=? WHERE id=?",
                    (timestamps.deadline(-10), "cx-5"),
                )
                task_repo._get_conn().commit()
                assert task_repo.reset_stale_running(timeout_seconds=3600) == 1
                assert task_repo.claim_next_task("w-new") is not None
                handed_over.append("w-new")
            return SimpleNamespace(image=image, artifacts={}, metadata={},
                                   metrics={}, message=None)

    repo.create_task("cx-5", "restore", "u", max_attempts=3)
    assert repo.claim_next_task("w-old") is not None

    stages = {"one": _Stage("one", hand_over=True), "two": _Stage("two")}
    orchestrator = orch.PipelineOrchestrator.__new__(orch.PipelineOrchestrator)
    orchestrator.planner = SimpleNamespace(build_stage=lambda name, kwargs: stages[name])
    orchestrator.model_manager = None
    orchestrator.topology = GpuTopology.single_device(-1)
    context = StageContext(task_id="cx-5", run_dir=str(tmp_path), gpu=-1)

    with pytest.raises(TaskCancelledError) as excinfo:
        orchestrator._run_plan(
            "cx-5", "img", SimpleNamespace(stages=[("one", {}), ("two", {})]), context,
            owner_worker_id="w-old",
        )

    assert handed_over == ["w-new"], "the handover in the stage did not happen"
    assert "w-new" in str(excinfo.value), "it stopped for the wrong reason"
    assert [r["stage_name"] for r in repo.get_task_stages("cx-5")] == ["one"],         "the losing worker started a stage after another worker took the task"
    assert repo.get_task("cx-5")["worker_id"] == "w-new", "the new owner lost the row"



def test_a_task_that_will_colorize_is_not_claimable_without_the_colour_model(
    isolated_db, tmp_path, monkeypatch
):
    """The routing hint has an effect, and this is it.

    A specialised worker (no DDColor) used to take a `restore` submitted with
    `{"auto_colorize": true}` because the hint was computed from the task type alone and
    so never mentioned `colorize` 鈥?the task then ran the colour model on a card that
    was not supposed to serve it. The opt-out direction matters just as much: a task
    that declared the face chain it would not run could be left waiting for a worker
    that does not exist.
    """
    import json

    from PIL import Image

    from fiximg.application.task_service import task_service
    from fiximg.config import settings
    from fiximg.inference.planner import PipelinePlanner

    monkeypatch.setattr(settings, "tasks_root", str(tmp_path / "tasks"))
    monkeypatch.setattr(settings, "inline_worker", False, raising=False)

    planner = PipelinePlanner()
    restorer = planner.required_capabilities("restore", {})
    assert "colorize" not in restorer, "the control case must not serve colour"

    task_id = task_service.enqueue(
        Image.new("RGB", (8, 8), "grey"), {"username": "alice"}, "restore",
        options={"auto_colorize": True},
    )
    declared = json.loads(repo.get_task(task_id)["required_capabilities"] or "[]")
    assert "colorize" in declared, declared

    assert repo.claim_next_task("w-restore", capabilities=restorer) is None, (
        "a worker without the colour model claimed a task that will colorize"
    )
    claimed = repo.claim_next_task("w-colour", capabilities=restorer | {"colorize"})
    assert claimed is not None and claimed["id"] == task_id


def test_an_opted_out_face_chain_does_not_reserve_face_workers(
    isolated_db, tmp_path, monkeypatch
):
    """`{"face_enhance": false}` must shrink the hint as well as the plan."""
    import json

    from PIL import Image

    from fiximg.application.task_service import task_service
    from fiximg.config import settings
    from fiximg.inference.planner import PipelinePlanner

    monkeypatch.setattr(settings, "tasks_root", str(tmp_path / "tasks"))
    monkeypatch.setattr(settings, "inline_worker", False, raising=False)

    planner = PipelinePlanner()
    full = planner.required_capabilities("restore", None)
    assert "face_detection" in full, "the default restore plan does serve faces"

    task_id = task_service.enqueue(
        Image.new("RGB", (8, 8), "grey"), {"username": "alice"}, "restore",
        options={"face_enhance": False},
    )
    declared = set(json.loads(repo.get_task(task_id)["required_capabilities"] or "[]"))
    assert "face_detection" not in declared, declared

    # A worker serving exactly that 鈥?and nothing for faces 鈥?can take it. Before the
    # fix the hint still declared the face chain, so this same worker was locked out
    # and the task waited for a specialist that was never needed.
    # Stated as a literal, not derived from the value under test: this is the whole
    # plan of a face-less restore (global restoration only), so the hint must be
    # exactly what those stages need.
    faceless = {"restore", "deblur", "denoise"}
    assert declared == faceless, declared
    claimed = repo.claim_next_task("w-plain", capabilities=faceless)
    assert claimed is not None and claimed["id"] == task_id, (
        "a plain restoration worker was locked out of a task it can finish"
    )
