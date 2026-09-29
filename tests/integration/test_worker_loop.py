"""Unit tests for the PipelineWorker loop (plan sections 12/28)."""
import threading

import pytest

from fiximg.infrastructure.db.repositories import task_repository as repo

# Cross-layer: worker loop consuming from the real DB-backed queue.
pytestmark = pytest.mark.integration


@pytest.fixture()
def db_env(tmp_path, monkeypatch):
    import fiximg.infrastructure.db.engine as legacy_db
    import fiximg.config as config_mod

    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(legacy_db, "DB_PATH", db_path)
    monkeypatch.setattr(legacy_db, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(legacy_db, "_conn", None)
    monkeypatch.setattr(config_mod.settings, "db_path", db_path)
    repo._DDL_DONE = False
    return db_path


class FakeOrchestrator:
    """Records claimed executions; optionally fails them."""

    def __init__(self, fail: bool = False):
        self.calls = []
        self.fail = fail
        self._lock = threading.Lock()

    def execute_queued(self, task_id: str, task_type: str) -> None:
        with self._lock:
            self.calls.append((task_id, task_type))
        if self.fail:
            raise RuntimeError("boom")


def test_worker_executes_queued_task(db_env):
    from fiximg.inference.worker import PipelineWorker

    repo.create_task("wq1", "restore", "u", input_path="unused.png")
    fake = FakeOrchestrator()
    worker = PipelineWorker(fake, poll_seconds=0.05, worker_id="w-test")
    worker.start()
    try:
        deadline = threading.Event()
        for _ in range(100):
            if fake.calls:
                break
            deadline.wait(0.05)
        assert fake.calls == [("wq1", "restore")]
        row = repo.get_task("wq1")
        # FakeOrchestrator doesn't finish tasks; the row stays running (claimed).
        assert row["status"] == "running"
    finally:
        worker.stop()
    assert not worker.is_running()


def test_worker_failure_keeps_polling(db_env):
    from fiximg.inference.worker import PipelineWorker

    repo.create_task("wq2", "colorize", "u", input_path="unused.png")
    repo.create_task("wq3", "restore", "u", input_path="unused.png")
    fake = FakeOrchestrator(fail=True)
    worker = PipelineWorker(fake, poll_seconds=0.05, worker_id="w-test")
    worker.start()
    try:
        deadline = threading.Event()
        for _ in range(200):
            if len(fake.calls) >= 2:
                break
            deadline.wait(0.05)
        assert {t for t, _ in fake.calls} == {"wq2", "wq3"}
    finally:
        worker.stop()


def test_worker_depth(db_env):
    from fiximg.inference.worker import PipelineWorker

    repo.create_task("wq4", "restore", "u")
    repo.create_task("wq5", "restore", "u")
    fake = FakeOrchestrator()
    worker = PipelineWorker(fake, poll_seconds=0.05)
    assert worker.depth() == 2


class _RecordingQueue:
    """The transport half of the loop, with the calls under test made visible."""

    kind = "recording"

    def __init__(self):
        self.acked = []
        self.retried = []
        #: (task_id, acks so far) captured *inside* the retry call, i.e. at the
        #: moment the worker decided this task will run again.
        self.attempts_left_behind = []

    def claim(self, worker_id, lease_seconds=3600.0):
        from fiximg.domain.tasks import Task

        row = repo.claim_next_task(worker_id, lease_seconds=lease_seconds)
        return Task.from_row(row) if row else None

    def ack(self, task_id):
        self.acked.append(task_id)

    def retry(self, task_id, delay_seconds=0.0):
        self.retried.append((task_id, delay_seconds))
        self.attempts_left_behind.append((task_id, list(self.acked)))
        return repo.retry_task(task_id, delay_seconds=delay_seconds)

    def heartbeat(self, task_id):
        return None

    def depth(self):
        return repo.queued_count()


def _drain(worker, predicate, attempts=200):
    """Poll until `predicate()` holds, so the assertion is about the loop's choices."""
    import time

    for _ in range(attempts):
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_a_finished_task_releases_its_transport_claim(db_env):
    """搂2.6 at-least-once has a cost that has to be paid back: the ack.

    Without it the queue entry stays pending until the lease expires, another
    worker is handed a task that is already complete, and the backend's
    task -> record map grows by one per finished task for the life of the process.
    """
    from fiximg.inference.worker import PipelineWorker

    repo.create_task("ack-ok", "restore", "u", input_path="unused.png")
    queue = _RecordingQueue()
    worker = PipelineWorker(FakeOrchestrator(), poll_seconds=0.05, queue=queue)
    worker.start()
    try:
        assert _drain(worker, lambda: queue.acked == ["ack-ok"]), queue.acked
    finally:
        worker.stop()


def test_a_task_that_will_run_again_is_not_acknowledged(db_env):
    """The retry path republishes, so the entry is still needed: acking here would
    leave a requeued task with nothing to wake a worker."""
    from fiximg.inference.worker import PipelineWorker

    repo.create_task("ack-retry", "restore", "u", input_path="unused.png", max_attempts=3)
    queue = _RecordingQueue()
    worker = PipelineWorker(FakeOrchestrator(fail=True), poll_seconds=0.05, queue=queue)
    worker.start()
    try:
        assert _drain(worker, lambda: queue.attempts_left_behind), queue.retried
        # Read the decision point, not the final state: the loop keeps going and
        # *should* ack the task once its budget really is exhausted.
        task_id, acked_when_retried = queue.attempts_left_behind[0]
        assert task_id == "ack-retry"
        assert acked_when_retried == [], (
            f"a task scheduled for another attempt was acknowledged: {acked_when_retried}"
        )
    finally:
        worker.stop()


def test_attempts_exhausted_is_terminal_and_gets_acknowledged(db_env):
    from fiximg.inference.worker import PipelineWorker

    repo.create_task("ack-dead", "restore", "u", input_path="unused.png", max_attempts=1)
    queue = _RecordingQueue()
    worker = PipelineWorker(FakeOrchestrator(fail=True), poll_seconds=0.05, queue=queue)
    worker.start()
    try:
        assert _drain(worker, lambda: "ack-dead" in queue.acked), (queue.acked, queue.retried)
        assert queue.retried == [], "no attempt budget was left to retry with"
        assert queue.acked == ["ack-dead"]
    finally:
        worker.stop()


def test_an_exception_outside_the_task_path_does_not_kill_the_worker(db_env, monkeypatch):
    """One bad task used to be enough to stop the whole worker.

    `_execute` protects the *pipeline*; anything raising before that guard 鈥?the
    claim's event write, the queue-wait accounting, a mistyped name in the
    correlation setup 鈥?escaped to the thread's target. The process stayed up, the
    worker stopped polling, and the task stayed `running` until its lease expired.
    What an operator sees is a queue that never drains, not a crash, so this is
    pinned as a behaviour rather than left to the exception log.
    """
    from fiximg.inference import worker as worker_module

    repo.create_task("bad-1", "restore", "u", input_path="unused.png", max_attempts=1)
    repo.create_task("good-1", "restore", "u", input_path="unused.png", max_attempts=3)

    original = worker_module.PipelineWorker._queue_wait_seconds

    def flaky(task):
        if task.id == "bad-1":
            raise NameError("simulated defect before the pipeline")
        return original(task)

    monkeypatch.setattr(worker_module.PipelineWorker, "_queue_wait_seconds",
                        staticmethod(flaky))

    fake = FakeOrchestrator()
    worker = worker_module.PipelineWorker(fake, poll_seconds=0.05)
    worker.start()
    try:
        ran = _drain(worker, lambda: any(t == "good-1" for t, _ in fake.calls))
        rescued = repo.get_task("bad-1")
    finally:
        worker.stop()

    assert ran, f"the worker stopped after one bad task: {fake.calls}"
    # Rescued, not stranded: the row reached a terminal state immediately instead
    # of sitting in `running` for a whole lease period.
    assert rescued["status"] == "failed", rescued
    assert rescued["error_code"] == "worker_error", rescued
    assert "simulated defect" in (rescued["error_message"] or ""), rescued


def _task_of(task_id):
    from fiximg.domain.tasks import Task

    return Task.from_row(repo.get_task(task_id))


class _AlwaysBusyQueue:
    """A transport that always hands out work, to test the busy path."""

    kind = "recording"

    def __init__(self, task):
        self._task = task
        self.acked = []
        self.retried = []
        self.beats = []

    def claim(self, worker_id, lease_seconds=3600.0):
        return self._task

    def ack(self, task_id):
        self.acked.append(task_id)

    def retry(self, task_id, delay_seconds=0.0):
        self.retried.append((task_id, delay_seconds))
        return False

    def heartbeat(self, task_id):
        self.beats.append(task_id)

    def depth(self):
        return 1


def test_housekeeping_runs_while_the_worker_is_busy(db_env, monkeypatch):
    """A loaded worker still has to do its periodic sweeps.

    `_scan_stale()` (expired leases back to the queue, expired artifacts reclaimed)
    used to live in the branch that only runs when a claim comes back empty, so a
    worker under continuous load never swept 鈥?the retention TTL was unread on
    exactly the deployment whose disk fills first.
    """
    from fiximg.inference import worker as worker_module

    repo.create_task("busy-1", "restore", "u", input_path="unused.png")
    scans = []
    monkeypatch.setattr(worker_module.PipelineWorker, "_scan_stale",
                        lambda self: scans.append(1))

    worker = worker_module.PipelineWorker(
        FakeOrchestrator(), poll_seconds=0.01, queue=_AlwaysBusyQueue(_task_of("busy-1"))
    )
    worker.start()
    try:
        swept = _drain(worker, lambda: len(scans) >= 2)
    finally:
        worker.stop()
    assert swept, f"no housekeeping happened while busy ({len(scans)} scans)"


def test_the_configured_retry_backoff_reaches_the_transport(db_env, monkeypatch):
    """`FIXIMG_TASK_RETRY_BACKOFF` was read by the repository and skipped by the caller."""
    from fiximg.config import settings
    from fiximg.inference.worker import PipelineWorker

    monkeypatch.setattr(settings, "task_retry_backoff_seconds", 12.5, raising=False)
    repo.create_task("backoff-1", "restore", "u", input_path="unused.png", max_attempts=3)
    repo.start_task("backoff-1")
    repo.update_progress("backoff-1", 5)

    queue = _RecordingQueue()
    worker = PipelineWorker(FakeOrchestrator(), poll_seconds=0.05, queue=queue)
    worker._handle_failure(_task_of("backoff-1"), RuntimeError("transient"))

    assert queue.retried == [("backoff-1", 12.5)], queue.retried
    assert repo.get_task("backoff-1")["retry_at"], "the delay was stored, not just passed"


class _CountingQueue(_AlwaysBusyQueue):
    """Alias kept explicit: the busy transport already records heartbeats."""


class _SlowOrchestrator:
    """Takes longer than one heartbeat interval, like a real restoration run."""

    def __init__(self, seconds: float = 1.4):
        self.seconds = seconds
        self.executed = []

    def execute_queued(self, task_id: str, task_type: str) -> None:
        import time

        time.sleep(self.seconds)
        self.executed.append(task_id)


def test_a_running_task_refreshes_its_lease_through_the_worker(db_env, monkeypatch):
    """A long task must not lose its lease, or the sweep requeues it mid-flight.

    `_Heartbeat` is the only thing standing between a slow restoration and a second
    worker claiming the same task, and it used to be covered solely by tests that
    called `queue.heartbeat()` themselves 鈥?deleting the thread from `_execute`
    left the suite green. This drives the real loop, so the wiring is what fails.
    """
    import fiximg.config as config_mod
    from fiximg.inference.worker import PipelineWorker

    monkeypatch.setattr(config_mod.settings, "worker_heartbeat_seconds", 1.0, raising=False)
    repo.create_task("hb-1", "restore", "u", input_path="unused.png")
    repo.start_task("hb-1")

    queue = _CountingQueue(_task_of("hb-1"))
    fake = _SlowOrchestrator()
    worker = PipelineWorker(fake, poll_seconds=0.05, queue=queue)
    worker.start()
    try:
        ran = _drain(worker, lambda: bool(fake.executed))
    finally:
        worker.stop()

    assert ran, "the task never executed"
    assert queue.beats, "the worker started a task without refreshing its lease"
    assert set(queue.beats) == {"hb-1"}, queue.beats


def test_a_cancelled_task_is_never_retried(db_env):
    """A cancellation is a decision, not a fault, so it must not spend an attempt.

    The worker's failure path requeues while `attempt_count < max_attempts`. Treating
    `TaskCancelledError` like any other exception would put the task the user just
    stopped back into the queue, and its own retry would run the pipeline again 鈥?with
    the row flipping between `cancelled` and `running` on the stream the user is
    watching.
    """
    from fiximg.domain.errors import TaskCancelledError
    from fiximg.inference.worker import PipelineWorker

    class _CancellingOrchestrator:
        """Stop the run the way the runtime does: the row is already cancelled.

        The user's HTTP request is what flips the row; the pipeline only notices at the
        next stage boundary and raises. Simulating only the raise would leave the row
        `running` and prove nothing about the state the user is told.
        """

        def execute_queued(self, task_id: str, task_type: str) -> None:
            assert repo.cancel_task(task_id) is True
            raise TaskCancelledError(f"{task_id} is no longer running")

    repo.create_task("cxw-1", "restore", "u", input_path="unused.png", max_attempts=3)
    queue = _RecordingQueue()
    worker = PipelineWorker(_CancellingOrchestrator(), poll_seconds=0.05, queue=queue)
    worker.start()
    try:
        drained = _drain(worker, lambda: queue.acked == ["cxw-1"])
    finally:
        worker.stop()

    assert drained, f"the claim was never released (acks={queue.acked})"
    assert queue.retried == [], f"a cancelled task was scheduled to run again: {queue.retried}"
    row = repo.get_task("cxw-1")
    assert row["status"] == "cancelled", row
    assert row["attempt_count"] == 1, "the cancellation spent another attempt"
    assert repo.queued_count() == 0, "the cancelled task went back to the queue"


def test_the_startup_report_names_the_claim_restriction_that_is_in_force(db_env):
    """`FIXIMG_WORKER_CAPABILITIES` must be visible as *enforced*, not as nothing.

    The line used to print only what the GPU topology said the pinned card serves, so a
    correctly restricted worker started with `served_capabilities: null` 鈥?and an
    operator (me, reading this line on a real boot) concluded the setting was ignored
    and went looking for a bug that was not there. The two knobs now get two fields.

    A handler is attached to the worker's own logger rather than using `caplog`,
    because the app configures that logger with a handler of its own and full-suite
    order decides whether records still propagate to the root 鈥?an earlier version of
    this test passed alone and failed in the suite for exactly that reason.
    """
    import logging
    import threading

    from fiximg.inference.worker import PipelineWorker

    class _RestrictedQueue:
        kind = "restricted"
        capabilities = frozenset({"restore", "deblur"})

        def __init__(self):
            self._stop = threading.Event()

        def claim(self, worker_id, lease_seconds=3600.0):
            self._stop.wait(0.05)
            return None

        def heartbeat(self, task_id):
            return None

        def depth(self):
            return 0

    collected: list = []

    class _Collect(logging.Handler):
        def emit(self, record):
            collected.append(record)

    logger = logging.getLogger("fiximg.worker")
    handler = _Collect()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    worker = None
    try:
        worker = PipelineWorker(FakeOrchestrator(), poll_seconds=0.05, queue=_RestrictedQueue())
        worker.start()
    finally:
        if worker is not None:
            worker.stop()
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    started = [r for r in collected if r.getMessage() == "gpu worker started"]
    assert started, f"no startup record; logger.disabled={logger.disabled}"
    fields = getattr(started[0], "extra_fields", {})
    assert fields["claim_serves"] == ["deblur", "restore"], fields
    assert fields["gpu_capabilities"] is None, fields
