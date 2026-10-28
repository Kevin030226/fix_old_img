"""Lifecycle of the deferred-evaluation pool: a stop must stop what it scheduled.

The defect these tests pin (Python 3.12 CI, "no such table: users"):

* ``PipelineWorker.stop()`` called ``shutdown_eval_pool(wait=False)``, which
  cleared the module-global pool before the executor had finished its work;
* the fixture's own ``shutdown_eval_pool(wait=True)`` then saw ``None`` and
  returned immediately, so the in-flight evaluation was never waited for;
* the next fixture repointed the process-wide database, and the leftover
  evaluation wrote into it.

Every test drives the interleaving with ``threading.Event`` and a second thread.
No sleep is used to make a race likely: the single timeout below bounds a
*negative* liveness assertion ("teardown must still be blocked"), and every
other wait has a generous upper bound so a broken implementation fails instead
of hanging the suite.

A second boundary is pinned here as well: ``PipelineWorker.stop(timeout)`` can
return while its thread is still alive. Deferred-evaluation intake must be
closed before the join, otherwise that surviving thread can schedule evaluation
after the drain — the same class of leak, one lifecycle later. These tests drive
that window with a controlled worker thread and assert the refusal, the honest
``False`` return value, and that a later lifecycle can schedule again.
"""
import threading

import pytest

from fiximg.inference import runtime as runtime_mod
from fiximg.inference.runtime import shutdown_eval_pool

#: How long a deliberately blocked teardown may take before we conclude it is
#: waiting. The operations it separates are a lock hand-off and an executor
#: shutdown (microseconds), so this is a bound on scheduling noise, not a sleep.
BLOCKED_TIMEOUT = 1.0
#: Upper bound for every positive wait, so a regression fails rather than hangs.
FINISH_TIMEOUT = 10.0


@pytest.fixture(autouse=True)
def _drain_eval_pool_between_tests():
    """Never inherit a pool, a job or a thread from the previous test."""
    shutdown_eval_pool(wait=True)
    yield
    shutdown_eval_pool(wait=True)


class _DatabaseProbe:
    """Orders the evaluation's database write against the teardown's close."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self._lock = threading.Lock()

    def touch(self, what: str) -> None:
        with self._lock:
            self.order.append(what)


def _submit_and_wait_for_start(job) -> threading.Event:
    """Queue *job*; return the event it sets on entry."""
    started = threading.Event()

    def _wrapped():
        started.set()
        job()

    runtime_mod._submit_eval_job(_wrapped)
    return started


def _schedule(orchestrator, task_id: str = "t-eval") -> bool:
    """Offer one deferred evaluation through the real orchestration entry point."""
    return orchestrator._evaluate_in_background(
        task_id, "restore", "input.png", "output.png", [], None, "tester",
    )


def _silence_deferred_writes(monkeypatch) -> None:
    """Keep a scheduled job's database writes out of the developer database."""
    monkeypatch.setattr(runtime_mod.task_repo, "set_evaluation", lambda *_a, **_k: None)
    monkeypatch.setattr(runtime_mod.task_repo, "add_event", lambda *_a, **_k: None)


class _IdleQueue:
    """Queue double for a worker whose polling loop is replaced by a no-op."""

    capabilities: tuple[str, ...] = ()

    def claim(self, *_args, **_kwargs):
        return None

    def depth(self) -> int:
        return 0


def test_a_blocking_shutdown_drains_a_pool_a_non_blocking_one_detached():
    """The regression: after ``wait=False`` a later ``wait=True`` must still wait."""
    database = _DatabaseProbe()
    release = threading.Event()

    def evaluation():
        release.wait(FINISH_TIMEOUT)
        database.touch("evaluation writes metrics")

    started = _submit_and_wait_for_start(evaluation)
    assert started.wait(FINISH_TIMEOUT), "the evaluation never started"

    shutdown_eval_pool(wait=False)  # what PipelineWorker.stop used to call

    teardown_done = threading.Event()

    def fixture_teardown():
        shutdown_eval_pool(wait=True)  # what the fixture calls next
        database.touch("database closed")
        teardown_done.set()

    teardown = threading.Thread(target=fixture_teardown, name="p37-teardown", daemon=True)
    teardown.start()
    try:
        assert not teardown_done.wait(BLOCKED_TIMEOUT), (
            "blocking shutdown returned while an evaluation was still running; "
            "the database can now be closed under that evaluation"
        )
    finally:
        release.set()
    teardown.join(FINISH_TIMEOUT)
    assert not teardown.is_alive(), "fixture teardown never finished"
    assert database.order == [
        "evaluation writes metrics",
        "database closed",
    ], f"teardown touched the database before the evaluation it waited for: {database.order}"


def test_pipeline_worker_stop_waits_for_the_evaluation_it_scheduled():
    """Worker teardown must not return while its deferred evaluation is running."""
    from fiximg.inference.worker import PipelineWorker

    worker = PipelineWorker(orchestrator=None, worker_id="p37-lifecycle")
    release = threading.Event()
    finished = threading.Event()

    def evaluation():
        release.wait(FINISH_TIMEOUT)
        finished.set()

    started = _submit_and_wait_for_start(evaluation)
    assert started.wait(FINISH_TIMEOUT), "the evaluation never started"

    stop_done = threading.Event()

    def stop_worker():
        worker.stop()
        stop_done.set()

    stopper = threading.Thread(target=stop_worker, name="p37-worker-stop", daemon=True)
    stopper.start()
    try:
        assert not stop_done.wait(BLOCKED_TIMEOUT), (
            "PipelineWorker.stop() returned with a deferred evaluation still in "
            "flight; teardown may then close the database under it"
        )
    finally:
        release.set()
    stopper.join(FINISH_TIMEOUT)
    assert stop_done.is_set(), "PipelineWorker.stop() did not finish"
    assert finished.is_set(), "the evaluation did not complete"


def test_blocking_shutdown_without_a_pool_creates_nothing():
    shutdown_eval_pool(wait=True)
    assert runtime_mod._eval_pool is None
    assert list(runtime_mod._draining_eval_pools) == []


def test_repeated_blocking_shutdown_is_idempotent_and_joins_the_thread():
    finished = threading.Event()
    started = _submit_and_wait_for_start(finished.set)
    assert started.wait(FINISH_TIMEOUT), "the evaluation never started"

    shutdown_eval_pool(wait=True)
    assert finished.is_set(), "wait=True returned before the queued job ran"

    shutdown_eval_pool(wait=True)
    shutdown_eval_pool(wait=True)
    assert runtime_mod._eval_pool is None
    assert list(runtime_mod._draining_eval_pools) == []
    assert not [
        thread for thread in threading.enumerate()
        if thread.name.startswith("fiximg-eval") and thread.is_alive()
    ], "a drained pool left its worker thread running"


def test_concurrent_submissions_and_shutdowns_keep_every_job():
    """Teardown and submission can overlap; neither may lose or reject a job."""
    workers = 8
    barrier = threading.Barrier(workers + 1)
    completed: list[int] = []
    errors: list[Exception] = []

    def record(index: int) -> None:
        completed.append(index)

    def run(index: int) -> None:
        try:
            barrier.wait(FINISH_TIMEOUT)
            runtime_mod._submit_eval_job(lambda: record(index))
            shutdown_eval_pool(wait=False)
        except Exception as exc:  # noqa: BLE001 — collected and asserted empty
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(index,), daemon=True)
        for index in range(workers)
    ]
    for thread in threads:
        thread.start()
    barrier.wait(FINISH_TIMEOUT)
    for thread in threads:
        thread.join(FINISH_TIMEOUT)

    shutdown_eval_pool(wait=True)

    assert errors == [], f"pool lifecycle raised instead of accepting the work: {errors!r}"
    assert sorted(completed) == list(range(workers)), (
        "jobs were lost when submission and shutdown overlapped: "
        f"{sorted(completed)} of {workers} completed"
    )
    assert runtime_mod._eval_pool is None
    assert list(runtime_mod._draining_eval_pools) == []


# ------------------------------------------------ stop timeout / late producer
def test_stop_timeout_refuses_eval_from_a_surviving_worker_thread(monkeypatch):
    """A worker thread that outlives ``stop(timeout)`` must not become a writer."""
    from fiximg.inference.worker import PipelineWorker

    orchestrator = runtime_mod.PipelineOrchestrator()
    probe = _DatabaseProbe()
    scheduled: dict[str, object] = {}

    def fake_evaluate(*_args, **_kwargs):
        probe.touch("evaluation ran")
        return "report", {"sharpness": 1.0}

    monkeypatch.setattr(orchestrator, "_evaluate", fake_evaluate)
    monkeypatch.setattr(
        runtime_mod.task_repo, "set_evaluation",
        lambda *_a, **_k: probe.touch("db write"),
    )
    monkeypatch.setattr(
        runtime_mod.task_repo, "add_event",
        lambda *_a, **_k: probe.touch("event write"),
    )

    worker = PipelineWorker(orchestrator, worker_id="p37-stop-timeout")
    thread_started = threading.Event()
    thread_release = threading.Event()

    def worker_thread():
        # The thread the worker owns, parked in the long-task phase that makes the
        # join time out. Finishing it runs the same deferral entry point that
        # `_execute` uses once it has marked a task completed.
        thread_started.set()
        thread_release.wait(FINISH_TIMEOUT)
        scheduled["queued"] = _schedule(orchestrator, "t-late")

    thread = threading.Thread(target=worker_thread, name="p37-worker-thread", daemon=True)
    worker._thread = thread  # what start() would have created
    thread.start()
    assert thread_started.wait(FINISH_TIMEOUT), "the controlled worker thread did not start"

    stopped = worker.stop(timeout=0.05)
    assert worker.is_running() is True, "a timed-out worker must not look stopped"
    probe.touch("database closed")  # teardown proceeds only because intake is closed

    thread_release.set()
    thread.join(FINISH_TIMEOUT)
    assert not thread.is_alive(), "the controlled worker thread did not finish"

    # Observation window: drain whatever a leaked submission may have created.
    shutdown_eval_pool(wait=True)

    assert probe.order == ["database closed"], (
        "deferred evaluation wrote after teardown: "
        f"{probe.order}. stop() returned with its thread alive and that thread then "
        "scheduled new evaluation work"
    )
    assert scheduled.get("queued") is False, (
        "a stopping worker scheduled deferred evaluation instead of being refused"
    )
    assert stopped is False, "stop() reported success although its thread was alive"


def test_submission_after_close_is_refused_with_evidence(monkeypatch):
    """The refusal is deterministic and names the task, not a silent drop."""
    orchestrator = runtime_mod.PipelineOrchestrator()
    events: list[tuple[str, str, dict]] = []

    def record(_logger, level, message, **fields):
        events.append((level, message, fields))

    monkeypatch.setattr(runtime_mod, "log_event", record)
    orchestrator.close_deferred_evaluation()

    assert _schedule(orchestrator, "t-refused") is False
    assert runtime_mod._eval_pool is None, "a refused evaluation created a pool"
    assert list(runtime_mod._draining_eval_pools) == []
    skipped = [event for event in events if event[1] == "async evaluation skipped"]
    assert skipped, f"the refusal left no evidence in the log: {events!r}"
    assert skipped[0][2]["task_id"] == "t-refused"


def test_close_and_submit_race_is_linearised(monkeypatch):
    """Either the job is in the pool the wait drains, or it is refused."""
    orchestrator = runtime_mod.PipelineOrchestrator()
    ran = threading.Event()

    def fake_evaluate(*_args, **_kwargs):
        ran.set()
        return "report", {}

    monkeypatch.setattr(orchestrator, "_evaluate", fake_evaluate)
    _silence_deferred_writes(monkeypatch)

    outcome: dict[str, object] = {}
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def submitter():
        try:
            barrier.wait(FINISH_TIMEOUT)
            outcome["accepted"] = _schedule(orchestrator, "t-race")
        except Exception as exc:  # noqa: BLE001 — collected and asserted empty
            errors.append(exc)

    def closer():
        try:
            barrier.wait(FINISH_TIMEOUT)
            orchestrator.close_deferred_evaluation()
            shutdown_eval_pool(wait=True)
        except Exception as exc:  # noqa: BLE001 — collected and asserted empty
            errors.append(exc)

    threads = [
        threading.Thread(target=submitter, name="p37-submitter", daemon=True),
        threading.Thread(target=closer, name="p37-closer", daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(FINISH_TIMEOUT)

    assert errors == [], f"the close/submit race raised: {errors!r}"
    accepted = outcome.get("accepted")
    assert isinstance(accepted, bool), (
        f"the deferral path did not report whether it queued the job: {accepted!r}"
    )
    if accepted:
        assert ran.is_set(), "an accepted evaluation was not drained before the wait returned"
    else:
        assert not ran.is_set(), "a refused evaluation still ran"
    assert runtime_mod._eval_pool is None
    assert list(runtime_mod._draining_eval_pools) == []


def test_a_new_orchestrator_lifecycle_can_schedule_again(monkeypatch):
    """A new app/worker gets a fresh admission gate, not the closed one."""
    _silence_deferred_writes(monkeypatch)
    first = runtime_mod.PipelineOrchestrator()
    first.close_deferred_evaluation()
    assert _schedule(first, "t-old") is False

    second = runtime_mod.PipelineOrchestrator()  # next app / worker lifecycle
    monkeypatch.setattr(second, "_evaluate", lambda *_a, **_k: ("report", {}))
    assert _schedule(second, "t-new") is True
    shutdown_eval_pool(wait=True)


def test_reopening_an_orchestrator_reenables_deferred_evaluation(monkeypatch):
    _silence_deferred_writes(monkeypatch)
    orchestrator = runtime_mod.PipelineOrchestrator()
    monkeypatch.setattr(orchestrator, "_evaluate", lambda *_a, **_k: ("report", {}))
    orchestrator.close_deferred_evaluation()
    assert _schedule(orchestrator, "t-closed") is False

    orchestrator.open_deferred_evaluation()
    assert _schedule(orchestrator, "t-open") is True
    shutdown_eval_pool(wait=True)


def test_restarting_a_worker_reopens_deferred_evaluation(monkeypatch):
    """The same worker instance may be started again for a new lifecycle."""
    from fiximg.inference.worker import PipelineWorker

    _silence_deferred_writes(monkeypatch)
    orchestrator = runtime_mod.PipelineOrchestrator()
    monkeypatch.setattr(orchestrator, "_evaluate", lambda *_a, **_k: ("report", {}))
    worker = PipelineWorker(orchestrator, worker_id="p37-restart", queue=_IdleQueue())
    monkeypatch.setattr(worker, "_warm_models", lambda: None)
    monkeypatch.setattr(worker, "_loop", lambda: None)

    worker.start()
    assert worker.stop(timeout=FINISH_TIMEOUT) is True
    assert _schedule(orchestrator, "t-after-stop") is False

    worker.start()
    try:
        assert _schedule(orchestrator, "t-after-restart") is True
    finally:
        assert worker.stop(timeout=FINISH_TIMEOUT) is True


def test_stop_reports_true_for_a_worker_that_never_started():
    from fiximg.inference.worker import PipelineWorker

    worker = PipelineWorker(orchestrator=None, worker_id="p37-idle")
    assert worker.stop(timeout=0.05) is True
    assert worker.stop(timeout=0.05) is True


def test_stop_is_incomplete_when_the_close_hook_raises(monkeypatch):
    """A failed close leaves the intake possibly open: stop must not claim success.

    ``stop()`` reports three things separately in its log (closed / thread alive /
    drained) and must not let the success path ignore any of them: the close hook
    is caught and returns False, so the same call must return False and record
    "stop incomplete" rather than "gpu worker stopped".
    """
    from fiximg.inference import worker as worker_mod
    from fiximg.inference.worker import PipelineWorker

    events: list[tuple[str, str, dict]] = []

    def record(_logger, level, message, **fields):
        events.append((level, message, fields))

    orchestrator = runtime_mod.PipelineOrchestrator()

    def refusing_close():
        raise RuntimeError("close hook exploded")

    monkeypatch.setattr(orchestrator, "close_deferred_evaluation", refusing_close)
    # The worker logs through its own module-level binding, so patch that one.
    monkeypatch.setattr(worker_mod, "log_event", record)

    worker = PipelineWorker(orchestrator, worker_id="p37-close-raises", queue=_IdleQueue())
    monkeypatch.setattr(worker, "_warm_models", lambda: None)
    monkeypatch.setattr(worker, "_loop", lambda: None)
    worker.start()

    assert worker.stop(timeout=FINISH_TIMEOUT) is False, (
        "stop() claimed success although the deferred-evaluation intake could not be closed"
    )

    messages = [event[1] for event in events]
    assert "could not close deferred-evaluation intake" in messages, (
        f"the close failure left no evidence: {events!r}"
    )
    incomplete = [event for event in events if event[1] == "gpu worker stop incomplete"]
    assert incomplete, f"stop did not report itself incomplete: {events!r}"
    assert incomplete[0][2]["deferred_evaluation_closed"] is False
    assert incomplete[0][2]["thread_alive"] is False
    assert "gpu worker stopped" not in messages, (
        "stop logged success although it reported a failed close"
    )
