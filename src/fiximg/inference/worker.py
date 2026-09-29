"""GPU execution worker — DB-backed queue consumer (plan §2.6 / §12 / §28).

Topology::

    API/Gradio  →  queue (tasks table by default)  →  worker  →  inference runtime

V3 changes vs V2:

* the worker talks to :class:`~fiximg.infrastructure.queue.base.QueueBackend`
  instead of calling the repository directly, so Redis Streams can be swapped in
  without touching this file (plan §5.5);
* claimed tasks get a **lease** and a periodic **heartbeat**, so a crashed
  worker is detected by lease expiry rather than a wall-clock guess (§2.6);
* failed attempts are **retried** with backoff until ``max_attempts`` (§2.6);
* queue wait, run duration and retries are recorded as metrics (§2.10).

One class serves two deployment modes:
  - standalone process: ``python worker.py`` (compose ``worker`` service)
  - in-process thread:  ``FIXIMG_INLINE_WORKER=true`` (single-container default)
"""
from __future__ import annotations

import threading
import time
from typing import Any

from fiximg.config import settings
from fiximg.domain.enums import EventType
from fiximg.domain.errors import TaskCancelledError
from fiximg.infrastructure.db import timestamps
from fiximg.infrastructure.observability.logging import (
    clear_context,
    get_logger,
    log_event,
    new_request_id,
    set_gpu_id,
    set_task_id,
    set_user_id,
    set_worker_id,
)
from fiximg.infrastructure.observability.metrics import MetricName, metrics
from fiximg.infrastructure.observability.tracing import tracer

logger = get_logger("fiximg.worker")


class PipelineWorker:
    """Polls the queue and executes claimed runs (single GPU slot by default)."""

    def __init__(
        self,
        orchestrator,
        poll_seconds: float = 0.5,
        worker_id: str | None = None,
        stale_timeout: float | None = None,
        queue=None,
        lease_seconds: float | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._poll_seconds = max(0.05, float(poll_seconds))
        self._stale_timeout = float(
            stale_timeout if stale_timeout is not None else settings.worker_lease_seconds
        )
        self._lease_seconds = float(
            lease_seconds if lease_seconds is not None else settings.worker_lease_seconds
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_id = worker_id or f"worker-{id(self):x}"
        self._last_stale_scan = 0.0
        self._last_sweep = 0.0
        self._queue = queue  # resolved lazily so tests can inject a fake
        self._topology: Any = None

    # ------------------------------------------------------------- lifecycle
    @property
    def queue(self):
        if self._queue is None:
            from fiximg.infrastructure.queue import get_queue

            self._queue = get_queue()
        return self._queue

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        # §3.5.3: this process is the one that executes inference, so it is the one
        # that has to preload. The call used to live only in `create_app()`, which
        # means a standalone GPU worker booted cold and paid the weight load plus
        # the first-request latency — while a pure API node, which runs nothing,
        # was the process holding the weights.
        self._warm_models()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="gpu-worker", daemon=True)
        self._thread.start()
        log_event(
            logger, "INFO", "gpu worker started",
            worker_id=self._worker_id,
            pinned_device=self.topology.pinned_device,
            # Two different knobs, and reading the wrong one as "not applied" costs an
            # operator a debugging session: `claim_serves` is what
            # `FIXIMG_WORKER_CAPABILITIES` restricts *this* worker's claims to, while
            # `gpu_capabilities` is what the GPU topology says the pinned card serves.
            # Logging only the second made a correctly restricted worker announce
            # `served_capabilities: null` and look as if its setting had been ignored.
            claim_serves=sorted(getattr(self.queue, "capabilities", ()) or ()) or None,
            gpu_capabilities=sorted(
                self.topology.capabilities_of(self.topology.pinned_device)
            ) or None,
        )

    def _warm_models(self) -> None:
        """Preload the models whose effective warmup strategy is `startup`."""
        from fiximg.inference.model_manager import model_manager

        try:
            warmed = model_manager.warm_at_process_start()
        except Exception as exc:  # noqa: BLE001 — a cold worker still serves work
            log_event(logger, "WARNING", "startup warmup skipped",
                      worker_id=self._worker_id, error=str(exc))
            return
        if warmed:
            log_event(logger, "INFO", "models warmed at worker start",
                      worker_id=self._worker_id, models=warmed)

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        # Deferred evaluation runs off the critical path (plan §3.16); give the
        # pool a moment to finish the metrics of the task that just completed.
        try:
            from fiximg.inference.runtime import shutdown_eval_pool

            shutdown_eval_pool(wait=False)
        except Exception:  # noqa: BLE001 — shutdown must never raise
            pass
        log_event(logger, "INFO", "gpu worker stopped", worker_id=self._worker_id)

    # ------------------------------------------------------------- execution
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                task = self.queue.claim(self._worker_id, lease_seconds=self._lease_seconds)
            except Exception as exc:  # noqa: BLE001 — a broken claim must not kill the loop
                log_event(logger, "ERROR", "claim failed", error=str(exc))
                time.sleep(max(self._poll_seconds, 1.0))
                continue
            # Housekeeping is time-guarded inside `_scan_stale`, so it runs whether
            # or not the queue is busy. It used to sit in the idle branch, which
            # meant a worker under continuous load never requeued expired leases
            # and never swept expired artifacts — the TTL was a number nothing read
            # precisely on the deployment where it fills up fastest.
            self._scan_stale()
            if task is None:
                self._stop.wait(self._poll_seconds)
                continue
            try:
                self._execute(task)
            except Exception as exc:  # noqa: BLE001 — one bad task must not end the worker
                # `_execute` guards the *pipeline*; anything it raises before that
                # guard — a broken log field, a failed event write, a typo in the
                # correlation setup — used to escape the thread's target and stop
                # this worker polling at all, with the task left `running` until
                # its lease expired an hour later. The failure looked like a hung
                # queue, not like a crashed worker.
                log_event(logger, "ERROR", "worker raised outside the task path",
                          task_id=task.id, error=str(exc))
                self._release_failed_claim(task, exc)

    def _release_failed_claim(self, task, exc: Exception) -> None:
        """Put a task that never reached the pipeline back into a sane state."""
        from fiximg.infrastructure.db.repositories import task_repository as task_repo

        try:
            task_repo.fail_task(task.id, str(exc), 0, error_code="worker_error")
        except Exception as fail_exc:  # noqa: BLE001 — the retry below still decides
            log_event(logger, "ERROR", "could not record the worker failure",
                      task_id=task.id, error=str(fail_exc))
        self._handle_failure(task, exc)

    def _execute(self, task) -> None:
        task_id = task.id
        task_type = task.task_type
        # §3.4.2 correlation: every log line of this run carries task/worker ids.
        new_request_id()
        set_user_id(task.user_id)
        set_task_id(task_id)
        set_worker_id(self._worker_id)
        set_gpu_id(self._gpu_id())

        wait_seconds = self._queue_wait_seconds(task)
        if wait_seconds is not None:
            metrics.observe(MetricName.QUEUE_WAIT_SECONDS, wait_seconds)

        log_event(
            logger, "INFO", "task claimed",
            task_id=task_id, task_type=task_type, attempt=task.attempt_count,
        )
        # §2.5: `task.started` is documented in `docs/api.md`, and the enum has
        # always had it, but nothing ever emitted it — an SSE client watching the
        # queued path jumped from `task.enqueued` straight to `stage.started`. The
        # claim is where this task became running, so that is where the event
        # belongs; the synchronous path emits the same name where it calls
        # `start_task`.
        from fiximg.infrastructure.db.repositories import task_repository as task_repo

        task_repo.add_event(
            EventType.TASK_STARTED.value, task_id=task_id, user_id=task.user_id,
            message=f"attempt {task.attempt_count}",
            data={"task_type": task_type, "attempt": task.attempt_count},
        )
        heartbeat = _Heartbeat(self.queue, task_id, settings.worker_heartbeat_seconds)
        heartbeat.start()
        started = time.perf_counter()
        try:
            # §2.10: one span per task, so the stage spans the runtime emits
            # nest under it in whatever backend the deployment installed.
            with tracer.span(
                "task.execute", task_id=task_id, task_type=task_type,
                worker_id=self._worker_id, user_id=task.user_id,
                attempt=task.attempt_count, queue_wait_seconds=wait_seconds,
            ):
                self._orchestrator.execute_queued(task_id, task_type)
        except TaskCancelledError:
            # The user's cancel is a decision, not a fault: no retry, and the claim is
            # released so no transport re-delivers a task that must not run again.
            log_event(logger, "INFO", "task cancelled; stopped at the stage boundary",
                      task_id=task_id, worker_id=self._worker_id)
            self._ack(task_id)
        except Exception as exc:  # noqa: BLE001
            # execute_queued already marked the row failed; decide on a retry.
            self._handle_failure(task, exc)
        else:
            self._ack(task_id)
        finally:
            heartbeat.stop()
            metrics.observe(MetricName.TASK_DURATION_SECONDS, time.perf_counter() - started)
            clear_context()

    def _ack(self, task_id: str) -> None:
        """Release the transport's claim once a task will not run again.

        Only Redis has something to do — XACK the stream entry; the SQLite queue
        finishes everything in the row. Skipping this left every completed task
        pending until its lease expired, where `XAUTOCLAIM` handed it to a worker
        for nothing, and kept a `task_id -> record id` map growing one entry per
        completed task for the life of the process.

        A task that is being *retried* is deliberately not acknowledged here:
        `queue.retry()` republishes it, and the manual-retry path does the same,
        so the entry it needs already exists.
        """
        try:
            self.queue.ack(task_id)
        except Exception as exc:  # noqa: BLE001 — an ack must not fail a finished task
            log_event(logger, "WARNING", "queue ack failed",
                      task_id=task_id, error=str(exc))

    def _handle_failure(self, task, exc: Exception) -> None:
        """Retry with backoff while attempts remain (plan §2.6)."""
        if task.attempt_count >= task.max_attempts:
            log_event(
                logger, "ERROR", "task failed (attempts exhausted)",
                task_id=task.id, attempt=task.attempt_count, error=str(exc),
            )
            # Terminal in the automatic sense; safe to release because a manual
            # retry republishes the task itself (see `TaskService.retry_task`).
            self._ack(task.id)
            return
        try:
            # `FIXIMG_TASK_RETRY_BACKOFF` is the configured value and
            # `retry_task`'s own default, but the transport signature defaults to
            # 0.0 and this call passed nothing — so the automatic retry path
            # re-claimed a hot-failing task immediately instead of waiting.
            requeued = self.queue.retry(task.id,
                                        delay_seconds=settings.task_retry_backoff_seconds)
        except Exception as retry_exc:  # noqa: BLE001
            log_event(logger, "ERROR", "retry scheduling failed", error=str(retry_exc))
            return
        if requeued:
            metrics.inc(MetricName.WORKER_RETRY_TOTAL)
            log_event(
                logger, "WARNING", "task requeued for retry",
                task_id=task.id, attempt=task.attempt_count, error=str(exc),
            )

    def _scan_stale(self) -> None:
        """Requeue running tasks whose lease expired (checked every ~60s)."""
        now = time.monotonic()
        if now - self._last_stale_scan < 60.0:
            return
        self._last_stale_scan = now
        try:
            from fiximg.infrastructure.db.repositories import task_repository as task_repo

            requeued = task_repo.reset_stale_running(self._stale_timeout)
            if requeued:
                log_event(logger, "WARNING", "requeued stale running tasks", count=requeued)
        except Exception as exc:  # noqa: BLE001
            log_event(logger, "ERROR", "stale scan failed", error=str(exc))
        self._sweep_artifacts(now)

    def _sweep_artifacts(self, now: float | None = None) -> bool:
        """Reclaim expired run directories on a fixed cadence (plan §2.8).

        The worker owns this, not the request path. Retention used to be reachable
        only from `PipelineOrchestrator.run()`, i.e. the synchronous path, so on
        the documented API + GPU-worker topology nothing ever swept and the TTL was
        a number nothing read. Returns whether a sweep ran.
        """
        moment = time.monotonic() if now is None else now
        interval = float(settings.artifact_sweep_seconds)
        if interval <= 0 or moment - self._last_sweep < interval:
            return False
        self._last_sweep = moment
        try:
            from fiximg.infrastructure.db import timestamps
            from fiximg.infrastructure.storage import local as artifact_service
            from fiximg.infrastructure.storage import purge_expired_published

            artifact_service.purge_stale_runs()
            # The object store has no directory to walk, so the same TTL is
            # applied through the database instead.
            purge_expired_published(timestamps.cutoff(settings.result_ttl))
        except Exception as exc:  # noqa: BLE001 — retention cannot fail a run
            log_event(logger, "WARNING", "artifact sweep failed", error=str(exc))
        return True

    # ------------------------------------------------------------ internals
    @property
    def topology(self):
        """GPU topology this worker runs under (plan §4.3 Step 4)."""
        if self._topology is None:
            from fiximg.inference.gpu import GpuTopology

            self._topology = GpuTopology.from_settings()
        return self._topology

    def _gpu_id(self) -> int:
        """Device recorded on this run's log lines.

        A pinned worker always reports its own device; otherwise the stage-level
        routing in the runtime decides per capability, and this is the default.
        """
        return self.topology.pinned_device if self.topology.pinned_device is not None \
            else self.topology.default_device

    @staticmethod
    def _queue_wait_seconds(task) -> float | None:
        """Seconds between task creation and claim (plan §2.10)."""
        if not task.created_at:
            return None
        try:
            return timestamps.age_seconds(task.created_at)
        except ValueError:
            return None

    # ------------------------------------------------------------ introspection
    def depth(self) -> int:
        """Approximate queue depth (queued rows not yet claimed)."""
        try:
            return self.queue.depth()
        except Exception:  # noqa: BLE001
            return -1

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def worker_id(self) -> str:
        return self._worker_id


class _Heartbeat:
    """Refreshes a running task's lease on a background thread (plan §2.6)."""

    def __init__(self, queue, task_id: str, interval: float) -> None:
        self._queue = queue
        self._task_id = task_id
        self._interval = max(1.0, float(interval))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="task-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._queue.heartbeat(self._task_id)
            except Exception:  # noqa: BLE001 — heartbeat is best-effort
                return


__all__ = ["PipelineWorker"]
