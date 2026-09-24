"""GPU execution worker — DB-backed queue consumer (plan sections 12 and 28).

Topology (Phase 4 complete):
  API/Gradio  ->  tasks table (status=queued)  ->  GPU worker process  ->  orchestrator

The queue is the `tasks` table itself: `POST /api/v1/tasks` (and the future
Gradio async path) persist the input image under the artifact storage and
insert a queued row; a worker claims rows atomically via
`task_repository.claim_next_task()` and executes them through the same
PipelineOrchestrator used by the synchronous path. Swapping this transport for
Redis/Celery later changes only claim/enqueue — stage and orchestrator code is
untouched.

One class serves two deployment modes:
  - standalone process: `python worker.py` (docker-compose `worker` service)
  - in-process thread:  FIXIMG_INLINE_WORKER=true (single-container default)
"""
import threading
import time

from app.core.logging import get_logger, log_event

logger = get_logger("fiximg.worker")


class PipelineWorker:
    """Polls the tasks table and executes queued runs serially (single GPU slot)."""

    def __init__(
        self,
        orchestrator,
        poll_seconds: float = 0.5,
        worker_id: str | None = None,
        stale_timeout: float = 3600.0,
    ) -> None:
        self._orchestrator = orchestrator
        self._poll_seconds = max(0.05, float(poll_seconds))
        self._stale_timeout = stale_timeout
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_id = worker_id or f"worker-{id(self):x}"
        self._last_stale_scan = 0.0

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="gpu-worker", daemon=True
        )
        self._thread.start()
        log_event(logger, "INFO", "gpu worker started", worker_id=self._worker_id)

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        log_event(logger, "INFO", "gpu worker stopped", worker_id=self._worker_id)

    # ------------------------------------------------------------- execution
    def _loop(self) -> None:
        from app.repositories import task_repository as task_repo

        while not self._stop.is_set():
            try:
                task = task_repo.claim_next_task(self._worker_id)
            except Exception as exc:  # noqa: BLE001
                log_event(logger, "ERROR", "claim failed", error=str(exc))
                time.sleep(max(self._poll_seconds, 1.0))
                continue
            if task is None:
                self._scan_stale(task_repo)
                self._stop.wait(self._poll_seconds)
                continue
            self._execute(task)

    def _execute(self, task: dict) -> None:
        task_id = task["id"]
        task_type = task["task_type"]
        # §25 correlation: each task run gets its own request_id/user_id in logs.
        from app.core.logging import new_request_id, set_gpu_id, set_user_id

        from app.inference.model_manager import resolve_gpu

        from app.core.config import settings

        new_request_id()
        set_user_id(task.get("user_id"))
        set_gpu_id(resolve_gpu(settings.device))
        log_event(logger, "INFO", "task claimed", task_id=task_id, task_type=task_type)
        try:
            self._orchestrator.execute_queued(task_id, task_type)
        except Exception as exc:  # noqa: BLE001
            # execute_queued already marked the row failed; keep the log only.
            log_event(
                logger, "ERROR", "task failed", task_id=task_id, error=str(exc)
            )

    def _scan_stale(self, task_repo) -> None:
        """Requeue running tasks from crashed workers (checked every ~60s)."""
        now = time.monotonic()
        if now - self._last_stale_scan < 60.0:
            return
        self._last_stale_scan = now
        try:
            requeued = task_repo.reset_stale_running(self._stale_timeout)
            if requeued:
                log_event(logger, "WARNING", "requeued stale running tasks", count=requeued)
        except Exception as exc:  # noqa: BLE001
            log_event(logger, "ERROR", "stale scan failed", error=str(exc))

    # ------------------------------------------------------------ introspection
    def depth(self) -> int:
        """Approximate queue depth (queued rows not yet claimed)."""
        from app.repositories import task_repository as task_repo

        try:
            return task_repo.queued_count()
        except Exception:  # noqa: BLE001
            return -1

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())
