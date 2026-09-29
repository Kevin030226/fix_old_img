"""Task service: the single entry the web/UI layers use (plan §8 / §12).

Hides whether execution is synchronous (Gradio callback, V1-compatible) or
queued (DB-backed GPU worker) — callers always get a documented result shape.

Queued path (plan §12/§28):

  1. persist the uploaded image under artifact storage (storage/tasks/...)
  2. insert a ``queued`` tasks row with input_path, priority and idempotency key
  3. a PipelineWorker (inline thread or standalone process) claims and runs it
"""
import json
import os
import time

import numpy as np
from PIL import Image

from fiximg.application.pipeline_modes import PIPELINE_MODES
from fiximg.domain.enums import TaskStatus
from fiximg.infrastructure.db.connection import integrity_errors
from fiximg.infrastructure.db.engine import get_conn
from fiximg.infrastructure.db.repositories import task_repository as task_repo
from fiximg.infrastructure.db.repositories.user_repository import get_user as task_repo_user_get
from fiximg.infrastructure.observability.logging import get_logger, log_event
from fiximg.inference.runtime import PipelineOrchestrator

logger = get_logger("fiximg.tasks")


class TaskService:
    """Create and query processing tasks; delegate execution to the orchestrator."""

    def __init__(self, orchestrator: PipelineOrchestrator | None = None) -> None:
        self.orchestrator = orchestrator or PipelineOrchestrator()

    # ---------------------------------------------------------------- create
    def submit(self, mode: str, input_image, user_state: dict, options: dict | None = None):
        """Run a pipeline synchronously for a Gradio callback (V1-compatible UX).

        mode: restore / restore_scratch / detect / colorize (see pipeline_modes).
        Returns (result_path, evaluation_text_or_None).
        """
        self._require_password_changed(user_state)
        task_type = self._task_type_for(mode)
        return self.orchestrator.run(input_image, user_state, task_type, options=options)

    def submit_queued(
        self, task_id: str, input_image, user_state: dict, mode: str,
        options: dict | None = None, ground_truth_image=None,
        idempotency_key: str | None = None, priority: int = 0,
    ) -> bool:
        """Enqueue a task for the GPU worker (Phase-4 path).

        The image is persisted under artifact storage before the row is created,
        so any worker process can pick the task up. Returns True on success;
        False when the backlog already has ``FIXIMG_WORKER_MAX_QUEUE`` tasks in it,
        in which case *no row and no artifact directory* are created and the caller
        turns the refusal into :class:`QueueFullError`.

        ``idempotency_key`` (plan §2.6) makes submission safely retryable: a
        repeated key returns the existing task instead of running the work twice.
        """
        self._require_password_changed(user_state)
        task_type = self._task_type_for(mode)
        username = (user_state or {}).get("username", "unknown") or "unknown"

        input_image = self._to_pil(input_image)

        from fiximg.config import settings
        from fiximg.infrastructure.observability.metrics import MetricName, metrics
        from fiximg.infrastructure.storage import local as artifact_service
        from fiximg.infrastructure.storage import publish_local_file

        if idempotency_key:
            existing = task_repo.find_by_idempotency_key(idempotency_key)
            if existing:
                log_event(
                    logger, "INFO", "duplicate submission ignored",
                    task_id=existing["id"], idempotency_key=idempotency_key,
                )
                return True

        max_queue = settings.worker_max_queue
        if max_queue > 0 and task_repo.queued_count() >= max_queue:
            task_repo.add_event(
                "task.rejected", level="warning", user_id=username,
                message="queue saturated; submission refused",
            )
            return False

        run_dir = artifact_service.create_run_dir(task_id)
        input_path = artifact_service.save_input_image(run_dir, task_id, input_image)
        # Register the input as an artifact at enqueue time so the API can list
        # it before the worker has run (plan §3.8 /artifacts contract).
        #
        # On a remote store the upload is published here too, and the key recorded.
        # Without it the split topology only half-worked: the *result* travelled,
        # but a worker on another node had no way to read the picture this node
        # received, so it claimed the task and failed inside stage 1.
        task_repo.add_artifact(task_id, "input", input_path, "image/png")
        input_key = publish_local_file(input_path, op="upload")
        if input_key:
            task_repo.set_artifact_uri(task_id, "input", input_key)

        # §16 Ground Truth mode: persist the optional reference photo next to
        # the input and hand its path to the orchestrator via options.
        gt_path = None
        if ground_truth_image is not None:
            from PIL import Image

            gt_dir = os.path.join(run_dir, "ground_truth")
            os.makedirs(gt_dir, exist_ok=True)
            gt_path = os.path.join(gt_dir, f"{task_id}_reference.png")
            if isinstance(ground_truth_image, Image.Image):
                ground_truth_image.convert("RGB").save(gt_path)
            else:
                Image.fromarray(ground_truth_image).convert("RGB").save(gt_path)
            task_repo.add_artifact(task_id, "ground_truth", gt_path, "image/png")

        if gt_path:
            options = dict(options or {})
            options["_ground_truth_path"] = gt_path
        # §4.3 Step 4: record what the task needs so a specialised worker can
        # skip it. Computed from the planner — and from the same options that decide
        # the plan, or the hint describes a task nobody submitted.
        try:
            required = self.orchestrator.planner.required_capabilities(task_type, options)
        except Exception:  # noqa: BLE001 — a routing hint must never block enqueue
            required = set()
        try:
            task_repo.create_task(
                task_id, task_type, username, input_path=input_path, options=options,
                priority=priority, idempotency_key=idempotency_key,
                required_capabilities=required,
            )
        except integrity_errors():
            # Lost a race on the same idempotency key: the other submission won.
            get_conn().rollback_after_conflict()
            existing = task_repo.find_by_idempotency_key(idempotency_key or "")
            if existing:
                log_event(
                    logger, "INFO", "duplicate submission raced; kept the first",
                    task_id=existing["id"], idempotency_key=idempotency_key,
                )
                return True
            raise
        task_repo.add_event(
            "task.enqueued", task_id=task_id, user_id=username,
            message=f"queued {task_type}",
            data={"task_type": task_type, "priority": priority},
        )
        metrics.inc(MetricName.TASK_SUBMIT_TOTAL, task_type=task_type)
        self._notify_queue(task_id)
        # §2.10's queue depth, recorded where it is actually meaningful: the
        # backlog this submission joined. A sampled gauge of the same name taken
        # on a timer would report idle polls as much as real pressure.
        try:
            from fiximg.infrastructure.queue import get_queue

            queue = get_queue()
            metrics.observe(MetricName.QUEUE_DEPTH, queue.depth(), backend=queue.kind)
        except Exception as exc:  # noqa: BLE001 — a metric never fails a submit
            log_event(logger, "DEBUG", "queue depth not observed",
                      task_id=task_id, error=str(exc))
        log_event(logger, "INFO", "task enqueued", task_id=task_id, task_type=task_type)
        return True

    @staticmethod
    def _notify_queue(task_id: str) -> None:
        """Tell the configured transport about a newly queued task.

        Writing the row is not enough on the Redis topology: ``claim()`` reads the
        stream, and the database only decides *what* a claim means (lease,
        attempts, capabilities). A submission that skipped this step left every
        worker on ``FIXIMG_QUEUE_BACKEND=redis`` polling an empty stream, so the
        task sat queued forever. The SQLite transport's ``enqueue`` is a documented
        no-op, which is why the gap was invisible without the Redis path.

        Best-effort by design: a Redis hiccup must not fail a submission the
        database already accepted, and a lost wake-up is recovered by the next
        ``retry()``/stale-lease sweep.
        """
        try:
            from fiximg.infrastructure.queue import get_queue

            get_queue().enqueue(task_id)
        except Exception as exc:  # noqa: BLE001 — transport is an optimisation
            log_event(
                logger, "WARNING", "queue notify failed",
                task_id=task_id, error=str(exc),
            )

    def enqueue(
        self, input_image, user_state: dict, mode: str, options: dict | None = None,
        ground_truth_image=None, idempotency_key: str | None = None,
        priority: int = 0,
    ) -> str:
        """Enqueue a task and return its task_id (UI/API async path, plan §23).

        ``priority`` is the queue position (plan §2.6): the claim orders
        ``priority DESC, created_at``, so a larger value is claimed first. It defaults
        to 0 — FIFO — which is all the web UI ever asks for; the JSON API exposes the
        field and refuses anything above ``FIXIMG_PRIORITY_MAX``.

        Raises :class:`QueueFullError` when the queue is saturated, so the HTTP
        layer maps it onto 503 without inspecting a boolean.
        """
        from fiximg.domain.errors import QueueFullError
        from fiximg.inference.runtime import new_task_id

        if idempotency_key:
            existing = task_repo.find_by_idempotency_key(idempotency_key)
            if existing:
                return existing["id"]

        task_id = new_task_id()
        ok = self.submit_queued(
            task_id, input_image, user_state, mode, options=options,
            ground_truth_image=ground_truth_image, idempotency_key=idempotency_key,
            priority=priority,
        )
        if not ok:
            raise QueueFullError("Task queue is saturated; please try again later")
        return task_id

    def has_worker(self) -> bool:
        """True when a queue consumer is available in this process.

        The inline worker thread polls the DB queue; a standalone worker
        process cannot be seen from here, so deployments running one must set
        FIXIMG_HAS_EXTERNAL_WORKER=true (or accept the sync fallback in the UI).
        """
        from fiximg.config import settings

        if settings.inline_worker:
            try:
                from fiximg.app_factory import get_inline_worker

                worker = get_inline_worker()
                return bool(worker and worker.is_running())
            except Exception:  # noqa: BLE001
                return False
        return settings.external_worker

    def queue_depth(self) -> int | None:
        """How many tasks are waiting to be claimed, or None when unknowable.

        Plan §3.9.1 puts a queue readout in the progress panel, and the number the
        UI must show is the *deployment's* backlog — which for a standalone worker is
        in the database, not in this process. The queue backends all answer through
        the repository for that reason, so this reads the same quantity `/health/ready`
        reports instead of a process-local view that would always say 0 on an API node.
        """
        try:
            from fiximg.infrastructure.queue import get_queue

            return get_queue().depth()
        except Exception:  # noqa: BLE001 — a progress panel is not a probe
            return None

    # ---------------------------------------------------------------- queries
    def get_task(self, task_id: str):
        return task_repo.get_task(task_id)

    def auto_preview(self, input_image, options: dict | None = None) -> dict:
        """What Auto Restore is about to do with *this* picture (plan §3.9.1's 自动分析).

        §3.9.1 puts an "Auto Analysis" row in the input area, above the switches; §2.11
        lists it as one of the three things the input column owns. Until now the analyzer
        ran only inside ``_execute_core``, so the user met its verdict for the first time
        after the GPU work, and the Auto tab's own caption ("only when the analysis finds
        a face") could not be checked against anything.

        The stages come from the same ``plan_auto_restore(analysis, options)`` call the
        worker will make, so the preview cannot drift from the run: a second copy of the
        thresholds is exactly how a preview starts lying. The two notes are the difference
        between what the image said and what the submitted switches decided.

        Cost is CPU-only CV on a 512 px copy (measured 7–11 ms per image on this box, with
        a face backend present); no GPU tensor, no restoration weights, so it is safe in
        the web process, which is where this is called from.
        """
        from fiximg.inference.analyzer import analyze_image, format_analysis_summary

        image = self._to_pil(input_image)
        try:
            analysis = analyze_image(image)
        except Exception as exc:  # noqa: BLE001 — an unreadable upload must not break the tab
            log_event(logger, "WARNING", "auto preview failed", error=str(exc))
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}

        plan = self.orchestrator.planner.plan_auto_restore(analysis, options)
        ran = self._stage_labels(plan)
        wanted = self._stage_labels(
            self.orchestrator.planner.plan_auto_restore(analysis)
        )
        return {
            "available": True,
            "analysis": analysis,
            "summary": format_analysis_summary(analysis),
            "stages": ran,
            #: Stages the image asked for that the submitted switches removed.
            "declined": [name for name in wanted if name not in ran],
            #: Stages the switches added even though the image did not ask for them.
            "forced": [name for name in ran if name not in wanted],
        }

    def _stage_labels(self, plan) -> list[str]:
        """The names this plan will be recorded under, taken from the stages themselves.

        A plan step is keyed by its registry name while the stage decides the name it
        reports: ``global_restore(with_scratch=True)`` records itself as ``scratch_repair``.
        A preview that printed registry names would show a list the history panel and
        ``GET /tasks/{id}`` then contradict, so the labels come from the built stage —
        construction only sets flags and never loads weights, which is what the enqueue
        capability hint already relies on.
        """
        from fiximg.domain.errors import StageNotAvailableError

        labels = []
        for name, kwargs in plan.stages:
            try:
                stage = self.orchestrator.planner.build_stage(name, kwargs or {})
            except StageNotAvailableError:
                labels.append(name)
                continue
            labels.append(getattr(stage, "name", name) or name)
        return labels

    def task_priority(self, task_id: str) -> int | None:
        """The queue position stored for this task, or None if there is no readable row.

        `POST /tasks` reports it so a caller can verify the position it bought, and it is
        read from the row rather than echoed from the request: a replay under the same
        ``Idempotency-Key`` returns the task that already existed, whose priority is its
        own. ``None`` (no row readable in this process) is reported as ``null`` rather
        than guessed at, because a reply that invents a queue position is worse than one
        that says it does not know.
        """
        row = task_repo.get_task(task_id)
        return None if row is None else row.get("priority")

    @staticmethod
    def planner_decisions(result_path: str | None) -> dict | None:
        """The analyzer/planner decisions a finished run recorded (§10).

        Lives here rather than in the report route because the web UI shows the
        same block, and two copies of "where the run wrote its decisions" is how
        the two views start disagreeing about what a task did.
        """
        if not result_path:
            return None
        decisions_path = os.path.join(
            os.path.dirname(os.path.dirname(result_path)), "report.json"
        )
        if not os.path.exists(decisions_path):
            return None
        try:
            with open(decisions_path, encoding="utf-8") as handle:
                return json.load(handle).get("planner_decisions")
        except (OSError, ValueError):
            return None

    def require_task(self, task_id: str):
        """Like :meth:`get_task` but raises :class:`TaskNotFoundError`."""
        from fiximg.domain.errors import TaskNotFoundError

        row = task_repo.get_task(task_id)
        if not row:
            raise TaskNotFoundError(f"Task not found: {task_id}", details={"task_id": task_id})
        return row

    def list_tasks(self, limit: int = 50, user_id: str | None = None):
        return task_repo.list_tasks(limit=limit, user_id=user_id)

    def stats_snapshot(self, window_days: int | None = None) -> dict:
        """The task/stage aggregates, computed in SQL over the real tables.

        Both readers go through here: ``GET /api/v1/stats`` adds its process-local
        probes on top, and the admin panel's statistics box renders this text-free
        shape. They used to be two different questions — the endpoint counted
        ``tasks``, the panel counted the legacy ``history`` table — so an admin could
        watch the two numbers disagree about the same deployment.
        """
        return task_repo.task_stats(window_days=window_days)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task that has not finished, and say so on the event stream.

        `task.cancelled` is one of the events the SSE stream treats as closing, so
        writing only the row left every live `GET /tasks/{id}/events` polling until its
        900-second deadline: the client could not tell "cancelled" from "hung"
        (plan §2.5-4).

        A task that was already running is reported with what actually happens to it —
        the worker stops at the next stage boundary — because "cancelled" arriving
        while stages are still being reported otherwise looks like a bug.
        """
        row = task_repo.get_task(task_id)
        was_running = bool(row) and row.get("status") == TaskStatus.RUNNING.value
        cancelled = task_repo.cancel_task(task_id)
        if cancelled:
            task_repo.add_event(
                "task.cancelled", task_id=task_id,
                user_id=(row or {}).get("user_id"),
                message=("cancelled while running; the worker stops at the next stage"
                         if was_running else "cancelled by request"),
                data={"effective": "stage boundary" if was_running else "immediate"},
            )
        return cancelled

    def retry_task(self, task_id: str) -> bool:
        """Requeue a failed/cancelled task (plan §3.8 ``POST /tasks/{id}/retry``).

        Attempts are reset so an operator can retry a task that exhausted its
        automatic retries; the plan's manual-retry endpoint is explicit about
        being an operator action.
        """
        row = task_repo.get_task(task_id)
        if not row:
            from fiximg.domain.errors import TaskNotFoundError

            raise TaskNotFoundError(f"Task not found: {task_id}", details={"task_id": task_id})
        if row["status"] in ("running", "queued"):
            from fiximg.domain.errors import TaskAlreadyFinishedError

            raise TaskAlreadyFinishedError(
                f"Task is still {row['status']}; wait for it to finish before retrying",
                details={"task_id": task_id, "status": row["status"]},
            )
        requeued = task_repo.requeue_for_manual_retry(task_id)
        if requeued:
            task_repo.add_event(
                "task.retrying", task_id=task_id, user_id=row.get("user_id"),
                message="manual retry requested",
            )
            # Same rule as a fresh submission (§2.6): the row says the task is
            # runnable, but on the Redis topology a worker only learns about it
            # from the stream. This path called the repository directly and
            # skipped the transport, so a retried task waited for
            # `XAUTOCLAIM`'s lease window to expire before anyone picked it up.
            self._notify_queue(task_id)
        return requeued

    def get_task_artifacts(self, task_id: str):
        return task_repo.get_artifacts(task_id)

    def get_task_stages(self, task_id: str):
        return task_repo.get_task_stages(task_id)

    def get_task_events(self, task_id: str, after_id: int = 0, limit: int = 200):
        """Ordered event log for a task (SSE replay + tailing, plan §3.8)."""
        return task_repo.list_task_events(task_id, after_id=after_id, limit=limit)

    def first_face_crop(self, task_id: str) -> str | None:
        """The first aligned crop the detection stage wrote, or None.

        Which directory a stage fills and what counts as an image in it is
        pipeline knowledge, so the view asks for the crop instead of walking the
        stage's output tree itself (plan §3.2: the UI talks to services).
        """
        from fiximg.inference.backends.legacy_cli import list_images

        for row in self.get_task_artifacts(task_id) or []:
            if row.get("kind") != "faces_dir":
                continue
            directory = row.get("uri") or row.get("path")
            if not directory or not os.path.isdir(directory):
                continue
            crops = list_images(directory)
            if crops:
                return os.path.join(directory, crops[0])
        return None

    #: Artifact kinds a remote store is consulted for. The finished image is the
    #: only file another node has to be able to serve; a stage's intermediates are
    #: meaningless next to a run that did not produce them.
    _PUBLISHED_KINDS = ("output",)

    def _recover_from_store(self, task_id: str, expected_path: str) -> bool:
        """Re-materialise a published artifact locally; False when not available."""
        from fiximg.domain.artifacts import ArtifactRef
        from fiximg.infrastructure.observability.metrics import MetricName, metrics
        from fiximg.infrastructure.storage import get_artifact_store

        row = next(
            (r for r in task_repo.get_artifacts(task_id) or []
             if r.get("kind") in self._PUBLISHED_KINDS and r.get("uri")),
            None,
        )
        if row is None:
            return False
        store = get_artifact_store()
        started = time.perf_counter()
        try:
            data = store.get_bytes(ArtifactRef(key=row["uri"], backend=store.backend))
        except Exception as exc:  # noqa: BLE001 — treated as "not there", then 410
            log_event(
                logger, "WARNING", "artifact recovery failed",
                task_id=task_id, uri=row["uri"], error=str(exc),
            )
            return False
        # The read half of §2.10's `artifact_io_seconds`; the write half is recorded
        # where the pipeline publishes. A cross-node download is the one request
        # type whose latency this dominates, so it is the number worth graphing.
        metrics.observe(
            MetricName.ARTIFACT_IO_SECONDS, time.perf_counter() - started,
            op="recover", store=store.backend,
        )
        os.makedirs(os.path.dirname(expected_path), exist_ok=True)
        with open(expected_path, "wb") as handle:
            handle.write(data)
        return True

    def task_result_path(self, task_id: str) -> str:
        """Path of the finished output, materialised locally if needed.

        The two "there is no file here" cases *raise* rather than returning None:
        not-yet-ready and gone-forever mean different things to a client (409 vs
        410), and an ``Optional`` return would ask every caller to invent a third
        answer for a case that cannot happen. A remote store gets one extra chance
        first — see :meth:`_recover_from_store`, which is what lets an API node
        without the worker's disk serve a download.
        """
        from fiximg.domain.errors import ArtifactExpiredError, TaskNotReadyError

        row = self.require_task(task_id)
        if row["status"] != "completed" or not row.get("result_path"):
            raise TaskNotReadyError(
                f"Task not completed (status={row['status']})",
                details={"task_id": task_id, "status": row["status"]},
            )
        path = row["result_path"]
        if not os.path.exists(path) and not self._recover_from_store(task_id, path):
            raise ArtifactExpiredError(
                "Result expired and was reclaimed by the retention sweeper",
                details={"task_id": task_id},
            )
        return path

    # ------------------------------------------------------------- internals
    @staticmethod
    def _to_pil(input_image):
        """Normalise Gradio (numpy) / PIL / path inputs to a PIL RGB image.

        Gradio 6 Image components hand callbacks a numpy.ndarray; the sync
        orchestrator already normalises internally, but the queued path saves
        the image to disk first, so the conversion must happen here.
        """
        if input_image is None:
            return None
        if isinstance(input_image, Image.Image):
            return input_image.convert("RGB")
        if isinstance(input_image, np.ndarray):
            return Image.fromarray(input_image).convert("RGB")
        return input_image

    @staticmethod
    def _require_password_changed(user_state: dict) -> None:
        """§21 force-change gate: block task submission for bootstrap accounts
        that have not replaced their initial password yet."""
        from fiximg.domain.errors import PasswordChangeRequiredError

        username = (user_state or {}).get("username")
        if not username:
            return
        user = task_repo_user_get(username)
        if user and user.get("must_change_password"):
            raise PasswordChangeRequiredError(
                "Please change the initial password first (Admin Panel → User Management)."
            )

    @staticmethod
    def _task_type_for(mode: str) -> str:
        cfg = PIPELINE_MODES.get(mode)
        if cfg:
            return cfg["task_type"]
        # The API accepts raw task types too (e.g. type=auto_restore or the
        # pre-existing type=detect_scratch), not just UI mode names.
        if any(cfg["task_type"] == mode for cfg in PIPELINE_MODES.values()):
            return mode
        raise ValueError(f"Unknown mode: {mode}")


task_service = TaskService()
