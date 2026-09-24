"""Task service: the single entry the web/UI layers use (plan sections 8 and 12).

Hides whether execution is synchronous (Gradio callback, V1-compatible) or
queued (DB-backed GPU worker) — callers always get a documented result shape.

Queued path (plan sections 12/28):
  1. persist the uploaded image under artifact storage (storage/tasks/...)
  2. insert a `queued` tasks row with input_path
  3. a PipelineWorker (inline thread or standalone process) claims and runs it
"""
import os

import numpy as np
from PIL import Image

from app.core.logging import get_logger, log_event
from app.inference.orchestrator import PipelineOrchestrator
from app.repositories import task_repository as task_repo
from app.repositories.user_repository import get_user as task_repo_user_get
from app.services.pipeline_modes import PIPELINE_MODES

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
    ) -> bool:
        """Enqueue a task for the GPU worker (Phase-4 path).

        The image is persisted under artifact storage before the row is created,
        so any worker process can pick the task up. Returns True on success;
        False when the queue is saturated (row is left failed with the reason).
        """
        self._require_password_changed(user_state)
        task_type = self._task_type_for(mode)
        username = (user_state or {}).get("username", "unknown") or "unknown"

        input_image = self._to_pil(input_image)

        from app.core.config import settings
        from app.services import artifact_service

        max_queue = settings.worker_max_queue
        if max_queue > 0 and task_repo.queued_count() >= max_queue:
            task_repo.add_event(
                "task.rejected", level="warning", user_id=username,
                message="queue saturated; submission refused",
            )
            return False

        run_dir = artifact_service.create_run_dir(task_id)
        input_path = artifact_service.save_input_image(run_dir, task_id, input_image)

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
        task_repo.create_task(
            task_id, task_type, username, input_path=input_path, options=options
        )
        task_repo.add_event(
            "task.enqueued", task_id=task_id, user_id=username,
            message=f"queued {task_type}",
        )
        log_event(logger, "INFO", "task enqueued", task_id=task_id, task_type=task_type)
        return True

    def enqueue(
        self, input_image, user_state: dict, mode: str, options: dict | None = None,
        ground_truth_image=None,
    ) -> str:
        """Enqueue a task and return its task_id (UI/API async path, plan §23).

        Raises InvalidRequestError-derived errors on validation failures and
        RuntimeError when the queue is saturated.
        """
        from app.inference.orchestrator import new_task_id

        task_id = new_task_id()
        ok = self.submit_queued(
            task_id, input_image, user_state, mode, options=options,
            ground_truth_image=ground_truth_image,
        )
        if not ok:
            raise RuntimeError("Task queue is saturated; please try again later")
        return task_id

    def has_worker(self) -> bool:
        """True when a queue consumer is available in this process.

        The inline worker thread polls the DB queue; a standalone worker
        process cannot be seen from here, so deployments running one must set
        FIXIMG_HAS_EXTERNAL_WORKER=true (or accept the sync fallback in the UI).
        """
        from app.core.config import settings

        if settings.inline_worker:
            try:
                from app.factory import get_inline_worker

                worker = get_inline_worker()
                return bool(worker and worker.is_running())
            except Exception:  # noqa: BLE001
                return False
        return settings.external_worker

    # ---------------------------------------------------------------- queries
    def get_task(self, task_id: str):
        return task_repo.get_task(task_id)

    def list_tasks(self, limit: int = 50, user_id: str | None = None):
        return task_repo.list_tasks(limit=limit, user_id=user_id)

    def cancel_task(self, task_id: str) -> bool:
        return task_repo.cancel_task(task_id)

    def get_task_artifacts(self, task_id: str):
        return task_repo.get_artifacts(task_id)

    def get_task_stages(self, task_id: str):
        return task_repo.get_task_stages(task_id)

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
        from app.core.exceptions import PasswordChangeRequiredError

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

    def plan_preview(self, mode: str, analysis: dict | None = None, options: dict | None = None):
        """Expose the planner for callers that need the decision details
        (e.g. auto_restore reports). Not used by the hot path."""
        task_type = self._task_type_for(mode)
        if analysis is not None:
            return self.orchestrator.planner.plan_auto_restore(analysis)
        return self.orchestrator.planner.plan(task_type, options=options)


task_service = TaskService()
