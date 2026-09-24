"""PipelineOrchestrator — plan stages, run them, record timings/artifacts (plan section 8).

The orchestrator is the only component that knows how a run flows:
  plan -> (stage run + timing + artifacts)* -> evaluation -> persistence.
Web/UI code never calls stages directly.

Two entry points:
  - run(): synchronous, used by Gradio callbacks (V1-compatible UX).
  - execute_queued(): DB-queue path; a worker claims a queued row and the
    input image is reloaded from artifact storage (storage/tasks/...).
"""
import json
import os
import threading
import time
import uuid
from datetime import datetime

import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.exceptions import InvalidRequestError
from app.core.logging import get_logger, log_event
from app.inference.model_manager import model_manager, resolve_gpu
from app.inference.planner import PipelinePlanner
from app.repositories import task_repository as task_repo
from app.services import artifact_service
from app.services.evaluation_service import (
    REFERENCE_TYPE,
    calculate_difference_metrics,
    degrade_note_from_metadata,
    format_difference_report,
)
from app.services.metrics_service import (
    REFERENCE_TYPE as NR_REFERENCE_TYPE,
    compute_no_reference_metrics,
    format_no_reference_report,
)
from app.inference.identity import REFERENCE_TYPE as identity_reference_type
from app.inference.identity import format_identity_report
from app.services.gt_metrics_service import (
    REFERENCE_TYPE as GT_REFERENCE_TYPE,
    compute_ground_truth_metrics,
    format_ground_truth_report,
)
from app.services.nr_iqa_service import (
    REFERENCE_TYPE as NR_IQA_REFERENCE_TYPE,
    compute_nr_iqa,
    format_nr_iqa_report,
)

logger = get_logger("fiximg.orchestrator")

#: task types whose result is a difference-metric evaluation
_EVALUATED_TYPES = {"restore", "restore_scratch"}

# One GPU execution at a time per process (V1 semantics; serializes the Gradio
# sync path and the in-process inline worker if both are enabled).
_EXEC_LOCK = threading.Lock()


def new_task_id() -> str:
    return "{}-{}".format(
        datetime.now().strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:8]
    )


class PipelineOrchestrator:
    """Execute a task type end-to-end and persist everything it produces."""

    def __init__(self, planner: PipelinePlanner | None = None) -> None:
        self.planner = planner or PipelinePlanner()
        self.model_manager = model_manager

    # ---------------------------------------------------------------- sync API
    def run(self, input_image, user_state, task_type: str, options: dict | None = None):
        """Synchronous execution used by the Gradio callbacks (V1-compatible UX).

        Returns (result_path, evaluation_text_or_None).
        """
        task_id = new_task_id()
        username = (user_state or {}).get("username", "unknown") or "unknown"
        run_dir = artifact_service.create_run_dir(task_id)
        context = self._build_context(task_id, username, task_type, run_dir, options)

        task_repo.create_task(task_id, task_type, username, options=options)
        task_repo.add_event(
            "task.created", task_id=task_id, user_id=username,
            message=f"sync task {task_type} submitted",
        )
        task_repo.start_task(task_id)
        started = time.perf_counter()
        try:
            image = self._normalize_input(input_image)
            input_path = artifact_service.save_input_image(run_dir, task_id, image)
            task_repo.add_artifact(task_id, "input", input_path, "image/png")

            result_path, evaluation_text, metrics, stage_meta, decisions = self._execute_core(
                task_id, image, task_type, context, input_path, run_dir
            )
            duration_ms = int((time.perf_counter() - started) * 1000)
            task_repo.finish_task(task_id, result_path, evaluation_text, duration_ms)
            artifact_service.write_report(
                run_dir,
                {
                    "task_id": task_id,
                    "task_type": task_type,
                    "stages": stage_meta,
                    "metrics": metrics,
                    "duration_ms": duration_ms,
                    "planner_decisions": decisions,
                },
            )
            artifact_service.maybe_purge()
            return result_path, evaluation_text
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            task_repo.fail_task(task_id, str(exc), duration_ms)
            task_repo.add_event(
                "task.failed", level="error", task_id=task_id, user_id=username,
                message=str(exc)[:500],
            )
            raise

    # ----------------------------------------------------------- queued API
    def execute_queued(self, task_id: str, task_type: str) -> None:
        """Async path: execute a claimed queued task; fills in status/artifacts.

        The input image is loaded from tasks.input_path (persisted at enqueue
        time), so worker and API processes share nothing but the database and
        the artifact storage.
        """
        row = task_repo.get_task(task_id)
        if not row:
            raise RuntimeError(f"Task {task_id} disappeared from the queue")
        username = row.get("user_id") or "unknown"
        input_path = row.get("input_path")
        try:
            options = json.loads(row.get("options_json") or "{}")
        except (TypeError, ValueError):
            options = {}
        run_dir = artifact_service.create_run_dir(task_id)
        context = self._build_context(task_id, username, task_type, run_dir, options)
        started = time.perf_counter()
        try:
            if not input_path or not os.path.exists(input_path):
                raise InvalidRequestError(
                    f"Input image missing for queued task {task_id}: {input_path}"
                )
            image = Image.open(input_path).convert("RGB")
            task_repo.add_artifact(task_id, "input", input_path, "image/png")

            result_path, evaluation_text, metrics, stage_meta, decisions = self._execute_core(
                task_id, image, task_type, context, input_path, run_dir
            )
            duration_ms = int((time.perf_counter() - started) * 1000)
            task_repo.finish_task(task_id, result_path, evaluation_text, duration_ms)
            artifact_service.write_report(
                run_dir,
                {
                    "task_id": task_id,
                    "task_type": task_type,
                    "stages": stage_meta,
                    "metrics": metrics,
                    "duration_ms": duration_ms,
                    "planner_decisions": decisions,
                },
            )
            task_repo.add_event(
                "task.completed", task_id=task_id, user_id=username,
                message=f"completed in {duration_ms} ms",
            )
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            task_repo.fail_task(task_id, str(exc), duration_ms)
            task_repo.add_event(
                "task.failed", level="error", task_id=task_id, user_id=username,
                message=str(exc)[:500],
            )
            raise

    # -------------------------------------------------------------- internals
    def _execute_core(self, task_id, image, task_type, context, input_path, run_dir):
        """Shared execution path: plan -> stages -> output -> evaluation."""
        # §10/§11 auto_restore (Phase 7): analyze first, then derive the plan.
        analysis = None
        if task_type == "auto_restore":
            from app.inference.analyzer import analyze_image, format_analysis_summary

            analysis = analyze_image(image)
            log_event(
                logger, "INFO", "auto analysis", task_id=task_id,
                analysis=format_analysis_summary(analysis),
            )
        plan = self.planner.plan_auto_restore(analysis) if analysis is not None else self.planner.plan(
            task_type, options=getattr(context, "options", None)
        )
        result_image, artifacts, stage_meta = self._run_plan(task_id, image, plan, context)
        result_path = self._persist_output(task_id, run_dir, result_image)
        for name, path in artifacts.items():
            task_repo.add_artifact(task_id, name, path)

        # §16 "with Ground Truth" mode: the caller supplied a reference photo
        # (persisted at enqueue time); metrics are computed against it.
        gt_path = (context.options or {}).get("_ground_truth_path")

        evaluation_text, metrics = self._evaluate(
            task_id, task_type, input_path, result_path, stage_meta, gt_path
        )
        return result_path, evaluation_text, metrics, stage_meta, plan.decisions

    def _build_context(self, task_id, username, task_type, run_dir, options=None):
        from app.inference.context import StageContext

        return StageContext(
            task_id=task_id,
            user=username,
            task_type=task_type,
            run_dir=run_dir,
            options=dict(options or {}),
            model_manager=self.model_manager,
            gpu=resolve_gpu(settings.device),
        )

    @staticmethod
    def _normalize_input(input_image) -> Image.Image:
        if input_image is None:
            raise InvalidRequestError("Please upload an image first.")
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")
        if max(input_image.size) > settings.max_image_side:
            raise InvalidRequestError(
                f"Image too large (long side exceeds {settings.max_image_side} px); please resize and retry."
            )
        return input_image

    def _run_plan(self, task_id, image, plan, context):
        artifacts = {}
        stage_meta = []
        # §25: record the executing device on every log line of this run.
        from app.core.logging import set_gpu_id

        set_gpu_id(context.gpu)
        with _EXEC_LOCK:
            total_stages = max(len(plan.stages), 1)
            for order, (stage_name, kwargs) in enumerate(plan.stages):
                stage = self.planner.build_stage(stage_name, kwargs)
                task_repo.record_stage(task_id, order, stage.name, "running")
                log_event(logger, "INFO", "stage start", task_id=task_id, stage=stage.name)

                # §24 live progress: each stage owns a slice of the bar, so a
                # long single-stage plan (restore/restore_scratch run the whole
                # 4-path CLI inside global_restore) still moves smoothly instead
                # of sitting at 0% until the stage ends.
                base = order / total_stages * 100.0
                span = 100.0 / total_stages

                def _on_stage_progress(
                    fraction, _message=None, _base=base, _span=span, _name=stage.name
                ):
                    pct = int(min(max(_base + _span * fraction, 0.0), 99.0))
                    task_repo.update_progress(task_id, pct, _name)

                context.progress_cb = _on_stage_progress
                task_repo.update_progress(task_id, int(min(base, 99.0)), stage.name)
                t0 = time.perf_counter()
                try:
                    stage_result = stage.run(image, context)
                    if stage_result.image is not None:
                        image = stage_result.image
                except Exception as exc:
                    duration_ms = int((time.perf_counter() - t0) * 1000)
                    task_repo.finish_stage(task_id, order, "failed", duration_ms, str(exc))
                    log_event(
                        logger, "ERROR", "stage failed",
                        task_id=task_id, stage=stage.name, duration_ms=duration_ms, error=str(exc),
                    )
                    raise
                duration_ms = int((time.perf_counter() - t0) * 1000)
                artifacts.update(stage_result.artifacts or {})
                stage_meta.append(
                    {
                        "stage": stage.name,
                        "status": "completed",
                        "duration_ms": duration_ms,
                        "metadata": stage_result.metadata,
                        "message": stage_result.message,
                    }
                )
                task_repo.finish_stage(task_id, order, "completed", duration_ms, stage_result.message)
                progress = int((order + 1) / total_stages * 100)
                task_repo.update_progress(task_id, min(progress, 99), stage.name)
                log_event(
                    logger, "INFO", "stage done",
                    task_id=task_id, stage=stage.name, duration_ms=duration_ms,
                )
            # No stage is running any more; drop the callback so nothing outside
            # the loop can write progress for this task.
            context.progress_cb = None
        return image, artifacts, stage_meta

    def _persist_output(self, task_id, run_dir, result_image) -> str:
        out_dir = artifact_service.output_dir_for(run_dir)
        os.makedirs(out_dir, exist_ok=True)
        result_path = os.path.join(out_dir, "final.png")
        result_image.save(result_path)
        task_repo.add_artifact(task_id, "output", result_path, "image/png")
        return result_path

    def _evaluate(self, task_id, task_type, input_path, result_path, stage_meta,
                  ground_truth_path=None):
        metrics = {}
        evaluation_text = None
        if task_type in _EVALUATED_TYPES:
            diff = calculate_difference_metrics(input_path, result_path)
            degrade_note = None
            for meta in stage_meta:
                if meta.get("stage") in ("global_restore", "scratch_repair"):
                    degrade_note = degrade_note_from_metadata(meta.get("metadata") or {})
            evaluation_text = format_difference_report(diff, degrade_note)
            metrics = {
                "psnr": diff["psnr"] if diff["psnr"] != float("inf") else "inf",
                "ssim": round(diff["ssim"], 4),
                "mae": round(diff["mae"], 4),
            }
            for name, value in metrics.items():
                task_repo.add_metric(task_id, name, value, REFERENCE_TYPE)

        # No-reference quality indicators of the output (plan §16): computed for
        # every task that produced an image, independent of task type.
        nr = compute_no_reference_metrics(result_path)
        if nr is not None:
            metrics.update(nr)
            for name, value in nr.items():
                task_repo.add_metric(task_id, name, value, NR_REFERENCE_TYPE)
            report = format_no_reference_report(nr)
            evaluation_text = (
                f"{evaluation_text}\n\n{report}" if evaluation_text else report
            )

        # Optional natural-scene statistics (plan §16 "can be added"): NIQE,
        # BRISQUE-style features and color statistics of the output.
        iqa = compute_nr_iqa(result_path)
        if iqa is not None:
            metrics["niqe"] = iqa["niqe"]
            task_repo.add_metric(
                task_id, "niqe", iqa["niqe"], NR_IQA_REFERENCE_TYPE
            )
            report = format_nr_iqa_report(iqa)
            evaluation_text = (
                f"{evaluation_text}\n\n{report}" if evaluation_text else report
            )

        # Face identity preservation (plan §17): for tasks whose face chain can
        # alter faces, compare input vs output face embeddings.
        if task_type in _EVALUATED_TYPES | {"auto_restore"}:
            identity = self._identity_metrics(input_path, result_path, task_id)
            if identity is not None and not identity.get("skipped"):
                metrics["face_count"] = identity.get("face_count", 0)
                metrics["enhanced_faces"] = identity.get("enhanced_faces", 0)
                if identity.get("identity_similarity") is not None:
                    metrics["identity_similarity"] = identity["identity_similarity"]
                task_repo.add_metric(
                    task_id, "identity_similarity", identity.get("identity_similarity"),
                    identity_reference_type,
                )
                report = format_identity_report(identity)
                evaluation_text = (
                    f"{evaluation_text}\n\n{report}" if evaluation_text else report
                )

        # §16 "with Ground Truth": when a reference photo is attached, LPIPS is
        # computed against it and prepended to the report (it is the only
        # metric here that actually compares to the caller's expectation).
        if ground_truth_path:
            gt = compute_ground_truth_metrics(ground_truth_path, result_path)
            if gt is not None:
                metrics["lpips"] = gt["lpips"]
                task_repo.add_metric(
                    task_id, "lpips", gt["lpips"], GT_REFERENCE_TYPE
                )
                report = format_ground_truth_report(gt)
                evaluation_text = (
                    f"{report}\n\n{evaluation_text}" if evaluation_text else report
                )
        return evaluation_text, metrics

    @staticmethod
    def _identity_metrics(input_path, result_path, task_id):
        """§17 identity comparison; None on unreadable images / disabled backend."""
        from app.inference.identity import compute_identity_similarity
        from PIL import Image

        try:
            with Image.open(input_path) as before, Image.open(result_path) as after:
                return compute_identity_similarity(before, after)
        except Exception as exc:  # noqa: BLE001 — identity must never fail a run
            log_event(
                logger, "WARNING", "identity metric skipped",
                task_id=task_id, error=str(exc)[:200],
            )
            return None
