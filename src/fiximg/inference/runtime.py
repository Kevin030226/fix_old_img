"""PipelineOrchestrator — plan stages, run them, record timings/artifacts (plan section 8).

The orchestrator is the only component that knows how a run flows:
  plan -> (stage run + timing + artifacts)* -> evaluation -> persistence.
Web/UI code never calls stages directly.

Two entry points:
  - run(): synchronous, used by Gradio callbacks (V1-compatible UX).
  - execute_queued(): DB-queue path; a worker claims a queued row and the
    input image is reloaded from artifact storage (storage/tasks/...).
"""
import os
import threading
import time
import uuid
from datetime import datetime

import numpy as np
from PIL import Image

from fiximg.config import settings
from fiximg.domain.enums import EventType, StageStatus
from fiximg.domain.errors import InvalidRequestError, TaskCancelledError
from fiximg.domain.tasks import decode_json_column
from fiximg.infrastructure.observability.logging import get_logger, log_event
from fiximg.infrastructure.observability.metrics import MetricName, metrics
from fiximg.infrastructure.observability.tracing import tracer
from fiximg.inference import gpu_utilization
from fiximg.inference.gpu import GpuTopology
from fiximg.inference.gpu_memory import MemorySampler, is_oom_error
from fiximg.inference.model_manager import model_manager
from fiximg.inference.planner import PipelinePlanner
from fiximg.inference.scheduler import default_scheduler
from fiximg.inference.size_policy import SizeDecision
from fiximg.infrastructure.db.repositories import task_repository as task_repo
from fiximg.infrastructure.storage import local as artifact_service
from fiximg.infrastructure.storage import fetch_to_local, publish_local_file
from fiximg.inference.evaluation.reference import REFERENCE_TYPE, calculate_difference_metrics, degrade_note_from_metadata, format_difference_report
from fiximg.inference.evaluation.no_reference import REFERENCE_TYPE as NR_REFERENCE_TYPE, compute_no_reference_metrics, format_no_reference_report
from fiximg.inference.identity import REFERENCE_TYPE as identity_reference_type
from fiximg.inference.identity import format_identity_report
from fiximg.inference.evaluation.ground_truth import REFERENCE_TYPE as GT_REFERENCE_TYPE, compute_ground_truth_metrics, format_ground_truth_report
from fiximg.inference.evaluation.nr_iqa import REFERENCE_TYPE as NR_IQA_REFERENCE_TYPE, compute_nr_iqa, format_nr_iqa_report

logger = get_logger("fiximg.orchestrator")

#: task types whose result is a difference-metric evaluation
#:
#: `auto_restore` belongs here too: it runs the same restoration stages, and its
#: §17 identity comparison already treats it as a restoration. What it must not
#: be is silent — when the analysis decided to colourise a grayscale input, the
#: input-output difference is that deliberate change, and the report says so
#: (see `_colorization_note`).
#: `colorize` and `detect_scratch` stay out: comparing a colourised output with
#: the grayscale original it came from is not a difference worth a number, and
#: `detect_scratch` outputs a mask rather than a picture.
_EVALUATED_TYPES = {"restore", "restore_scratch", "auto_restore"}

#: Single-worker pool for deferred evaluation (plan §3.16). Metrics are
#: CPU-bound, so serialising them keeps them from competing with the next
#: inference for cores — and bounds the thread count.
_eval_pool = None
_eval_pool_lock = threading.Lock()


def _get_eval_pool():
    """Lazily create the background evaluation pool."""
    global _eval_pool
    with _eval_pool_lock:
        if _eval_pool is None:
            from concurrent.futures import ThreadPoolExecutor

            _eval_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fiximg-eval")
        return _eval_pool


def shutdown_eval_pool(wait: bool = False) -> None:
    """Stop the background evaluation pool (worker shutdown / tests)."""
    global _eval_pool
    with _eval_pool_lock:
        pool, _eval_pool = _eval_pool, None
    if pool is not None:
        pool.shutdown(wait=wait, cancel_futures=False)


def _attempt_count(task_id: str):
    """Attempt number for log correlation (§3.4.2); None when unavailable."""
    try:
        return (task_repo.get_task(task_id) or {}).get("attempt_count")
    except Exception:  # noqa: BLE001 — a log field must never break a run
        return None


#: Metric reference type for values a stage measured itself (plan §5.3).
STAGE_REFERENCE_TYPE = "stage_self_report"


def error_code_of(exc: BaseException) -> str:
    """Machine-readable code for a failure (plan §3.4.1 / §6).

    ``ErrorCode`` is a namespace of string constants (not an enum), so both
    branches already yield a plain string. Anything that is not an ``AppError``
    is an unexpected failure and is reported as ``INTERNAL_ERROR``, so clients
    always get a stable value to branch on.
    """
    from fiximg.domain.errors import AppError, ErrorCode

    if isinstance(exc, AppError):
        return str(exc.code)
    return ErrorCode.INTERNAL_ERROR


#: Minimum percentage movement between two `task.progress` events (§2.5 point 4).
_PROGRESS_EVENT_STEP = 5

#: Metrics whose task-level value is not "whatever the last stage reported".
#: A peak is a high-water mark: two stages that each reached 4 GB produced a task
#: that peaked at 4 GB, not at whichever stage happened to finish last.
_METRIC_MAXIMA = frozenset({"gpu_peak_mb", "gpu_util_max_pct"})

#: Metrics whose task-level value is a *sample-weighted* mean. Utilisation comes
#: from a window of readings, so a 30 s stage and a 200 ms stage must not count
#: equally — and letting the last stage's figure stand for the whole task is the
#: same "last writer wins" error the maxima above already refuse.
_METRIC_WEIGHTED_MEANS = frozenset({"gpu_util_pct"})
#: Where each weighted mean finds its own weight, inside the same stage's metrics.
_METRIC_WEIGHTS = {"gpu_util_pct": "gpu_samples"}
#: Metrics that add up across stages. The sample count belongs to the mean it
#: weighted: keeping the last stage's count beside a task-level weighted mean makes
#: the number unverifiable, so the task reports how many readings it pooled.
_METRIC_SUMS = frozenset({"gpu_samples"})


def _stage_label(meta: dict) -> str | None:
    """`"global_restore@2.0"` for one stage's metadata row, or None if unnamed."""
    name = meta.get("stage")
    if not name:
        return None
    version = meta.get("version")
    return f"{name}@{version}" if version else str(name)


def _artifact_producers(stage_meta: list) -> dict:
    """``{artifact kind: "stage@version"}`` for the rows this run wrote.

    Only the stage knows what it produced, and only the stage knows which version
    of its model was resident — `stage.version` is what the backend reports after a
    load, not what the manifest declares it will be (plan §2.6's artifact-version
    control, which had no writer: every row read back NULL).
    """
    producers: dict[str, str] = {}
    for meta in stage_meta or []:
        label = _stage_label(meta)
        if not label:
            continue
        for kind in meta.get("artifacts") or []:
            producers[kind] = label
    return producers


def _metric_weight(name: str, stage_metrics: dict) -> float:
    """The sample count a weighted mean should use (1.0 when unlabelled)."""
    raw = stage_metrics.get(_METRIC_WEIGHTS.get(name, ""), 1)
    try:
        weight = float(raw)
    except (TypeError, ValueError):
        return 1.0
    return weight if weight > 0 else 1.0


def _collect_stage_metrics(stage_meta: list) -> dict:
    """Merge the metrics every stage reported into one flat dict (plan §5.3)."""
    merged: dict = {}
    weights: dict[str, float] = {}
    for meta in stage_meta or []:
        stage_metrics = meta.get("metrics") or {}
        for name, value in stage_metrics.items():
            weight = _metric_weight(name, stage_metrics)
            if name in _METRIC_SUMS and name in merged:
                try:
                    merged[name] = merged[name] + value
                    continue
                except TypeError:  # a non-numeric value is not summed
                    pass
            if name in _METRIC_WEIGHTED_MEANS and name in merged:
                previous = weights.get(name, 1.0)
                try:
                    merged[name] = round(
                        (merged[name] * previous + value * weight) / (previous + weight), 1
                    )
                    weights[name] = previous + weight
                    continue
                except TypeError:  # a non-numeric value is not averaged
                    pass
            if name in _METRIC_MAXIMA and name in merged:
                try:
                    merged[name] = max(merged[name], value)
                    continue
                except TypeError:  # a non-comparable value is not aggregated
                    pass
            merged[name] = value
            weights[name] = weight
    return merged


def _colorization_note(stage_meta: list) -> str | None:
    """Name it when the input-output difference includes a deliberate recoloring.

    ``auto_restore`` appends the colorization stage for a grayscale input, and
    then PSNR/SSIM/MAE compare a colored picture against the grayscale original —
    a large, correct, and otherwise unexplained difference. The metrics are still
    worth reporting; hiding them (which is what used to happen) threw away the
    restoration numbers for every non-grayscale auto_restore run too.
    """
    if any((meta or {}).get("stage") == "colorization" for meta in stage_meta or []):
        return (
            "This run also colorized a grayscale input: PSNR/SSIM/MAE measure that "
            "deliberate color change as well as the restoration, so a low value here "
            "is not by itself a restoration regression."
        )
    return None


def _persist_stage_metrics(task_id: str, metrics: dict) -> None:
    """Store stage-reported metrics in the metrics table (best effort)."""
    for name, value in (metrics or {}).items():
        try:
            task_repo.add_metric(task_id, name, value, STAGE_REFERENCE_TYPE)
        except Exception as exc:  # noqa: BLE001 — metrics must never fail a run
            log_event(logger, "WARNING", "stage metric not persisted",
                      task_id=task_id, metric=name, error=str(exc))



def new_task_id() -> str:
    return "{}-{}".format(
        datetime.now().strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:8]
    )


class PipelineOrchestrator:
    """Execute a task type end-to-end and persist everything it produces."""

    #: Class-level fallback so an instance built via ``__new__`` (tests, DI
    #: harnesses) still resolves a device; ``__init__`` replaces it with the
    #: configured topology.
    topology: GpuTopology = GpuTopology.single_device()

    def __init__(self, planner: PipelinePlanner | None = None,
                 topology: GpuTopology | None = None) -> None:
        self.planner = planner or PipelinePlanner()
        self.model_manager = model_manager
        #: §4.3 Step 4: which device serves which capability.
        self.topology = topology or GpuTopology.from_settings()

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
        task_repo.add_event(
            EventType.TASK_STARTED.value, task_id=task_id, user_id=username,
            message="synchronous start",
            data={"task_type": task_type, "attempt": _attempt_count(task_id)},
        )
        started = time.perf_counter()
        try:
            image, size_decision, original_size = self._normalize_input(input_image)
            input_path = artifact_service.save_input_image(run_dir, task_id, image)
            task_repo.add_artifact(task_id, "input", input_path, "image/png")

            result_path, evaluation_text, metrics, stage_meta, decisions = self._execute_core(
                task_id, image, task_type, context, input_path, run_dir,
                size_decision=size_decision, original_size=original_size,
            )
            duration_ms = int((time.perf_counter() - started) * 1000)
            if not task_repo.finish_task(task_id, result_path, evaluation_text, duration_ms):
                # The row was no longer running — cancelled through the API, most
                # likely. The image stays on disk; it does not get to decide the
                # task's status.
                log_event(logger, "WARNING", "discarded a terminal write",
                          task_id=task_id, reason="task was not running any more")
            artifact_service.write_report(
                run_dir,
                {
                    "task_id": task_id,
                    "task_type": task_type,
                    "stages": stage_meta,
                    "metrics": metrics,
                    "duration_ms": duration_ms,
                    "planner_decisions": decisions,
                    "size_policy": size_decision.to_dict(),
                },
            )
            artifact_service.maybe_purge()
            return result_path, evaluation_text
        except TaskCancelledError as exc:
            # The synchronous (Gradio) task can be cancelled too — its id is listed in
            # the history panel and `POST /tasks/{id}/cancel` takes any row. Same rule
            # as the queued path: the row says `cancelled`, so it must not be
            # overwritten with `failed` here.
            log_event(logger, "INFO", "synchronous run cancelled",
                      task_id=task_id, detail=str(exc)[:300])
            raise
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            task_repo.fail_task(task_id, str(exc), duration_ms, error_code=error_code_of(exc))
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
        # The claim stamped this worker into the row; every terminal write is
        # fenced by it so a stalled original cannot overwrite the attempt that
        # took the task over after its lease expired.
        owner_worker_id = row.get("worker_id")
        input_path = row.get("input_path")
        #: `options_json` is a JSON column: text on SQLite, a structure on PostgreSQL.
        options = decode_json_column(row.get("options_json"), {}) or {}
        run_dir = artifact_service.create_run_dir(task_id)
        context = self._build_context(task_id, username, task_type, run_dir, options)
        started = time.perf_counter()
        try:
            if not input_path or not (self._recover_input(task_id, input_path)):
                raise InvalidRequestError(
                    f"Input image missing for queued task {task_id}: {input_path}"
                )
            # §3.5.5: the size policy is applied on the execution path too, so a
            # queued task and a synchronous one see identical input.
            image, size_decision, original_size = self._normalize_input(
                Image.open(input_path)
            )
            # The input artifact is registered at enqueue time; only re-register
            # for tasks that predate that (or the synchronous path).
            if not task_repo.artifact_exists(task_id, "input"):
                task_repo.add_artifact(task_id, "input", input_path, "image/png")

            result_path, evaluation_text, metrics, stage_meta, decisions = self._execute_core(
                task_id, image, task_type, context, input_path, run_dir,
                size_decision=size_decision, original_size=original_size,
                # §3.16: only the queued path may defer — `run()` returns the
                # evaluation text to its caller and therefore always evaluates.
                defer_evaluation=settings.eval_mode == "async",
                owner_worker_id=owner_worker_id,
            )
            duration_ms = int((time.perf_counter() - started) * 1000)
            finished = task_repo.finish_task(task_id, result_path, evaluation_text,
                                     duration_ms, worker_id=owner_worker_id)
            if not finished:
                # The lease expired and another worker took this task over (or the user
                # cancelled it). This run's image stays on disk, but it does not get to
                # decide the status of a row it no longer owns.
                log_event(logger, "WARNING", "discarded a terminal write",
                          task_id=task_id, worker_id=owner_worker_id,
                          reason="row no longer running under this worker")
            artifact_service.write_report(
                run_dir,
                {
                    "task_id": task_id,
                    "task_type": task_type,
                    "stages": stage_meta,
                    "metrics": metrics,
                    "duration_ms": duration_ms,
                    "planner_decisions": decisions,
                    "size_policy": size_decision.to_dict(),
                },
            )
            task_repo.add_event(
                "task.completed", task_id=task_id, user_id=username,
                message=f"completed in {duration_ms} ms",
            )
            # §3.16: the result is already downloadable; only the *metrics* are
            # still pending when evaluation runs asynchronously.
            #
            # The signal is `evaluation_text is None`, not "metrics are empty":
            # stages report their own metrics (§5.3), so a non-empty dict says
            # nothing about whether the *evaluation* ran. Inline evaluation
            # always produces text (the difference or no-reference report).
            if evaluation_text is None and settings.eval_mode == "async":
                self._evaluate_in_background(
                    task_id, task_type, input_path, result_path, stage_meta,
                    (context.options or {}).get("_ground_truth_path"), username,
                )
        except TaskCancelledError as exc:
            # Not a failure. The row already says `cancelled`, so writing `failed`
            # over it — or putting `task.failed` on a stream that just carried
            # `task.cancelled` — would contradict what the client was told. Re-raised
            # so the worker releases the claim without spending another attempt.
            log_event(logger, "INFO", "run stopped by a cancellation",
                      task_id=task_id, detail=str(exc)[:300])
            raise
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            task_repo.fail_task(task_id, str(exc), duration_ms,
                                error_code=error_code_of(exc),
                                worker_id=owner_worker_id)
            task_repo.add_event(
                "task.failed", level="error", task_id=task_id, user_id=username,
                message=str(exc)[:500],
            )
            raise

    # ----------------------------------------------------------- evaluation
    def _evaluate_in_background(self, task_id, task_type, input_path, result_path,
                                stage_meta, gt_path, user_id) -> None:
        """Run §16/§17 evaluation off the critical path (plan §3.16).

        The plan is explicit that evaluation must not delay the result: the task
        is already ``completed`` and the image downloadable when this is
        scheduled. Metrics land later as a ``task.evaluated`` event.

        A single-worker pool serialises evaluation: the metrics are CPU-bound
        and would otherwise compete with the next inference for cores.
        """
        def _job():
            try:
                text, metrics = self._evaluate(
                    task_id, task_type, input_path, result_path, stage_meta, gt_path
                )
                task_repo.set_evaluation(task_id, text)
                task_repo.add_event(
                    "task.evaluated", task_id=task_id, user_id=user_id,
                    message="quality metrics ready",
                    data={"metrics": sorted(metrics) if isinstance(metrics, dict) else []},
                )
                log_event(logger, "INFO", "async evaluation done",
                          task_id=task_id, metrics=len(metrics or {}))
            except Exception as exc:  # noqa: BLE001 — metrics are never fatal
                log_event(logger, "ERROR", "async evaluation failed",
                          task_id=task_id, error=str(exc))

        _get_eval_pool().submit(_job)

    # -------------------------------------------------------------- internals
    def _execute_core(self, task_id, image, task_type, context, input_path, run_dir,
                      defer_evaluation: bool = False, size_decision=None,
                      original_size=None, owner_worker_id=None):
        """Shared execution path: plan -> stages -> output -> evaluation.

        ``defer_evaluation`` is set only by the queued path when
        ``FIXIMG_EVAL_MODE=async``: the evaluation is then scheduled by the
        caller *after* the task is marked completed, so the result becomes
        downloadable immediately (plan §3.16). The synchronous ``run()`` path
        always evaluates inline because it returns the text to its caller.
        """
        # §10/§11 auto_restore (Phase 7): analyze first, then derive the plan.
        analysis = None
        if task_type == "auto_restore":
            from fiximg.inference.analyzer import analyze_image, format_analysis_summary

            analysis = analyze_image(image)
            log_event(
                logger, "INFO", "auto analysis", task_id=task_id,
                analysis=format_analysis_summary(analysis),
            )
        plan = (
            self.planner.plan_auto_restore(
                analysis, options=getattr(context, "options", None)
            )
            if analysis is not None
            else self.planner.plan(task_type, options=getattr(context, "options", None))
        )
        result_image, artifacts, stage_meta = self._run_plan(
            task_id, image, plan, context, owner_worker_id=owner_worker_id
        )
        # §3.5.5: the user downloads what they uploaded. This is *not* only about
        # the adaptive band — the vendored restoration chain rounds each side to a
        # multiple of 4 (and the detector's to 16), so a 298×450 photo comes back
        # 296×448 on the "direct" path too, where no policy resize happened at all.
        # Measured live: the result was 2 px smaller than the stored input, which
        # both resized the user's download and made the PSNR/SSIM report compare
        # two different geometries (SSIM read -0.20 for a normal restoration).
        # `restore_original_size` no-ops when the geometry already matches, so the
        # tiled and already-correct paths pay nothing.
        if original_size is not None:
            from fiximg.inference.size_policy import restore_original_size

            result_image = restore_original_size(result_image, original_size)
        result_path = self._persist_output(
            task_id, run_dir, result_image,
            model_version=_stage_label(stage_meta[-1]) if stage_meta else None,
        )
        producers = _artifact_producers(stage_meta)
        for name, path in artifacts.items():
            task_repo.add_artifact(task_id, name, path, model_version=producers.get(name))

        # Stage-reported metrics (plan §5.3): face counts, tile counts, degrade
        # counts. Persisted alongside the runtime's evaluation metrics so the
        # report is a single place to look.
        stage_metrics = _collect_stage_metrics(stage_meta)

        if defer_evaluation:
            # Deferred: `execute_queued` schedules it once the task is completed.
            _persist_stage_metrics(task_id, stage_metrics)
            return result_path, None, stage_metrics, stage_meta, plan.decisions

        # §16 "with Ground Truth" mode: the caller supplied a reference photo
        # (persisted at enqueue time); metrics are computed against it.
        gt_path = (context.options or {}).get("_ground_truth_path")

        evaluation_text, metrics = self._evaluate(
            task_id, task_type, input_path, result_path, stage_meta, gt_path
        )
        if stage_metrics:
            metrics = {**stage_metrics, **(metrics or {})}
            _persist_stage_metrics(task_id, stage_metrics)
        return result_path, evaluation_text, metrics, stage_meta, plan.decisions

    def _retry_on_another_device(self, exc, task_id, stage, capabilities,
                                 failed_device, context, image):
        """Retry a stage on a different device after an OOM (plan §7.1).

        Returns ``(StageResult, device)`` on success, or None when the failure
        was not an OOM, no alternative device exists, or the retry also failed.
        The caller then reports the original failure.
        """
        if not is_oom_error(exc):
            return None
        alternatives = [
            device for device in self.topology.candidate_devices(capabilities)
            if device != failed_device
        ]
        if not alternatives:
            return None

        retry_device = alternatives[0]
        log_event(
            logger, "WARNING", "stage ran out of memory; retrying on another device",
            task_id=task_id, stage=stage.name,
            from_device=failed_device, to_device=retry_device,
        )
        task_repo.add_event(
            "stage.retried", level="warning", task_id=task_id,
            message=f"{stage.name}: out of memory on device {failed_device}",
            data={"stage": stage.name, "from_device": failed_device,
                  "to_device": retry_device},
        )
        previous = context.gpu
        context.gpu = retry_device
        try:
            with default_scheduler.slot(capabilities, device=retry_device):
                result = stage.run(image, context)
        except Exception as retry_exc:  # noqa: BLE001 — report the original failure
            log_event(logger, "ERROR", "retry after OOM also failed",
                      task_id=task_id, stage=stage.name,
                      device=retry_device, error=str(retry_exc))
            return None
        finally:
            context.gpu = previous
        return result, retry_device

    def _build_context(self, task_id, username, task_type, run_dir, options=None):
        from fiximg.inference.context import StageContext

        return StageContext(
            task_id=task_id,
            user=username,
            task_type=task_type,
            run_dir=run_dir,
            options=dict(options or {}),
            model_manager=self.model_manager,
            # The initial device is the topology default; `_run_plan` overrides
            # it per stage according to the stage's capabilities.
            gpu=self.topology.default_device,
        )

    @staticmethod
    def _normalize_input(input_image) -> tuple[Image.Image, "SizeDecision", tuple[int, int]]:
        """Coerce to RGB and classify the size (plan §3.5.5).

        Returns the (possibly resized) image together with the policy decision,
        so the caller can restore the original dimensions after the stages run
        and record the choice in the report.
        """
        from fiximg.inference.size_policy import (
            decide_from_settings,
            resize_for_inference,
        )

        if input_image is None:
            raise InvalidRequestError("Please upload an image first.")
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        decision = decide_from_settings(input_image.size)
        if decision.rejects:
            # The policy's reason is already user-facing; keep it verbatim so
            # the API returns the same explanation the report records.
            raise InvalidRequestError(decision.reason)

        prepared, original_size = resize_for_inference(input_image, decision)
        if decision.resizes:
            log_event(
                logger, "INFO", "input resized for inference",
                original=f"{input_image.width}x{input_image.height}",
                target=f"{prepared.width}x{prepared.height}",
                reason=decision.reason,
            )
        return prepared, decision, original_size

    def _run_plan(self, task_id, image, plan, context, owner_worker_id=None):
        """Run the planned stages, stopping when the row says this run is over.

        `owner_worker_id` is the claim this process holds. Without it the check can
        only see an explicit cancellation, which is the right answer for the
        synchronous path — and for a harness that never claimed anything.
        """
        artifacts: dict[str, str] = {}
        stage_meta: list[dict] = []
        # §25: record the executing device on every log line of this run.
        from fiximg.infrastructure.observability.logging import set_gpu_id

        set_gpu_id(context.gpu)
        total_stages = max(len(plan.stages), 1)
        for order, (stage_name, kwargs) in enumerate(plan.stages):
            # §3.8 cancellation. A cancel taken while an earlier stage was in flight
            # already flipped the row out of `running`, so this is the point where the
            # run notices — better than burning another model pass on work nobody is
            # waiting for. Not mid-stage: no stage in this pipeline is written to
            # survive being interrupted half way through.
            reason = task_repo.interruption(task_id, owner_worker_id)
            if reason:
                raise TaskCancelledError(
                    f"Task {task_id} interrupted before stage {stage_name}: {reason}",
                    details={"task_id": task_id, "stage": stage_name, "reason": reason},
                )
            stage = self.planner.build_stage(stage_name, kwargs)
            task_repo.record_stage(
                task_id, order, stage.name, "running",
                stage_version=getattr(stage, "version", None),
            )
            task_repo.add_event(
                "stage.started", task_id=task_id,
                message=stage.name,
                data={"stage": stage.name, "order": order,
                      "version": getattr(stage, "version", None)},
            )
            log_event(
                logger, "INFO", "stage start",
                task_id=task_id, stage=stage.name,
                # §3.4.2 log fields: which model/version this stage drives.
                model_name=stage.name,
                model_version=getattr(stage, "version", None),
                attempt=_attempt_count(task_id),
            )

            # §24 live progress: each stage owns a slice of the bar, so a
            # long single-stage plan (restore/restore_scratch run the whole
            # 4-path CLI inside global_restore) still moves smoothly instead
            # of sitting at 0% until the stage ends.
            base = order / total_stages * 100.0
            span = 100.0 / total_stages

            # §2.5 point 4: `task.progress` existed as an event name with no
            # emitter, so an SSE client watching a single-stage plan (where the
            # whole vendored chain runs inside one stage) saw nothing until that
            # stage ended. Throttled to whole percentage steps: the bar is already
            # in the task row, and an event per callback would write a row per
            # tile.
            _state = {"pct": -100}

            def _on_stage_progress(
                fraction, _message=None, _base=base, _span=span, _name=stage.name,
                _seen=_state,
            ):
                pct = int(min(max(_base + _span * fraction, 0.0), 99.0))
                task_repo.update_progress(task_id, pct, _name)
                if pct - _seen["pct"] < _PROGRESS_EVENT_STEP and pct < 99:
                    return
                _seen["pct"] = pct
                task_repo.add_event(
                    EventType.TASK_PROGRESS.value, task_id=task_id,
                    message=_message or _name,
                    data={"progress": pct, "stage": _name, "detail": _message},
                )

            context.progress_cb = _on_stage_progress
            task_repo.update_progress(task_id, int(min(base, 99.0)), stage.name)
            # §4.3 Step 4 / §7.1: route the stage to the device that serves its
            # capabilities, preferring the roomiest one when memory-aware
            # scheduling is enabled. Single-device deployments get the default.
            stage_capabilities = getattr(stage, "capabilities", ())
            stage_device = self.topology.device_for(
                stage_capabilities,
                memory_aware=settings.gpu_memory_aware,
                required_mb=settings.gpu_memory_headroom_mb,
            )
            previous_device = context.gpu
            context.gpu = stage_device
            set_gpu_id(stage_device)
            t0 = time.perf_counter()
            sampler = MemorySampler(stage_device)
            # §7.1: open the utilisation window on the probe's own clock, so the
            # readings the stage later aggregates were taken while it ran. A CPU
            # stage has no driver to ask, so it opens no window at all.
            probe = gpu_utilization.start_probe() if stage_device >= 0 else None
            util_started = 0.0
            if probe is not None:
                probe.open_window()
                util_started = probe.now()
            try:
                # §2.4: one slot per stage, sized by the stage's own
                # capabilities — not a single global lock for the pipeline.
                with sampler, default_scheduler.slot(stage_capabilities, device=stage_device), \
                        tracer.span("inference.stage", task_id=task_id, stage=stage.name,
                                    stage_order=order, device=stage_device):
                    stage_result = stage.run(image, context)
                if stage_result.image is not None:
                    image = stage_result.image
            except Exception as exc:
                # §7.1: an out-of-memory failure is recoverable when another
                # device can serve the same capability — retry there once
                # instead of failing the whole task.
                retried = self._retry_on_another_device(
                    exc, task_id, stage, stage_capabilities, stage_device, context, image,
                )
                if retried is not None:
                    stage_result, stage_device = retried
                    image = stage_result.image if stage_result.image is not None else image
                    context.gpu = stage_device
                else:
                    duration_ms = int((time.perf_counter() - t0) * 1000)
                    task_repo.finish_stage(task_id, order, "failed", duration_ms, str(exc))
                    task_repo.add_event(
                        "stage.failed", level="error", task_id=task_id,
                        message=str(exc)[:500],
                        data={"stage": stage.name, "order": order, "duration_ms": duration_ms,
                              "device": stage_device},
                    )
                    log_event(
                        logger, "ERROR", "stage failed",
                        task_id=task_id, stage=stage.name, duration_ms=duration_ms,
                        device=stage_device, error=str(exc),
                    )
                    raise
            finally:
                if probe is not None:
                    # Release the demand even when the stage failed: a window left
                    # open would keep the sampler querying for the whole process.
                    probe.close_window()
                context.gpu = previous_device
            duration_ms = int((time.perf_counter() - t0) * 1000)
            artifacts.update(stage_result.artifacts or {})
            stage_metadata = stage_result.metadata or {}
            # §7.1: the stage's own peak VRAM, measured as a delta so a peak
            # reached earlier in the run cannot inflate this stage's figure.
            stage_metrics = dict(getattr(stage_result, "metrics", None) or {})
            stage_metrics.update(sampler.metrics())
            if probe is not None:
                stage_metrics.update(
                    gpu_utilization.stage_window(util_started, probe.now(), stage_device)
                )
            util_pct = stage_metrics.get(gpu_utilization.STAGE_UTILIZATION_METRIC)
            if isinstance(util_pct, (int, float)):
                # §2.10 / §7.1: the same window aggregate the task report carries,
                # as a series, so a Grafana graph of "was the card working during
                # `global_restore`" reads one definition rather than a second
                # measurement taken at a different moment.
                metrics.observe(
                    MetricName.GPU_UTILIZATION_PERCENT, float(util_pct),
                    stage=stage.name, device=str(stage_device),
                )
            peak_mb = stage_metrics.get("gpu_peak_mb")
            if isinstance(peak_mb, (int, float)):
                # §2.10: the per-stage VRAM high-water mark, which the task report
                # already carries, on the metrics endpoint too — one definition
                # (bytes) derived from the sampler's own number rather than a
                # second measurement.
                metrics.observe(
                    MetricName.GPU_MEMORY_BYTES, float(peak_mb) * 1024 * 1024,
                    stage=stage.name, device=str(stage_device),
                )
            # §3.9: a stage that waved off its own work (`skipped` in its
            # metadata — no dlib, no aligned crop, no scratch to look for) is not
            # a stage that did the work. `StageStatus.SKIPPED` had no writer, so
            # the report said "completed" for both and a client had to read the
            # prose message to tell them apart.
            stage_status = (
                StageStatus.SKIPPED if stage_metadata.get("skipped")
                else StageStatus.COMPLETED
            )
            stage_meta.append(
                {
                    "stage": stage.name,
                    "version": getattr(stage, "version", None),
                    "status": stage_status.value,
                    "duration_ms": duration_ms,
                    "device": stage_device,
                    "metadata": stage_result.metadata,
                    # Which artifact rows this stage wrote: `artifacts.model_version`
                    # is answered by the producer, not guessed afterwards from the
                    # file name (plan §2.6's "artifact version" control).
                    "artifacts": sorted(stage_result.artifacts or {}),
                    "metrics": stage_metrics,
                    "message": stage_result.message,
                }
            )
            task_repo.finish_stage(
                task_id, order, stage_status.value, duration_ms, stage_result.message,
                metrics=stage_metrics,
            )
            task_repo.add_event(
                EventType.STAGE_SKIPPED.value if stage_status is StageStatus.SKIPPED
                else EventType.STAGE_COMPLETED.value,
                task_id=task_id,
                message=stage.name,
                data={"stage": stage.name, "order": order, "duration_ms": duration_ms,
                      "device": stage_device, "status": stage_status.value},
            )
            metrics.observe(
                MetricName.STAGE_DURATION_SECONDS, duration_ms / 1000,
                stage=stage.name, device=stage_device,
            )
            progress = int((order + 1) / total_stages * 100)
            task_repo.update_progress(task_id, min(progress, 99), stage.name)
            log_event(
                logger, "INFO", "stage done",
                task_id=task_id, stage=stage.name,
                # `duration_ms` and `latency_ms` carry the same value: the plan
                # names the field `latency_ms`, V2 code and the API use
                # `duration_ms`. Both are emitted so neither reader breaks.
                duration_ms=duration_ms, latency_ms=duration_ms,
                device=stage_device,
                status=stage_status.value,
                model_name=stage_metadata.get("model") or stage.name,
                model_version=getattr(stage, "version", None),
            )
        # No stage is running any more; drop the callback so nothing outside
        # the loop can write progress for this task.
        context.progress_cb = None
        return image, artifacts, stage_meta

    def _persist_output(self, task_id, run_dir, result_image, model_version=None) -> str:
        out_dir = artifact_service.output_dir_for(run_dir)
        os.makedirs(out_dir, exist_ok=True)
        result_path = os.path.join(out_dir, "final.png")
        result_image.save(result_path)
        uri = self._publish_output(result_path)
        task_repo.add_artifact(task_id, "output", result_path, "image/png",
                               uri=uri, model_version=model_version)
        return result_path

    @staticmethod
    def _publish_output(path: str) -> str | None:
        """Upload a finished result when the store is remote; else return None.

        The rule (key derivation, local-store short circuit, swallow-on-failure,
        `artifact_io_seconds`) lives once in
        :func:`fiximg.infrastructure.storage.publish_local_file`, because the input
        image is published by the same rule at the other end of the pipeline.
        """
        return publish_local_file(path)

    @staticmethod
    def _recover_input(task_id: str, input_path: str) -> bool:
        """Pull the input image from the store when this node does not have it.

        The API node writes the upload and publishes it (§2.8); on a split
        topology the worker's own disk has never seen it. Without this step the
        remote store supported a *download* on another node but not a *run* on it,
        and the failure surfaced as a missing-file error inside stage 1 rather than
        as a storage question.
        """
        if os.path.isfile(input_path):
            return True
        row = next(
            (r for r in (task_repo.get_artifacts(task_id) or [])
             if r.get("kind") == "input" and r.get("uri")),
            None,
        )
        if row is None:
            return False
        recovered = fetch_to_local(row["uri"], input_path)
        if recovered:
            log_event(logger, "INFO", "input recovered from the store",
                      task_id=task_id, key=row["uri"])
        return recovered

    def _evaluate(self, task_id, task_type, input_path, result_path, stage_meta,
                  ground_truth_path=None):
        metrics = {}
        evaluation_text = None
        if task_type in _EVALUATED_TYPES:
            diff = calculate_difference_metrics(input_path, result_path)
            # The degrade report comes from whichever stage ran the legacy chain:
            # `warp_back` when the face chain ran (§3.7), `global_restore` /
            # `scratch_repair` otherwise. The first stage that has something to
            # say wins: the list is in plan order, so a later stage with no report
            # — a skipped warp_back, say — cannot erase the reason an earlier one
            # already gave.
            degrade_note = None
            for meta in stage_meta:
                if meta.get("stage") in ("warp_back", "global_restore", "scratch_repair"):
                    degrade_note = degrade_note_from_metadata(meta.get("metadata") or {})
                    if degrade_note:
                        break
            notes = [n for n in (degrade_note, _colorization_note(stage_meta)) if n]
            evaluation_text = format_difference_report(diff, notes)
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
        if task_type in _EVALUATED_TYPES:
            identity = self._identity_metrics(input_path, result_path, task_id)
            if identity is not None and not identity.get("skipped"):
                # Namespaced deliberately: `face_count` / `enhanced_count` are the
                # face chain's stage metrics and stay untouched here. The two
                # measure different things (what the pipeline processed vs. what
                # this metric's detector sees in the two pictures) and on a
                # dlib-less install they differ — sharing a key made report.json
                # claim face_count 1 while the stages reported 0.
                metrics["identity_faces_input"] = identity.get("input_faces", 0)
                metrics["identity_faces_output"] = identity.get("output_faces", 0)
                metrics["identity_faces_paired"] = identity.get("paired_faces", 0)
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
        from fiximg.inference.identity import compute_identity_similarity
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
