"""PipelinePlanner — map task types to ordered stage lists (plan sections 9/10/12).

Adding a new capability (super resolution, deblur, ...) means registering a
stage here; nothing else in the system changes.

Task types are static plans (`_TASKS`); `auto_restore` (plan §10, Phase 7)
instead derives its plan at runtime from the ImageAnalyzer output via
`plan_auto_restore(analysis)`. The `plan()` overload accepts the task options
(plan §12) so `face_enhance` / `auto_colorize` switches control the flow.
"""
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.exceptions import StageNotAvailableError


@dataclass
class PipelinePlan:
    """An ordered sequence of stage names plus task metadata."""

    task_type: str
    stages: list = field(default_factory=list)
    #: extra planner decisions surfaced to the report (e.g. auto analysis)
    decisions: dict = field(default_factory=dict)


class PipelinePlanner:
    """Resolve task types into PipelinePlan objects."""

    #: task_type -> list of (stage_name, kwargs)
    _TASKS = {
        "restore": [("global_restore", {"with_scratch": False})],
        "restore_scratch": [("global_restore", {"with_scratch": True})],
        "detect_scratch": [("scratch_detection", {})],
        "colorize": [("colorization", {})],
    }

    #: auto_restore decision thresholds (plan §10/§11); live values come from
    #: settings so they can be calibrated via env vars.
    @property
    def AUTO_SCRATCH_THRESHOLD(self) -> float:
        return settings.auto_scratch_threshold

    @property
    def AUTO_BLUR_THRESHOLD(self) -> float:
        return settings.auto_blur_threshold

    def plan(self, task_type: str, options: dict | None = None) -> PipelinePlan:
        """Build the plan for `task_type`, honouring the §12 option switches.

        options (all optional):
          face_enhance: append face_detection + face_enhancement stages
          auto_colorize: append a colorization stage (for B&W inputs)
        """
        if task_type == "auto_restore":
            # Deferred to plan_auto_restore — requires the analysis first.
            raise StageNotAvailableError(
                "auto_restore requires an image analysis; use plan_auto_restore(analysis)"
            )
        if task_type not in self._TASKS:
            raise StageNotAvailableError(f"Unknown task type: {task_type}")

        stages = list(self._TASKS[task_type])
        options = options or {}

        # §12 option switches: the legacy run.py already covers the face chain
        # inside global_restore; face_enhance/auto_colorize act on top of it.
        if options.get("face_enhance") and task_type in ("restore", "restore_scratch"):
            stages += [("face_detection", {}), ("face_enhancement", {})]
        if options.get("auto_colorize") and task_type in ("restore", "restore_scratch"):
            stages += [("colorization", {})]

        applied = {
            k: bool(options.get(k))
            for k in ("face_enhance", "auto_colorize")
            if options.get(k)
        }
        return PipelinePlan(task_type=task_type, stages=stages, decisions={"options": applied})

    def plan_auto_restore(self, analysis: dict) -> PipelinePlan:
        """Derive a pipeline from the ImageAnalyzer output (plan §10 flow).

        Decision table (thresholds from settings, tunable via env vars):

          grayscale?                 -> append colorization
          scratch_score >= 0.5       -> scratch repair (with_scratch=True)
          otherwise                  -> plain global restoration
          blur_score >= 0.5          -> noted in decisions (model handles it)
          face_count > 0             -> face_detection + face_enhancement
        """
        if not analysis:
            raise StageNotAvailableError("auto_restore requires a non-empty analysis")

        stages: list = []
        decisions: dict = {"analysis": analysis}

        scratched = float(analysis.get("scratch_score") or 0.0) >= self.AUTO_SCRATCH_THRESHOLD
        if scratched:
            stages.append(("global_restore", {"with_scratch": True}))
        else:
            stages.append(("global_restore", {"with_scratch": False}))

        if analysis.get("face_count"):
            stages += [("face_detection", {}), ("face_enhancement", {})]
        if analysis.get("is_grayscale"):
            stages.append(("colorization", {}))

        decisions.update(
            {
                "used_scratch_repair": scratched,
                "face_enhancement": bool(analysis.get("face_count")),
                "colorization": bool(analysis.get("is_grayscale")),
                "blurry_input": float(analysis.get("blur_score") or 0.0)
                >= self.AUTO_BLUR_THRESHOLD,
            }
        )
        return PipelinePlan(task_type="auto_restore", stages=stages, decisions=decisions)

    def available_tasks(self) -> list:
        return sorted([*self._TASKS.keys(), "auto_restore"])

    def build_stage(self, name: str, kwargs: dict):
        """Instantiate a stage by name (single registration point)."""
        from app.inference.stages.colorization import ColorizationStage
        from app.inference.stages.face_enhancement import (
            FaceDetectionStage,
            FaceEnhancementStage,
        )
        from app.inference.stages.global_restore import GlobalRestoreStage
        from app.inference.stages.scratch import ScratchDetectionStage

        registry = {
            "global_restore": GlobalRestoreStage,
            "scratch_repair": lambda: GlobalRestoreStage(with_scratch=True),
            "scratch_detection": ScratchDetectionStage,
            "face_detection": FaceDetectionStage,
            "face_enhancement": FaceEnhancementStage,
            "colorization": ColorizationStage,
        }
        factory = registry.get(name)
        if factory is None:
            raise StageNotAvailableError(f"Unknown stage: {name}")
        return factory(**kwargs) if kwargs else factory()
