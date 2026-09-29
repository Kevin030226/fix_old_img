"""PipelinePlanner — map task types to ordered stage lists (plan §9/§10/§12).

Adding a new capability (super resolution, deblur, ...) means registering a
stage in :mod:`fiximg.inference.registry`; the planner itself does not change.

Task types are static plans (`_TASKS`); `auto_restore` (plan §10, Phase 7)
instead derives its plan at runtime from the ImageAnalyzer output via
`plan_auto_restore(analysis, options)`. Both entry points read the same task
options (plan §12), so `face_enhance` / `auto_colorize` control the flow whether
the plan came from the task type or from the image analysis.
"""
from dataclasses import dataclass, field

from fiximg.config import settings
from fiximg.domain.errors import StageNotAvailableError


@dataclass
class PipelinePlan:
    """An ordered sequence of stage names plus task metadata."""

    task_type: str
    stages: list = field(default_factory=list)
    #: extra planner decisions surfaced to the report (e.g. auto analysis)
    decisions: dict = field(default_factory=dict)


#: One step of a plan: the stage name and the kwargs its entry point takes.
StageSpec = tuple[str, dict]


class PipelinePlanner:
    """Resolve task types into PipelinePlan objects."""

    #: The task options whose effect is *which stages run*. Declared here because the
    #: UI and the option tables have to offer exactly these keys and no others.
    #: `hr` is deliberately not one of them: it changes what a face stage does, not
    #: whether the planner schedules it, so those stages read it from
    #: ``StageContext.options`` instead.
    PLAN_SWITCHES = ("face_enhance", "auto_colorize")

    #: task_type -> list of (stage_name, kwargs)
    _TASKS: dict[str, list[StageSpec]] = {
        "restore": [("global_restore", {"with_scratch": False})],
        "restore_scratch": [("global_restore", {"with_scratch": True})],
        "detect_scratch": [("scratch_detection", {})],
        "colorize": [("colorization", {})],
    }

    #: The face chain, in order (plan §3.6/§3.7). Each stage does exactly one
    #: job, so nothing is processed twice. `warp_back` is what produces the
    #: final composited image.
    FACE_CHAIN: list[StageSpec] = [
        ("face_detection", {}),
        ("face_enhancement", {}),
        ("warp_back", {}),
    ]

    #: auto_restore decision thresholds (plan §10/§11); live values come from
    #: settings so they can be calibrated via env vars.
    @property
    def auto_scratch_threshold(self) -> float:
        return settings.auto_scratch_threshold

    @property
    def auto_blur_threshold(self) -> float:
        return settings.auto_blur_threshold

    def plan(self, task_type: str, options: dict | None = None) -> PipelinePlan:
        """Build the plan for `task_type`, honouring the §12 option switches.

        options (all optional):
          face_enhance: append the face chain (detection → enhancement →
              warp-back). ``None`` means "task-type default", which is ``True``
              for the restoration types — that preserves the V2 behaviour where
              ``run.py`` performed the face chain inside global restoration.
          auto_colorize: append a colorization stage (for B&W inputs)

        Stage responsibilities are explicit and non-overlapping (§3.6/§3.7):
        ``global_restore`` only restores, ``face_detection`` only finds faces,
        ``face_enhancement`` only enhances crops, ``warp_back`` only composites.
        """
        if task_type == "auto_restore":
            # Deferred to plan_auto_restore — requires the analysis first.
            raise StageNotAvailableError(
                "auto_restore requires an image analysis; use plan_auto_restore(analysis, options)"
            )
        if task_type not in self._TASKS:
            raise StageNotAvailableError(f"Unknown task type: {task_type}")

        stages = list(self._TASKS[task_type])
        options = options or {}
        restores = task_type in ("restore", "restore_scratch")

        # Tri-state: absent means "use the task-type default" so a client that
        # sends {"face_enhance": false} really does opt out.
        face_enhance = options.get("face_enhance")
        if face_enhance is None:
            face_enhance = restores
        if face_enhance and restores:
            stages += self.FACE_CHAIN

        if options.get("auto_colorize") and restores:
            stages.append(("colorization", {}))

        applied = {
            "face_enhance": bool(face_enhance and restores),
            "auto_colorize": bool(options.get("auto_colorize") and restores),
        }
        return PipelinePlan(task_type=task_type, stages=stages, decisions={"options": applied})

    @staticmethod
    def _override(options: dict, key: str, decided: bool) -> bool:
        """The explicit option wins over what was already decided.

        Tri-state on purpose: ``None`` (key absent) leaves the decision where it
        belongs — with the task type in :meth:`plan`, with the analysis here — while a
        submitted boolean is an instruction, in both directions.
        """
        value = options.get(key)
        return decided if value is None else bool(value)

    def plan_auto_restore(self, analysis: dict, options: dict | None = None) -> PipelinePlan:
        """Derive a pipeline from the ImageAnalyzer output (plan §10 flow).

        Decision table (thresholds from settings, tunable via env vars):

          grayscale?                 -> append colorization
          scratch_score >= 0.5       -> scratch repair (with_scratch=True)
          otherwise                  -> plain global restoration
          blur_score >= 0.5          -> noted in decisions (model handles it)
          face_count > 0             -> face_detection + face_enhancement

        ``options`` overrides two of those rows (plan §12 applied to the dynamic plan):

          face_enhance / auto_colorize absent -> the analysis decides (the rows above)
          true                                -> run the stage even if the analysis says
                                                 it is not needed
          false                               -> never run it, whatever the analysis says

        The override is not a convenience: the analysis runs on a copy downscaled to
        512 px, so small faces in a large scan are invisible to it, and a faded black
        and white photo can carry enough hand-tint to read as colour. Without this the
        user's only way to say "there IS a face in this one" was to leave the Auto tab
        for the fixed restoration plans, losing the scratch decision with it.
        """
        if not analysis:
            raise StageNotAvailableError("auto_restore requires a non-empty analysis")

        options = options or {}
        stages: list = []
        decisions: dict = {"analysis": analysis}

        scratched = float(analysis.get("scratch_score") or 0.0) >= self.auto_scratch_threshold
        if scratched:
            stages.append(("global_restore", {"with_scratch": True}))
        else:
            stages.append(("global_restore", {"with_scratch": False}))

        face_enhance = self._override(options, "face_enhance", bool(analysis.get("face_count")))
        colorize = self._override(options, "auto_colorize", bool(analysis.get("is_grayscale")))
        if face_enhance:
            stages += self.FACE_CHAIN
        if colorize:
            stages.append(("colorization", {}))

        decisions.update(
            {
                "used_scratch_repair": scratched,
                "face_enhancement": face_enhance,
                "colorization": colorize,
                "blurry_input": float(analysis.get("blur_score") or 0.0)
                >= self.auto_blur_threshold,
                #: Same shape as :meth:`plan`'s block: what the switches ended up
                #: deciding, next to the raw analysis in ``decisions["analysis"]``, so
                #: a report shows an override ("faces=0, face_enhancement=true")
                #: instead of just the outcome.
                "options": {"face_enhance": face_enhance, "auto_colorize": colorize},
            }
        )
        return PipelinePlan(task_type="auto_restore", stages=stages, decisions=decisions)

    def available_tasks(self) -> list:
        return sorted([*self._TASKS.keys(), "auto_restore"])

    #: capability → default stage name, so new models are picked by *what they
    #: do* rather than by hard-coded name (plan §3.14.2).
    _CAPABILITY_STAGES = {
        "restore": "global_restore",
        "scratch_repair": "scratch_repair",
        "scratch_detection": "scratch_detection",
        "face_detection": "face_detection",
        "face_enhance": "face_enhancement",
        "colorize": "colorization",
    }

    def stage_for_capability(self, capability: str) -> str:
        """Resolve a capability to a registered stage name.

        The registry wins (so a plugin can supersede a built-in provider); the
        table above is the fallback for capabilities with no registered owner.
        """
        from fiximg.inference.registry import stage_registry

        matches = stage_registry.find_by_capability(capability)
        if matches:
            return matches[0]
        stage = self._CAPABILITY_STAGES.get(capability)
        if stage is None:
            raise StageNotAvailableError(f"No stage provides capability: {capability}")
        return stage

    def build_stage(self, name: str, kwargs: dict):
        """Instantiate a stage by name through the registry (single source of truth)."""
        from fiximg.inference.registry import stage_registry

        return stage_registry.create(name, kwargs)

    def required_capabilities(self, task_type: str, options: dict | None = None) -> set[str]:
        """Capabilities a task will need, computed from the plan it will actually run.

        Used at enqueue time so a worker pinned to one GPU (or specialised in a
        subset of capabilities) can skip tasks it could not complete.

        ``options`` is not optional to get right. `plan()` appends stages from the §12
        switches, so asking by task type alone declared the *default* plan:
        `restore` with `{"auto_colorize": true}` claimed it needed no `colorize`, and
        a worker serving only the restoration chain claimed that task and then colorized
        anyway — the exact opposite of what §4.3 Step 4 is for. The switch cuts both
        ways: `{"face_enhance": false}` must *stop* declaring the face chain, or a
        face-less specialised worker would be told it cannot take a task it can finish,
        and the task would sit unclaimed while every queue gauge said healthy.

        Dynamic task types stay unrestricted even with options: an `auto_restore`
        submission that declines both switches still branches on `scratch_score`, so no
        subset of capabilities is guaranteed sufficient for it. Overstating the hint is
        the failure mode that starves a queue, so this returns the empty set and the
        worker's own stage reporting (skipped, not fabricated) handles the rest.

        The stages are instantiated to read their capabilities, because a constructor
        can widen them (``GlobalRestoreStage(with_scratch=True)`` advertises
        ``scratch_repair``). Building a stage is cheap: it only sets flags, it never
        loads weights.
        """
        if task_type == "auto_restore":
            return set()

        try:
            plan = self.plan(task_type, options)
        except StageNotAvailableError:
            return set()

        capabilities: set[str] = set()
        for stage_name, kwargs in plan.stages:
            try:
                stage = self.build_stage(stage_name, kwargs or {})
            except StageNotAvailableError:
                continue
            capabilities |= set(getattr(stage, "capabilities", ()))
        return capabilities
