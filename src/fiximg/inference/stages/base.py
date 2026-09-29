"""Stage contract and shared helpers (plan §5.3, §3.6).

Every stage declares ``name`` / ``version`` / ``capabilities`` and implements
:meth:`run`. The contract the plan states explicitly:

    Stage 不直接写数据库 / 用户历史 / 不知道 HTTP / Gradio / Redis

A stage only returns a :class:`StageResult`; the runtime owns persistence.

Why there is no ``NormalizeStage`` / ``AnalyzeStage`` (plan §3.6)
-----------------------------------------------------------------
The plan's target pipeline lists ``NormalizeStage`` and ``AnalyzeStage`` ahead of
``RestorationStage``. V3 keeps them as the runtime's **pre-plan phase** rather
than stages, because the plan itself depends on them:

    image -> normalize (size policy, RGB)  -> analyze -> planner -> stages

* the size policy (§3.5.5) must run *before* a plan exists — it can reject an
  input outright, and an adaptive resize changes what the stages receive;
* ``analyze_image`` feeds ``plan_auto_restore``, so the plan cannot be built
  before the analysis finishes.

Modelling either as a stage would mean computing the plan twice or letting a
stage mutate the plan, which is exactly the coupling the stage contract exists to
prevent. The behaviour the plan is after — one place decides normalization, one
place decides analysis, both separately testable — is satisfied by
:mod:`fiximg.inference.size_policy` and :mod:`fiximg.inference.analyzer`, and both
are recorded in the run report.

``EvaluationStage`` is likewise deliberately *not* a stage: plan §3.16 requires
evaluation not to block the result, so it runs after the task is marked completed
(see ``runtime._evaluate_in_background``).
"""
from abc import ABC, abstractmethod

from fiximg.inference.context import StageContext, StageResult

#: Directory inside a run where the legacy stage chain shares its artefacts.
#: All four legacy stages read and write this one tree, which is what lets them
#: run as separate invocations (plan §3.6).
_LEGACY_PIPELINE_DIR = "legacy_pipeline"


def legacy_pipeline_root(context: StageContext) -> str:
    """Shared output root for the legacy restoration chain (plan §3.6).

    ``global_restore`` writes ``stage_1_restore_output/`` here, ``face_detection``
    reads it and writes ``stage_2_detection_output/``, and so on. Sharing one
    root is what makes the stages composable without a shared process.
    """
    return context.path_in_run("stages", _LEGACY_PIPELINE_DIR)


class BaseStage(ABC):
    """A single processing capability (restore, detect, enhance, colorize...).

    Stages receive an RGB PIL image plus a StageContext and return a StageResult.
    Heavy model handles come from context.model_manager so models are loaded once
    and reused across runs (plan §6).

    Contract (plan §3.6) — a stage must NOT:
      * write to the database or to user history,
      * know about HTTP, Gradio, Redis or the queue,
      * mutate the image it receives.
    It only returns a :class:`StageResult`; the runtime owns persistence.
    """

    name: str = "base"
    #: semantic version of this stage's behaviour, recorded per task stage (§6)
    version: str = "1.0"
    #: capabilities advertised to the planner/registry (plan §3.14.2)
    capabilities: frozenset[str] = frozenset()

    @abstractmethod
    def run(self, image, context: StageContext) -> StageResult:
        """Process `image` and return a StageResult (never mutates the input)."""
        raise NotImplementedError

    def describe(self) -> dict:
        """Metadata used by the registry / models API."""
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": sorted(self.capabilities),
            "class": type(self).__name__,
        }


__all__ = ["BaseStage", "legacy_pipeline_root"]
