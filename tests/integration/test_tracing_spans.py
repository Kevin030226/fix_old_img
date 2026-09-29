"""The tracing seam is wired, not just defined (plan 搂2.10).

A tracer that nothing calls is indistinguishable from no tracing at all, so this
runs a real plan through the orchestrator with the recording tracer installed and
asserts the spans the deployment is supposed to get: one per stage, timed, with
the correlation ids a trace is useless without.
"""
import pytest
from PIL import Image

from fiximg.inference.stages.base import BaseStage, StageResult
from fiximg.infrastructure.observability.tracing import (
    _RecordingTracer,
    get_tracer,
    install_tracer,
)

pytestmark = pytest.mark.integration

_STUBBED_STAGES = (
    "global_restore", "scratch_repair", "scratch_detection",
    "face_detection", "face_enhancement", "warp_back", "colorization",
)


class _StubStage(BaseStage):
    """Passes the image through; the point is the plumbing, not the pixels.

    The registry instantiates stages with the plan's options (``with_scratch``
    and friends), so the stub accepts and ignores them. ``name`` is what the
    runtime reports in logs and spans, so each registered stage gets its own
    subclass carrying the name the plan asked for.
    """

    name = "global_restore"
    version = "stub-1.0"
    capabilities = frozenset({"restore", "scratch_repair", "face_detection",
                              "face_restore", "face_composite", "warp_back",
                              "colorize", "deblur", "denoise"})

    def __init__(self, **kwargs) -> None:
        self.options = kwargs

    def run(self, image, context):
        return StageResult(image=image, metadata={"stub": True})


class _FailingStage(_StubStage):
    """Explodes inside the span, to prove the span still closes."""

    def run(self, image, context):
        raise RuntimeError("stage exploded")


def _stub_for(name: str, base: type[_StubStage] = _StubStage) -> type[_StubStage]:
    return type(f"Stub_{name}", (base,), {"name": name})


@pytest.fixture()
def recorder():
    original = get_tracer()
    tracer = _RecordingTracer()
    install_tracer(tracer)
    try:
        yield tracer
    finally:
        install_tracer(original)


@pytest.fixture()
def stub_plan(monkeypatch, tmp_path):
    import fiximg.config as config_mod
    from fiximg.inference import registry as registry_module

    monkeypatch.setattr(config_mod.settings, "tasks_root", str(tmp_path / "storage" / "tasks"))
    monkeypatch.setattr(config_mod.settings, "eval_mode", "inline")

    registry = registry_module.build_default_registry()
    for name in _STUBBED_STAGES:
        registry.register(name, _stub_for(name), replace=True)
    monkeypatch.setattr(registry_module, "stage_registry", registry)
    return tmp_path


def _span_names(tracer):
    return [span.name for span in tracer.recorded()]


def test_a_run_pipeline_opens_a_span_per_stage(recorder, stub_plan, isolated_db):
    from fiximg.inference.runtime import PipelineOrchestrator

    PipelineOrchestrator().run(Image.new("RGB", (32, 24), "blue"), {"username": "alice"}, "restore")

    stage_spans = [s for s in recorder.recorded() if s.name == "inference.stage"]
    assert stage_spans, "the runtime must open one span per executed stage"
    assert stage_spans[0].attributes["stage"] == "global_restore"
    assert stage_spans[0].attributes["task_id"], "a trace without a task id cannot be joined"
    assert stage_spans[0].duration_s is not None, "the span must be timed, not just named"


def test_the_colorize_pipeline_is_traced_too(recorder, stub_plan, isolated_db):
    from fiximg.inference.runtime import PipelineOrchestrator

    PipelineOrchestrator().run(Image.new("RGB", (32, 24), "gray"), {"username": "alice"}, "colorize")

    stages = [s.attributes["stage"] for s in recorder.recorded() if s.name == "inference.stage"]
    assert stages == ["colorization"]


def test_a_failing_stage_marks_its_span_and_still_closes_it(recorder, stub_plan, isolated_db, monkeypatch):
    from fiximg.inference import registry as registry_module
    from fiximg.inference.runtime import PipelineOrchestrator

    registry = registry_module.build_default_registry()
    for name in _STUBBED_STAGES:
        base = _FailingStage if name == "global_restore" else _StubStage
        registry.register(name, _stub_for(name, base), replace=True)
    monkeypatch.setattr(registry_module, "stage_registry", registry)

    with pytest.raises(RuntimeError):
        PipelineOrchestrator().run(Image.new("RGB", (32, 24), "blue"), {"username": "alice"}, "restore")

    stage_spans = [s for s in recorder.recorded() if s.name == "inference.stage"]
    assert stage_spans and stage_spans[0].duration_s is not None, (
        "the span must close even when the stage raises 鈥?an unclosed span hides the failure"
    )
