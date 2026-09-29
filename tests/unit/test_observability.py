"""Observability tests (plan 搂2.10, 搂3.4.2).

Metrics registry, the JSON log correlation fields and the tracing seam.
"""
import json
import logging
import sys

import pytest

from fiximg.infrastructure.observability.logging import (
    JsonFormatter,
    clear_context,
    current_context,
    new_request_id,
    set_gpu_id,
    set_task_id,
    set_user_id,
    set_worker_id,
)
from fiximg.infrastructure.observability.metrics import (
    MetricName,
    MetricsRegistry,
    metrics,
)
from fiximg.infrastructure.observability.tracing import (
    NoOpTracer,
    OtelTracer,
    _RecordingTracer,
    get_tracer,
    install_tracer,
    tracer as tracer_handle,
)


# ------------------------------------------------------------------ metrics
def test_counters_accumulate_and_are_labelled():
    registry = MetricsRegistry()
    registry.inc("task_submit_total")
    registry.inc("task_submit_total")
    registry.inc("task_submit_total", task_type="restore")

    counters = registry.snapshot()["counters"]
    assert counters["task_submit_total"] == 2.0
    assert counters["task_submit_total{task_type=restore}"] == 1.0


def test_histogram_reports_percentiles():
    registry = MetricsRegistry()
    for value in range(1, 101):
        registry.observe("stage_duration_seconds", value / 100)

    summary = registry.snapshot()["histograms"]["stage_duration_seconds"]
    assert summary["count"] == 100
    assert summary["avg"] == 0.505
    assert summary["p50"] == 0.50
    assert summary["p95"] == 0.95
    assert summary["p99"] == 0.99


def test_two_samples_do_not_report_the_minimum_as_the_p95():
    """The case the old `int(n * pct) - 1` index got wrong 鈥?and n=2 is what CI runs.

    A p95 below the p50 is not a conservative estimate, it is a different
    quantity, and `/metrics` would have published it as the same name.
    """
    registry = MetricsRegistry()
    registry.observe("queue_wait_seconds", 0.9)
    registry.observe("queue_wait_seconds", 0.1)

    summary = registry.snapshot()["histograms"]["queue_wait_seconds"]
    assert summary["p50"] == 0.1  # nearest rank of two samples is the lower one
    assert summary["p95"] == 0.9
    assert summary["p99"] == 0.9
    assert summary["p95"] >= summary["p50"]


def test_timer_context_manager_records_a_sample():
    registry = MetricsRegistry()
    with registry.timer("artifact_io_seconds", kind="output"):
        pass
    assert registry.snapshot()["histograms"]["artifact_io_seconds{kind=output}"]["count"] == 1


def test_histogram_window_is_bounded():
    registry = MetricsRegistry()
    for value in range(2000):
        registry.observe("bounded", value)
    summary = registry.snapshot()["histograms"]["bounded"]
    # The count is cumulative but the sample window is capped.
    assert summary["count"] == 2000
    assert summary["p99"] <= 2000


def test_prometheus_exposition_is_well_formed():
    registry = MetricsRegistry()
    registry.inc("task_submit_total", 3)
    registry.observe("task_duration_seconds", 1.5)

    text = registry.render_prometheus()
    assert "task_submit_total 3.0" in text
    assert "task_duration_seconds 1" in text
    assert text.endswith("\n")


def test_labelled_series_are_valid_exposition():
    """Quoted label values, and the labels repeated on the derived series.

    Two defects, both invisible to a test that only checked a counter with no
    labels. Found by mutation-testing the deployment gap-filler: two series that
    should have compared equal did not.

    * `stage=global_restore` is not valid exposition syntax. A scraper rejects the
      scrape or that series, so every labelled series the registry recorded was
      being served malformed.
    * Emitting `stage_duration_seconds{stage=x}` and then a bare
      `stage_duration_seconds_p50` drops the label from the percentile, collapsing
      every stage's p95 into one number — the opposite of the per-stage breakdown
      `docs/deployment.md` promises at this endpoint.
    """
    import re

    from fiximg.infrastructure.observability.metrics import MetricsRegistry

    registry = MetricsRegistry()
    registry.observe("stage_duration_seconds", 1.5, stage="global_restore")
    registry.inc("task_submit_total", 2, task_type="restore")
    text = registry.render_prometheus()

    # name{label="value",...} float  — the whole exposition, not just one line.
    line_re = re.compile(
        r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})? -?[0-9.eE+]+$'
    )
    label_re = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*="[^"]*"$')

    emitted = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert emitted, text
    for line in emitted:
        match = line_re.match(line)
        assert match, f"not valid exposition syntax: {line!r}\n{text}"
        labels = match.group(2)
        if labels:
            for pair in labels.strip("{}").split(","):
                assert label_re.match(pair), (
                    f"label {pair!r} is not quoted; the exposition is malformed: {line!r}"
                )

    # The label must survive onto the derived percentile, or the per-stage view is
    # a single number with the stages averaged together.
    p95 = [ln for ln in emitted if ln.startswith("stage_duration_seconds_p95")]
    assert p95, text
    assert all('stage="global_restore"' in ln for ln in p95), p95


def test_prometheus_exposition_carries_every_percentile_the_summary_has():
    """`/metrics` used to publish only avg and p95 of the three it computes."""
    registry = MetricsRegistry()
    for value in (1.0, 2.0, 3.0, 4.0):
        registry.observe("stage_duration_seconds", value)

    text = registry.render_prometheus()
    for name in ("avg", "p50", "p95", "p99"):
        assert f"stage_duration_seconds_{name} " in text, name
    assert "stage_duration_seconds_p50 2.0" in text


def test_reset_clears_everything():
    registry = MetricsRegistry()
    registry.inc("x")
    registry.observe("y", 1)
    registry.reset()
    snapshot = registry.snapshot()
    assert snapshot["counters"] == {}
    assert snapshot["histograms"] == {}


def test_module_singleton_exposes_the_documented_metric_names():
    assert MetricName.TASK_SUBMIT_TOTAL == "task_submit_total"
    assert MetricName.STAGE_DURATION_SECONDS == "stage_duration_seconds"
    assert MetricName.MODEL_INFERENCE_SECONDS == "model_inference_seconds"
    assert MetricName.QUEUE_WAIT_SECONDS == "queue_wait_seconds"
    assert metrics is not None


def test_every_declared_metric_name_has_a_production_call_site():
    """A declared series nobody records is decoration, not observability (plan 搂2.10).

    The test above asserts these names *by string equality*, which still passes
    after every producer is deleted 鈥?and five of the ten names (`queue_depth`,
    `model_load_seconds`, `model_inference_seconds`, `gpu_memory_bytes`,
    `artifact_io_seconds`) were in exactly that state: declared, documented, never
    recorded, and therefore absent from `/api/v1/stats` and the Prometheus text.

    Scanning `src/` is deliberately coarse 鈥?a docstring naming the constant would
    pass. What this gate is for is the failure it actually caught: a name that
    exists in one file and nowhere else.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "src"
    sources = [
        path for path in root.rglob("*.py")
        if "__pycache__" not in str(path) and path.name != "metrics.py"
    ]
    declared = {
        name: value for name, value in vars(MetricName).items()
        if not name.startswith("_") and isinstance(value, str)
    }

    for const, value in sorted(declared.items()):
        writers = [
            path for path in sources
            if re.search(rf"MetricName\.{const}\b",
                         path.read_text(encoding="utf-8", errors="replace"))
        ]
        assert writers, f"{value}: declared in MetricName but recorded nowhere in src/"


def test_an_inference_call_records_its_own_duration():
    """`model_inference_seconds` must be the forward pass, not the stage around it."""
    from fiximg.inference.backends.base import BaseModelBackend, ModelRequest, ModelResult

    registry = MetricsRegistry()
    calls = {"load": 0, "infer": 0}

    class _Stub(BaseModelBackend):
        name = "stub_model"
        version = "1.0"
        implementation = "native"
        capabilities = frozenset({"restore"})

        def _do_load(self, device: str) -> None:
            calls["load"] += 1

        def _do_infer(self, request: ModelRequest) -> ModelResult:
            calls["infer"] += 1
            return ModelResult(image=request.image)

    import fiximg.inference.backends.base as base_module
    import numpy as np
    from PIL import Image

    request = ModelRequest(image=Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)))
    original = base_module.metrics
    base_module.metrics = registry
    try:
        _Stub().infer(request)
    finally:
        base_module.metrics = original

    keys = registry.snapshot()["histograms"]
    assert any(key.startswith("model_inference_seconds{") for key in keys), keys
    assert calls == {"load": 1, "infer": 1}, "the stub did not run the real path"
    # The timer belongs to the forward pass only: loading is a different series,
    # and folding it in would make one name mean two things.
    assert not any(key.startswith("model_load_seconds") for key in keys), keys


# ---------------------------------------------------------------- log context
def test_json_formatter_includes_correlation_fields():
    new_request_id()
    set_user_id("alice")
    set_task_id("t-1")
    set_worker_id("w-1")
    set_gpu_id(0)

    record = logging.LogRecord("fiximg.test", logging.INFO, __file__, 1, "hello", None, None)
    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "hello"
    assert payload["user_id"] == "alice"
    assert payload["task_id"] == "t-1"
    assert payload["worker_id"] == "w-1"
    assert payload["gpu_id"] == "0"
    assert payload["request_id"]
    clear_context()


def test_clear_context_removes_every_field():
    new_request_id()
    set_user_id("bob")
    set_task_id("t-2")
    clear_context()
    assert current_context() == {}


def test_current_context_only_lists_set_fields():
    clear_context()
    set_task_id("t-3")
    assert current_context() == {"task_id": "t-3"}
    clear_context()


def test_log_event_attaches_extra_fields():
    from fiximg.infrastructure.observability.logging import get_logger, log_event

    logger = get_logger("fiximg.test.extra")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):  # noqa: D102
            records.append(record)

    handler = _Capture()
    logger.addHandler(handler)
    try:
        log_event(logger, "INFO", "stage done", stage="colorization", duration_ms=12)
    finally:
        logger.removeHandler(handler)

    assert records[0].extra_fields == {"stage": "colorization", "duration_ms": 12}


# ------------------------------------------------------------------ tracing
def test_default_tracer_is_a_noop():
    tracer = NoOpTracer()
    with tracer.span("test", task_id="t") as span:
        pass
    assert span.duration_s is not None
    assert tracer.enabled is False


def test_recording_tracer_keeps_spans():
    tracer = _RecordingTracer()
    with tracer.span("a"):
        pass
    with tracer.span("b", stage="x"):
        pass
    assert [s.name for s in tracer.recorded()] == ["a", "b"]
    assert tracer.recorded()[1].attributes == {"stage": "x"}


def test_install_tracer_swaps_the_active_tracer():
    original = get_tracer()
    recorder = _RecordingTracer()
    try:
        install_tracer(recorder)
        assert get_tracer() is recorder
    finally:
        install_tracer(original)


# ------------------------------------------------------- tracer selection (搂2.10)
@pytest.fixture()
def clean_tracer():
    """get_tracer() builds lazily, so a test must restore what it changed."""
    import fiximg.infrastructure.observability.tracing as tracing_module

    original = tracing_module._tracer
    yield tracing_module
    tracing_module._tracer = original


def test_tracing_off_is_a_noop_by_default(clean_tracer, monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "tracing", "off")
    assert clean_tracer.default_tracer().enabled is False


def test_internal_mode_keeps_spans_in_memory(clean_tracer, monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "tracing", "internal")
    tracer = clean_tracer.default_tracer()
    assert isinstance(tracer, _RecordingTracer)
    assert tracer.enabled is True


def test_unknown_modes_are_rejected_at_config_time(monkeypatch):
    """A typo must not silently mean "untraced" (plan 搂3.3 fail-fast)."""
    from fiximg.config import Settings

    monkeypatch.setenv("FIXIMG_TRACING", "yes-please")
    with pytest.raises(ValueError, match="FIXIMG_TRACING"):
        Settings()


def test_off_sdk_and_internal_are_all_accepted(monkeypatch):
    from fiximg.config import Settings

    for mode in ("off", "sdk", "otel", "internal"):
        monkeypatch.setenv("FIXIMG_TRACING", mode)
        assert Settings().tracing == mode


def test_asking_for_otel_without_the_package_says_so(clean_tracer, monkeypatch, caplog):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "tracing", "sdk")
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    with caplog.at_level("WARNING"):
        tracer = clean_tracer.default_tracer()
    assert isinstance(tracer, NoOpTracer), "degrade, but do not pretend"
    assert "opentelemetry is not installed" in caplog.text
    assert "fiximg[otel]" in caplog.text


def _fake_otel(monkeypatch, log):
    """A stand-in for opentelemetry that records what a span was told to do."""
    import types

    class _FakeOTelSpan:
        def record_exception(self, exc, **kwargs):
            log.append(("exception", type(exc).__name__))

        def set_status(self, code, description=""):
            log.append(("status", code, description))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is not None:
                self.record_exception(exc)
                self.set_status("ERROR", str(exc) or exc_type.__name__)
            return False

    class _FakeOTelTracer:
        def start_as_current_span(self, name, attributes=None, **kwargs):
            log.append(("start", name, dict(attributes or {})))
            return _FakeOTelSpan()

    trace_module = types.ModuleType("opentelemetry.trace")
    trace_module.get_tracer = lambda name: _FakeOTelTracer()

    class _StatusCode:
        ERROR = "ERROR"
        OK = "OK"

    trace_module.StatusCode = _StatusCode
    otel_module = types.ModuleType("opentelemetry")
    otel_module.trace = trace_module
    monkeypatch.setitem(sys.modules, "opentelemetry", otel_module)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_module)


def test_otel_tracer_forwards_the_span_and_its_attributes(clean_tracer, monkeypatch):
    log: list = []
    _fake_otel(monkeypatch, log)

    with OtelTracer().span("inference.stage", task_id="t-1", stage="global_restore",
                           device=None, payload={"nested": 1}):
        pass

    started = [entry for entry in log if entry[0] == "start"]
    assert started == [("start", "inference.stage",
                        {"task_id": "t-1", "stage": "global_restore", "payload": "{'nested': 1}"})], (
        "None attributes are dropped and non-scalars stringified 鈥?OTel rejects them"
    )


def test_otel_tracer_records_a_failing_span(clean_tracer, monkeypatch):
    log: list = []
    _fake_otel(monkeypatch, log)

    with pytest.raises(RuntimeError):
        with OtelTracer().span("model.load", model_name="ddcolor"):
            raise RuntimeError("weight missing")

    assert ("exception", "RuntimeError") in log
    assert ("status", "ERROR", "weight missing") in log


def test_otel_tracer_keeps_the_correlation_context(clean_tracer, monkeypatch):
    log: list = []
    _fake_otel(monkeypatch, log)
    assert OtelTracer().current_context() == NoOpTracer().current_context()


def test_the_module_handle_follows_install_tracer(clean_tracer):
    """`from ... import tracer` must not pin call sites to the first tracer.

    The module-level alias used to be bound at import time, so install_tracer()
    silently changed nothing for every call site that imported it.
    """
    original = get_tracer()
    recorder = _RecordingTracer()
    try:
        install_tracer(recorder)
        with tracer_handle.span("task.execute", task_id="t-9"):
            pass
        assert [s.name for s in recorder.recorded()] == ["task.execute"]
        assert tracer_handle.enabled is True
    finally:
        install_tracer(original)

