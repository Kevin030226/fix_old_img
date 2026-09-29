"""GPU utilisation measurement (plan 搂7.1) 鈥?parsing, windowing, giving up honestly.

Nothing here needs a GPU: the driver tool is behind an injectable ``runner``, so
the parsing is tested against captured output and the window maths against a
counter clock. What the seam guarantees is that the numbers on `/health/ready`, in
the task report and in the metrics series all come from the same readings.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from fiximg.inference import gpu_utilization as gu
from fiximg.inference.gpu import GpuTopology
from fiximg.inference.gpu_utilization import (
    DeviceReading,
    UtilizationProbe,
    parse_query_output,
)

# Captured from `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,
# memory.total --format=csv,noheader,nounits` on the RTX 5060 box (single line,
# ", " separated, no units); the second row is the shape a two-card host answers.
CAPTURED = "0, 0, 496, 8151\n1, 73, 2048, 8151\n"


def _probe(rows: str, ticks=None, **kwargs):
    """A probe on a counter clock: every reading gets its own stamp."""
    counter = ticks if ticks is not None else _Ticks()
    return UtilizationProbe(runner=lambda: rows, clock=counter, **kwargs)


class _Ticks:
    def __init__(self, start=0.0, step=1.0):
        self.value = start
        self.step = step

    def __call__(self) -> float:
        stamp = self.value
        self.value += self.step
        return stamp


def _filings(probe: UtilizationProbe, device: int | None = None) -> list[dict]:
    """The readings a probe holds, through the same path the endpoint uses."""
    wanted = None if device is None else [device]
    return probe.describe(wanted)["devices"]


# ------------------------------------------------------------------- parsing
def test_captured_driver_output_parses_every_field():
    rows = parse_query_output(CAPTURED)
    assert rows == [
        DeviceReading(0, 0, 496, 8151),
        DeviceReading(1, 73, 2048, 8151),
    ], rows
    assert rows[1].to_dict() == {
        "device": 1, "utilization_pct": 73, "memory_used_mb": 2048, "memory_total_mb": 8151,
    }


def test_rows_the_driver_cannot_answer_are_skipped_not_fatal():
    """MIG/vGPU rows answer `[N/A]`; one of them must not cost the healthy cards.

    A parser that raised here would take the whole reading down for a row nobody
    can use, and the probe would report the healthy GPU as unavailable.
    """
    text = (
        "0, 55, 496, 8151\n"
        "1, [N/A], [N/A], [N/A]\n"
        "GPU busy id, pid\n"          # a banner, not a row
        "\n"
        "2, 12, 512\n"                 # fewer fields than queried
    )
    assert parse_query_output(text) == [DeviceReading(0, 55, 496, 8151)]


def test_carriage_returns_and_padding_survive_the_parser():
    assert parse_query_output("0,    42 ,  100 ,  200 \r\n") == [
        DeviceReading(0, 42, 100, 200)
    ]


# ------------------------------------------------------------------ windows
def test_window_aggregates_only_this_device_inside_the_interval():
    ticks = _Ticks(start=10.0)
    probe = _probe("0, 20, 496, 8151\n1, 90, 496, 8151\n", ticks=ticks)
    started = 12.0
    probe.poll_once()                     # stamp 10 鈥?before the window
    inside = probe.now()                  # stamp 11
    probe.poll_once()                     # stamp 12 鈥?inside
    ended = probe.now()                   # stamp 13
    probe.poll_once()                     # stamp 14 鈥?after
    assert started == 12.0 and inside == 11.0 and ended == 13.0

    window = probe.window(12.0, 13.0, 0)
    assert window == {
        gu.STAGE_UTILIZATION_METRIC: 20.0,
        "gpu_util_max_pct": 20,
        "gpu_samples": 1,
        "gpu_device": 0,
    }, window
    assert probe.window(12.0, 13.0, 1)["gpu_util_max_pct"] == 90


def test_window_means_the_readings_it_found_and_reports_each_of_them():
    probe = _probe("0, 100, 1, 2\n")
    stamps = []
    for _ in range(3):
        stamps.append(probe.now())
        probe.poll_once()
    window = probe.window(stamps[0] - 1, stamps[-1] + 1, 0)
    assert window["gpu_samples"] == 3, window
    assert window[gu.STAGE_UTILIZATION_METRIC] == 100.0


def test_no_window_for_a_device_that_has_no_driver_to_ask():
    """CPU (-1) and unknown devices answer "not measured", never 0 %.

    A fabricated 0 here is the failure mode this module exists to avoid: it reads
    as "the scheduler left the card idle" on a machine that has no card.
    """
    probe = _probe("0, 60, 496, 8151\n")
    probe.poll_once()
    assert probe.window(0.0, 1000.0, -1) == {}
    assert probe.window(0.0, 1000.0, 7) == {}
    assert _filings(probe, 7) == []


# ---------------------------------------------------------------- give-up
def test_repeated_query_failings_stop_the_probe_for_the_process():
    def boom():
        raise FileNotFoundError("nvidia-smi")

    probe = UtilizationProbe(runner=boom, max_failures=2)
    assert probe.poll_once() == []
    assert probe.status == gu.STATUS_IDLE, "one failure is not a missing tool"
    assert probe.poll_once() == []
    assert probe.status == gu.STATUS_UNAVAILABLE
    assert probe._stop.is_set()
    # And it must not come back: every stage of every task would re-fork the tool.
    probe.ensure_running()
    assert probe.status == gu.STATUS_UNAVAILABLE
    assert not probe.thread_alive


def test_a_query_with_no_rows_counts_as_a_failure():
    """The tool installed but reporting nothing is "no device", not "idle at 0 %"."""
    probe = UtilizationProbe(runner=lambda: "", max_failures=2)
    probe.poll_once()
    assert probe.status != gu.STATUS_UNAVAILABLE
    probe.poll_once()
    assert probe.status == gu.STATUS_UNAVAILABLE
    assert _filings(probe) == []


def test_a_success_resets_the_failure_streak():
    answers = iter(["", "0, 5, 1, 2", "", "", ""])
    probe = UtilizationProbe(runner=lambda: answers.__next__(), max_failures=2)
    probe.poll_once()          # 1 failure
    probe.poll_once()          # success resets
    assert probe.status == gu.STATUS_MEASURING
    probe.poll_once()          # 1
    probe.poll_once()          # 2 -> gives up
    assert probe.status == gu.STATUS_UNAVAILABLE
    assert _filings(probe, 0) == [DeviceReading(0, 5, 1, 2).to_dict()]


def test_the_probe_only_forks_while_someone_cares():
    """A driver query is a subprocess: idle time must not cost one per tick.

    Asserted on the decision function with a counter clock, because "it did *not*
    poll" is exactly the claim a wall-clock sleep cannot prove.
    """
    ticks = _Ticks(start=0.0, step=1.0)          # every call moves 1 s
    probe = _probe("0, 5, 1, 2\n", ticks=ticks, interval=1.0)
    assert probe._should_poll() is False, "nobody is looking"

    probe.open_window()
    assert probe._should_poll() is True

    probe.close_window()
    # A window leaves two intervals of demand behind, so the reading that follows
    # the last stage is still fresh rather than one taken minutes ago.
    assert probe._should_poll() is True
    assert probe._should_poll() is False, "the demand must expire, not linger"

    probe.request_reading()
    assert probe._should_poll() is True


def test_a_window_closed_too_often_never_goes_negative():
    """A leaked close would park the sampler on a count that can never reopen."""
    probe = _probe("0, 5, 1, 2\n")
    probe.close_window()
    assert probe._windows == 0
    assert probe._should_poll() is False


def test_the_thread_samples_and_stops_without_leaking():
    probe = UtilizationProbe(runner=lambda: "0, 40, 1, 2\n", interval=0.01)
    probe.ensure_running()
    first = probe._thread
    assert first is not None
    probe.ensure_running()
    assert probe._thread is first, "the second call must not start another sampler"
    probe.open_window()
    sampler_thread = first
    deadline = time.monotonic() + 5.0
    while not _filings(probe, 0) and time.monotonic() < deadline:
        time.sleep(0.005)
    assert _filings(probe, 0), "the sampler thread never filed a reading"
    assert probe.status == gu.STATUS_MEASURING
    probe.stop()
    # Identity, not a process-wide thread count: other runtime tests legitimately
    # open real windows on a CUDA host, so counting threads would depend on order.
    assert not sampler_thread.is_alive(), sampler_thread
    assert not probe.thread_alive


# ------------------------------------------------------------------ report
def test_describe_reports_the_freshest_reading_per_device():
    probe = _probe("0, 0, 496, 8151\n1, 77, 512, 8151\n")
    probe.poll_once()
    report = probe.describe([0, 1])
    assert report["status"] == gu.STATUS_MEASURING
    assert [row["device"] for row in report["devices"]] == [0, 1]
    assert report["busy_devices"] == 1, report
    assert report["interval_seconds"] == pytest.approx(gu.DEFAULT_INTERVAL_SECONDS)
    # Freshness travels with the reading: the sampler stops when nobody is looking,
    # so a reader must be able to tell a 0 % that just happened from one from minutes
    # ago. Without this key `utilization_pct: 0` on an idle API node is unmarkable.
    assert isinstance(report["age_seconds"], float), report
    assert report["age_seconds"] >= 0.0


def test_describe_of_a_cpu_only_topology_invents_no_device_row():
    probe = _probe("0, 30, 1, 2\n")
    probe.poll_once()
    report = probe.describe([-1])
    assert report["devices"] == []
    assert "busy_devices" not in report, report


def test_describe_without_a_filter_lists_each_device_once():
    probe = _probe("0, 30, 1, 2\n1, 0, 1, 2\n")
    for _ in range(3):
        probe.poll_once()
    report = probe.describe()
    assert sorted(row["device"] for row in report["devices"]) == [0, 1]


def test_module_helpers_delegate_to_the_shared_probe(monkeypatch):
    """`start_probe`/`stage_window` are the seam the runtime and API call through."""
    calls: list = []

    class _Stub:
        def ensure_running(self):
            calls.append("ensure")

        def window(self, started_at, ended_at, device):
            calls.append((started_at, ended_at, device))
            return {"gpu_util_pct": 5.0}

        def describe(self, devices=None):
            calls.append(("describe", devices))
            return {"status": "measuring"}

    monkeypatch.setattr(gu, "default_probe", _Stub())
    assert gu.start_probe() is not None
    assert gu.stage_window(1.0, 2.0, 0) == {"gpu_util_pct": 5.0}
    assert gu.describe([0]) == {"status": "measuring"}
    assert calls == ["ensure", (1.0, 2.0, 0), "ensure", ("describe", [0])], calls


# ---------------------------------------------------------- the runtime's window
class _Stage:
    name = "global_restore"
    version = "test"
    capabilities = frozenset({"restore"})

    def run(self, image, context):
        return SimpleNamespace(
            image=image, artifacts={}, metadata={}, metrics={}, message=None
        )


def _orchestrator(monkeypatch, topology, stage, registry=None):
    from fiximg.inference import runtime as orch

    monkeypatch.setattr(orch.task_repo, "record_stage", lambda *a, **k: None)
    monkeypatch.setattr(orch.task_repo, "add_event", lambda *a, **k: None)
    monkeypatch.setattr(orch.task_repo, "update_progress", lambda *a, **k: None)
    finished: list[dict] = []
    monkeypatch.setattr(
        orch.task_repo, "finish_stage",
        lambda task_id, order, status, duration_ms, message, metrics=None: finished.append(
            {"status": status, "metrics": metrics}
        ),
    )
    monkeypatch.setattr(orch.task_repo, "interruption", lambda *a, **k: None)
    if registry is not None:
        monkeypatch.setattr(orch, "metrics", registry)
    orchestrator = orch.PipelineOrchestrator(
        planner=SimpleNamespace(build_stage=lambda name, kwargs: stage),
        topology=topology,
    )
    return orchestrator, finished


class _WindowProbe:
    """Stand-in for the shared probe: the maths is tested above, the wiring here."""

    def __init__(self, window=None):
        self.window_result = dict(window or {})
        self.started = False
        self.asked: list[tuple[float, float, int]] = []
        self.opened = 0
        self.closed = 0
        self._ticks = _Ticks(start=100.0)

    def ensure_running(self):
        self.started = True

    def open_window(self):
        self.opened += 1

    def close_window(self):
        self.closed += 1

    def now(self) -> float:
        return self._ticks()

    def window(self, started_at, ended_at, device):
        self.asked.append((started_at, ended_at, device))
        return dict(self.window_result)


def _run_one_stage(monkeypatch, topology, probe, registry=None):
    from fiximg.inference.context import StageContext

    orchestrator, finished = _orchestrator(monkeypatch, topology, _Stage(), registry)
    monkeypatch.setattr(gu, "default_probe", probe)
    orchestrator._run_plan("t", "img", SimpleNamespace(stages=[("global_restore", {})]),
                           StageContext(task_id="t", run_dir=".", options={}))
    return finished


READING = {
    gu.STAGE_UTILIZATION_METRIC: 62.5,
    "gpu_util_max_pct": 90,
    "gpu_samples": 4,
    "gpu_device": 0,
}


def test_a_gpu_stage_opens_a_window_and_publishes_its_utilisation(monkeypatch):
    """搂7.1: the stage's own utilisation reaches the report, the rows and `/metrics`.

    Asserted through the real runtime: the probe is asked for the *routed* device
    over an interval the runtime stamped on the probe's own clock, the window is
    opened and released around exactly that interval, and the same aggregate lands
    in `stage_metrics`, in the persisted row and in the series.
    """
    from fiximg.infrastructure.observability.metrics import MetricsRegistry

    registry = MetricsRegistry()
    probe = _WindowProbe(READING)
    finished = _run_one_stage(monkeypatch, GpuTopology.single_device(0), probe, registry)

    assert probe.started, "the runtime must warm the sampler before it needs it"
    assert (probe.opened, probe.closed) == (1, 1), (probe.opened, probe.closed)
    assert len(probe.asked) == 1, probe.asked
    started_at, ended_at, device = probe.asked[0]
    assert device == 0
    assert ended_at > started_at, (started_at, ended_at)

    metrics_row = finished[0]["metrics"]
    assert metrics_row[gu.STAGE_UTILIZATION_METRIC] == 62.5, metrics_row
    assert metrics_row["gpu_util_max_pct"] == 90
    assert metrics_row["gpu_samples"] == 4

    snapshot = registry.snapshot()
    key = f'gpu_utilization_percent{{device=0,stage={_Stage.name}}}'
    assert key in snapshot["histograms"], sorted(snapshot["histograms"])
    assert snapshot["histograms"][key]["count"] == 1


def test_a_cpu_stage_asks_no_utilisation_at_all(monkeypatch):
    """No driver, no reading 鈥?and no sampler thread started to fail at one.

    This is what keeps a CPU-only deployment honest: the alternative is a stage
    metric that says 0 % for work that never touched a GPU.
    """
    probe = _WindowProbe(READING)
    finished = _run_one_stage(monkeypatch, GpuTopology.single_device(-1), probe)

    assert not probe.started
    assert (probe.opened, probe.closed) == (0, 0)
    assert probe.asked == []
    assert not [k for k in (finished[0]["metrics"] or {}) if "util" in k], finished[0]


def test_a_failed_stage_still_releases_the_window(monkeypatch):
    """The sampler must not keep forking for a stage that stopped existing.

    A window left open by an exception is a permanent demand: the probe would
    query the driver every interval for the rest of the process's life, on a box
    running nothing.
    """
    from fiximg.inference.context import StageContext

    class _Boom(_Stage):
        def run(self, image, context):
            raise RuntimeError("model crashed")

    probe = _WindowProbe(READING)
    orchestrator, _finished = _orchestrator(
        monkeypatch, GpuTopology.single_device(0), _Boom()
    )
    monkeypatch.setattr(gu, "default_probe", probe)
    with pytest.raises(RuntimeError, match="model crashed"):
        orchestrator._run_plan(
            "t", "img", SimpleNamespace(stages=[("global_restore", {})]),
            StageContext(task_id="t", run_dir=".", options={}),
        )
    assert (probe.opened, probe.closed) == (1, 1), (probe.opened, probe.closed)


def test_task_level_utilisation_is_the_sample_weighted_mean_of_its_stages():
    """Two stages of different length must not count equally (plan 搂7.1, 搂5.3).

    The merge this replaced took the last stage's figure: a 4-sample restore at
    100 % followed by a 1-sample finishing stage at 20 % reported 20 % for a task
    that spent almost all of its GPU time flat out.
    """
    from fiximg.inference.runtime import _collect_stage_metrics

    stages = [
        {"stage": "global_restore", "metrics": {
            gu.STAGE_UTILIZATION_METRIC: 100.0, "gpu_util_max_pct": 100, "gpu_samples": 4}},
        {"stage": "colorization", "metrics": {
            gu.STAGE_UTILIZATION_METRIC: 20.0, "gpu_util_max_pct": 90, "gpu_samples": 1}},
    ]
    merged = _collect_stage_metrics(stages)
    assert merged[gu.STAGE_UTILIZATION_METRIC] == 84.0, merged
    assert merged["gpu_util_max_pct"] == 100, merged
    assert merged["gpu_samples"] == 5, "the task reports the pool it averaged over"


def test_a_single_stage_task_keeps_its_own_utilisation():
    from fiximg.inference.runtime import _collect_stage_metrics

    merged = _collect_stage_metrics([{"stage": "s", "metrics": dict(READING)}])
    assert merged[gu.STAGE_UTILIZATION_METRIC] == 62.5


def test_a_missing_weight_does_not_break_the_mean():
    """A stage that reported a percentage but no sample count still counts."""
    from fiximg.inference.runtime import _collect_stage_metrics

    merged = _collect_stage_metrics([
        {"stage": "a", "metrics": {gu.STAGE_UTILIZATION_METRIC: 100.0, "gpu_samples": 0}},
        {"stage": "b", "metrics": {gu.STAGE_UTILIZATION_METRIC: 0.0}},
    ])
    assert merged[gu.STAGE_UTILIZATION_METRIC] == 50.0, merged
