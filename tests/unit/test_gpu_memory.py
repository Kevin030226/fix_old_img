"""Memory-aware scheduling and per-stage VRAM metrics (plan 搂7.1, 搂4.3 Step 4).

The report lists two related gaps: "GPU peak memory 鈥?鍔犲叆姣?task/stage 缁村害" and
the multi-GPU item's note that capability routing alone is not enough without
memory awareness.

Everything here must also work on a CPU-only host, where probes return None: the
selection falls back to the first candidate and the sampler reports no peak. That
fallback is asserted explicitly, because it is what keeps a CPU deployment on its
previous behaviour.
"""
import torch
from types import SimpleNamespace

import pytest

from fiximg.inference.gpu import DeviceAssignment, GpuTopology
from fiximg.inference.gpu_memory import (
    DeviceMemory,
    MemorySampler,
    all_devices,
    device_memory,
    is_oom_error,
    memory_summary,
    peak_memory_report,
    select_device,
)


def _snap(device: int, free_mb: float, total_mb: float = 24_000.0) -> DeviceMemory:
    return DeviceMemory(device=device, total_mb=total_mb, free_mb=free_mb)


def _probe(mapping: dict[int, float]):
    """A fake probe returning a snapshot only for the listed devices."""
    return lambda device: _snap(device, mapping[device]) if device in mapping else None


# ------------------------------------------------------------ device snapshots
def test_device_memory_of_cpu_is_none():
    assert device_memory(-1) is None


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(_cuda_available(), reason="this host has a usable GPU")
def test_all_devices_is_empty_without_cuda():
    """On a CPU-only host the probe must return nothing rather than raise."""
    assert all_devices() == []


@pytest.mark.skipif(_cuda_available(), reason="this host has a usable GPU")
def test_peak_memory_report_is_empty_without_cuda():
    assert peak_memory_report() == {}


@pytest.mark.skipif(not _cuda_available(), reason="needs a usable GPU")
def test_all_devices_reports_every_visible_card():
    snapshots = all_devices()
    assert snapshots, "CUDA is available but no device was reported"
    for snapshot in snapshots:
        assert snapshot.total_mb > 0
        assert 0 <= snapshot.free_mb <= snapshot.total_mb


def test_memory_summary_is_json_friendly():
    summary = memory_summary()
    assert set(summary) == {"devices", "peak_allocated_mb"}
    assert isinstance(summary["devices"], list)


def test_device_memory_to_dict_reports_usage():
    payload = _snap(0, 10_000.0, 24_000.0).to_dict()
    assert payload["free_mb"] == 10_000.0
    assert payload["used_mb"] == 14_000.0
    assert payload["device"] == 0


# ---------------------------------------------------------------- OOM detection
def test_is_oom_error_recognises_torch_messages():
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert is_oom_error(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"))
    assert is_oom_error(RuntimeError("hip out of memory"))


def test_is_oom_error_ignores_other_failures():
    assert is_oom_error(RuntimeError("shape mismatch")) is False
    assert is_oom_error(ValueError("bad input")) is False
    assert is_oom_error(None) is False


def test_is_oom_error_matches_torch_exception_type():
    torch = pytest.importorskip("torch")
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is None:
        pytest.skip("torch build has no OutOfMemoryError")
    assert is_oom_error(oom_type("out of memory"))


# ------------------------------------------------------------ device selection
def test_select_device_picks_the_roomiest():
    chosen = select_device([0, 1], probe=_probe({0: 2_000.0, 1: 9_000.0}))
    assert chosen == 1


def test_select_device_respects_the_required_headroom():
    # Device 1 is roomiest but cannot fit; device 0 can.
    chosen = select_device(
        [0, 1], required_mb=4_000.0, headroom_mb=100.0,
        probe=_probe({0: 5_000.0, 1: 3_000.0}),
    )
    assert chosen == 0


def test_select_device_falls_back_to_the_first_without_probe_data():
    assert select_device([2, 5], probe=_probe({})) == 2
    assert select_device([], probe=_probe({})) == -1


def test_select_device_returns_the_roomiest_when_nothing_fits():
    """A heuristic must not turn a transient spike into a hard failure."""
    chosen = select_device(
        [0, 1], required_mb=99_000.0, probe=_probe({0: 500.0, 1: 900.0})
    )
    assert chosen == 1


def test_select_device_ignores_devices_without_a_snapshot():
    chosen = select_device([0, 1, 2], probe=_probe({0: 100.0, 2: 8_000.0}))
    assert chosen == 2


# ------------------------------------------------------- topology integration
def test_candidate_devices_lists_every_match():
    topology = GpuTopology(
        assignments=[
            DeviceAssignment(0, frozenset({"restore"})),
            DeviceAssignment(1, frozenset({"restore"})),
        ],
        default_device=0,
    )
    assert topology.candidate_devices(["restore"]) == [0, 1]


def test_candidate_devices_of_a_pinned_worker_is_single():
    topology = GpuTopology(default_device=0, pinned_device=1)
    assert topology.candidate_devices(["restore"]) == [1]


def test_candidate_devices_falls_back_to_the_default():
    topology = GpuTopology(
        assignments=[DeviceAssignment(1, frozenset({"colorize"}))], default_device=0
    )
    assert topology.candidate_devices(["restore"]) == [0]


def test_candidate_devices_are_de_duplicated():
    topology = GpuTopology(
        assignments=[
            DeviceAssignment(0, frozenset({"restore"})),
            DeviceAssignment(0, frozenset({"deblur"})),
        ],
        default_device=0,
    )
    assert topology.candidate_devices(["restore", "deblur"]) == [0]


def test_device_for_without_memory_awareness_keeps_the_first_match():
    """Default behaviour must not change: first declared match wins."""
    topology = GpuTopology(
        assignments=[
            DeviceAssignment(0, frozenset({"restore"})),
            DeviceAssignment(1, frozenset({"restore"})),
        ],
        default_device=0,
    )
    assert topology.device_for(["restore"]) == 0


def test_device_for_with_memory_awareness_picks_the_roomiest(monkeypatch):
    import fiximg.inference.gpu_memory as mem

    monkeypatch.setattr(mem, "device_memory", _probe({0: 1_000.0, 1: 12_000.0}))
    topology = GpuTopology(
        assignments=[
            DeviceAssignment(0, frozenset({"restore"})),
            DeviceAssignment(1, frozenset({"restore"})),
        ],
        default_device=0,
    )
    assert topology.device_for(["restore"], memory_aware=True) == 1


def test_memory_aware_selection_ignores_a_single_candidate(monkeypatch):
    """Nothing to choose between 鈥?the probe must not even be consulted."""
    calls = []

    def _spy(device):
        calls.append(device)
        return _snap(device, 1.0)

    topology = GpuTopology.single_device(0)
    assert topology.device_for(["restore"], memory_aware=True, probe=_spy) == 0
    assert calls == []


def test_memory_aware_selection_honours_pinning(monkeypatch):
    """A pinned worker must not be moved to a roomier card."""
    topology = GpuTopology(default_device=0, pinned_device=1)
    assert topology.device_for(["anything"], memory_aware=True,
                               probe=_probe({0: 99_000.0, 1: 10.0})) == 1


def test_device_for_accepts_an_explicit_probe():
    topology = GpuTopology(
        assignments=[
            DeviceAssignment(0, frozenset({"restore"})),
            DeviceAssignment(1, frozenset({"restore"})),
        ],
        default_device=0,
    )
    assert topology.device_for(
        ["restore"], memory_aware=True, probe=_probe({0: 900.0, 1: 7_000.0})
    ) == 1


# ------------------------------------------------------------- memory sampler
def test_sampler_reports_no_peak_without_cuda():
    with MemorySampler(-1) as sampler:
        pass
    assert sampler.peak_mb is None
    assert sampler.metrics() == {}


def test_sampler_metrics_shape_when_a_peak_is_recorded():
    sampler = MemorySampler(0)
    sampler.peak_mb = 1234.5
    assert sampler.metrics() == {"gpu_peak_mb": 1234.5, "gpu_device": 0}


def test_sampler_is_usable_as_a_context_manager_on_a_device_index():
    """A CUDA device index on a CPU-only host must still be harmless.

    Deterministically *no* measurement: `__enter__` bails out because there is no
    CUDA, and `__exit__` finds `max_memory_allocated` raising, so the sampler
    degrades to "nothing recorded" rather than propagating the error into a run.
    """
    with MemorySampler(0) as sampler:
        pass

    if torch.cuda.is_available():
        # On a CUDA host the sampler really can read device 0, and the honest
        # contract is "a number, and no exception" 鈥?not "None". Asserting None
        # here was a CPU-only-machine assumption that read as a failure the first
        # time the suite ran on the GPU it was written for.
        assert isinstance(sampler.peak_mb, float), sampler.peak_mb
        assert set(sampler.metrics()) == {"gpu_peak_mb", "gpu_device"}
    else:
        assert sampler.peak_mb is None
        assert sampler.metrics() == {}


# ------------------------------------------- wiring into the runtime's stage run
class _Stage:
    name = "global_restore"
    version = "test"
    capabilities = frozenset({"restore"})

    def __init__(self, seen, fail_times=0):
        self._seen = seen
        self._fail_times = fail_times

    def run(self, image, context):
        self._seen.append(context.gpu)
        if len(self._seen) <= self._fail_times:
            raise RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB")
        return SimpleNamespace(image=image, artifacts={}, metadata={}, metrics={}, message=None)


def _orchestrator(monkeypatch, topology, stages):
    from fiximg.inference import runtime as orch

    monkeypatch.setattr(orch.task_repo, "record_stage", lambda *a, **k: None)
    monkeypatch.setattr(orch.task_repo, "finish_stage", lambda *a, **k: None)
    monkeypatch.setattr(orch.task_repo, "add_event", lambda *a, **k: None)
    monkeypatch.setattr(orch.task_repo, "update_progress", lambda *a, **k: None)
    return orch.PipelineOrchestrator(
        planner=SimpleNamespace(build_stage=lambda name, kwargs: stages[name]),
        topology=topology,
    )


def _plan(*names):
    return SimpleNamespace(stages=[(name, {}) for name in names])


def test_stage_metrics_carry_the_peak_vram(monkeypatch):
    """搂7.1: the peak is attributed to the stage that caused it.

    The sampler cannot measure anything on a CPU-only host, so the *reading* is
    replaced with one that did record a peak. What is under test is the
    device-independent half: the runtime merging it into the stage's metrics and
    handing that to the repository 鈥?which is what puts `gpu_peak_mb` into
    `task_stages.metrics_json` and then the API. The previous version of this test
    asserted only `"metrics" in stage_meta[0]`, which held whether or not the
    merge existed.
    """
    from fiximg.inference import runtime as orch
    from fiximg.inference.context import StageContext

    monkeypatch.setattr(
        orch.MemorySampler, "metrics",
        lambda self: {"gpu_peak_mb": 2048.0, "gpu_device": self.device},
    )

    topology = GpuTopology.single_device(0)
    stage = _Stage([])
    orchestrator = _orchestrator(monkeypatch, topology, {"global_restore": stage})
    # Patched *after* the helper, which replaces `finish_stage` with its own
    # no-op: patching first would be silently overwritten and the list stays
    # empty for the wrong reason.
    finished: list[dict] = []
    monkeypatch.setattr(
        orch.task_repo, "finish_stage",
        lambda task_id, order, status, duration_ms, message, metrics=None: finished.append(
            {"status": status, "metrics": metrics}
        ),
    )
    context = StageContext(task_id="t", run_dir="/tmp/x", options={})

    _image, _artifacts, stage_meta = orchestrator._run_plan(
        "t", "img", _plan("global_restore"), context
    )
    assert stage_meta[0]["device"] == 0
    assert stage_meta[0]["metrics"]["gpu_peak_mb"] == 2048.0
    assert finished, "the stage must still be persisted"
    assert finished[0]["status"] == "completed"
    assert finished[0]["metrics"]["gpu_peak_mb"] == 2048.0


def test_oom_moves_the_stage_to_another_device(monkeypatch):
    """搂7.1: an out-of-memory failure retries on a capable device."""
    from fiximg.inference.context import StageContext

    topology = GpuTopology(
        assignments=[
            DeviceAssignment(0, frozenset({"restore"})),
            DeviceAssignment(1, frozenset({"restore"})),
        ],
        default_device=0,
    )
    seen: list[int] = []
    stage = _Stage(seen, fail_times=1)          # fails once, then succeeds
    orchestrator = _orchestrator(monkeypatch, topology, {"global_restore": stage})
    context = StageContext(task_id="t", run_dir="/tmp/x", options={})

    _image, _artifacts, stage_meta = orchestrator._run_plan(
        "t", "img", _plan("global_restore"), context
    )
    assert seen == [0, 1]                       # ran on 0, retried on 1
    assert stage_meta[0]["device"] == 1
    assert stage_meta[0]["status"] == "completed"


def test_oom_without_an_alternative_device_still_fails(monkeypatch):
    from fiximg.inference.context import StageContext

    seen: list[int] = []
    stage = _Stage(seen, fail_times=1)
    orchestrator = _orchestrator(monkeypatch, GpuTopology.single_device(0),
                                 {"global_restore": stage})
    context = StageContext(task_id="t", run_dir="/tmp/x", options={})

    with pytest.raises(RuntimeError, match="out of memory"):
        orchestrator._run_plan("t", "img", _plan("global_restore"), context)
    assert seen == [0]                          # no second attempt


def test_non_oom_failures_are_not_retried(monkeypatch):
    from fiximg.inference.context import StageContext

    class _Broken(_Stage):
        def run(self, image, context):
            raise RuntimeError("shape mismatch")

    topology = GpuTopology(
        assignments=[
            DeviceAssignment(0, frozenset({"restore"})),
            DeviceAssignment(1, frozenset({"restore"})),
        ],
        default_device=0,
    )
    orchestrator = _orchestrator(monkeypatch, topology, {"global_restore": _Broken([])})
    context = StageContext(task_id="t", run_dir="/tmp/x", options={})

    with pytest.raises(RuntimeError, match="shape mismatch"):
        orchestrator._run_plan("t", "img", _plan("global_restore"), context)


def test_oom_retry_on_a_device_that_also_fails_reports_the_original(monkeypatch):
    from fiximg.inference.context import StageContext

    topology = GpuTopology(
        assignments=[
            DeviceAssignment(0, frozenset({"restore"})),
            DeviceAssignment(1, frozenset({"restore"})),
        ],
        default_device=0,
    )
    stage = _Stage([], fail_times=99)           # always OOM
    orchestrator = _orchestrator(monkeypatch, topology, {"global_restore": stage})
    context = StageContext(task_id="t", run_dir="/tmp/x", options={})

    with pytest.raises(RuntimeError, match="out of memory"):
        orchestrator._run_plan("t", "img", _plan("global_restore"), context)


def test_context_device_is_restored_after_an_oom_retry(monkeypatch):
    from fiximg.inference.context import StageContext

    topology = GpuTopology(
        assignments=[
            DeviceAssignment(0, frozenset({"restore"})),
            DeviceAssignment(1, frozenset({"restore"})),
        ],
        default_device=0,
    )
    stage = _Stage([], fail_times=1)
    orchestrator = _orchestrator(monkeypatch, topology, {"global_restore": stage})
    context = StageContext(task_id="t", run_dir="/tmp/x", options={}, gpu=0)

    orchestrator._run_plan("t", "img", _plan("global_restore"), context)
    assert context.gpu == 0                     # the override was scoped


# --------------------------------------------- task-level aggregation of peaks
def _meta(stage: str, peak: float | None, **extra):
    metrics = dict(extra)
    if peak is not None:
        metrics["gpu_peak_mb"] = peak
    return {"stage": stage, "metrics": metrics}


def test_task_peak_is_the_highest_stage_not_the_last_one():
    """The per-stage peak is a high-water mark, so 'last writer wins' is wrong.

    Two stages that each reached 4 GB made a task that peaked at 4 GB 鈥?under the
    plain merge the task-level number was whichever stage reported last, which
    reads as if the run had been small.
    """
    from fiximg.inference.runtime import _collect_stage_metrics

    merged = _collect_stage_metrics([
        _meta("global_restore", 4096.0),
        _meta("face_enhancement", 512.0),
    ])
    assert merged["gpu_peak_mb"] == 4096.0
    # Order must not matter.
    assert _collect_stage_metrics([
        _meta("face_enhancement", 512.0),
        _meta("global_restore", 4096.0),
    ])["gpu_peak_mb"] == 4096.0


def test_ordinary_metrics_still_take_the_last_reporters_value():
    """Only the maxima set is special: `face_count` is per-stage by meaning."""
    from fiximg.inference.runtime import _collect_stage_metrics

    merged = _collect_stage_metrics([
        _meta("face_detection", None, face_count=3),
        _meta("face_enhancement", None, face_count=2),
    ])
    assert merged["face_count"] == 2


# --------------------------------------- process peak vs the per-stage reset (搂7.1)
class _FakeCuda:
    """PyTorch's counter semantics: a peak that `reset_peak_memory_stats()` rewinds.

    `max_memory_allocated` is "highest value since the last reset", not "highest
    ever". A fake that forgets the reset cannot catch the defect this models: the
    sampler resets on entry so a stage measures only its own work, and every reader
    that then asks the live counter gets the *last window's* peak.
    """

    def __init__(self) -> None:
        self.allocated = 0.0
        self.peak_since_reset = 0.0
        self.resets = 0
        self.broken = False

    def _check(self) -> None:
        if self.broken:
            raise RuntimeError("CUDA driver error: device is gone")

    def is_available(self) -> bool:
        return True

    def init(self) -> None:
        return None

    def device_count(self) -> int:
        return 1
    def current_device(self) -> int:
        return 0

    def reset_peak_memory_stats(self, device=None) -> None:
        self._check()
        self.resets += 1
        self.peak_since_reset = self.allocated

    def max_memory_allocated(self, device=None) -> int:
        self._check()
        return int(self.peak_since_reset * 1024 * 1024)

    def alloc(self, mb: float) -> None:
        self.allocated += mb
        self.peak_since_reset = max(self.peak_since_reset, self.allocated)

    def free(self, mb: float) -> None:
        self.allocated -= mb


@pytest.fixture()
def fake_cuda(monkeypatch):
    from fiximg.inference import gpu_memory as gm

    fake = _FakeCuda()
    monkeypatch.setattr(gm, "_torch", lambda: SimpleNamespace(cuda=fake))
    monkeypatch.setattr(gm, "_process_peak_mb", {}, raising=False)
    return fake


def test_the_process_peak_is_not_rewound_by_the_next_stages_reset(fake_cuda):
    fake_cuda.alloc(300.0)
    with MemorySampler(0) as heavy:
        pass  # entered after the allocation, exits immediately
    assert heavy.peak_mb == 300.0

    fake_cuda.free(300.0)
    fake_cuda.alloc(100.0)
    with MemorySampler(0) as light:
        pass
    assert light.peak_mb == 100.0

    from fiximg.inference import gpu_memory as gm

    # The live counter now reads 100 because `light` rewound it; the aggregate must
    # still remember 300, or the dashboard reports a peak the process never had.
    assert gm.process_peak_mb(0) == 300.0
    assert gm.peak_memory_report() == {"0": 300.0}
    assert gm.process_peak_mb(0) >= max(heavy.peak_mb, light.peak_mb)


def test_work_outside_any_sampler_window_still_counts(fake_cuda):
    """A warm-up that allocates between stages must not fall off the aggregate."""
    from fiximg.inference import gpu_memory as gm

    with MemorySampler(0):
        fake_cuda.alloc(200.0)
    assert gm.process_peak_mb(0) == 200.0

    fake_cuda.alloc(250.0)  # no sampler around it
    assert gm.process_peak_mb(0) == 450.0


def test_a_sampler_window_that_never_raised_keeps_the_previous_peak(fake_cuda):
    from fiximg.inference import gpu_memory as gm

    fake_cuda.alloc(500.0)
    with MemorySampler(0):
        pass
    fake_cuda.free(500.0)
    with MemorySampler(0):
        pass
    assert gm.process_peak_mb(0) == 500.0


def test_a_peak_reached_between_windows_is_not_lost_by_the_next_reset(fake_cuda):
    """The fold has to happen *before* `reset_peak_memory_stats()`, not after.

    A model loaded outside any sampler window raises PyTorch's peak; when it is
    freed again the only copy of that number is the counter the next stage is about
    to rewind. Reading it after the reset reports the new window instead, and the
    process quietly forgets a peak it really had.
    """
    from fiximg.inference import gpu_memory as gm

    fake_cuda.alloc(400.0)   # warm-up / model load, outside every window
    fake_cuda.free(400.0)    # released before the next stage claims the device

    with MemorySampler(0) as window:
        fake_cuda.alloc(50.0)
    assert window.peak_mb == 50.0
    assert gm.process_peak_mb(0) == 400.0


def test_a_window_folds_its_peak_before_the_device_can_become_unreadable(fake_cuda):
    """The fold belongs to the window, not to whoever reads the report later.

    After an out-of-memory the device stops answering `max_memory_allocated`, so a
    lazily-folded peak is simply gone: the stage that OOMed 鈥?the one whose number
    matters most 鈥?would be the one missing from the aggregate.
    """
    from fiximg.inference import gpu_memory as gm

    with MemorySampler(0) as window:
        fake_cuda.alloc(300.0)
    assert window.peak_mb == 300.0

    fake_cuda.broken = True
    assert gm.process_peak_mb(0) == 300.0
    assert gm.peak_memory_report() == {"0": 300.0}


def test_the_stats_peak_is_never_below_a_stage_peak(fake_cuda):
    """`GET /stats` and the stage metrics have to tell the same story.

    They used to ask the same counter at different moments, so the aggregate came out
    *smaller* than a stage figure the same process had just published (measured on a
    real run: stage 334.8 MB, stats 146.3 MB).
    """
    from fiximg.inference.model_manager import ModelManager

    with MemorySampler(0) as heavy:
        fake_cuda.alloc(400.0)
    fake_cuda.free(400.0)
    with MemorySampler(0) as light:
        fake_cuda.alloc(20.0)

    assert heavy.peak_mb == 400.0 and light.peak_mb == 20.0
    stats = ModelManager().gpu_stats()
    assert stats["gpu_memory_peak_mb"] == 400.0
    assert stats["gpu_memory_peak_mb"] >= max(heavy.peak_mb, light.peak_mb)


class _UninitialisedCuda(_FakeCuda):
    """The state a fresh process is really in: peak stats refuse to reset.

    `torch.cuda.reset_peak_memory_stats()` raises "Invalid device argument" until the
    CUDA context exists, which is exactly the first sampler window of a worker 鈥?the one
    that loads the model. This fake reproduces that so the ordering (and the noise when
    it still fails) can be tested on a host with no GPU.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def init(self) -> None:
        self.calls.append("init")

    def reset_peak_memory_stats(self, device=None) -> None:
        self.calls.append("reset")
        if "init" not in self.calls:
            raise RuntimeError("Invalid device argument ")
        super().reset_peak_memory_stats(device)



def test_the_sampler_initialises_cuda_before_resetting_the_peak(fake_cuda, monkeypatch):
    """Order matters: reset before the context exists is the bug, not the exception.

    Measured on a real RTX 5060 in the deployment environment: the first window of a
    process reported no VRAM at all, and the silent fallback made the heaviest stage of
    every run look like it had used none.
    """
    from fiximg.inference import gpu_memory as gm

    cuda = _UninitialisedCuda()
    monkeypatch.setattr(gm, "_torch", lambda: SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(gm, "_process_peak_mb", {}, raising=False)
    gm.reset_lost_measurement_report()

    with MemorySampler(0) as sampler:
        cuda.alloc(120.0)

    assert "init" in cuda.calls and "reset" in cuda.calls, cuda.calls
    assert cuda.calls.index("init") < cuda.calls.index("reset"), (
        f"the peak was reset before the CUDA context existed: {cuda.calls}"
    )
    assert sampler.peak_mb == 120.0, (cuda.calls, sampler.peak_mb)


def test_a_lost_measurement_is_reported_once_instead_of_being_silent(fake_cuda, monkeypatch):
    """A missing number has to be distinguishable from a small one."""
    from fiximg.inference import gpu_memory as gm

    class _Broken(_UninitialisedCuda):
        def init(self) -> None:
            self.calls.append("init")
            raise RuntimeError("no CUDA context")

    cuda = _Broken()
    monkeypatch.setattr(gm, "_torch", lambda: SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(gm, "_process_peak_mb", {}, raising=False)
    gm.reset_lost_measurement_report()

    warnings = []
    import logging

    class _Catch(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                warnings.append(record.getMessage())

    root = logging.getLogger("fiximg.gpu.memory")
    handler = _Catch()
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    try:
        with MemorySampler(0) as first:
            pass
        with MemorySampler(0) as second:
            pass
    finally:
        root.removeHandler(handler)

    assert first.peak_mb is None and second.peak_mb is None
    assert any("measurement unavailable" in w for w in warnings), warnings
    assert len([w for w in warnings if "measurement unavailable" in w]) == 1, warnings
