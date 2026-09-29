"""Multi-GPU routing tests (plan 搂4.3 Step 4).

Two mechanisms are under test:

* **in-process routing** 鈥?``FIXIMG_GPU_ROUTING`` assigns capabilities to
  devices, so different stages of one task can run on different cards;
* **worker pinning + capability-aware claiming** 鈥?a worker specialised for a
  subset of capabilities must not claim a task it cannot complete.
"""
import json

import pytest

from fiximg.domain.tasks import Task
from fiximg.infrastructure.db.repositories import task_repository as repo
from fiximg.infrastructure.queue import SqliteQueueBackend
from fiximg.inference.gpu import DeviceAssignment, GpuTopology, parse_routing


# ------------------------------------------------------------------- parsing
def test_parse_routing_reads_devices_and_capabilities():
    assignments = parse_routing("0:restore,scratch_repair;1:colorize,face_restore")
    assert assignments == [
        DeviceAssignment(device=0, capabilities=frozenset({"restore", "scratch_repair"})),
        DeviceAssignment(device=1, capabilities=frozenset({"colorize", "face_restore"})),
    ]


def test_parse_routing_tolerates_whitespace_and_empty_entries():
    assignments = parse_routing("  0 : restore , denoise ;; 1:colorize ;")
    assert [(a.device, sorted(a.capabilities)) for a in assignments] == [
        (0, ["denoise", "restore"]),
        (1, ["colorize"]),
    ]


def test_parse_routing_skips_malformed_entries():
    """A typo in a deployment variable must not stop the service."""
    assignments = parse_routing("gpu0:restore;1:colorize")
    assert [a.device for a in assignments] == [1]


def test_parse_routing_accepts_a_device_without_capabilities():
    assignments = parse_routing("0:")
    assert assignments == [DeviceAssignment(device=0, capabilities=frozenset())]


def test_parse_routing_of_empty_string():
    assert parse_routing("") == []
    assert parse_routing(None) == []


# ------------------------------------------------------------------ topology
def test_unrouted_topology_uses_the_default_device():
    topology = GpuTopology.single_device(0)
    assert topology.device_for(["restore"]) == 0
    assert topology.device_for(["colorize"]) == 0
    assert topology.device_for([]) == 0
    assert topology.is_routed is False


def test_routing_picks_the_device_that_serves_the_capability():
    topology = GpuTopology(
        assignments=[
            DeviceAssignment(0, frozenset({"restore", "scratch_repair"})),
            DeviceAssignment(1, frozenset({"colorize"})),
        ],
        default_device=0,
    )
    assert topology.device_for(["restore"]) == 0
    assert topology.device_for(["scratch_repair"]) == 0
    assert topology.device_for(["colorize"]) == 1


def test_unlisted_capability_falls_back_to_the_default():
    topology = GpuTopology(
        assignments=[DeviceAssignment(1, frozenset({"colorize"}))],
        default_device=0,
    )
    assert topology.device_for(["face_restore"]) == 0
    assert topology.device_for([]) == 0


def test_first_matching_assignment_wins():
    topology = GpuTopology(
        assignments=[
            DeviceAssignment(0, frozenset({"restore"})),
            DeviceAssignment(1, frozenset({"restore"})),
        ],
        default_device=0,
    )
    assert topology.device_for(["restore"]) == 0


def test_devices_lists_every_routable_device():
    topology = GpuTopology(
        assignments=[DeviceAssignment(1, frozenset({"colorize"})), DeviceAssignment(2, frozenset())],
        default_device=0,
    )
    assert topology.devices() == [0, 1, 2]


def test_capabilities_of_a_device():
    topology = GpuTopology(
        assignments=[
            DeviceAssignment(0, frozenset({"restore"})),
            DeviceAssignment(0, frozenset({"denoise"})),
            DeviceAssignment(1, frozenset({"colorize"})),
        ]
    )
    assert topology.capabilities_of(0) == frozenset({"restore", "denoise"})
    assert topology.capabilities_of(1) == frozenset({"colorize"})
    assert topology.capabilities_of(9) == frozenset()


def test_describe_reports_the_configuration():
    topology = GpuTopology(
        assignments=[DeviceAssignment(1, frozenset({"colorize"}))],
        default_device=0,
        pinned_device=1,
    )
    described = topology.describe()
    assert described["default_device"] == 0
    assert described["pinned_device"] == 1
    assert described["routed"] is True
    assert described["devices"] == [0, 1]
    assert described["assignments"] == [{"device": 1, "capabilities": ["colorize"]}]


# ------------------------------------------------------------------ pinning
def test_pinned_device_wins_over_every_assignment():
    topology = GpuTopology(
        assignments=[DeviceAssignment(0, frozenset({"restore"}))],
        default_device=0,
        pinned_device=2,
    )
    assert topology.device_for(["restore"]) == 2
    assert topology.device_for(["anything"]) == 2


def test_served_capabilities_is_true_when_not_pinned():
    topology = GpuTopology.single_device(0)
    assert topology.served_capabilities(["colorize"]) is True


def test_pinned_worker_serves_only_its_declared_capabilities():
    topology = GpuTopology(
        assignments=[DeviceAssignment(1, frozenset({"colorize", "face_restore"}))],
        default_device=0,
        pinned_device=1,
    )
    assert topology.served_capabilities(["colorize"]) is True
    assert topology.served_capabilities(["colorize", "face_restore"]) is True
    # A restore task needs a capability this worker does not provide.
    assert topology.served_capabilities(["restore"]) is False
    assert topology.served_capabilities(["restore", "colorize"]) is False


def test_pinned_worker_without_declared_capabilities_serves_everything():
    topology = GpuTopology(default_device=0, pinned_device=1)
    assert topology.served_capabilities(["restore"]) is True


def test_served_capabilities_with_no_requirements():
    topology = GpuTopology(
        assignments=[DeviceAssignment(1, frozenset({"colorize"}))], pinned_device=1
    )
    assert topology.served_capabilities([]) is True


# -------------------------------------------------------------------- config
def test_from_settings_reads_the_routing_variable(monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "gpu_routing", "0:restore;1:colorize")
    monkeypatch.setattr(config_mod.settings, "worker_gpu", "")
    monkeypatch.setattr(config_mod.settings, "device", "cpu")

    topology = GpuTopology.from_settings()
    assert topology.device_for(["restore"]) == 0
    assert topology.device_for(["colorize"]) == 1
    assert topology.default_device == -1


def test_from_settings_reads_the_pinned_worker_device(monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "gpu_routing", "")
    monkeypatch.setattr(config_mod.settings, "worker_gpu", "1")
    topology = GpuTopology.from_settings()
    assert topology.pinned_device == 1
    assert topology.device_for(["anything"]) == 1


def test_from_settings_ignores_a_negative_pin(monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "worker_gpu", "-1")
    assert GpuTopology.from_settings().pinned_device is None


def test_from_settings_ignores_an_empty_pin(monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "worker_gpu", "")
    assert GpuTopology.from_settings().pinned_device is None


# ------------------------------------------------- capability-aware claiming
@pytest.fixture()
def queue(isolated_db):
    return SqliteQueueBackend()


def _create(task_id, capabilities=(), **kwargs):
    repo.create_task(task_id, "restore", "alice",
                     required_capabilities=capabilities, **kwargs)


def test_required_capabilities_are_persisted(isolated_db):
    _create("t-1", ["restore", "face_restore"])
    row = repo.get_task("t-1")
    assert json.loads(row["required_capabilities"]) == ["face_restore", "restore"]


def test_task_without_requirements_stores_null(isolated_db):
    _create("t-1")
    assert repo.get_task("t-1")["required_capabilities"] is None


def test_claim_without_a_filter_takes_the_head_of_the_queue(isolated_db, queue):
    _create("t-1", ["restore"])
    _create("t-2", ["colorize"])
    assert queue.claim("w-1").id in ("t-1", "t-2")


def test_specialised_worker_skips_tasks_it_cannot_serve(isolated_db):
    _create("t-color", ["colorize"])
    _create("t-restore", ["restore"])

    color_worker = SqliteQueueBackend(capabilities={"colorize"})
    claimed = color_worker.claim("w-color")
    assert claimed is not None
    assert claimed.id == "t-color"

    restore_worker = SqliteQueueBackend(capabilities={"restore"})
    assert restore_worker.claim("w-restore").id == "t-restore"


def test_specialised_worker_returns_none_when_nothing_matches(isolated_db):
    _create("t-restore", ["restore"])
    color_worker = SqliteQueueBackend(capabilities={"colorize"})
    assert color_worker.claim("w-color") is None


def test_worker_claiming_a_superset_of_capabilities(isolated_db):
    _create("t-face", ["face_restore"])
    worker = SqliteQueueBackend(capabilities={"restore", "face_restore", "colorize"})
    assert worker.claim("w-1").id == "t-face"


def test_worker_with_no_declared_capabilities_claims_anything(isolated_db):
    _create("t-1", ["colorize"])
    worker = SqliteQueueBackend(capabilities=set())
    assert worker.claim("w-1").id == "t-1"


def test_task_with_no_requirements_is_claimable_by_anyone(isolated_db):
    _create("t-any")
    worker = SqliteQueueBackend(capabilities={"colorize"})
    assert worker.claim("w-1").id == "t-any"


def test_claim_filter_respects_priority(isolated_db):
    """Ids are chosen so the tie-break cannot fake the result.

    `ORDER BY priority DESC, created_at, id`: rows inserted in the same clock tick share a
    `created_at` (Windows' clock is millisecond-granular, so three inserts routinely get
    the same instant), which leaves `id` to decide the order. With the original ids
    (`t-low` / `t-high`) the alphabetical tie-break happened to agree with the priority
    order, so this test stayed green after `priority DESC` was deleted from the statement.
    Now the low-priority row sorts *first* by id and by insertion, so only the priority
    ordering can produce the asserted result.
    """
    _create("a-first", ["colorize"], priority=0)
    _create("z-second", ["colorize"], priority=10)
    _create("m-other", ["restore"], priority=99)

    worker = SqliteQueueBackend(capabilities={"colorize"})
    # The high-priority restore task is skipped even though it outranks both.
    assert worker.claim("w-1").id == "z-second"
    assert worker.claim("w-1").id == "a-first"
    assert worker.claim("w-1") is None


def test_claim_filter_still_claims_atomically(isolated_db):
    _create("t-1", ["colorize"])
    first = SqliteQueueBackend(capabilities={"colorize"})
    second = SqliteQueueBackend(capabilities={"colorize"})
    assert first.claim("w-1") is not None
    assert second.claim("w-2") is None


def test_queue_backend_exposes_its_capabilities(isolated_db):
    backend = SqliteQueueBackend(capabilities={"colorize", "face_restore"})
    assert backend.capabilities == frozenset({"colorize", "face_restore"})


def test_task_domain_model_survives_the_new_column(isolated_db):
    _create("t-1", ["restore"])
    task = Task.from_row(repo.claim_next_task("w-1"))
    assert task.id == "t-1"


# ----------------------------------------------------------- planner contract
def test_planner_reports_required_capabilities_per_task_type():
    from fiximg.inference.planner import PipelinePlanner

    planner = PipelinePlanner()
    restore_caps = planner.required_capabilities("restore")
    assert "restore" in restore_caps

    scratch_caps = planner.required_capabilities("restore_scratch")
    assert "scratch_repair" in scratch_caps

    colorize_caps = planner.required_capabilities("colorize")
    assert "colorize" in colorize_caps


def test_planner_reports_no_restriction_for_dynamic_task_types():
    """auto_restore picks its plan from the image, so it cannot be pre-filtered."""
    from fiximg.inference.planner import PipelinePlanner

    assert PipelinePlanner().required_capabilities("auto_restore") == set()


def test_planner_reports_no_restriction_for_an_unknown_task_type():
    from fiximg.inference.planner import PipelinePlanner

    assert PipelinePlanner().required_capabilities("not-a-task") == set()


# ------------------------------------------------------------- end-to-end
def test_specialised_workers_split_the_queue(isolated_db):
    """The report's topology: GPU0 restores, GPU1 colorizes."""
    _create("t-restore-1", ["restore"])
    _create("t-restore-2", ["restore"])
    _create("t-color-1", ["colorize"])

    gpu0 = SqliteQueueBackend(capabilities={"restore"})
    gpu1 = SqliteQueueBackend(capabilities={"colorize"})

    assert {gpu0.claim("w0").id for _ in range(2)} == {"t-restore-1", "t-restore-2"}
    assert gpu0.claim("w0") is None
    assert gpu1.claim("w1").id == "t-color-1"
    assert gpu1.claim("w1") is None

# --------------------------------------------- runtime per-stage device routing
class _DeviceRecordingStage:
    """A stage that records the device it was handed via the context."""

    def __init__(self, name, capabilities, seen):
        self.name = name
        self.version = "test"
        self.capabilities = frozenset(capabilities)
        self._seen = seen

    def run(self, image, context):
        from types import SimpleNamespace

        self._seen.append((self.name, context.gpu))
        return SimpleNamespace(image=image, artifacts={}, metadata={}, message=None)


def _run_two_stage_plan(monkeypatch, topology):
    """Execute a restore -> colorize plan and return [(stage, device), ...]."""
    from types import SimpleNamespace

    from fiximg.inference import runtime as orch
    from fiximg.inference.context import StageContext

    seen: list = []
    stages = {
        "global_restore": _DeviceRecordingStage("global_restore", {"restore"}, seen),
        "colorization": _DeviceRecordingStage("colorization", {"colorize"}, seen),
    }

    monkeypatch.setattr(orch.task_repo, "record_stage", lambda *a, **k: None)
    monkeypatch.setattr(orch.task_repo, "finish_stage", lambda *a, **k: None)
    monkeypatch.setattr(orch.task_repo, "add_event", lambda *a, **k: None)
    monkeypatch.setattr(orch.task_repo, "update_progress", lambda *a, **k: None)

    orchestrator = orch.PipelineOrchestrator(
        planner=SimpleNamespace(build_stage=lambda name, kwargs: stages[name]),
        topology=topology,
    )
    plan = SimpleNamespace(stages=[("global_restore", {}), ("colorization", {})])
    context = StageContext(task_id="t", gpu=topology.default_device)
    orchestrator._run_plan("t", "img", plan, context)
    return seen, context


def test_runtime_routes_each_stage_to_its_device(monkeypatch):
    topology = GpuTopology(
        assignments=[
            DeviceAssignment(0, frozenset({"restore"})),
            DeviceAssignment(1, frozenset({"colorize"})),
        ],
        default_device=0,
    )
    seen, _context = _run_two_stage_plan(monkeypatch, topology)
    assert seen == [("global_restore", 0), ("colorization", 1)]


def test_runtime_restores_the_context_device_after_each_stage(monkeypatch):
    topology = GpuTopology(
        assignments=[DeviceAssignment(1, frozenset({"colorize"}))],
        default_device=0,
    )
    _seen, context = _run_two_stage_plan(monkeypatch, topology)
    # The per-stage override is scoped to the stage, not leaked to the caller.
    assert context.gpu == 0


def test_single_device_topology_keeps_every_stage_together(monkeypatch):
    seen, _context = _run_two_stage_plan(monkeypatch, GpuTopology.single_device(0))
    assert seen == [("global_restore", 0), ("colorization", 0)]


def test_runtime_accepts_an_injected_topology():
    from fiximg.inference.runtime import PipelineOrchestrator

    topology = GpuTopology.single_device(3)
    assert PipelineOrchestrator(topology=topology).topology.default_device == 3


def test_orchestrator_has_a_class_level_topology_fallback():
    """Instances built via __new__ (tests, DI) must still resolve a device."""
    from fiximg.inference.runtime import PipelineOrchestrator

    bare = PipelineOrchestrator.__new__(PipelineOrchestrator)
    assert bare.topology.device_for(["restore"]) == -1
