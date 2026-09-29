"""Stage / model backend registry tests (plan 搂2.2, 搂3.14).

The point of the registry is that adding a model does not require editing the
planner. These tests pin that property: capability lookup, plugin registration
and the "capabilities, not names" rule.
"""
import pytest

from fiximg.domain.errors import ModelUnavailableError, StageNotAvailableError
from fiximg.inference.backends.base import BaseModelBackend, ModelRequest, ModelResult
from fiximg.inference.backends.registry import (
    ModelBackendRegistry,
    backend_registry,
    build_default_backend_registry,
)
from fiximg.inference.context import StageResult
from fiximg.inference.planner import PipelinePlanner
from fiximg.inference.registry import StageRegistry, build_default_registry, stage_registry
from fiximg.inference.stages.base import BaseStage


class _FakeStage(BaseStage):
    name = "fake"
    version = "2.0"
    capabilities = frozenset({"fake_capability"})

    def run(self, image, context):  # pragma: no cover - not executed here
        return StageResult(image=image)


# ------------------------------------------------------------------- registry
def test_default_registry_registers_every_builtin_stage():
    names = stage_registry.names()
    for expected in (
        "global_restore",
        "scratch_repair",
        "scratch_detection",
        "face_detection",
        "face_enhancement",
        "colorization",
    ):
        assert expected in names


def test_registry_rejects_duplicate_registration():
    registry = StageRegistry()
    registry.register("a", _FakeStage)
    with pytest.raises(ValueError):
        registry.register("a", _FakeStage)
    # replace=True is the explicit override path used by plugins.
    registry.register("a", _FakeStage, replace=True)
    assert len(registry) == 1


def test_registry_lookup_by_capability():
    registry = StageRegistry()
    registry.register("sr", _FakeStage, capabilities={"upscale", "restore"})
    registry.register("restore", _FakeStage, capabilities={"restore"})

    assert registry.find_by_capability("upscale") == ["sr"]
    assert registry.find_by_capability("restore") == ["restore", "sr"]
    assert registry.find_by_capability("nope") == []
    assert registry.all_capabilities() == ["restore", "upscale"]


def test_registry_create_unknown_stage_raises_domain_error():
    registry = StageRegistry()
    with pytest.raises(StageNotAvailableError):
        registry.create("missing")
    with pytest.raises(StageNotAvailableError):
        registry.get("missing")


def test_registry_passes_kwargs_to_builder():
    class _Configured(BaseStage):
        name = "configured"

        def __init__(self, with_scratch: bool = False) -> None:
            self.with_scratch = with_scratch

        def run(self, image, context):  # pragma: no cover
            return StageResult(image=image)

    registry = StageRegistry()
    registry.register("configured", _Configured)
    assert registry.create("configured", {"with_scratch": True}).with_scratch is True


def test_build_default_registry_is_independent_of_the_singleton():
    fresh = build_default_registry()
    assert fresh.names() == stage_registry.names()
    assert fresh is not stage_registry


def test_stage_describe_exposes_capabilities():
    assert _FakeStage().describe() == {
        "name": "fake",
        "version": "2.0",
        "capabilities": ["fake_capability"],
        "class": "_FakeStage",
    }


# ------------------------------------------------------------------- planner
def test_planner_builds_stages_through_the_registry():
    """build_stage must not carry its own table any more."""
    planner = PipelinePlanner()
    stage = planner.build_stage("global_restore", {"with_scratch": True})
    assert stage.name == "scratch_repair"  # GlobalRestoreStage renames itself
    assert "scratch_repair" in stage.capabilities

    with pytest.raises(StageNotAvailableError):
        planner.build_stage("does_not_exist", {})


def test_planner_resolves_capability_to_stage():
    planner = PipelinePlanner()
    assert planner.stage_for_capability("colorize") == "colorization"
    assert planner.stage_for_capability("face_restore") == "face_enhancement"
    with pytest.raises(StageNotAvailableError):
        planner.stage_for_capability("teleportation")


def test_planner_prefers_a_registered_plugin_over_the_builtin_table(monkeypatch):
    """A plugin declaring a capability must win over the hard-coded fallback."""
    from fiximg.inference import registry as registry_module

    patched = StageRegistry()
    patched.register("global_restore", _FakeStage, capabilities={"restore"})
    patched.register("super_restore", _FakeStage, capabilities={"restore"})
    monkeypatch.setattr(registry_module, "stage_registry", patched)

    # "global_restore" sorts before "super_restore", so the registry result is
    # deterministic and still capability-driven.
    assert PipelinePlanner().stage_for_capability("restore") == "global_restore"


# ------------------------------------------------------------- backend registry
class _FakeBackend(BaseModelBackend):
    name = "fake"
    version = "1.0"
    capabilities = frozenset({"fake"})

    def _do_infer(self, request: ModelRequest) -> ModelResult:  # pragma: no cover
        return ModelResult(image=request.image)


def test_backend_registry_instantiates_lazily_and_caches():
    registry = ModelBackendRegistry()
    created = []

    def factory():
        created.append(1)
        return _FakeBackend()

    registry.register("fake", factory)
    assert registry.names() == ["fake"]
    assert created == []  # nothing built until first use
    first = registry.get("fake")
    second = registry.get("fake")
    assert first is second
    assert created == [1]


def test_backend_registry_unknown_model_raises_domain_error():
    registry = ModelBackendRegistry()
    with pytest.raises(ModelUnavailableError):
        registry.get("nope")


def test_backend_registry_health_never_raises():
    registry = ModelBackendRegistry()
    registry.register("broken", lambda: (_ for _ in ()).throw(RuntimeError("no weights")))
    report = registry.health_all()
    assert report["broken"]["healthy"] is False
    assert "no weights" in report["broken"]["detail"]


def test_default_backend_registry_covers_the_documented_models():
    registry = build_default_backend_registry()
    # One entry per stage of the legacy chain plus DDColor (plan 搂3.6/搂3.7).
    assert registry.names() == [
        "ddcolor",
        "face_detection",
        "face_enhancement",
        "global_restore",
        "scratch_repair",
        "warp_back",
    ]


def test_every_documented_model_reports_which_implementation_serves_it():
    """Plan 搂3.5.4's "visible fallback" is only visible if every row carries it.

    Measured live before this was fixed: the two face stages reported `null` and
    DDColor reported `"unknown"`, so the API could not tell a caller whether the
    native or the subprocess path would serve them.
    """
    by_name = {e["name"]: e for e in build_default_backend_registry().describe_all()}

    assert set(by_name) == {"ddcolor", "face_detection", "face_enhancement",
                            "global_restore", "scratch_repair", "warp_back"}
    for name, entry in sorted(by_name.items()):
        assert entry["implementation"] in ("native", "legacy-cli"), (
            f"{name} reports implementation {entry['implementation']!r}"
        )


def test_describe_all_reports_the_registry_key_not_the_instance_name():
    """One adapter class serves several models, and the API keys its view by name.

    A backend that reports a different name than the key it is registered under
    used to *delete* that model from `GET /api/v1/models`: the last duplicate won
    the dict, so `face_detection` vanished and its implementation read as `null`.
    """
    registry = ModelBackendRegistry()

    class _SharedAdapter(BaseModelBackend):
        name = "global_restore"  # the first model this class was ever written for
        version = "1.0"
        implementation = "legacy-cli"
        capabilities = frozenset({"restore"})

    registry.register("face_detection", _SharedAdapter)
    registry.register("face_enhancement", _SharedAdapter)

    by_name = {e["name"]: e for e in registry.describe_all()}
    assert set(by_name) == {"face_detection", "face_enhancement"}
    assert all(e["implementation"] == "legacy-cli" for e in by_name.values())


def test_default_registry_registers_the_split_face_chain():
    """Each legacy path is its own stage, so nothing is processed twice."""
    names = stage_registry.names()
    for expected in ("face_detection", "face_enhancement", "warp_back"):
        assert expected in names

    warp_back = stage_registry.create("warp_back")
    assert "warp_back" in warp_back.capabilities
    # The face stages must not advertise the restoration capability: that is
    # what keeps the planner from running the same work in two places.
    assert "restore" not in stage_registry.create("face_detection").capabilities
    assert "restore" not in stage_registry.create("face_enhancement").capabilities


# --------------------------------------------------- folder-bridge conformance
def _folder_driving_stage_names() -> set[str]:
    """Which stages hand their backend a folder 鈥?read from the source, not a list.

    A hand-maintained list would go stale the moment a stage started or stopped
    using the bridge, which is the same "declaration drifts from what runs" failure
    the protocol itself is meant to prevent.
    """
    import pathlib
    import re

    stage_dir = (pathlib.Path(__file__).resolve().parents[2]
                 / "src" / "fiximg" / "inference" / "stages")
    names: set[str] = set()
    for module in sorted(stage_dir.glob("*.py")):
        source = module.read_text(encoding="utf-8", errors="replace")
        if ".run_folder(" not in source:
            continue
        names.update(re.findall(r'^\s{4}name\s*=\s*"([^"]+)"', source, re.MULTILINE))
    return names


def test_the_folder_driving_stages_are_discovered_not_assumed():
    """Guards the scanner itself: it must find the four known bridge stages."""
    found = _folder_driving_stage_names()
    assert {"global_restore", "face_detection", "face_enhancement", "warp_back"} <= found, found
    # Colourization runs a resident pipeline, not a folder bridge.
    assert "colorization" not in found


@pytest.mark.parametrize("stage_name", sorted(_folder_driving_stage_names()))
def test_a_folder_driving_stage_gets_a_folder_bridge_backend(stage_name):
    """Plan 搂3.5.1's "one interface" has two halves, and stages use both.

    `ModelBackend` covers the single-image call; the split legacy chain (plan
    搂3.6/搂3.7) hands a backend an input directory and reads the shared root back.
    A backend registered for such a stage without that half used to fail inside the
    stage at request time with an AttributeError 鈥?nothing bound the registration to
    the contract the caller assumes.
    """
    from fiximg.inference.backends.base import FolderBridgeBackend, ModelBackend

    backend = backend_registry.get(stage_name)

    assert isinstance(backend, ModelBackend), stage_name
    assert isinstance(backend, FolderBridgeBackend), (
        f"{stage_name} is folder-driven but its backend has no run_folder/produced_dir"
    )
    assert callable(backend.run_folder) and callable(backend.produced_dir)
    # 搂3.5.4: whichever implementation is selected must say which it is.
    assert backend.implementation in ("native", "legacy-cli"), stage_name


def test_ddcolor_is_not_claimed_to_be_a_folder_bridge():
    """The protocol is optional on purpose; asserting the negative keeps it honest."""
    from fiximg.inference.backends.base import FolderBridgeBackend, ModelBackend

    ddcolor = backend_registry.get("ddcolor")
    assert isinstance(ddcolor, ModelBackend)
    assert not isinstance(ddcolor, FolderBridgeBackend), (
        "if DDColor ever grows a run_folder, this test and the stage scanner agree to update"
    )
