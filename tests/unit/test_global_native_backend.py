"""Native Global backend selection and guards (plan 搂3.5.1, 搂3.5.4).

These run without weights and without loading a model: what is under test is the
*decision* (native when possible, subprocess adapter otherwise) and the guards
that turn a misconfiguration into an error rather than into a plausible-looking
image.

Numeric equivalence of the native path is proven separately in
``tests/gpu/test_global_equivalence.py``; this file deliberately does not assert
what it cannot check.
"""
import os
import sys
import types

import pytest

from fiximg.domain.errors import ModelUnavailableError
from fiximg.inference.backends import global_restore as gr
from fiximg.inference.backends import registry as registry_module
from fiximg.inference.backends.base import ModelRequest
from fiximg.inference.backends.global_restore import GlobalRestoreBackend
from fiximg.inference.backends.legacy_cli import LegacyCliBackend


class _Opt:
    """Just the fields the guards read."""

    def __init__(self, **overrides):
        self.Quality_restore = True
        self.Scratch_and_Quality_restore = False
        self.checkpoints_dir = ""
        self.name = "mapping_quality"
        self.load_pretrainA = ""
        self.load_pretrainB = ""
        self.gpu_ids = []
        self.test_mode = "Full"
        for key, value in overrides.items():
            setattr(self, key, value)


@pytest.fixture()
def fake_native(monkeypatch):
    """Substitute the environment, not the decision.

    Only the model class (needs torch + weights) and the filesystem/interpreter
    probes are replaced; ``native_available()``, the ownership rule and the
    selection logic run as written, so these tests would notice a change to any of
    them 鈥?a stubbed availability check made the "switch off" case vacuous once
    already.
    """
    made: list = []

    import fiximg.config as config_mod
    from fiximg.inference.backends import legacy_tree

    monkeypatch.setattr(config_mod.settings, "native_tree", "global", raising=False)
    monkeypatch.setattr(legacy_tree, "foreign_tree_loaded", lambda tree: None)
    monkeypatch.setattr(gr, "quality_weights_present", lambda: True)

    class _Fake:
        implementation = "native"

        def __init__(self, config=None):
            made.append(config)

    monkeypatch.setattr(registry_module, "_GLOBAL_BACKEND_CLS", _Fake)
    monkeypatch.setattr(registry_module, "_native_global", None)
    return made


# ============================ availability guards ==========================
def test_native_is_available_when_weights_and_torch_are_there(monkeypatch):
    monkeypatch.setattr(gr, "quality_weights_present", lambda: True)
    monkeypatch.setattr(gr, "legacy_tree_conflict", lambda: None)
    assert gr.native_available() == (True, "")


def test_missing_weights_are_reported_as_a_reason_not_an_exception(monkeypatch):
    monkeypatch.setattr(gr, "quality_weights_present", lambda: False)
    monkeypatch.setattr(gr, "legacy_tree_conflict", lambda: None)
    available, reason = gr.native_available()
    assert available is False
    assert "weights" in reason


def test_a_tree_without_torch_falls_back(monkeypatch):
    """The reason string is what makes a fallback debuggable later."""
    monkeypatch.setattr(gr, "quality_weights_present", lambda: True)
    monkeypatch.setattr(gr, "legacy_tree_conflict", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", None)
    available, reason = gr.native_available()
    assert available is False
    assert "torch" in reason


def test_a_conflicting_legacy_tree_is_detected(monkeypatch):
    """Global/ and Face_Enhancement/ both own ``options``/``models``/``util``/``data``.

    The first tree imported owns those names for the process, so importing the
    second would silently mix two code bases 鈥?the detector has to see the
    intruder even when it is already in sys.modules.
    """
    from fiximg.inference.backends import legacy_tree

    intruder = types.ModuleType("models")
    intruder.__file__ = os.path.join(legacy_tree.FACE_ROOT, "models", "__init__.py")
    monkeypatch.setitem(sys.modules, "models", intruder)
    assert gr.legacy_tree_conflict() == "models"


def test_an_unrelated_third_party_package_is_not_a_conflict(monkeypatch):
    """Only the *other legacy tree* counts, or any dependency would disable native."""
    intruder = types.ModuleType("models")
    intruder.__file__ = os.path.join(os.sep, "site-packages", "models", "__init__.py")
    monkeypatch.setitem(sys.modules, "models", intruder)
    assert gr.legacy_tree_conflict() is None


def test_a_module_from_global_is_not_a_conflict(monkeypatch):
    owned = types.ModuleType("models")
    owned.__file__ = os.path.join(gr.GLOBAL_DIR, "models", "__init__.py")
    monkeypatch.setitem(sys.modules, "models", owned)
    assert gr.legacy_tree_conflict() is None


# ================================ selection ================================
def test_the_quality_path_picks_the_native_backend(fake_native, monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "native_tree", "global", raising=False)
    monkeypatch.setattr(gr, "native_available", lambda: (True, ""))
    backend = registry_module.select_restore_backend(with_scratch=False)
    assert backend.implementation == "native"
    assert len(fake_native) == 1


def test_scratch_uses_the_adapter_without_its_own_weights(fake_native, monkeypatch):
    """Scratch needs one more network than quality (the detector + VAE_B_scratch).

    A checkout that only has the quality weights must keep running the subprocess
    version for scratched photos rather than half-loading the branch.
    """
    monkeypatch.setattr(gr, "scratch_native_available",
                        lambda: (False, "scratch branch weights are missing"))
    backend = registry_module.select_restore_backend(with_scratch=True)
    assert isinstance(backend, LegacyCliBackend)
    assert backend.implementation == "legacy-cli"
    assert backend.with_scratch is True
    assert fake_native == [], "the scratch path must not build the quality backend"


def test_a_disabled_switch_falls_back(fake_native, monkeypatch):
    """No native_available stub here: the ownership rule is what is under test."""
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "native_tree", "none", raising=False)
    assert registry_module.select_restore_backend().implementation == "legacy-cli"
    assert fake_native == [], "an unwanted native backend must not even be constructed"


def test_owning_the_other_tree_disables_global_native(fake_native, monkeypatch):
    """One process, one native tree 鈥?an exclusivity rule, not a preference.

    Face enhancement needs Face_Enhancement/, which declares the same top-level
    packages as Global/, so a face worker necessarily runs restoration through
    the subprocess adapter.
    """
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "native_tree", "face", raising=False)
    assert registry_module.select_restore_backend().implementation == "legacy-cli"
    assert fake_native == []


def test_an_unavailable_environment_falls_back_with_a_reason(fake_native, monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "native_tree", "global", raising=False)
    monkeypatch.setattr(gr, "native_available",
                        lambda: (False, "quality weights are missing under Global/checkpoints/restoration"))
    assert registry_module.select_restore_backend().implementation == "legacy-cli"
    assert fake_native == []


def test_the_native_backend_is_reused_across_requests(fake_native, monkeypatch):
    """Resident weights are the whole point: one instance per process."""
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "native_tree", "global", raising=False)
    monkeypatch.setattr(gr, "native_available", lambda: (True, ""))
    first = registry_module.select_restore_backend()
    second = registry_module.select_restore_backend()
    assert first is second
    assert len(fake_native) == 1


def test_the_backend_registry_and_the_stage_agree(fake_native, monkeypatch):
    """GET /api/v1/models must describe the implementation that actually serves."""
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "native_tree", "global", raising=False)
    monkeypatch.setattr(gr, "native_available", lambda: (True, ""))
    monkeypatch.setattr(gr, "scratch_native_available", lambda: (True, ""))
    registry = registry_module.build_default_backend_registry()
    assert registry.get("global_restore").implementation == "native"
    # Both branches live in the Global tree, so one worker can host both.
    assert registry.get("scratch_repair").implementation == "native"


def test_the_registry_falls_back_in_lockstep(fake_native, monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "native_tree", "global", raising=False)
    monkeypatch.setattr(gr, "native_available", lambda: (False, "no torch"))
    monkeypatch.setattr(gr, "scratch_native_available", lambda: (False, "no torch"))
    registry = registry_module.build_default_backend_registry()
    assert registry.get("global_restore").implementation == "legacy-cli"


# ================================ opt building =============================
def test_opt_uses_absolute_weight_paths():
    """The vendored default is ``./checkpoints`` relative to cwd.

    Only the subprocess version is guaranteed to start inside Global/; a library
    call inheriting the relative path loads *nothing*, and the vendored loader
    reports that by printing a line and continuing with random weights.
    """
    opt = GlobalRestoreBackend()._build_opt(-1)

    assert os.path.isabs(opt.checkpoints_dir)
    assert os.path.isabs(opt.load_pretrainA) and os.path.isabs(opt.load_pretrainB)
    assert opt.gpu_ids == [], "CPU means an empty id list 鈥?.cuda() is gated on it"
    assert opt.test_mode == "Full" and opt.Quality_restore is True


def test_opt_reuses_the_vendored_architecture_switches():
    """parameter_set() is called, not re-derived: upstream changes cannot drift."""
    opt = GlobalRestoreBackend()._build_opt(-1)
    assert opt.mapping_n_block == 6 and opt.map_mc == 512 and opt.mc == 64
    assert opt.n_downsample_global == 3 and opt.no_instance is True
    assert opt.serial_batches is True and opt.no_flip is True


def test_opt_for_gpu_keeps_the_index():
    opt = GlobalRestoreBackend()._build_opt(1)
    assert opt.gpu_ids == [1]


def test_the_scratch_branch_is_refused():
    backend = GlobalRestoreBackend()
    with pytest.raises(ModelUnavailableError, match="quality-restoration branch"):
        backend._validate_branch(_Opt(Scratch_and_Quality_restore=True))


def test_a_missing_quality_flag_is_refused():
    backend = GlobalRestoreBackend()
    with pytest.raises(ModelUnavailableError):
        backend._validate_branch(_Opt(Quality_restore=False))


# ======================= refuse to run on random weights ===================
def _tmp_opt(tmp_path):
    (tmp_path / "VAE_A_quality").mkdir(exist_ok=True)
    (tmp_path / "VAE_B_quality").mkdir(exist_ok=True)
    (tmp_path / "mapping_quality").mkdir(exist_ok=True)
    return _Opt(
        checkpoints_dir=str(tmp_path),
        name="mapping_quality",
        load_pretrainA=str(tmp_path / "VAE_A_quality"),
        load_pretrainB=str(tmp_path / "VAE_B_quality"),
    )


def test_loading_refuses_when_a_weight_is_missing(tmp_path):
    opt = _tmp_opt(tmp_path)
    (tmp_path / "VAE_A_quality" / "latest_net_G.pth").write_bytes(b"x")
    # VAE B and the mapping net are deliberately absent.

    with pytest.raises(ModelUnavailableError) as excinfo:
        GlobalRestoreBackend._require_weights(opt)
    missing = excinfo.value.details["missing"]
    assert len(missing) == 2
    assert any("latest_net_mapping_net.pth" in p for p in missing)
    assert "randomly initialised" in str(excinfo.value)


def test_all_three_weights_present_passes_the_guard(tmp_path):
    opt = _tmp_opt(tmp_path)
    (tmp_path / "VAE_A_quality" / "latest_net_G.pth").write_bytes(b"x")
    (tmp_path / "VAE_B_quality" / "latest_net_G.pth").write_bytes(b"x")
    (tmp_path / "mapping_quality" / "latest_net_mapping_net.pth").write_bytes(b"x")
    GlobalRestoreBackend._require_weights(opt)


# ============================== device mapping =============================
@pytest.mark.parametrize("hint,expected", [
    ("cpu", -1),
    (-1, -1),
    ("none", -1),
    ("cuda", 0),
    ("cuda:1", 1),
    ("0", 0),
    ("2", 2),
])
def test_device_hints_map_to_the_legacy_gpu_ids_convention(hint, expected):
    assert gr._device_index(hint) == expected


def test_auto_follows_torch_availability(monkeypatch):
    class _FakeCuda:
        @staticmethod
        def is_available():
            return _FakeCuda.answer

    class _FakeTorch:
        cuda = _FakeCuda()

    _FakeCuda.answer = False
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    assert gr._device_index("auto") == -1

    _FakeCuda.answer = True
    assert gr._device_index("auto") == 0
    assert gr._device_index("") == 0


# ================================= lifecycle ===============================
def test_unload_releases_the_networks():
    backend = GlobalRestoreBackend()
    backend._model = object()
    backend._opt = object()
    backend._loaded = True
    backend.unload()
    assert backend._model is None and backend._opt is None
    assert backend.is_loaded is False


def test_infer_without_a_loaded_model_is_a_domain_error(monkeypatch):
    from PIL import Image

    backend = GlobalRestoreBackend()
    with pytest.raises(ModelUnavailableError):
        backend._do_infer(ModelRequest(image=Image.new("RGB", (8, 8))))


def test_native_reports_which_implementation_it_is(monkeypatch):
    backend = GlobalRestoreBackend()
    monkeypatch.setattr(gr, "native_available", lambda: (False, "quality weights are missing"))
    info = backend.describe()
    assert info["implementation"] == "native"
    assert info["weights"] and all(p.endswith(".pth") for p in info["weights"])
    health = backend.health()
    assert health.extra["available"] is False
    assert "weights" in health.extra["reason"]


# ====================== what the models API reports =======================
class _Versions:
    def current_version(self, name):
        return None

    def previous_version(self, name):
        return None

    def residents(self, name):
        return []


class _Manager:
    registry = _Versions()

    def loaded_models(self):
        return []


class _Registry:
    def __init__(self, implementation):
        self._implementation = implementation

    def describe_all(self):
        return [{
            "name": "global_restore",
            "version": "1.0",
            "capabilities": ["restore"],
            "loaded": False,
            "device": "cpu",
            "implementation": self._implementation,
        }]


def test_the_models_api_reports_the_serving_implementation():
    """`framework: legacy-cli` in the manifest cannot say where weights live.

    The declared framework and the actual implementation differ as soon as the
    native backend serves the chain, so the API must expose both 鈥?otherwise an
    operator checking GPU residency reads the wrong thing.
    """
    from fiximg.application.model_service import ModelService
    from fiximg.inference.manifest import ModelManifest

    manifest = ModelManifest(models={})
    for implementation in ("native", "legacy-cli"):
        service = ModelService(
            registry=_Registry(implementation), manager=_Manager(),
            manifest_provider=lambda: manifest,
        )
        entry = service.list_models()[0]
        assert entry["name"] == "global_restore"
        assert entry["implementation"] == implementation


def test_the_models_api_reports_which_optimisations_landed():
    """"Declared" and "in force" are different facts, and operators need the second.

    A manifest key that no backend applied used to be invisible here, so a
    deployment could declare `channels_last: true` for a chain that ignored it and
    read the models API as confirmation (plan 搂3.5.4).
    """
    from fiximg.application.model_service import ModelService
    from fiximg.inference.manifest import ModelManifest

    record = {
        "inference_mode": True,
        "precision": "fp16",
        "channels_last": True,
        "compile": False,
        "compile_mode": None,
        "accelerated": True,
        "applied": {"channels_last": False},
    }

    class _OptimisedRegistry(_Registry):
        def describe_all(self):
            entry = _Registry.describe_all(self)[0]
            entry["optimisations"] = record
            return [entry]

    service = ModelService(registry=_OptimisedRegistry("native"), manager=_Manager(),
                           manifest_provider=lambda: ModelManifest(models={}))
    entry = service.list_models()[0]
    assert entry["optimisations"]["precision"] == "fp16"
    assert entry["optimisations"]["applied"] == {"channels_last": False}, (
        "the declined layout must be named, not omitted"
    )


def test_model_health_exposes_why_native_was_skipped(monkeypatch):
    """The reason string has to reach the operator, not only the log."""
    backend = GlobalRestoreBackend()
    monkeypatch.setattr(gr, "native_available",
                        lambda: (False, "'models' is already imported from another legacy tree"))
    extra = backend.health().extra
    assert extra["available"] is False
    assert "legacy tree" in extra["reason"]


# ========================= the scratch branch =============================
# Weights-free: the detector and the triplet are never built here. What is under
# test is which branch an opt describes, what the backend refuses, and that the
# selection reaches it at all.
def _scratch_opt():
    return _Opt(Scratch_and_Quality_restore=True, Quality_restore=False)


def test_scratch_availability_needs_one_more_network(monkeypatch, tmp_path):
    from fiximg.inference.backends import global_restore as gr_module

    monkeypatch.setattr(gr_module, "native_available", lambda: (True, ""))
    monkeypatch.setattr(gr_module, "scratch_weights_present", lambda: False)
    available, reason = gr_module.scratch_native_available()
    assert available is False
    assert "scratch" in reason.lower()

    monkeypatch.setattr(gr_module, "scratch_weights_present", lambda: True)
    assert gr_module.scratch_native_available() == (True, "")


def test_quality_weights_do_not_imply_scratch_weights(monkeypatch, tmp_path):
    """The detector is a separate file; a partial install must not select scratch."""
    from fiximg.inference.backends import global_restore as gr_module

    monkeypatch.setattr(gr_module, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(gr_module, "DETECTION_CHECKPOINT", str(tmp_path / "absent.pt"))
    (tmp_path / "VAE_A_quality").mkdir()
    (tmp_path / "VAE_B_quality").mkdir()
    (tmp_path / "mapping_quality").mkdir()
    for name in ("VAE_A_quality", "VAE_B_quality"):
        (tmp_path / name / "latest_net_G.pth").write_bytes(b"x")
    (tmp_path / "mapping_quality" / "latest_net_mapping_net.pth").write_bytes(b"x")

    assert gr_module.quality_weights_present() is True
    assert gr_module.scratch_weights_present() is False


def test_scratch_selection_builds_the_native_scratch_backend(monkeypatch):
    import fiximg.config as config_mod
    from fiximg.inference.backends import global_restore as gr_module

    monkeypatch.setattr(config_mod.settings, "native_tree", "global", raising=False)
    monkeypatch.setattr(gr_module, "scratch_native_available", lambda: (True, ""))

    made = []

    class _Fake:
        implementation = "native"

        def __init__(self, config=None):
            made.append(config)

    monkeypatch.setattr(registry_module, "_SCRATCH_BACKEND_CLS", _Fake)
    monkeypatch.setattr(registry_module, "_native_scratch", None)

    first = registry_module.select_restore_backend(with_scratch=True)
    assert first.implementation == "native"
    assert registry_module.select_restore_backend(with_scratch=True) is first
    assert len(made) == 1


def test_scratch_falls_back_when_its_weights_are_absent(monkeypatch):
    from fiximg.inference.backends import global_restore as gr_module

    monkeypatch.setattr(gr_module, "scratch_native_available",
                        lambda: (False, "scratch branch weights are missing"))
    backend = registry_module.select_restore_backend(with_scratch=True)
    assert backend.implementation == "legacy-cli"
    assert backend.with_scratch is True


def test_the_registry_entry_for_scratch_reports_the_live_implementation(monkeypatch):
    import fiximg.config as config_mod
    from fiximg.inference.backends import global_restore as gr_module

    monkeypatch.setattr(config_mod.settings, "native_tree", "global", raising=False)
    monkeypatch.setattr(gr_module, "native_available", lambda: (True, ""))
    monkeypatch.setattr(gr_module, "scratch_native_available", lambda: (True, ""))
    registry = registry_module.build_default_backend_registry()
    assert registry.get("scratch_repair").implementation == "native"


def test_scratch_backend_identity():
    backend = gr.NativeScratchRepairBackend()
    assert backend.name == "scratch_repair"
    assert backend.implementation == "native"
    assert "scratch_repair" in backend.capabilities
    assert "restore" in backend.capabilities


def test_the_quality_branch_is_refused_by_the_scratch_backend():
    backend = gr.NativeScratchRepairBackend()
    with pytest.raises(ModelUnavailableError, match="scratch-and-quality branch"):
        backend._validate_branch(_Opt())


def test_the_scratch_branch_is_refused_by_the_quality_backend():
    backend = gr.GlobalRestoreBackend()
    with pytest.raises(ModelUnavailableError, match="quality-restoration branch"):
        backend._validate_branch(_scratch_opt())


def test_hr_scratch_refuses_before_loading_anything(tmp_path):
    from fiximg.domain.errors import ModelUnavailableError as MUE

    backend = gr.NativeScratchRepairBackend()
    with pytest.raises(MUE, match="mapping_Patch_Attention"):
        backend.run_folder(str(tmp_path), str(tmp_path), hr=True)


def test_run_folder_needs_input_images(tmp_path, monkeypatch):
    from fiximg.domain.errors import PipelineFailedError

    backend = gr.NativeScratchRepairBackend()
    monkeypatch.setattr(backend, "load", lambda device=None: None)
    monkeypatch.setattr(backend, "_model", object())
    with pytest.raises(PipelineFailedError, match="no input images"):
        backend.run_folder(str(tmp_path), str(tmp_path / "pipe"))


def test_restoring_without_a_model_is_refused(tmp_path):
    backend = gr.NativeScratchRepairBackend()
    with pytest.raises(ModelUnavailableError, match="not loaded"):
        backend._restore_from_files(str(tmp_path / "a.png"), str(tmp_path / "m.png"))


def test_detecting_without_the_detector_is_refused(monkeypatch):
    import numpy as np
    from PIL import Image

    backend = gr.NativeScratchRepairBackend()
    monkeypatch.setattr(backend, "_opt", _Opt())
    with pytest.raises(ModelUnavailableError, match="not loaded"):
        backend._detect_mask(Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)))


def test_scratch_health_names_the_detector(monkeypatch):
    from fiximg.inference.backends import global_restore as gr_module

    monkeypatch.setattr(gr_module, "scratch_native_available", lambda: (True, ""))
    backend = gr.NativeScratchRepairBackend()
    extra = backend.health().extra
    assert extra["implementation"] == "native"
    assert extra["detector"] == gr_module.DETECTION_CHECKPOINT
