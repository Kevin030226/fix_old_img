"""Native-tree ownership and the stage-3 selector (plan 搂2.9, 搂3.13, 搂4.2 Step 2).

Two things are pinned here that are easy to get wrong silently:

* ownership is *exclusive* 鈥?declaring ``face`` must make Global restoration fall
  back, not attempt to load a second tree into the same interpreter;
* the stage-3 selector must not construct the native class unless the tree is owned
  and the checkpoint is really there.

``_build_opt`` is stubbed rather than called: it imports the ``Face_Enhancement``
tree's ``options`` package, and in a test process that may already have imported
``Global``'s same-named package, resolution would depend on test order. That
dependency is the whole reason ownership exists, so faking it here is honest 鈥?the real import path is exercised by ``tests/gpu/test_face_enhance_equivalence.py``
in a child process that loads only one tree.
"""
import os
import sys
import types

import pytest

from fiximg.domain.errors import ModelUnavailableError
from fiximg.inference.backends import face_enhance_native as fen
from fiximg.inference.backends import legacy_tree as lt
from fiximg.inference.backends import registry as registry_module
from fiximg.inference.backends.face_enhance_native import (
    NativeFaceEnhancementBackend,
)
from fiximg.inference.backends.legacy_cli import LegacyCliBackend, each_img_dir


@pytest.fixture()
def clean_ownership(monkeypatch):
    """Deterministic ownership: no tree imported yet, setting controlled."""
    import fiximg.config as config_mod

    for name in lt.SHARED_PACKAGES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(config_mod.settings, "native_tree", "global", raising=False)
    return config_mod


# ================================ ownership ================================
def test_default_is_the_verified_tree(clean_ownership):
    assert lt.declared_tree() == "global"
    assert lt.owns("global") is True
    assert lt.owns("face") is False, "the two trees cannot both be resident"


def test_none_disables_both(clean_ownership, monkeypatch):
    monkeypatch.setattr(clean_ownership.settings, "native_tree", "none", raising=False)
    assert lt.owns("global") is False
    assert lt.owns("face") is False


def test_face_owns_and_global_yields(clean_ownership, monkeypatch):
    monkeypatch.setattr(clean_ownership.settings, "native_tree", "face", raising=False)
    assert lt.owns("face") is True
    assert lt.owns("global") is False


def test_an_unknown_value_still_falls_back_cleanly(clean_ownership, monkeypatch):
    """Config validates, but a stray object must not crash inference selection."""
    monkeypatch.setattr(clean_ownership.settings, "native_tree", "sideways", raising=False)
    assert lt.declared_tree() == "global"


def test_declared_tree_survives_unreadable_settings(monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod, "settings", None, raising=False)
    assert lt.declared_tree() == "global"


def test_the_loaded_face_tree_blocks_global_native(clean_ownership, monkeypatch):
    """Uses the clean-ownership fixture: other suites import the Global tree for
    their own opt-building, and a leftover ``options`` would change the answer."""
    face_models = types.ModuleType("models")
    face_models.__file__ = os.path.join(lt.FACE_ROOT, "models", "__init__.py")
    monkeypatch.setitem(sys.modules, "models", face_models)

    assert lt.imported_from("face") == "models"
    assert lt.foreign_tree_loaded("global") == "models"
    assert lt.foreign_tree_loaded("face") is None


def test_nothing_imported_means_no_conflict(monkeypatch):
    for name in lt.SHARED_PACKAGES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert lt.foreign_tree_loaded("global") is None
    assert lt.foreign_tree_loaded("face") is None


def test_third_party_names_are_not_a_legacy_conflict(monkeypatch):
    intruder = types.ModuleType("models")
    intruder.__file__ = os.path.join(os.sep, "site-packages", "models", "__init__.py")
    monkeypatch.setitem(sys.modules, "models", intruder)
    assert lt.foreign_tree_loaded("global") is None


def test_ownership_report_is_what_an_operator_reads(clean_ownership, monkeypatch):
    monkeypatch.setattr(clean_ownership.settings, "native_tree", "face", raising=False)
    report = lt.ownership_report()
    assert report["declared"] == "face"
    assert report["face_native"] is True
    assert report["global_native"] is False
    assert report["conflict"] is None


def test_the_setting_is_validated_at_startup(monkeypatch):
    """A typo would silently mean 'nothing resident' (plan 搂3.3 fail-fast)."""
    from fiximg.config import Settings

    monkeypatch.setenv("FIXIMG_NATIVE_TREE", "Global")  # wrong case is wrong value
    with pytest.raises(ValueError, match="FIXIMG_NATIVE_TREE"):
        Settings()


def test_the_three_accepted_values_load(monkeypatch):
    from fiximg.config import Settings

    for value in ("global", "face", "none"):
        monkeypatch.setenv("FIXIMG_NATIVE_TREE", value)
        assert Settings().native_tree == value


# ======================= the stage-3 backend selector ======================
@pytest.fixture()
def stage3_env(monkeypatch, clean_ownership, tmp_path):
    """Native stage 3 reachable without touching torch or the weights."""
    made: list = []

    monkeypatch.setattr(fen, "checkpoint_present", lambda: True)

    class _Fake:
        implementation = "native"

        def __init__(self, config=None):
            made.append(config)

    monkeypatch.setattr(registry_module, "_FACE_ENHANCE_BACKEND_CLS", _Fake)
    monkeypatch.setattr(registry_module, "_native_face_enhance", None)
    return clean_ownership, made


def test_stage_three_is_on_the_adapter_by_default(stage3_env):
    config_mod, made = stage3_env
    backend = registry_module.select_face_enhancement_backend()
    assert isinstance(backend, LegacyCliBackend)
    assert backend.implementation == "legacy-cli"
    assert made == [], "a global-owning worker must not build the face model"


def test_stage_three_goes_native_on_a_face_worker(stage3_env):
    config_mod, made = stage3_env
    config_mod.settings.native_tree = "face"
    backend = registry_module.select_face_enhancement_backend()
    assert backend.implementation == "native"
    assert len(made) == 1


def test_stage_three_hr_always_uses_the_adapter(stage3_env):
    """HR selects FaceSR_512, which this install does not ship."""
    config_mod, _made = stage3_env
    config_mod.settings.native_tree = "face"
    backend = registry_module.select_face_enhancement_backend(hr=True)
    assert isinstance(backend, LegacyCliBackend)


def test_a_missing_checkpoint_keeps_the_adapter(stage3_env, monkeypatch):
    config_mod, made = stage3_env
    config_mod.settings.native_tree = "face"
    monkeypatch.setattr(fen, "checkpoint_present", lambda: False)
    assert registry_module.select_face_enhancement_backend().implementation == "legacy-cli"
    assert made == []


def test_the_registry_reports_the_serving_implementation(stage3_env):
    config_mod, _made = stage3_env
    config_mod.settings.native_tree = "face"
    registry = registry_module.build_default_backend_registry()
    assert registry.get("face_enhancement").implementation == "native"

    config_mod.settings.native_tree = "global"
    registry_module._native_face_enhance = None
    registry = registry_module.build_default_backend_registry()
    assert registry.get("face_enhancement").implementation == "legacy-cli"


# ============================== the backend ================================
class _Opt:
    def __init__(self, checkpoints_dir, name):
        self.checkpoints_dir = checkpoints_dir
        self.name = name


def test_loading_refuses_a_missing_generator(tmp_path):
    opt = _Opt(str(tmp_path), "Setting_9_epoch_100")
    with pytest.raises(ModelUnavailableError) as excinfo:
        NativeFaceEnhancementBackend._require_weights(opt)
    assert "randomly initialised" in str(excinfo.value)
    assert excinfo.value.details["expected"].endswith("latest_net_G.pth")


def test_the_present_generator_passes(tmp_path):
    directory = tmp_path / "Setting_9_epoch_100"
    directory.mkdir()
    (directory / "latest_net_G.pth").write_bytes(b"x")
    NativeFaceEnhancementBackend._require_weights(_Opt(str(tmp_path), "Setting_9_epoch_100"))


def test_enhancing_without_a_model_is_refused(tmp_path):
    backend = NativeFaceEnhancementBackend()
    backend._model = None
    with pytest.raises(ModelUnavailableError, match="not loaded"):
        backend._enhance(str(tmp_path), str(tmp_path / "out"), "cpu")


def test_hr_is_refused_at_run_time_not_silently_approximated(tmp_path):
    backend = NativeFaceEnhancementBackend()
    with pytest.raises(ModelUnavailableError, match="FaceSR_512"):
        backend.run_folder(str(tmp_path), str(tmp_path), hr=True)


def test_no_crops_is_a_skip_not_a_failure(tmp_path):
    """batch.py returns early when nothing was detected; the bridge must match."""
    backend = NativeFaceEnhancementBackend()
    root = str(tmp_path)
    assert backend.run_folder(root, root) == root
    assert not os.path.isdir(each_img_dir(root))


def test_produced_dir_is_the_each_img_folder(tmp_path):
    backend = NativeFaceEnhancementBackend()
    assert backend.produced_dir(str(tmp_path)) == each_img_dir(str(tmp_path))


def test_unload_drops_the_model():
    backend = NativeFaceEnhancementBackend()
    backend._model = object()
    backend._opt = object()
    backend._loaded = True
    backend.unload()
    assert backend._model is None and backend._opt is None


def test_health_explains_why_native_is_unavailable(clean_ownership, monkeypatch):
    """The reason has to reach /api/v1/models, not only the log."""
    health = NativeFaceEnhancementBackend().health()
    assert health.extra["available"] is False
    assert "does not own" in health.extra["reason"]

    clean_ownership.settings.native_tree = "face"
    monkeypatch.setattr(lt, "foreign_tree_loaded", lambda tree: "models")
    health = NativeFaceEnhancementBackend().health()
    assert health.extra["available"] is False
    assert "Global tree is loaded" in health.extra["reason"]


def test_checkpoint_constants_point_at_the_shipped_weight():
    assert fen.CHECKPOINT_NAME == "Setting_9_epoch_100"
    assert fen.GENERATOR_WEIGHTS.endswith(os.path.join("Setting_9_epoch_100", "latest_net_G.pth"))
    assert fen.CHECKPOINTS_DIR.startswith(fen.FACE_ROOT)


def test_warmup_is_declared_a_no_op_with_a_reason():
    """A real warmup needs a dataset item, so the first request pays for it."""
    backend = NativeFaceEnhancementBackend()
    backend._do_warmup()  # must not raise
