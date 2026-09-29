"""Native face-detection backend: contract tests (plan 搂3.5.1, 搂4.2 Step 2).

dlib cannot be compiled in this environment (no C++ toolchain), so the numbers
produced by the real detector are verified elsewhere 鈥?``tests/gpu/test_face_detect_equivalence.py`` compares crops against the vendored
script wherever dlib is installed. What these tests pin instead is everything that
does not need a trained model:

* the crop geometry is *delegated* to the vendored ``search`` /
  ``compute_transformation_matrix`` with the arguments ``detect_all_dlib.py``
  itself passes 鈥?not re-derived here;
* the folder contract the stages rely on (read the restoration output of the
  shared root, write ``stage_2_detection_output``);
* the vendored naming rule ``<stem>_<n>.png`` that stage 4 pairs crops back by;
* refusal of the HR geometry instead of approximating it;
* selection and fallback, which is off by default precisely because the numerics
  are unverified here.
"""
import os
import sys
import types

import numpy as np
import pytest
from PIL import Image

from fiximg.domain.errors import ModelUnavailableError, PipelineFailedError
from fiximg.inference.backends import face_detect_native as fdn
from fiximg.inference.backends import registry as registry_module
from fiximg.inference.backends.base import ModelRequest
from fiximg.inference.backends.face_detect_native import NativeFaceDetectionBackend
from fiximg.inference.backends.legacy_cli import LegacyCliBackend, detection_dir, restored_image_dir


class _Box:
    """A dlib rectangle is passed straight to the shape predictor."""

    def __init__(self, left):
        self.left = left

    def __repr__(self):
        return f"Box({self.left})"


class _FakeVendored:
    """Records the geometry calls and returns a predictable crop."""

    def __init__(self):
        self.search_calls = 0
        self.matrix_calls = []
        self.landmarks = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
                                   [7.0, 8.0], [9.0, 10.0]], dtype=np.float64)

    def search(self, face_landmarks):
        self.search_calls += 1
        return self.landmarks

    def compute_transformation_matrix(self, img, landmark, normalize, target_face_scale=1.0):
        self.matrix_calls.append({
            "shape": img.shape, "normalize": normalize, "scale": target_face_scale,
        })
        return {"params": landmark}


@pytest.fixture()
def fake_dlib(monkeypatch, tmp_path):
    """dlib + the vendored helpers + skimage's warp, all stubbed."""
    boxes = {"list": [_Box(0), _Box(40)]}

    dlib = types.ModuleType("dlib")
    dlib.get_frontal_face_detector = lambda: (lambda array: list(boxes["list"]))

    class _Predictor:
        """dlib's predictor is called as ``predictor(image, box)``."""

        def __init__(self, path):
            self.path = path

        def __call__(self, image, box):
            return box

    dlib.shape_predictor = _Predictor
    monkeypatch.setitem(sys.modules, "dlib", dlib)

    vendored = _FakeVendored()
    monkeypatch.setattr(fdn, "_load_vendored", lambda: vendored)
    landmark = tmp_path / "shape_predictor_68.dat"
    landmark.write_bytes(b"fake predictor")  # availability checks the file, dlib is stubbed
    monkeypatch.setattr(fdn, "LANDMARK_MODEL", str(landmark))
    # The selection seam caches a process-wide instance; each test starts clean.
    monkeypatch.setattr(registry_module, "_native_face", None)
    monkeypatch.setattr(registry_module, "_FACE_BACKEND_CLS", None)

    import skimage.util
    import skimage.transform

    def fake_warp(image, affine, output_shape=None, **kwargs):
        height, width = output_shape[0], output_shape[1]
        crop = np.zeros((height, width, 3), dtype=np.float64)
        # Encode the box order so a test can tell the crops apart.
        crop[..., 0] = float(len(affine) if isinstance(affine, dict) else 0) + width
        return crop

    monkeypatch.setattr(skimage.transform, "warp", fake_warp)
    # Patched where production looks: it imports `img_as_ubyte` from
    # `skimage.util` (the declared home of the symbol), so a function-scoped
    # import re-reads the attribute on every call 鈥?which is what makes this
    # patch effective at all.
    monkeypatch.setattr(skimage.util, "img_as_ubyte", lambda a: np.clip(a, 0, 255).astype(np.uint8))
    return vendored, boxes


@pytest.fixture()
def backend(fake_dlib):
    vendored, _boxes = fake_dlib
    instance = NativeFaceDetectionBackend({"stem": "task-1"})
    instance.load("cpu")
    return instance, vendored


# ================================ geometry =================================
def test_load_opens_the_landmark_model_by_absolute_path(fake_dlib, monkeypatch):
    _vendored, _boxes = fake_dlib
    opened = []
    dlib = sys.modules["dlib"]
    monkeypatch.setattr(dlib, "shape_predictor", lambda path: opened.append(path) or object())
    backend = NativeFaceDetectionBackend()
    backend.load("cpu")
    assert opened == [fdn.LANDMARK_MODEL]
    assert os.path.isabs(opened[0]), "the vendored script's bare filename relies on cwd"


def test_the_vendored_geometry_is_delegated_not_reimplemented(backend):
    instance, vendored = backend
    array = np.zeros((64, 64, 3), dtype=np.uint8)
    instance._align(array)

    assert vendored.search_calls == 2, "one landmark set per detected box"
    assert len(vendored.matrix_calls) == 2
    for call in vendored.matrix_calls:
        # detect_all_dlib.py:173 鈥?normalize=False, target_face_scale=1.3
        assert call["normalize"] is False
        assert call["scale"] == fdn.TARGET_FACE_SCALE == 1.3


def test_crops_are_the_size_the_vendored_script_produces(backend, monkeypatch):
    instance, _vendored = backend
    seen = {}

    import skimage.transform

    def spy(image, affine, output_shape=None, **kwargs):
        seen["output_shape"] = output_shape
        return np.zeros(output_shape, dtype=np.float64)

    monkeypatch.setattr(skimage.transform, "warp", spy)
    instance._align(np.zeros((64, 64, 3), dtype=np.uint8))
    assert seen["output_shape"] == (fdn.CROP_SIZE, fdn.CROP_SIZE, 3) == (256, 256, 3)


# ============================ the folder contract ==========================
def test_run_folder_reads_the_restoration_output_of_the_shared_root(backend, tmp_path):
    instance, _vendored = backend
    root = str(tmp_path / "pipeline")
    source = os.path.join(restored_image_dir(root), "a.png")
    os.makedirs(os.path.dirname(source), exist_ok=True)
    Image.new("RGB", (64, 64), "grey").save(source)

    instance.run_folder(str(tmp_path / "ignored"), root)

    produced = sorted(os.listdir(detection_dir(root)))
    assert produced == ["a_1.png", "a_2.png"], "two boxes -> the vendored naming rule"


def test_run_folder_fails_when_stage_1_never_ran(backend, tmp_path):
    instance, _vendored = backend
    root = str(tmp_path / "empty")
    with pytest.raises(PipelineFailedError, match="requires the restoration output"):
        instance.run_folder(str(tmp_path), root)


def test_run_folder_refuses_the_hr_geometry(backend, tmp_path):
    """HR maps onto a 512 canvas 鈥?a different transform, not a flag."""
    instance, _vendored = backend
    root = str(tmp_path / "pipeline")
    os.makedirs(restored_image_dir(root), exist_ok=True)
    Image.new("RGB", (32, 32), "grey").save(os.path.join(restored_image_dir(root), "a.png"))

    with pytest.raises(ModelUnavailableError, match="HR"):
        instance.run_folder(str(tmp_path), root, hr=True)


def test_progress_markers_match_the_stage_label(backend, tmp_path):
    instance, _vendored = backend
    root = str(tmp_path / "pipeline")
    os.makedirs(restored_image_dir(root), exist_ok=True)
    for name in ("a.png", "b.png"):
        Image.new("RGB", (32, 32), "grey").save(os.path.join(restored_image_dir(root), name))

    seen = []
    instance.run_folder(str(tmp_path), root,
                        on_progress=lambda step, total, label: seen.append((step, total, label)))
    assert seen == [(1, 2, "face detection"), (2, 2, "face detection")]


def test_produced_dir_is_the_detection_folder(backend, tmp_path):
    instance, _vendored = backend
    assert instance.produced_dir(str(tmp_path)) == detection_dir(str(tmp_path))


# ============================== infer surface ==============================
def test_infer_needs_a_work_dir(backend, tmp_path):
    instance, _vendored = backend
    with pytest.raises(ModelUnavailableError, match="work_dir"):
        instance.infer(ModelRequest(image=Image.new("RGB", (32, 32)), device="cpu"))


def test_infer_returns_crops_and_counts(backend, tmp_path):
    instance, _vendored = backend
    result = instance.infer(ModelRequest(
        image=Image.new("RGB", (64, 64), "grey"),
        device="cpu", work_dir=str(tmp_path), options={},
    ))
    assert result.metadata["face_count"] == 2
    assert result.metadata["backend"] == "native"
    assert result.metadata["model"] == "dlib-68"
    # Detection must not alter the flowing image 鈥?stage 3 consumes the crops.
    assert result.image.size == (64, 64)
    assert sorted(os.listdir(result.artifacts["faces_dir"])) == [
        "task-1_1.png", "task-1_2.png",
    ]


def test_infer_refuses_hr(backend, tmp_path):
    instance, _vendored = backend
    with pytest.raises(ModelUnavailableError, match="256-canvas"):
        instance.infer(ModelRequest(image=Image.new("RGB", (16, 16)), device="cpu",
                                    work_dir=str(tmp_path), options={"hr": True}))


def test_infer_before_load_is_a_domain_error(monkeypatch, tmp_path):
    monkeypatch.setattr(fdn, "native_available", lambda: (True, ""))
    with pytest.raises(ModelUnavailableError):
        NativeFaceDetectionBackend()._do_infer(
            ModelRequest(image=Image.new("RGB", (8, 8)), device="cpu", work_dir=str(tmp_path))
        )


def test_unload_releases_the_detector_and_predictor(backend):
    instance, _vendored = backend
    instance.unload()
    assert instance._detector is None and instance._predictor is None
    assert instance.is_loaded is False


# =============================== availability ==============================
def test_a_missing_landmark_model_is_a_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(fdn, "LANDMARK_MODEL", str(tmp_path / "absent.dat"))
    available, reason = fdn.native_available()
    assert available is False
    assert "landmark model" in reason


def test_a_missing_dlib_is_a_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(fdn, "LANDMARK_MODEL", str(tmp_path / "here.dat"))
    (tmp_path / "here.dat").write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "dlib", None)
    available, reason = fdn.native_available()
    assert available is False
    assert "dlib" in reason


def test_available_when_both_are_present(monkeypatch, tmp_path):
    monkeypatch.setattr(fdn, "LANDMARK_MODEL", str(tmp_path / "here.dat"))
    (tmp_path / "here.dat").write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "dlib", types.ModuleType("dlib"))
    assert fdn.native_available() == (True, "")


def test_load_reports_unavailability_rather_than_crashing(monkeypatch):
    monkeypatch.setattr(fdn, "native_available", lambda: (False, "dlib is not installed"))
    with pytest.raises(ModelUnavailableError, match="dlib is not installed"):
        NativeFaceDetectionBackend().load("cpu")


# ================================= selection ===============================
@pytest.fixture()
def face_selection(monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(fdn, "native_available", lambda: (True, ""))

    class _Fake:
        implementation = "native"

        def __init__(self, config=None):
            self.config = config

    monkeypatch.setattr(registry_module, "_FACE_BACKEND_CLS", _Fake)
    monkeypatch.setattr(registry_module, "_native_face", None)
    return config_mod, _Fake


def test_turning_the_flag_off_selects_the_subprocess_adapter(face_selection, monkeypatch):
    config_mod, _Fake = face_selection
    monkeypatch.setattr(config_mod.settings, "face_detect_native", False, raising=False)
    backend = registry_module.select_face_detection_backend()
    assert isinstance(backend, LegacyCliBackend)
    assert backend.implementation == "legacy-cli"


def test_detection_is_independent_of_which_legacy_tree_the_process_owns(
    face_selection, monkeypatch
):
    """`FIXIMG_NATIVE_TREE` says what the process imports; detection imports nothing.

    A face-detection worker may be a ``global`` worker or a ``face`` one, and the
    escape hatch for *detection* is `FIXIMG_FACE_DETECT_NATIVE` 鈥?not the tree
    setting. If `none` also disabled detection there would be two switches for one
    decision, and the subprocess-chain test (which sets `none` to keep the vendored
    stages out of process) would silently stop testing stage 2 at all.
    """
    config_mod, _Fake = face_selection
    monkeypatch.setattr(config_mod.settings, "face_detect_native", True, raising=False)
    for tree in ("global", "face", "none"):
        monkeypatch.setattr(config_mod.settings, "native_tree", tree, raising=False)
        assert registry_module.select_face_detection_backend().implementation == "native", tree

    monkeypatch.setattr(config_mod.settings, "face_detect_native", False, raising=False)
    assert registry_module.select_face_detection_backend().implementation == "legacy-cli"


def test_detection_is_native_by_default_now_that_the_crops_were_compared(monkeypatch):
    """The shipped default, read from a fresh Settings 鈥?not from a patched one.

    The flag existed as "off until the pixel comparison has run somewhere". That
    comparison is `tests/gpu/test_face_detect_equivalence.py`, and it passes on an
    install with dlib 20.0.1 and `shape_predictor_68_face_landmarks.dat`, so the
    default moved. Asserting `settings.face_detect_native` on the module-level
    instance would only have reported whatever the test process's environment
    happened to say, so this constructs a Settings of its own.
    """
    from fiximg.config import Settings

    monkeypatch.delenv("FIXIMG_FACE_DETECT_NATIVE", raising=False)
    assert Settings().face_detect_native is True

    monkeypatch.setenv("FIXIMG_FACE_DETECT_NATIVE", "0")
    assert Settings().face_detect_native is False, "the switch still means no"


def test_detection_uses_native_when_enabled(face_selection, monkeypatch):
    config_mod, _Fake = face_selection
    monkeypatch.setattr(config_mod.settings, "face_detect_native", True, raising=False)
    assert registry_module.select_face_detection_backend().implementation == "native"


def test_hr_always_goes_to_the_adapter(face_selection, monkeypatch):
    config_mod, _Fake = face_selection
    monkeypatch.setattr(config_mod.settings, "face_detect_native", True, raising=False)
    backend = registry_module.select_face_detection_backend(hr=True)
    assert isinstance(backend, LegacyCliBackend)


def test_an_unavailable_environment_falls_back(face_selection, monkeypatch):
    config_mod, _Fake = face_selection
    monkeypatch.setattr(config_mod.settings, "face_detect_native", True, raising=False)
    monkeypatch.setattr(fdn, "native_available", lambda: (False, "dlib is not installed"))
    assert registry_module.select_face_detection_backend().implementation == "legacy-cli"


def test_the_dlib_models_stay_resident(face_selection, monkeypatch):
    """One instance per process: that is what 'resident' means here."""
    config_mod, _Fake = face_selection
    monkeypatch.setattr(config_mod.settings, "face_detect_native", True, raising=False)
    assert registry_module.select_face_detection_backend() is \
        registry_module.select_face_detection_backend()


def test_the_registry_reports_what_serves_detection(face_selection, monkeypatch):
    config_mod, _Fake = face_selection
    monkeypatch.setattr(config_mod.settings, "face_detect_native", True, raising=False)
    registry = registry_module.build_default_backend_registry()
    assert registry.get("face_detection").implementation == "native"

    monkeypatch.setattr(config_mod.settings, "face_detect_native", False, raising=False)
    registry_module._native_face = None
    registry = registry_module.build_default_backend_registry()
    assert registry.get("face_detection").implementation == "legacy-cli"


# ============================ health reporting =============================
def test_health_names_the_implementation_and_the_model(backend, monkeypatch):
    instance, _vendored = backend
    extra = instance.health().extra
    assert extra["implementation"] == "native"
    assert extra["crop_size"] == 256
    assert extra["available"] is True
    assert extra["reason"] is None


def test_health_explains_an_unavailable_native_path(monkeypatch):
    monkeypatch.setattr(fdn, "native_available", lambda: (False, "dlib is not installed"))
    extra = NativeFaceDetectionBackend().health().extra
    assert extra["available"] is False
    assert "dlib" in extra["reason"]


# =============================== stage wiring ==============================
def test_the_stage_records_which_implementation_ran(fake_dlib, monkeypatch, tmp_path):
    """Stage metadata must not claim legacy-cli when dlib ran in-process.

    The stage never calls load() itself 鈥?the bridge is expected to load on
    demand, which is the whole point of a resident backend being handed a folder.
    """
    import fiximg.config as config_mod
    from fiximg.inference.context import StageContext
    from fiximg.inference.stages.face_enhancement import FaceDetectionStage

    vendored, _boxes = fake_dlib
    monkeypatch.setattr(config_mod.settings, "face_detect_native", True, raising=False)
    monkeypatch.setattr(config_mod.settings, "native_tree", "none", raising=False)

    from fiximg.inference.stages.base import legacy_pipeline_root

    run_dir = str(tmp_path / "run")
    context = StageContext(task_id="task-1", run_dir=run_dir, options={})
    # The stage resolves its own shared root; write where it will look, not where
    # the test would like it to look.
    root = legacy_pipeline_root(context)
    os.makedirs(restored_image_dir(root), exist_ok=True)
    Image.new("RGB", (48, 48), "grey").save(os.path.join(restored_image_dir(root), "task-1.png"))
    os.makedirs(os.path.join(run_dir, "input"), exist_ok=True)

    result = FaceDetectionStage().run(Image.new("RGB", (48, 48), "grey"), context)

    assert result.metadata["backend"] == "native"
    assert result.metadata["face_count"] == 2
    assert vendored.search_calls == 2


def test_run_folder_loads_on_demand(fake_dlib, tmp_path):
    """A stage that only knows run_folder must not have to manage the lifecycle."""
    instance = NativeFaceDetectionBackend({"stem": "task-1"})
    assert instance.is_loaded is False

    root = str(tmp_path / "pipeline")
    os.makedirs(restored_image_dir(root), exist_ok=True)
    Image.new("RGB", (32, 32), "grey").save(os.path.join(restored_image_dir(root), "a.png"))

    instance.run_folder(str(tmp_path), root)
    assert instance.is_loaded is True
    # sorted(): os.listdir returns directory order, which is creation order on
    # some filesystems and something else on others. Asserting the raw listing
    # passed on NTFS and failed on ext4, for a pipeline that is correct either
    # way.
    assert sorted(os.listdir(detection_dir(root))) == ["a_1.png", "a_2.png"]


def test_aligning_without_loading_is_refused(fake_dlib):
    """The bridge loads on demand, so _align should never see a None detector."""
    import numpy as np

    with pytest.raises(ModelUnavailableError, match="not loaded"):
        NativeFaceDetectionBackend()._align(np.zeros((8, 8, 3), dtype=np.uint8))


def test_warmup_on_an_unloaded_backend_is_harmless(fake_dlib):
    import numpy as np  # noqa: F401

    backend = NativeFaceDetectionBackend()
    backend._do_warmup()  # must not raise: warmup is never fatal
    assert backend.is_loaded is False
