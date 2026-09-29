"""Native in-process face detection (plan §3.5.1, §4.2 Step 2).

Stage 2 of the vendored pipeline (``Face_Detection/detect_all_dlib.py``) ran as a
subprocess even though it is one dlib HOG scan plus one 68-point shape predictor
and a similarity warp — nothing that needs a fresh interpreter. Making it native
keeps dlib's models resident, which matters because the face chain runs it for
every photo that has faces.

Unlike the Global tree, this one imports no top-level ``options``/``models``/
``util``/``data`` package, so it coexists with the native Global backend in the
same process (plan §2.6 wanted the boundaries explicit rather than accidental).

The geometry is **not reimplemented**: ``search``,
``compute_transformation_matrix`` and the ``warp``/``img_as_ubyte`` pair are the
vendored ones, so an aligned crop here is the aligned crop the CLI produced. Two
deliberate deviations, both to avoid a silent failure:

* the landmark model is opened by absolute path — the vendored script relies on
  its cwd being ``Face_Detection/``, which is only true of the subprocess;
* only the non-HR geometry is implemented. ``detect_all_dlib_HR.py`` maps the
  standard points onto a 512 canvas (``* 512.0``) rather than 256, so it is a
  different function, not a flag; ``hr=True`` refuses rather than approximating.
"""
from __future__ import annotations

import importlib.util
import os

from fiximg.domain.errors import ModelUnavailableError, PipelineFailedError
from fiximg.inference.backends.base import BaseModelBackend, ModelRequest, ModelResult
from fiximg.inference.backends.legacy_cli import detection_dir, list_images, restored_image_dir
from fiximg.paths import PROJECT_ROOT

FACE_DETECTION_DIR = os.path.join(PROJECT_ROOT, "Face_Detection")
LANDMARK_MODEL = os.path.join(FACE_DETECTION_DIR, "shape_predictor_68_face_landmarks.dat")

#: Crop geometry the vendored script uses for the non-HR path.
CROP_SIZE = 256
TARGET_FACE_SCALE = 1.3

_vendored = None
_dlib = None


def _load_vendored():
    """``Face_Detection/detect_all_dlib.py`` as a private module.

    Loaded by file location under a unique name: importing it as ``detect_all_dlib``
    would work, but ``Face_Detection`` is not on ``sys.path`` and adding it there
    would put its ``json``/``os`` -free but generically named files in scope for the
    whole application.
    """
    global _vendored

    if _vendored is None:
        path = os.path.join(FACE_DETECTION_DIR, "detect_all_dlib.py")
        spec = importlib.util.spec_from_file_location("_fiximg_legacy_face_detect", path)
        if spec is None or spec.loader is None:
            raise ModelUnavailableError(
                "Cannot load the vendored face detection script",
                details={"path": path},
            )
        module = importlib.util.module_from_spec(spec)
        import sys

        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 — a broken tree must not half-load
            sys.modules.pop(spec.name, None)
            raise ModelUnavailableError(
                f"The vendored face detection script failed to import: {exc}",
                details={"path": path},
            ) from exc
        _vendored = module
    return _vendored


def native_available() -> tuple[bool, str]:
    """Whether dlib-based detection can run in this process, and why not."""
    if not os.path.isfile(LANDMARK_MODEL):
        return False, f"landmark model is missing: {LANDMARK_MODEL}"
    try:
        import dlib  # noqa: F401
    except ImportError:
        return False, "dlib is not installed (pip install 'fiximg[gpu]' builds it from source)"
    return True, ""


class NativeFaceDetectionBackend(BaseModelBackend):
    """dlib 68-point detection + similarity alignment, weights resident."""

    name = "face_detection"
    version = "2.0"
    implementation = "native"
    capabilities = frozenset({"face_detection"})

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._detector = None
        self._predictor = None

    # ------------------------------------------------------------- lifecycle
    def _do_load(self, device: str) -> None:
        available, reason = native_available()
        if not available:
            raise ModelUnavailableError(
                f"Native face detection is unavailable: {reason}",
                details={"backend": self.name},
            )
        import dlib

        self._detector = dlib.get_frontal_face_detector()
        # Absolute path: the vendored script passes a bare filename and relies on
        # cwd == Face_Detection/, which only the subprocess version guarantees.
        self._predictor = dlib.shape_predictor(LANDMARK_MODEL)
        _load_vendored()

    def _do_warmup(self) -> None:
        import numpy as np

        detector = self._detector
        if detector is None:
            return
        try:
            detector(np.zeros((64, 64, 3), dtype=np.uint8))
        except Exception:  # noqa: BLE001 — warmup must never be fatal
            pass

    def _do_unload(self) -> None:
        self._detector = None
        self._predictor = None

    # --------------------------------------------------------------- geometry
    def _align(self, array):
        """Aligned 256×256 crops, in the order the vendored script emits them."""
        # `skimage.util`, not `skimage`: the top-level package re-exports it
        # without declaring it in ``__all__``, which makes the import unofficial.
        from skimage.transform import warp
        from skimage.util import img_as_ubyte

        detector, predictor = self._detector, self._predictor
        if detector is None or predictor is None:
            raise ModelUnavailableError(
                f"{self.name} is not loaded", details={"backend": self.name}
            )

        vendored = _load_vendored()
        crops = []
        for box in detector(array):
            landmarks = vendored.search(predictor(array, box))
            affine = vendored.compute_transformation_matrix(
                array, landmarks, False, target_face_scale=TARGET_FACE_SCALE
            )
            crops.append(img_as_ubyte(warp(array, affine, output_shape=(CROP_SIZE, CROP_SIZE, 3))))
        return crops

    def _detect_into(self, array, stem: str, target_dir: str) -> int:
        """Detect, align and write the crops of one image; return the count."""
        from PIL import Image

        os.makedirs(target_dir, exist_ok=True)
        crops = self._align(array)
        for index, crop in enumerate(crops, start=1):
            # ``x[:-4] + "_" + n`` is the vendored naming rule
            # (detect_all_dlib.py:175) — reproduced exactly, because stage 4 pairs
            # each crop back onto its source by these names.
            Image.fromarray(crop).save(os.path.join(target_dir, f"{stem}_{index}.png"))
        return len(crops)

    # --------------------------------------------------------------- inference
    def _do_infer(self, request: ModelRequest) -> ModelResult:
        import numpy as np

        if request.options.get("hr"):
            raise ModelUnavailableError(
                "Native face detection implements the 256-canvas geometry only; "
                "HR alignment is a different transform",
                details={"backend": self.name, "hr": True},
            )
        if self._detector is None:
            raise ModelUnavailableError(f"{self.name} is not loaded", details={"backend": self.name})
        if not request.work_dir:
            raise ModelUnavailableError(
                "FaceDetectionBackend requires a work_dir", details={"backend": self.name}
            )

        target = os.path.join(request.work_dir, "detection")
        count = self._detect_into(np.asarray(request.image.convert("RGB")),
                                  self.config.get("stem", "image"), target)
        return ModelResult(
            image=request.image,
            metadata={
                "model": "dlib-68",
                "backend": self.implementation,
                "face_count": count,
                "stage": "face_detect",
            },
            artifacts={"faces_dir": target},
        )

    # ---------------------------------------------------- folder bridge for stages
    def run_folder(
        self,
        input_dir: str,
        output_dir: str,
        *,
        gpu: int = -1,
        hr: bool = False,
        on_progress=None,
    ) -> str:
        """Same surface as the adapter's stage-2 call.

        ``input_dir`` is ignored on purpose: the vendored detection stage reads the
        *restoration output* from the shared pipeline root (``batch.py``'s
        ``run_face_detection`` recomputes it the same way), so honoring the argument
        would detect faces in a different picture than stage 4 composites back.
        """
        from PIL import Image

        if hr:
            raise ModelUnavailableError(
                "Native face detection does not implement the HR geometry; "
                "run this stage on the legacy-cli backend",
                details={"backend": self.name, "hr": True},
            )

        source_dir = restored_image_dir(output_dir)
        names = list_images(source_dir)
        if not names:
            raise PipelineFailedError(
                f"Face detection requires the restoration output: {source_dir} is empty.",
                details={"backend": self.name, "input_dir": source_dir},
            )

        # The stage calls this bridge directly, so it owns loading: infer() goes
        # through BaseModelBackend.infer which loads on demand, run_folder must too.
        if not self.is_loaded:
            self.load("cpu" if gpu is None or int(gpu) < 0 else str(gpu))

        import numpy as np

        target = detection_dir(output_dir)
        os.makedirs(target, exist_ok=True)
        for position, name in enumerate(names, start=1):
            image = Image.open(os.path.join(source_dir, name)).convert("RGB")
            stem = os.path.splitext(name)[0]
            # Zero crops is not an error: the vendored script warns and continues,
            # and stage 3 treats an empty crop set as "nothing to enhance".
            self._detect_into(np.asarray(image), stem, target)
            if on_progress is not None:
                on_progress(position, len(names), "face detection")
        return output_dir

    def produced_dir(self, output_dir: str) -> str:
        return detection_dir(output_dir)

    def health(self):  # noqa: D102 - see BaseModelBackend
        health = super().health()
        available, reason = native_available()
        health.extra = {
            "implementation": self.implementation,
            "available": available,
            "reason": reason or None,
            "landmark_model": LANDMARK_MODEL,
            "crop_size": CROP_SIZE,
        }
        return health


__all__ = [
    "FACE_DETECTION_DIR",
    "LANDMARK_MODEL",
    "NativeFaceDetectionBackend",
    "native_available",
]
