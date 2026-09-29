"""Native in-process face enhancement (plan §3.5.1, §4.2 Step 2 — last of the four).

Stage 3 of the vendored pipeline (``Face_Enhancement/test_face.py``) loads a
92M-parameter SPADE generator per run: interpreter start, imports, weight load and
teardown, for a handful of 256px crops. This backend builds the same model once
and keeps it resident.

Reuse, not reimplementation, for anything that decides pixels:

* the preprocessing is the vendored ``data.create_dataloader(opt)`` /
  ``FaceTestDataset`` — resize policy, label handling and tensor normalisation
  come from the tree itself;
* the network is the vendored ``Pix2PixModel(opt)`` with ``mode="inference"``;
* the write is ``torchvision.utils.save_image((generated[b] + 1) / 2, path)``,
  the same call ``test_face.py:42`` makes (no ``normalize``, unlike stage 1).

Two differences from the script, both deliberate:

* paths are absolute. ``test_face.py`` relies on its cwd being ``Face_Enhancement/``
  for ``checkpoints_dir`` and for the label folder; here they are resolved from the
  project root so a library call cannot silently read the wrong directory;
* only the non-HR checkpoint is implemented. ``--HR`` selects ``FaceSR_512``, which
  is not part of the shipped weights, so HR requests go back to the adapter.

This tree declares the same top-level packages as ``Global/``, so hosting it
natively is a per-process decision — see
:mod:`fiximg.inference.backends.legacy_tree` for why it cannot be done by
swapping ``sys.modules``.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any

from fiximg.domain.errors import ModelUnavailableError
from fiximg.inference.backends import device_index
from fiximg.inference.backends.base import BaseModelBackend, ModelRequest, ModelResult
from fiximg.inference.backends.legacy_cli import detection_dir, each_img_dir, list_images
from fiximg.inference.backends.legacy_tree import owns
from fiximg.inference.precision import cached_policy, inference_context
from fiximg.paths import PROJECT_ROOT

FACE_ROOT = os.path.join(PROJECT_ROOT, "Face_Enhancement")
CHECKPOINTS_DIR = os.path.join(FACE_ROOT, "checkpoints")

#: Checkpoint selected by the vendored CLI for the non-HR path.
CHECKPOINT_NAME = "Setting_9_epoch_100"
GENERATOR_WEIGHTS = os.path.join(CHECKPOINTS_DIR, CHECKPOINT_NAME, "latest_net_G.pth")


def checkpoint_present() -> bool:
    """True when the generator weights this branch needs exist."""
    return os.path.isfile(GENERATOR_WEIGHTS)


def native_available() -> tuple[bool, str]:
    """Whether face enhancement can run in-process here, and why not."""
    if not owns("face"):
        return False, (
            "this process does not own the Face_Enhancement tree "
            "(FIXIMG_NATIVE_TREE, or the Global tree is loaded)"
        )
    if not checkpoint_present():
        return False, f"generator weights are missing: {GENERATOR_WEIGHTS}"
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch is not installed"
    return True, ""


_argv_lock = threading.Lock()


@contextmanager
def _argv_guarded(argv: list):
    """Swap ``sys.argv`` for the vendored parser, serialised process-wide.

    ``BaseOptions.gather_options()`` reads the real argv with no way to inject one,
    so building options mutates process state for the length of the call. The lock
    is what makes that survivable: two models loading at once would otherwise read
    each other's flags and each get the other's device id.
    """
    with _argv_lock:
        original = sys.argv
        sys.argv = list(argv)
        try:
            yield
        finally:
            sys.argv = original


def _ensure_tree_importable() -> None:
    """Put ``Face_Enhancement/`` first on sys.path, as the script's cwd did."""
    if FACE_ROOT not in sys.path:
        sys.path.insert(0, FACE_ROOT)


class NativeFaceEnhancementBackend(BaseModelBackend):
    """Progressive face restoration over aligned crops, weights resident."""

    name = "face_enhancement"
    version = "2.0"
    implementation = "native"
    capabilities = frozenset({"face_restore", "face_enhance"})

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        #: The resident Pix2PixModel and the option namespace it was built from.
        #: ``Any`` is honest here: both classes come from the Face_Enhancement tree,
        #: which is imported at runtime and is therefore not visible to mypy.
        self._model: Any = None
        self._opt: Any = None

    # ------------------------------------------------------------- lifecycle
    @property
    def policy(self):
        """Declared execution policy for this model (plan §3.5.4)."""
        return cached_policy(self.name)

    def _build_opt(self, crops_dir: str, output_dir: str, device_index) -> object:
        """The option namespace ``batch.py`` passes to ``test_face.py``.

        Mirrored flag for flag: same ``load_size``/``batchSize``/``label_nc``, same
        ``preprocess_mode resize``, same ``--no_parsing_map``. Only the paths become
        absolute, and ``nThreads`` stays 0 (its default) so no data-worker processes
        are spawned inside the GPU worker.

        ``BaseOptions.gather_options()`` parses the *process* argv
        (``parser.parse_known_args()``, base_options.py:192) and then derives
        ``semantic_nc``, the gpu-id list and the batch/GPU assertions in ``parse()``
        — there is no argv-parameter entry point. So the argv is swapped under a
        lock for the duration of ``parse()`` rather than reimplementing that
        post-processing, which is exactly the kind of transcription this backend
        otherwise avoids.
        """
        _ensure_tree_importable()
        from options.test_options import TestOptions

        argv = [
            "test_face.py",
            f"--dataroot={FACE_ROOT}",
            f"--old_face_folder={crops_dir}",
            "--old_face_label_folder=./",
            "--tensorboard_log",
            f"--name={CHECKPOINT_NAME}",
            f"--checkpoints_dir={CHECKPOINTS_DIR}",
            f"--gpu_ids={device_index}",
            "--load_size", "256",
            "--batchSize", "4",
            "--label_nc", "18",
            "--no_instance",
            "--preprocess_mode", "resize",
            f"--results_dir={output_dir}",
            "--no_parsing_map",
        ]
        with _argv_guarded(argv):
            return TestOptions().parse(save=False)

    @staticmethod
    def _require_weights(opt) -> None:
        """Refuse to run the branch whose generator is not installed.

        ``util.load_network`` raises for a missing file, but the path is assembled
        from options that used to be cwd-relative — checking the resolved location
        is what turns a wrong path into an error rather than a wrong picture.
        """
        expected = os.path.join(opt.checkpoints_dir, opt.name, "latest_net_G.pth")
        if not os.path.isfile(expected):
            raise ModelUnavailableError(
                "Face enhancement weights are missing; refusing to run on "
                "randomly initialised networks",
                details={"expected": expected, "checkpoint": opt.name},
            )

    def _do_load(self, device: str) -> None:
        available, reason = native_available()
        if not available:
            raise ModelUnavailableError(
                f"Native face enhancement is unavailable: {reason}",
                details={"backend": self.name},
            )
        _ensure_tree_importable()
        from models.pix2pix_model import Pix2PixModel

        opt = self._build_opt(
            self.config.get("crops_dir") or "",
            self.config.get("output_dir") or "",
            device_index(device),
        )
        self._require_weights(opt)
        model = Pix2PixModel(opt)
        model.eval()
        self._opt = opt
        # Declared optimisations are applied here too (plan §3.5.4): the manifest
        # keys must reach every native backend, not only the one that happened to
        # call the applier first. Nothing is declared for this model today, so this
        # is a no-op until someone asks for it — and then it is honoured rather
        # than silently ignored.
        self._model = self.apply_policy(model)

    def _do_unload(self) -> None:
        self._model = None
        self._opt = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    def _do_warmup(self) -> None:
        """No synthetic warmup: this model needs a dataset item, not a tensor.

        Running one real crop through it would write an output file, so the first
        request pays the usual first-batch cost; stage timing records it (§3.4.2).
        """

    # --------------------------------------------------------------- inference
    def _enhance(self, crops_dir: str, target_dir: str, device, on_progress=None) -> int:
        if self._model is None:
            # Before any vendored import: raising must not leave Face_Enhancement/
            # on sys.path for the rest of the process.
            raise ModelUnavailableError(
                f"{self.name} is not loaded", details={"backend": self.name}
            )

        import torchvision.utils as vutils

        _ensure_tree_importable()
        from data import create_dataloader

        opt = self._opt
        opt.old_face_folder = crops_dir
        opt.results_dir = target_dir
        os.makedirs(target_dir, exist_ok=True)

        loader = create_dataloader(opt)
        total = max(len(list_images(crops_dir)), 1)
        written = 0
        for batch in loader:
            # inference_context() already wraps torch.inference_mode(); the vendored
            # script used no_grad, which is the weaker form of the same guarantee.
            with inference_context(self.policy, device):
                generated = self._model(batch, mode="inference")
            paths = batch["path"]
            for index in range(generated.shape[0]):
                name = os.path.split(str(paths[index]))[-1]
                vutils.save_image((generated[index] + 1) / 2,
                                  os.path.join(target_dir, name))
                written += 1
            if on_progress is not None:
                on_progress(min(written, total), total, "face enhancement")
        return written

    def _do_infer(self, request: ModelRequest) -> ModelResult:
        started = time.perf_counter()
        if request.options.get("hr"):
            raise ModelUnavailableError(
                "Native face enhancement implements the shipped Setting_9 checkpoint "
                "only; HR selects FaceSR_512, which is not part of this install",
                details={"backend": self.name, "hr": True},
            )
        if not request.work_dir:
            raise ModelUnavailableError(
                "FaceEnhancementBackend requires a work_dir", details={"backend": self.name}
            )

        if self._model is None:
            self.load(request.device)
        target = each_img_dir(request.work_dir)
        count = self._enhance(detection_dir(request.work_dir), target, request.device)
        return ModelResult(
            image=request.image,
            metadata={
                "model": f"Face_Enhancement/{CHECKPOINT_NAME}",
                "backend": self.implementation,
                "enhanced_count": count,
                "stage": "face_enhance",
                "duration_s": round(time.perf_counter() - started, 3),
            },
            artifacts={"each_img_dir": target},
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
        """Same surface as the adapter's stage-3 call.

        Reads ``stage_2_detection_output`` and writes ``stage_3_face_output``
        inside the shared pipeline root; ``input_dir`` is ignored, exactly as the
        vendored stage ignores it (both recompute the crop directory from the root).
        """
        if hr:
            raise ModelUnavailableError(
                "Native face enhancement does not implement the HR checkpoint "
                "(FaceSR_512); run this stage on the legacy-cli backend",
                details={"backend": self.name, "hr": True},
            )

        crops = detection_dir(output_dir)
        if not list_images(crops):
            # Not an error upstream: batch.py returns early and stage 4 copies the
            # restoration output through. Mirror that, rather than failing a task.
            return output_dir

        if self._model is None:
            self.load(str(gpu))
        target_dir = each_img_dir(output_dir)
        self._enhance(crops, target_dir, str(gpu), on_progress=on_progress)
        return output_dir

    def produced_dir(self, output_dir: str) -> str:
        return each_img_dir(output_dir)

    def health(self):  # noqa: D102 - see BaseModelBackend
        health = super().health()
        available, reason = native_available()
        health.extra = {
            "implementation": self.implementation,
            "available": available,
            "reason": reason or None,
            "checkpoint": CHECKPOINT_NAME,
            "weights": GENERATOR_WEIGHTS,
        }
        return health

    def describe(self) -> dict:  # noqa: D102 - see BaseModelBackend
        info = super().describe()
        info["checkpoint"] = CHECKPOINT_NAME
        return info


__all__ = [
    "CHECKPOINTS_DIR",
    "CHECKPOINT_NAME",
    "GENERATOR_WEIGHTS",
    "NativeFaceEnhancementBackend",
    "checkpoint_present",
    "native_available",
]
