"""Native in-process Global restoration backend (plan §3.5.1, §4.2 Step 1-2).

This is the item the report names as the highest-value change (§9): until now
every whole-image restoration spawned ``python run.py``, which re-read the three
quality networks from disk per request — no resident weights, no warmup, no
per-stage device control, and ~2 s of interpreter + import cost on top.

Here the networks are built once and kept in the process, and inference calls the
*vendored* functions rather than reimplementing them, so the numbers cannot drift:
:data:`Global/test.py` supplies ``parameter_set`` (the architecture switches),
``data_transforms`` (the geometry) and ``Pix2PixHDModel_Mapping.inference``, while
the frame is written by the same ``torchvision.utils.save_image(..., normalize=
True)`` call the CLI used — that ``normalize`` is not a no-op: it stretches each
image by its own min/max, and reimplementing it by hand is how a "refactor" ends up
with different output.

**Constraint that shapes the design**: ``Global/`` and ``Face_Enhancement/`` each
ship top-level ``options`` / ``models`` / ``util`` / ``data`` packages, so the two
trees cannot be imported into one interpreter at all. Native loading is therefore
per *tree*, and this backend claims the process for ``Global``; the face chain
stays on :class:`~fiximg.inference.backends.legacy_cli.LegacyCliBackend` (which is
also why the scratch path, whose settings live in the same tree but needs the mask
plumbing, is opt-in separately below).
"""
from __future__ import annotations

import importlib.util
import io
import os
import sys
import time
from typing import Any

from fiximg.domain.errors import ModelUnavailableError, PipelineFailedError
from fiximg.inference.backends.base import BaseModelBackend, ModelRequest, ModelResult
from fiximg.inference.backends.legacy_cli import list_images, restored_image_dir
from fiximg.inference.precision import cached_policy, inference_context
from fiximg.paths import PROJECT_ROOT

GLOBAL_DIR = os.path.join(PROJECT_ROOT, "Global")
CHECKPOINTS_DIR = os.path.join(GLOBAL_DIR, "checkpoints", "restoration")

#: The weights the quality (no-scratch) path actually reads. Anything else in
#: those directories is training state (discriminators, optimizers).
QUALITY_WEIGHTS = (
    os.path.join("VAE_A_quality", "latest_net_G.pth"),
    os.path.join("VAE_B_quality", "latest_net_G.pth"),
    os.path.join("mapping_quality", "latest_net_mapping_net.pth"),
)

#: Modules both legacy trees declare at top level; the collision rules and their
#: rationale live in :mod:`fiximg.inference.backends.legacy_tree`.

_legacy_test = None
_legacy_detection = None


def quality_weights_present() -> bool:
    """True when all three quality networks exist on disk."""
    return all(os.path.isfile(os.path.join(CHECKPOINTS_DIR, rel)) for rel in QUALITY_WEIGHTS)


#: Networks the scratch branch adds on top of the quality pair.
SCRATCH_EXTRA_WEIGHTS = (
    os.path.join("VAE_B_scratch", "latest_net_G.pth"),
    os.path.join("mapping_scratch", "latest_net_mapping_net.pth"),
)
#: Scratch mask detector (a separate UNet, loaded by Global/detection.py).
DETECTION_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "Global", "checkpoints", "detection", "FT_Epoch_latest.pt"
)


def scratch_weights_present() -> bool:
    """True when the quality nets, the scratch nets and the detector all exist."""
    return (quality_weights_present()
            and all(os.path.isfile(os.path.join(CHECKPOINTS_DIR, rel)) for rel in SCRATCH_EXTRA_WEIGHTS)
            and os.path.isfile(DETECTION_CHECKPOINT))


def scratch_native_available() -> tuple[bool, str]:
    """Availability of the scratch branch (it needs one more network than quality)."""
    available, reason = native_available()
    if not available:
        return False, reason
    if not scratch_weights_present():
        return False, (
            "scratch branch weights are missing (VAE_B_scratch, mapping_scratch "
            "and checkpoints/detection/FT_Epoch_latest.pt)"
        )
    return True, ""


def legacy_tree_conflict() -> str | None:
    """Name of a top-level legacy package already imported from the other tree.

    Thin delegation, kept as this module's public wording because the health
    payload and tests report it as the fallback reason.
    """
    from fiximg.inference.backends.legacy_tree import foreign_tree_loaded

    return foreign_tree_loaded("global")


def _detection_module():
    """``Global/detection.py`` as a private module, for ``data_transforms`` and
    ``scale_tensor``.

    Those two decide the geometry the detector sees, and therefore the mask. They
    are reused rather than transcribed so a change upstream cannot leave this
    backend quietly producing a different mask from the subprocess version.
    """
    global _legacy_detection

    if _legacy_detection is None:
        path = os.path.join(GLOBAL_DIR, "detection.py")
        spec = importlib.util.spec_from_file_location("_fiximg_legacy_scratch_detect", path)
        if spec is None or spec.loader is None:
            raise ModelUnavailableError(
                "Cannot load the vendored scratch detector",
                details={"path": path},
            )
        module = importlib.util.module_from_spec(spec)
        if GLOBAL_DIR not in sys.path:
            sys.path.insert(0, GLOBAL_DIR)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 — a broken tree must not half-load
            sys.modules.pop(spec.name, None)
            raise ModelUnavailableError(
                f"The vendored scratch detector failed to import: {exc}",
                details={"path": path},
            ) from exc
        _legacy_detection = module
    return _legacy_detection


def native_available() -> tuple[bool, str]:
    """Whether this process can run Global restoration in-process, and why not."""
    from fiximg.inference.backends.legacy_tree import declared_tree, owns

    if not owns("global"):
        return False, (
            "this process does not own the Global tree "
            f"(FIXIMG_NATIVE_TREE={declared_tree()!r}, or the face tree is loaded)"
        )
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch is not installed"
    if not quality_weights_present():
        return False, "quality weights are missing under Global/checkpoints/restoration"
    conflict = legacy_tree_conflict()
    if conflict:
        return False, f"'{conflict}' is already imported from another legacy tree"
    return True, ""


def _legacy_module():
    """Load ``Global/test.py`` as a private module (its name would shadow stdlib).

    ``test.py`` is importable: everything executable sits inside ``__main__``, so
    this gives access to the authoritative preprocessing helpers instead of a
    re-derivation of them.
    """
    global _legacy_test
    if _legacy_test is not None:
        return _legacy_test

    if GLOBAL_DIR not in sys.path:
        sys.path.insert(0, GLOBAL_DIR)
    spec = importlib.util.spec_from_file_location(
        "_fiximg_legacy_global_test", os.path.join(GLOBAL_DIR, "test.py")
    )
    if spec is None or spec.loader is None:
        raise ModelUnavailableError(
            "Cannot load the vendored Global pipeline",
            details={"path": os.path.join(GLOBAL_DIR, "test.py")},
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — a broken tree must not half-load
        sys.modules.pop(spec.name, None)
        raise ModelUnavailableError(
            f"The vendored Global pipeline failed to import: {exc}",
            details={"path": os.path.join(GLOBAL_DIR, "test.py")},
        ) from exc
    _legacy_test = module
    return module


def _device_index(device) -> str | int:
    """See :func:`fiximg.inference.backends.device_index` — shared with the face tree."""
    from fiximg.inference.backends import device_index

    return device_index(device)


class GlobalRestoreBackend(BaseModelBackend):
    """Overall quality restoration, resident in the worker process."""

    name = "global_restore"
    version = "1.0"
    implementation = "native"
    capabilities = frozenset({"restore", "deblur", "denoise"})

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        # Both come from the vendored tree (`test.py`'s model + parsed options),
        # which is untyped upstream code — `Any` is the honest annotation.
        self._model: Any = None
        self._opt: Any = None

    # ------------------------------------------------------------- lifecycle
    def _check_available(self) -> tuple[bool, str]:
        """Branch-specific availability; overridden by the scratch subclass."""
        return native_available()

    @property
    def policy(self):
        """Declared execution policy (plan §3.5.4 — never a global opt-out)."""
        return cached_policy(self.name)

    def _build_opt(self, device_index):
        legacy = _legacy_module()
        from options.test_options import TestOptions  # vendored package, Global/ on sys.path

        options = TestOptions()
        if not options.initialized:
            # BaseOptions builds its parser lazily inside parse(); going straight to
            # .parser without this yields an argparse object with no arguments at all.
            options.initialize()
        opt = options.parser.parse_args([
            "--test_mode=Full",
            "--Quality_restore",
            # "=" form because argparse reads a bare "-1" as another option, and
            # "-1" is exactly what "run on CPU" means in this convention.
            f"--gpu_ids={device_index}",
            # Absolute, because the vendored default ("./checkpoints") is resolved
            # against the *current directory* — which only the subprocess version
            # guarantees. Without this the weights silently fail to load.
            f"--checkpoints_dir={CHECKPOINTS_DIR}",
        ])
        opt.isTrain = options.isTrain
        # BaseOptions.parse() post-processing, reproduced: comma string -> list of
        # non-negative ids. Skipping it leaves opt.gpu_ids a str.
        opt.gpu_ids = [int(x) for x in str(opt.gpu_ids).split(",") if x.strip() and int(x) >= 0]
        legacy.parameter_set(opt)
        # parameter_set() hard-codes checkpoints_dir as "./checkpoints/restoration",
        # which only resolves inside Global/ — true for the subprocess version, which
        # starts there, but not for a library call. Re-point it at the absolute root
        # and re-derive the VAE paths exactly as test.py:72-75 does.
        opt.checkpoints_dir = CHECKPOINTS_DIR
        self._validate_branch(opt)
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_quality")
        return opt

    def _validate_branch(self, opt) -> None:
        """Refuse any branch but the quality one this backend implements.

        The scratch branch switches the networks, the mapping name and the
        non-local settings (test.py:76-90); loading quality weights under those
        settings would be quietly wrong rather than loudly wrong.
        """
        if not opt.Quality_restore or opt.Scratch_and_Quality_restore:
            raise ModelUnavailableError(
                f"{self.name} only implements the quality-restoration branch",
                details={
                    "backend": self.name,
                    "Quality_restore": bool(opt.Quality_restore),
                    "Scratch_and_Quality_restore": bool(opt.Scratch_and_Quality_restore),
                    "hint": "scratch repair is NativeScratchRepairBackend in this module",
                },
            )

    @staticmethod
    def _require_weights(opt) -> None:
        """Fail before loading if the branch's own weights are not where we say.

        ``base_model.load_network`` prints "…not exists yet" and carries on with
        randomly initialised networks, so a mistyped path produces a plausible
        looking image instead of an error. Checking the resolved paths is what
        turns that into a startup failure (plan §3.5.2 verify-then-load).
        """
        wanted = (
            (opt.load_pretrainA, "latest_net_G.pth"),
            (opt.load_pretrainB, "latest_net_G.pth"),
            (os.path.join(opt.checkpoints_dir, opt.name), "latest_net_mapping_net.pth"),
        )
        missing = [
            os.path.join(directory, filename)
            for directory, filename in wanted
            if not os.path.isfile(os.path.join(directory, filename))
        ]
        if missing:
            raise ModelUnavailableError(
                "Global restoration weights are missing; refusing to run on "
                "randomly initialised networks",
                details={"missing": missing, "hint": "make weights"},
            )

    def _do_load(self, device: str) -> None:
        available, reason = self._check_available()
        if not available:
            raise ModelUnavailableError(
                f"Native Global restoration is unavailable: {reason}",
                details={"backend": self.name},
            )
        _legacy_module()
        from models.mapping_model import Pix2PixHDModel_Mapping  # vendored package

        opt = self._build_opt(_device_index(device))
        self._require_weights(opt)
        model = Pix2PixHDModel_Mapping()
        try:
            model.initialize(opt)
        except Exception as exc:  # noqa: BLE001 — surface a broken tree as unavailable
            raise ModelUnavailableError(
                f"Global restoration weights failed to load: {exc}",
                details={"backend": self.name, "checkpoints": CHECKPOINTS_DIR},
            ) from exc
        model.eval()
        self._opt = opt
        # The declared precision optimisations are applied here, not in the
        # vendored loader: `channels_last` / `compile` are per-model decisions
        # (plan §3.5.4), and a backend that never applied them made the manifest
        # key decorative for this chain while /models still reported the model as
        # configured.
        self._model = self.apply_policy(model)

    def _do_warmup(self) -> None:
        """One tiny forward pass so the first real request is not the cold one."""
        from PIL import Image

        try:
            self._infer(Image.new("RGB", (64, 64), "black"))
        except Exception:  # noqa: BLE001 — warmup must never be fatal
            pass

    def _do_unload(self) -> None:
        self._model = None
        self._opt = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    # --------------------------------------------------------------- inference
    def _encode(self, tensor):
        """Turn the model's [-1,1] output into the image the CLI would have written.

        Deliberately the *same* call as ``Global/test.py:182`` —
        ``save_image(..., normalize=True)`` stretches by the tensor's own min/max
        before quantising to uint8, which is a per-image contrast change, not a
        clip. Writing to an in-memory buffer keeps the byte-for-byte behaviour
        without reintroducing a temp file per request.
        """
        import torchvision.utils as vutils
        from PIL import Image

        buffer = io.BytesIO()
        vutils.save_image(
            (tensor.data.cpu() + 1.0) / 2.0, buffer,
            format="png", nrow=1, padding=0, normalize=True,
        )
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def _infer(self, image):
        import torch
        import torchvision.transforms as transforms
        from PIL import Image

        legacy = _legacy_module()
        opt = self._opt

        # test_mode == "Full": round each side to a multiple of 4 and resize — the
        # only geometry change on this path (no crop, no tiling here).
        prepared = legacy.data_transforms(image, method=Image.Resampling.BILINEAR, scale=False)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        tensor = transform(prepared).unsqueeze(0)
        if opt.gpu_ids:
            tensor = tensor.cuda(opt.gpu_ids[0])
        mask = torch.zeros_like(tensor)

        # inference_context() is the repo-wide policy: torch.inference_mode()
        # always, autocast only when the manifest declares a dtype for this model
        # and the device is CUDA (the legacy CLI used torch.no_grad()).
        with inference_context(self.policy, self.device):
            generated = self._model.inference(tensor, mask)
        out = generated.data.cpu() if opt.gpu_ids else generated
        return self._encode(out), prepared

    def _do_infer(self, request: ModelRequest) -> ModelResult:
        if self._model is None:
            raise ModelUnavailableError(
                f"{self.name} is not loaded", details={"backend": self.name}
            )
        started = time.perf_counter()
        restored, prepared = self._infer(request.image.convert("RGB"))
        return ModelResult(
            image=restored,
            metadata={
                "model": "Global/triplet-domain-translation",
                "backend": self.implementation,
                "precision": self.policy.describe(),
                "duration_s": round(time.perf_counter() - started, 3),
                # The geometry the model actually saw — a size change here is a
                # change in output, so it belongs in the record.
                "model_input_size": [prepared.width, prepared.height],
            },
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
        """Same surface as :class:`LegacyCliBackend.run_folder`.

        The stages (and the tiling path) are folder-based because the pipeline they
        replaced was; keeping this bridge means switching backends is a selection
        decision rather than a rewrite of the stage. ``hr`` is accepted and ignored
        on the quality path — in the vendored code ``--HR`` only takes effect
        together with ``--Scratch_and_Quality_restore``.
        """
        target = restored_image_dir(output_dir)
        os.makedirs(target, exist_ok=True)
        names = list_images(input_dir)
        for position, name in enumerate(names, start=1):
            from PIL import Image

            path = os.path.join(input_dir, name)
            if not os.path.isfile(path):
                continue
            result = self.infer(ModelRequest(image=Image.open(path).convert("RGB"),
                                             device=str(gpu), options={"hr": hr}))
            if result.image is None:
                from fiximg.domain.errors import PipelineFailedError

                raise PipelineFailedError(
                    f"Global restoration returned no image for {name}",
                    details={"backend": self.implementation, "input": path},
                )
            # Legacy naming: whatever the extension, the stage output is PNG.
            out_name = (name[:-4] if name.lower().endswith(".jpg") else os.path.splitext(name)[0]) + ".png"
            result.image.save(os.path.join(target, out_name))
            if on_progress is not None:
                on_progress(position, len(names), "restore")
        return output_dir

    def produced_dir(self, output_dir: str) -> str:
        """Where this stage writes its result inside a pipeline root."""
        return restored_image_dir(output_dir)

    def health(self):  # noqa: D102 - see BaseModelBackend
        health = super().health()
        available, reason = self._check_available()
        health.extra = {
            "implementation": self.implementation,
            "available": available,
            "reason": reason or None,
            "checkpoints": CHECKPOINTS_DIR,
            "precision": self.policy.describe(),
        }
        return health

    def describe(self) -> dict:  # noqa: D102 - see BaseModelBackend
        info = super().describe()
        info["weights"] = [os.path.join(CHECKPOINTS_DIR, rel) for rel in QUALITY_WEIGHTS]
        return info


class NativeScratchRepairBackend(GlobalRestoreBackend):
    """Scratch-mask detection + joint scratch/quality restoration, resident.

    The same Global tree as the quality path, with one extra network in front: the
    scratch detector (``Global/detection.py``'s UNet) produces a binary mask, the
    image is hole-synthesised from it, and the triplet runs with the
    ``mapping_scratch`` / ``VAE_B_scratch`` weights and the non-local settings
    ``parameter_set`` selects. Both detectors and the triplet stay loaded, so a
    scratched photo no longer pays two interpreter starts and two weight reads.

    Intermediates are still written where the vendored script writes them
    (``stage_1_restore_output/masks/{input,mask}``), and the restoration step reads
    them back from there. That is deliberate: reusing the on-disk round-trip is
    what makes the output bit-identical to the subprocess version, and those
    directories are also what the UI shows as the scratch mask.
    """

    name = "scratch_repair"
    version = "1.0"
    implementation = "native"
    capabilities = frozenset({"restore", "scratch_repair"})

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._scratch_detector = None

    # ------------------------------------------------------------- lifecycle
    def _check_available(self) -> tuple[bool, str]:
        return scratch_native_available()

    def _validate_branch(self, opt) -> None:
        """The inverse of the quality branch's guard."""
        if not opt.Scratch_and_Quality_restore or opt.Quality_restore:
            raise ModelUnavailableError(
                f"{self.name} only implements the scratch-and-quality branch",
                details={
                    "backend": self.name,
                    "Scratch_and_Quality_restore": bool(opt.Scratch_and_Quality_restore),
                    "Quality_restore": bool(opt.Quality_restore),
                },
            )

    def _build_opt(self, device_index):
        legacy = _legacy_module()
        from options.test_options import TestOptions  # vendored package, Global/ on sys.path

        options = TestOptions()
        if not options.initialized:
            options.initialize()
        opt = options.parser.parse_args([
            "--Scratch_and_Quality_restore",
            f"--gpu_ids={device_index}",
            f"--checkpoints_dir={CHECKPOINTS_DIR}",
        ])
        opt.isTrain = options.isTrain
        opt.gpu_ids = [int(x) for x in str(opt.gpu_ids).split(",") if x.strip() and int(x) >= 0]
        # parameter_set() selects the non-local architecture switches and the
        # scratch checkpoints in one go (test.py:76-85) — including NL_use_mask,
        # which is what makes the mask reach the mapping network at all.
        legacy.parameter_set(opt)
        opt.checkpoints_dir = CHECKPOINTS_DIR
        self._validate_branch(opt)
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_scratch")
        return opt

    def _load_detector(self, device_index) -> None:
        _legacy_module()  # puts Global/ on sys.path for detection_models/detection_util
        import torch

        from detection_models import networks

        model = networks.UNet(
            in_channels=1,
            out_channels=1,
            depth=4,
            conv_num=2,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode="upsample",
            with_tanh=False,
            sync_bn=True,
            antialiasing=True,
        )
        checkpoint = torch.load(DETECTION_CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        # Place the model *after* construction. The vendored `UNet(sync_bn=True)`
        # wraps itself in DataParallel inside its own __init__ (networks.py:85), and
        # that wrapper moves the freshly built weights onto the GPU — so a worker
        # that asked for CPU kept a CUDA detector and then fed it a CPU tensor:
        # "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)
        # should be the same". Only a machine with CUDA can see this, which is why
        # it survived every CPU-only run of this file.
        target = "cpu" if device_index is None or int(device_index) < 0 else int(device_index)
        model.to(target)
        # The detector is part of this model's chain, so the declared optimisations
        # cover it too - and if it declines one, the record says the declaration is
        # not fully honoured rather than quoting the mapping net that accepted it.
        self._scratch_detector = self.apply_policy(model)

    def _do_load(self, device: str) -> None:
        super()._do_load(device)
        index = _device_index(device)
        self._load_detector(int(index) if isinstance(index, int) else -1)

    def _do_unload(self) -> None:
        self._scratch_detector = None
        super()._do_unload()

    # ------------------------------------------------------------ scratch mask
    def _detect_mask(self, image):
        """(transformed_image, mask_tensor) as ``Global/detection.py`` produces them.

        The preprocessing is that script's own ``data_transforms``/``scale_tensor``,
        and ``--input_size`` is ``full_size`` because that is what ``batch.py``
        passes — not the script's default.
        """
        import torch
        import torchvision as tv

        detection = _detection_module()
        transformed = detection.data_transforms(image, "full_size")

        tensor = tv.transforms.ToTensor()(transformed.convert("L"))
        tensor = tv.transforms.Normalize([0.5], [0.5])(tensor)
        tensor = torch.unsqueeze(tensor, 0)
        _, _, ow, oh = tensor.shape

        scaled = detection.scale_tensor(tensor)
        if self._scratch_detector is None:
            raise ModelUnavailableError(
                f"{self.name} is not loaded", details={"backend": self.name}
            )
        if self._opt is not None and self._opt.gpu_ids:
            scaled = scaled.to(self._opt.gpu_ids[0])
        with torch.no_grad():
            probability = torch.sigmoid(self._scratch_detector(scaled))
        probability = probability.data.cpu()
        return transformed, torch.nn.functional.interpolate(
            probability, [ow, oh], mode="nearest"
        )

    # --------------------------------------------------------------- inference
    def _write_intermediates(self, image, stem: str, mask_dir: str, input_dir: str):
        """Detect the scratch mask, then persist what the vendored script persisted.

        The mask goes out through the same ``save_image(..., normalize=True)`` call
        ``detection.py:152`` makes, and the restoration step reads both files back.
        Round-tripping them instead of threading tensors through is what keeps the
        result identical to the subprocess version — and the mask file is an
        artifact the UI shows, so it has to exist either way.
        """
        import torchvision.utils as vutils

        transformed, probability = self._detect_mask(image)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(input_dir, exist_ok=True)

        mask_path = os.path.join(mask_dir, stem + ".png")
        input_path = os.path.join(input_dir, stem + ".png")
        vutils.save_image(
            (probability >= 0.4).float(), mask_path, nrow=1, padding=0, normalize=True,
        )
        transformed.save(input_path)
        return input_path, mask_path

    def _restore_from_files(self, input_path: str, mask_path: str):
        """The mask branch of ``Global/test.py:137-167``, in-process.

        ``irregular_hole_synthesize`` and the single-channel mask slice
        (``mask[:1]``) come from the vendored code: whether the mask is applied to
        the image *before* normalisation, and which of its channels reach the
        mapping network, both change pixels.
        """
        import numpy as np
        import torchvision.transforms as transforms
        from PIL import Image

        legacy = _legacy_module()
        opt = self._opt
        if opt is None or self._model is None:
            raise ModelUnavailableError(
                f"{self.name} is not loaded", details={"backend": self.name}
            )

        image = Image.open(input_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        if getattr(opt, "mask_dilation", 0):
            import cv2

            kernel = np.ones((3, 3), np.uint8)
            mask = Image.fromarray(
                cv2.dilate(np.array(mask), kernel, iterations=opt.mask_dilation).astype("uint8")
            )

        holed = legacy.irregular_hole_synthesize(image, mask)
        normalise = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        input_tensor = normalise(holed).unsqueeze(0)
        mask_tensor = transforms.ToTensor()(mask)[:1].unsqueeze(0)
        if opt.gpu_ids:
            input_tensor = input_tensor.cuda(opt.gpu_ids[0])
            mask_tensor = mask_tensor.cuda(opt.gpu_ids[0])

        with inference_context(self.policy, self.device):
            generated = self._model.inference(input_tensor, mask_tensor)
        out = generated.data.cpu() if opt.gpu_ids else generated
        return self._encode(out), holed

    def _infer(self, image):
        """Scratch path for one image, through the same intermediates the folders use."""
        import shutil
        import tempfile

        stem = self.config.get("stem") or "image"
        scratch = tempfile.mkdtemp(prefix="fiximg_scratch_")
        try:
            input_path, mask_path = self._write_intermediates(
                image, stem, os.path.join(scratch, "mask"), os.path.join(scratch, "input")
            )
            restored, prepared = self._restore_from_files(input_path, mask_path)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        # ``prepared`` is the image the network actually consumed (mask applied),
        # so its size is what the stage should record as the model input.
        return restored, prepared

    # ---------------------------------------------------- folder bridge
    def run_folder(self, input_dir: str, output_dir: str, *, gpu: int = -1,
                   hr: bool = False, on_progress=None) -> str:
        """Stage 1 of the scratched-photo path, in one process.

        Lays out the directories ``batch.py`` uses —
        ``stage_1_restore_output/masks/{input,mask}`` plus ``restored_image`` — so
        no later stage can tell which implementation produced them.
        """
        from PIL import Image

        from fiximg.inference.backends.legacy_cli import restored_image_dir

        if hr:
            raise ModelUnavailableError(
                "Native scratch repair does not implement the HR branch: it selects "
                "mapping_Patch_Attention, whose weights are not shipped, and the "
                "vendored loader would silently run it uninitialised",
                details={"backend": self.name, "hr": True},
            )
        if self._model is None:
            self.load(str(gpu))

        names = list_images(input_dir)
        if not names:
            raise PipelineFailedError(
                f"Scratch repair received no input images: {input_dir} is empty.",
                details={"backend": self.name, "input_dir": input_dir},
            )

        restore_root = os.path.dirname(restored_image_dir(output_dir))
        mask_dir = os.path.join(restore_root, "masks", "mask")
        scratch_input = os.path.join(restore_root, "masks", "input")
        target = restored_image_dir(output_dir)
        os.makedirs(target, exist_ok=True)

        for position, name in enumerate(names, start=1):
            path = os.path.join(input_dir, name)
            if not os.path.isfile(path):
                continue
            stem = os.path.splitext(name)[0]
            input_path, mask_path = self._write_intermediates(
                Image.open(path).convert("RGB"), stem, mask_dir, scratch_input
            )
            restored, _ = self._restore_from_files(input_path, mask_path)
            restored.save(os.path.join(target, stem + ".png"))
            if on_progress is not None:
                on_progress(position, len(names), "scratch repair + quality restoration")
        return output_dir

    def health(self):  # noqa: D102 - see BaseModelBackend
        health = super().health()
        health.extra = dict(health.extra or {}, detector=DETECTION_CHECKPOINT)
        return health


__all__ = [
    "CHECKPOINTS_DIR",
    "DETECTION_CHECKPOINT",
    "NativeScratchRepairBackend",
    "QUALITY_WEIGHTS",
    "GlobalRestoreBackend",
    "legacy_tree_conflict",
    "native_available",
    "quality_weights_present",
    "scratch_native_available",
    "scratch_weights_present",
]
