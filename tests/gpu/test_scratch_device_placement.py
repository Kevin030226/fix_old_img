"""The scratch detector must land on the device the process asked for (搂3.5.4).

``Global/detection_models/networks.py:85`` wraps the UNet in ``DataParallel`` inside
its own ``__init__`` when ``sync_bn=True``, and that wrapper moves the freshly built
weights to the GPU. So a worker started with ``FIXIMG_DEVICE=cpu`` (``gpu=-1``) got a
CUDA detector while the mask step still built its input on CPU, and the first
inference died with "Input type (torch.FloatTensor) and weight type
(torch.cuda.FloatTensor) should be the same". Every CPU-only run of this repository is
blind to it 鈥?the tensors agree by accident 鈥?which is why this file is marked
``gpu``: the assertion only has teeth where a GPU exists to refuse.
"""
import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="needs a CUDA device: without one there is no second device to be put "
               "on by mistake, so the assertion cannot fail",
    ),
]


def _scratch_backend():
    from fiximg.inference.backends.global_restore import NativeScratchRepairBackend

    available, reason = NativeScratchRepairBackend({"stem": "d"})._check_available()
    if not available:
        pytest.skip(f"scratch weights are absent: {reason}")
    return NativeScratchRepairBackend({"stem": "d"})


def _detector_device(backend):
    return next(backend._scratch_detector.parameters()).device


def test_a_cpu_request_keeps_the_whole_chain_on_cpu(tmp_path):
    """Asking for CPU must not reserve VRAM, and the mask step must still run."""
    from PIL import Image

    backend = _scratch_backend()
    backend.load("-1")
    assert _detector_device(backend).type == "cpu", _detector_device(backend)

    _image, mask = backend._detect_mask(Image.new("RGB", (64, 64), "white"))
    assert tuple(mask.shape) == (1, 1, 64, 64), mask.shape


def test_a_gpu_request_puts_the_detector_on_that_gpu():
    from PIL import Image

    backend = _scratch_backend()
    backend.load("0")
    assert _detector_device(backend) == torch.device("cuda:0"), _detector_device(backend)

    _image, mask = backend._detect_mask(Image.new("RGB", (64, 64), "white"))
    assert tuple(mask.shape) == (1, 1, 64, 64), mask.shape


def test_auto_follows_the_manifest_device_not_the_constructor_side_effect():
    """`auto` on a CUDA host means CUDA 鈥?the same answer the main model gives."""
    backend = _scratch_backend()
    backend.load("auto")
    detector = _detector_device(backend)
    main = next(backend._model.parameters()).device
    assert detector == main, f"detector {detector} vs restoration model {main}"
