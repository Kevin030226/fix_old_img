"""Real-model golden baseline for Global quality restoration (plan 搂3.10.2).

The synthetic stage in ``tests/inference/test_golden_image.py`` protects the
plumbing. This protects the **model output**: it runs the resident native backend
on a fixed input and pins the resulting characteristics 鈥?mode, size, shape,
dynamic range, mean/std, PSNR/SSIM against the input.

Why characteristics and not pixels: the plan says so, and because
``cudnn.benchmark=True`` lets different machines pick different CUDA kernels. On
CPU the run is deterministic (the equivalence suite proves native == legacy
byte-for-byte), so this baseline is tight; on GPU the tolerances in
``assert_matches_golden`` absorb kernel differences while still catching a real
quality shift 鈥?a swapped checkpoint, a changed normalisation, a dropped
contrast stretch.

Needs the Global quality weights, hence the ``gpu`` marker. Refresh deliberately
with ``FIXIMG_UPDATE_GOLDEN=1`` and review the JSON diff: this file is committed
data describing what the product is supposed to produce.
"""
import pytest
from PIL import Image

from fiximg.inference.backends.base import ModelRequest
from fiximg.inference.backends.global_restore import (
    CHECKPOINTS_DIR,
    GlobalRestoreBackend,
    quality_weights_present,
)
from tests.fixtures.golden_kit import (
    GOLDEN_FIELDS,
    assert_matches_golden,
    characteristics,
    load_golden,
    make_input,
    save_golden,
    update_requested,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not quality_weights_present(),
                       reason=f"Global quality weights are absent under {CHECKPOINTS_DIR}"),
]

BASELINE = "global_quality"


@pytest.fixture(scope="module")
def restored():
    """One fixed input, restored once per module 鈥?the model run is the cost."""
    source = make_input(seed=7, size=(96, 64))
    backend = GlobalRestoreBackend()
    backend.load("cpu")
    result = backend.infer(ModelRequest(image=source, device="cpu"))
    assert result.image is not None
    return source, result


def test_real_model_output_matches_the_golden_baseline(restored):
    source, result = restored
    actual = characteristics(result.image, source)

    if update_requested():
        save_golden(BASELINE, actual)
        pytest.skip(f"golden baseline refreshed: {BASELINE}.json")

    expected = load_golden(BASELINE)
    assert_matches_golden(actual, expected)


def test_the_baseline_records_every_planned_field(restored):
    """Guard the assertion surface: a dropped field is a silent gap."""
    source, result = restored
    actual = characteristics(result.image, source)
    assert set(GOLDEN_FIELDS) <= set(actual), sorted(set(GOLDEN_FIELDS) - set(actual))
    assert set(GOLDEN_FIELDS) <= set(load_golden(BASELINE))


def test_the_output_geometry_is_the_pipelines_rounding(restored):
    """96脳64 is already a multiple of 4, so nothing should move.

    Pinned because geometry is the one characteristic a size policy or a resize
    change would silently alter, and the mean/std tolerances would not catch it.
    """
    _, result = restored
    assert result.image.size == (96, 64)
    assert result.image.mode == "RGB"


def test_the_restoration_actually_changed_the_image(restored):
    """A baseline captured from a no-op model would pass forever.

    PSNR vs the input must stay finite and in a band: too high means the network
    did nothing (random weights would still look "stable" against its own
    baseline), and the identity check below makes that impossible.
    """
    import numpy as np

    source, result = restored
    assert not np.array_equal(np.asarray(source), np.asarray(result.image))
    metrics = characteristics(result.image, source)
    assert 8.0 < metrics["psnr_vs_input"] < 60.0
    assert 0.0 < metrics["ssim_vs_input"] <= 1.0


def test_the_baseline_is_attributable_to_a_model_version(restored):
    """Without this, a weight swap looks like an unexplained quality drift."""
    _, result = restored
    assert result.metadata["model"] == "Global/triplet-domain-translation"
    assert result.metadata["backend"] == "native"


def test_repeated_runs_on_one_instance_are_reproducible():
    """The resident model must not drift between requests."""
    import numpy as np

    source = make_input(seed=7, size=(96, 64))
    backend = GlobalRestoreBackend()
    backend.load("cpu")
    first = np.asarray(backend.infer(ModelRequest(image=source, device="cpu")).image)
    second = np.asarray(backend.infer(ModelRequest(image=source, device="cpu")).image)
    assert np.array_equal(first, second)
    assert Image.fromarray(first).size == (96, 64)
