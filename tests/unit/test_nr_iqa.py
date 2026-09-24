"""Unit tests for the optional no-reference IQA metrics (plan §16 additions)."""
import numpy as np
import pytest

from app.services import nr_iqa_service as nr


@pytest.fixture()
def natural_photo(tmp_path):
    """Smooth colour-gradient photo with mild noise (naturalness-like)."""
    rng = np.random.default_rng(3)
    base = np.tile(np.linspace(40, 220, 120).astype(np.uint8), (90, 1))
    rgb = np.stack([base, np.roll(base, 15, axis=1), np.roll(base, 30, axis=1)], axis=-1)
    rgb = np.clip(rgb.astype(int) + rng.integers(-8, 9, rgb.shape), 0, 255).astype(np.uint8)
    path = tmp_path / "nat.png"
    _save(rgb, path)
    return str(path)


@pytest.fixture()
def flat_photo(tmp_path):
    path = tmp_path / "flat.png"
    _save(np.full((90, 120, 3), 128, dtype=np.uint8), path)
    return str(path)


def _save(rgb, path):
    from PIL import Image

    Image.fromarray(rgb).save(path)


# ------------------------------------------------------------------- NIQE
def test_niqe_prefers_natural_over_flat(natural_photo, flat_photo):
    nat = nr.compute_nr_iqa(natural_photo)["niqe"]
    flat = nr.compute_nr_iqa(flat_photo)["niqe"]
    assert nat < flat


def test_niqe_is_deterministic(natural_photo):
    a = nr.compute_nr_iqa(natural_photo)["niqe"]
    b = nr.compute_nr_iqa(natural_photo)["niqe"]
    assert a == b


def test_niqe_finite_and_positive(flat_photo):
    value = nr.compute_nr_iqa(flat_photo)["niqe"]
    assert np.isfinite(value) and value >= 0


# ------------------------------------------------------------- MSCN internals
def test_mscn_zero_mean_unit_variance():
    rng = np.random.default_rng(5)
    gray = rng.normal(128, 30, size=(64, 64))
    m = nr._mscn(gray)
    assert abs(m.mean()) < 0.15
    assert 0.4 < m.std() < 1.6


def test_aggd_params_stable_on_constant_input():
    result = nr._aggd_params(np.zeros((32, 32)))
    assert all(np.isfinite(v) for v in result)


# --------------------------------------------------------- Color statistics
def test_color_statistics_greyish_image(flat_photo):
    from PIL import Image
    import numpy as np

    rgb = np.asarray(Image.open(flat_photo).convert("RGB"))
    color = nr.compute_color_statistics(rgb)
    assert color["mean_saturation"] < 0.05
    assert color["colorfulness"] < 5


def test_color_statistics_saturated_image(tmp_path):
    rgb = np.zeros((60, 60, 3), dtype=np.uint8)
    rgb[:, :, 0] = 220  # strong red
    path = tmp_path / "red.png"
    _save(rgb, path)
    from PIL import Image

    color = nr.compute_color_statistics(np.asarray(Image.open(path).convert("RGB")))
    assert color["mean_saturation"] > 0.8
    assert color["dominant_hue_deg"] == 0


# ------------------------------------------------------------------ facade
def test_facade_returns_all_blocks(natural_photo):
    result = nr.compute_nr_iqa(natural_photo)
    assert {"niqe", "color", "naturalness", "brisque_note", "notice"} <= set(result)
    assert "⚠" in result["notice"]


def test_facade_unreadable_returns_none(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    assert nr.compute_nr_iqa(str(bad)) is None


def test_report_contains_notice(natural_photo):
    text = nr.format_nr_iqa_report(nr.compute_nr_iqa(natural_photo))
    assert "NIQE" in text and "Colorfulness" in text and "⚠" in text


def test_reference_type_distinct():
    from app.services.gt_metrics_service import REFERENCE_TYPE as GT
    from app.services.metrics_service import REFERENCE_TYPE as NR

    assert len({nr.REFERENCE_TYPE, GT, NR}) == 3
