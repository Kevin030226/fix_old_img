"""Unit tests for no-reference quality metrics (plan §16)."""
import numpy as np
import pytest
from PIL import Image

from app.services.metrics_service import (
    LIMITATION_NOTICE,
    REFERENCE_TYPE,
    compute_no_reference_metrics,
    format_no_reference_report,
)


@pytest.fixture()
def smooth_png(tmp_path):
    """Flat gray image: low sharpness, low noise."""
    p = str(tmp_path / "smooth.png")
    Image.new("RGB", (64, 64), (128, 128, 128)).save(p)
    return p


@pytest.fixture()
def textured_png(tmp_path):
    """Dense random noise: high sharpness, high contrast, high noise."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    p = str(tmp_path / "textured.png")
    Image.fromarray(arr).save(p)
    return p


def test_returns_none_for_unreadable_path(tmp_path):
    assert compute_no_reference_metrics(str(tmp_path / "missing.png")) is None


def test_keys_and_reference_type(smooth_png):
    m = compute_no_reference_metrics(smooth_png)
    assert set(m) == {"sharpness", "contrast", "brightness", "noise"}
    assert REFERENCE_TYPE == "no_reference"


def test_flat_image_has_low_sharpness_and_noise(smooth_png):
    m = compute_no_reference_metrics(smooth_png)
    assert m["sharpness"] < 1.0
    assert m["noise"] < 1.0
    assert abs(m["brightness"] - 128) < 1.0


def test_textured_image_scores_higher(smooth_png, textured_png):
    a = compute_no_reference_metrics(smooth_png)
    b = compute_no_reference_metrics(textured_png)
    assert b["sharpness"] > a["sharpness"]
    assert b["contrast"] > a["contrast"]
    assert b["noise"] > a["noise"]


def test_report_contains_notice_and_values(smooth_png):
    m = compute_no_reference_metrics(smooth_png)
    text = format_no_reference_report(m)
    assert LIMITATION_NOTICE in text
    assert str(m["sharpness"]) in text
