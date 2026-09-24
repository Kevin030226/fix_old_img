"""Unit tests for the input-output difference evaluation service (plan section 16)."""

import numpy as np
from PIL import Image

from app.services.evaluation_service import (
    calculate_difference_metrics,
    format_difference_report,
    degrade_note_from_metadata,
)


def _write(tmp_path, name, arr):
    path = str(tmp_path / name)
    Image.fromarray(arr).save(path)
    return path


def test_identical_images_infinite_psnr(tmp_path):
    arr = np.full((32, 32, 3), 128, dtype=np.uint8)
    a = _write(tmp_path, "a.png", arr)
    b = _write(tmp_path, "b.png", arr)
    metrics = calculate_difference_metrics(a, b)
    assert metrics["identical"] is True
    assert metrics["psnr"] == float("inf")
    report = format_difference_report(metrics)
    assert "PSNR: ∞" in report
    assert "pixel-identical" in report


def test_different_images_metrics(tmp_path):
    base = np.full((64, 64, 3), 100, dtype=np.uint8)
    modified = base.copy()
    modified[:16, :, :] = 200
    a = _write(tmp_path, "a.png", base)
    b = _write(tmp_path, "b.png", modified)
    metrics = calculate_difference_metrics(a, b)
    assert metrics["identical"] is False
    assert 0 < metrics["psnr"] < 60
    assert 0 < metrics["ssim"] <= 1
    assert 0 < metrics["mae"] < 1
    report = format_difference_report(metrics)
    assert "PSNR:" in report and "SSIM:" in report and "MAE:" in report


def test_degrade_note_translation():
    assert degrade_note_from_metadata(
        {"report": {"degraded_count": 1, "degrade_reason": "no_face_detected"}}
    ) == "No face detected; face enhancement skipped, result is overall quality restoration only."
    assert degrade_note_from_metadata({"report": {"degraded_count": 0}}) is None
    assert degrade_note_from_metadata({}) is None


def test_grayscale_input_resized(tmp_path):
    gray = np.full((20, 20), 90, dtype=np.uint8)
    rgb = np.full((20, 20, 3), 90, dtype=np.uint8)
    a = _write(tmp_path, "gray.png", gray)
    b = _write(tmp_path, "rgb.png", rgb)
    metrics = calculate_difference_metrics(a, b)
    assert metrics["identical"] is True
