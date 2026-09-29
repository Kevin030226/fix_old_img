"""Golden-image quality regression tests (plan 搂3.10.2).

The measurement code lives in :mod:`tests.fixtures.golden_kit` so that these
synthetic baselines and the real-model baselines in ``tests/gpu`` compare the same
numbers. Everything here runs in CI: the stage under test is a deterministic
stand-in, so no weights and no GPU are needed.
"""
import dataclasses
import io
import os

import numpy as np
import pytest
from PIL import Image

from fiximg.inference.context import StageResult
from fiximg.inference.stages.base import BaseStage
from tests.fixtures.golden_kit import (
    GOLDEN_DIR,
    GOLDEN_FIELDS,
    assert_matches_golden,
    characteristics,
    load_golden,
    make_input,
    psnr,
    save_golden,
    ssim,
)



# --------------------------------------------------------------- the fixture
class _SyntheticGoldenStage(BaseStage):
    """A deterministic stand-in for the restoration model.

    It reproduces the *shape* of a real restoration: mild denoise, contrast
    lift, slight warm shift. That is enough for the golden assertions to have
    teeth 鈥?a change in the runtime's image plumbing (channel order, dtype,
    resize) shows up immediately.
    """

    name = "global_restore"
    version = "golden-1.0"
    capabilities = frozenset({"restore"})

    def run(self, image, context):
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        blurred = (array + np.roll(array, 1, axis=0) + np.roll(array, 1, axis=1)) / 3.0
        lifted = np.clip((blurred - 128.0) * 1.08 + 132.0, 0, 255)
        warmed = lifted.copy()
        warmed[..., 0] = np.clip(warmed[..., 0] * 1.01, 0, 255)
        warmed[..., 2] = np.clip(warmed[..., 2] * 0.99, 0, 255)
        return StageResult(
            image=Image.fromarray(warmed.astype(np.uint8), "RGB"),
            metadata={"model": "synthetic-golden"},
            metrics={"golden_stage": 1},
        )


@pytest.fixture()
def golden(tmp_path, monkeypatch):
    """Run a plan through the orchestrator with the synthetic golden stage."""
    from fiximg.inference import registry as registry_module

    monkeypatch.setattr(
        registry_module, "stage_registry",
        _stubbed(registry_module, _SyntheticGoldenStage),
    )
    monkeypatch.setattr("fiximg.config.settings.tasks_root",
                        str(tmp_path / "storage" / "tasks"))
    return tmp_path


def _stubbed(registry_module, restore_stage):
    registry = registry_module.build_default_registry()
    registry.register("global_restore", restore_stage, replace=True)
    registry.register("scratch_repair", restore_stage, replace=True)
    for name, caps in (
        ("face_detection", {"face_detection"}),
        ("face_enhancement", {"face_restore", "face_enhance"}),
        ("warp_back", {"warp_back", "face_composite"}),
    ):
        registry.register(name, _PassThrough, replace=True, capabilities=caps)
    return registry


class _PassThrough(BaseStage):
    """Face-chain stand-in: returns the image untouched."""

    name = "face_detection"
    capabilities = frozenset({"face_detection"})

    def run(self, image, context):
        return StageResult(image=image, metadata={"stub": True})


# ------------------------------------------------------------------- the tests
def test_golden_input_is_deterministic():
    """A golden test is only useful if the input never moves."""
    first = np.asarray(make_input(), dtype=np.int64)
    second = np.asarray(make_input(), dtype=np.int64)
    assert np.array_equal(first, second)


def test_golden_input_has_the_expected_character():
    image = make_input()
    assert image.mode == "RGB"
    assert image.size == (96, 64)
    stats = characteristics(image)
    assert 0 <= stats["dtype_range"][0]
    assert stats["dtype_range"][1] <= 255
    # A gradient with grain: not flat, not saturated.
    assert 10.0 < stats["std"] < 120.0


def test_synthetic_stage_output_matches_its_golden_fingerprint(golden):
    """The end-to-end characteristic set is stable (plan 搂3.10.2)."""
    from fiximg.inference.context import StageContext

    source = make_input()
    context = StageContext(task_id="golden", run_dir=str(golden), options={})
    result = _SyntheticGoldenStage().run(source, context)

    actual = characteristics(result.image, source)

    if os.environ.get("FIXIMG_UPDATE_GOLDEN"):
        save_golden("synthetic_restore", actual)
        pytest.skip("golden baseline refreshed")

    expected = load_golden("synthetic_restore")
    assert set(expected) <= set(actual) | {"psnr_vs_input", "ssim_vs_input"}
    assert_matches_golden(actual, expected)


def test_golden_fingerprint_covers_every_planned_field(golden):
    """Guard the assertion surface itself: a dropped field is a silent gap."""
    from fiximg.inference.context import StageContext

    source = make_input()
    actual = characteristics(_SyntheticGoldenStage().run(
        source, StageContext(task_id="g", run_dir=str(golden), options={})
    ).image, source)
    for field in GOLDEN_FIELDS:
        assert field in actual, f"golden fingerprint is missing {field}"


def test_stage_result_reports_metrics_for_the_artifact_schema(golden):
    """plan 搂3.10.2 asks for an artifact-schema assertion too."""
    from fiximg.inference.context import StageContext

    result = _SyntheticGoldenStage().run(
        make_input(), StageContext(task_id="g", run_dir=str(golden), options={})
    )
    assert result.metrics == {"golden_stage": 1}
    assert result.metadata["model"] == "synthetic-golden"


def test_restoration_preserves_size_and_mode(golden):
    """The most common regression: a resize or channel swap slipping in."""
    from fiximg.inference.context import StageContext

    source = make_input(size=(120, 90))
    result = _SyntheticGoldenStage().run(
        source, StageContext(task_id="g", run_dir=str(golden), options={})
    )
    assert result.image.size == source.size
    assert result.image.mode == "RGB"


def test_restoration_stays_within_the_expected_quality_band(golden):
    """The output must stay close to the input: a wild shift is a bug, not a fix."""
    from fiximg.inference.context import StageContext

    source = make_input()
    result = _SyntheticGoldenStage().run(
        source, StageContext(task_id="g", run_dir=str(golden), options={})
    )
    assert psnr(source, result.image) > 20.0
    assert ssim(source, result.image) > 0.5


def test_characteristics_helper_is_sensitive_to_a_real_change(golden):
    """Sanity check: the fingerprint must actually notice differences."""
    source = make_input()
    baseline = characteristics(source)
    shifted = characteristics(Image.fromarray(
        np.clip(np.asarray(source, dtype=np.int32) + 20, 0, 255).astype(np.uint8), "RGB"
    ))
    assert abs(shifted["mean"] - baseline["mean"]) > 5.0
    assert psnr(source, Image.fromarray(
        np.clip(np.asarray(source, dtype=np.int32) + 20, 0, 255).astype(np.uint8), "RGB"
    )) < 30.0


def test_ssim_of_an_image_with_itself_is_one():
    source = make_input()
    assert ssim(source, source) == pytest.approx(1.0, abs=1e-6)
    assert psnr(source, source) == float("inf")


def test_golden_baseline_file_is_committed_and_readable():
    """The baseline must exist in the repo, not be generated on the fly."""
    path = os.path.join(GOLDEN_DIR, "synthetic_restore.json")
    assert os.path.exists(path), (
        "missing golden baseline; run "
        "FIXIMG_UPDATE_GOLDEN=1 pytest tests/inference/test_golden_image.py"
    )
    baseline = load_golden("synthetic_restore")
    assert baseline["mode"] == "RGB"
    assert baseline["size"] == [96, 64]


def test_golden_baseline_roundtrip(tmp_path, monkeypatch):
    """Refreshing the baseline must be reproducible."""
    payload = {"mode": "RGB", "size": [8, 8], "mean": 12.5}
    save_golden("_tmp_roundtrip", payload)
    try:
        assert load_golden("_tmp_roundtrip") == payload
    finally:
        os.remove(os.path.join(GOLDEN_DIR, "_tmp_roundtrip.json"))


def test_artifact_schema_of_a_persisted_result(golden):
    """The artifact record has the fields the report's schema promises."""
    from fiximg.domain.artifacts import build_artifact
    from fiximg.inference.context import StageContext

    result = _SyntheticGoldenStage().run(
        make_input(), StageContext(task_id="g", run_dir=str(golden), options={})
    )
    path = os.path.join(str(golden), "out.png")
    result.image.save(path)

    artifact = build_artifact("golden", "output", path, "image/png")
    payload = dataclasses.asdict(artifact)
    for field in ("task_id", "kind", "uri", "mime_type", "width", "height",
                  "size_bytes", "sha256", "created_at"):
        assert field in payload, field
    assert payload["width"] == 96 and payload["height"] == 64
    assert len(payload["sha256"]) == 64


def test_png_bytes_are_stable_for_the_same_pixels(golden):
    """A golden artifact must not change just because it was re-encoded."""
    from fiximg.inference.context import StageContext

    result = _SyntheticGoldenStage().run(
        make_input(), StageContext(task_id="g", run_dir=str(golden), options={})
    )
    first = io.BytesIO()
    second = io.BytesIO()
    result.image.save(first, format="PNG")
    result.image.save(second, format="PNG")
    assert first.getvalue() == second.getvalue()
