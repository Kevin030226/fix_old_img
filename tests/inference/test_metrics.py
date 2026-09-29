"""Unit tests for the input-output difference evaluation service (plan section 16).

Two of these exist because a *reference* metric is only worth reporting if the
number means what its label says, and both ways of being wrong here look
perfectly healthy in a report:

* SSIM computed ``sigma2_sq`` as ``E[y^2] - mu_x * mu_y`` — the cross term
  belongs to ``sigma12``, so the correct term is ``E[y^2] - mu_y^2``. The error
  pushed a real restoration down to 0.099 (true 0.652) and pushed unrelated noise
  up to 0.0875 (true 0.0093): it failed in both directions at once, and neither
  was visible without a second implementation to compare against.
* the face chain's own ``face_count`` and this metric's detector count shared a
  key, so a report could say "1 face enhanced" beside a stage list reporting that
  no face was processed.
"""
import numpy as np
import pytest
from PIL import Image

from fiximg.inference.evaluation.reference import (
    _calculate_ssim,
    _read_image,
    calculate_difference_metrics,
    degrade_note_from_metadata,
    format_difference_report,
)


def _write(tmp_path, name, arr):
    path = str(tmp_path / name)
    Image.fromarray(arr).save(path)
    return path


def _textured(size=96, seed=1):
    """A deterministic, structured image.

    Flat fields make SSIM degenerate (every variance term is zero), so the
    comparison needs real local structure.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    base = (128 + 40 * np.sin(x / 3.0) * np.cos(y / 5.0)).astype(np.float64)
    return np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)


def _rgb(arr):
    return np.stack([arr] * 3, -1)


def _shift(arr, amount, seed=None):
    if seed is None:
        noise = np.zeros(arr.shape, dtype=np.int16)
    else:
        noise = np.random.default_rng(seed).normal(0, amount, arr.shape)
    return np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ------------------------------------------------------- the three base metrics
def test_identical_images_infinite_psnr(tmp_path):
    arr = np.full((32, 32, 3), 128, dtype=np.uint8)
    metrics = calculate_difference_metrics(
        _write(tmp_path, "a.png", arr), _write(tmp_path, "b.png", arr)
    )
    assert metrics["identical"] is True
    assert metrics["psnr"] == float("inf")
    report = format_difference_report(metrics)
    assert "PSNR: ∞" in report
    assert "pixel-identical" in report


def test_different_images_metrics(tmp_path):
    base = np.full((64, 64, 3), 100, dtype=np.uint8)
    modified = base.copy()
    modified[:16, :, :] = 200
    metrics = calculate_difference_metrics(
        _write(tmp_path, "a.png", base), _write(tmp_path, "b.png", modified)
    )
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
    """A grayscale input and its RGB twin are the same picture."""
    gray = np.full((20, 20), 90, dtype=np.uint8)
    rgb = np.full((20, 20, 3), 90, dtype=np.uint8)
    metrics = calculate_difference_metrics(
        _write(tmp_path, "gray.png", gray), _write(tmp_path, "rgb.png", rgb)
    )
    assert metrics["identical"] is True


# ------------------------------------------------------------------ SSIM itself
def test_ssim_matches_scikit_image_on_a_textured_pair(tmp_path):
    """The platform's SSIM must equal the reference implementation's.

    The tolerance is 1e-3, and it is the *border convention* that needs it: the
    platform fills the 5-pixel frame with `filter2D`'s reflected edge, while
    scikit-image crops to the region the window fully covers. Measured on a
    textured pair the two differ by ~2e-4. That is three orders of magnitude
    below the tolerance and two below the sign error this test exists to catch
    (which moved the value by 0.55, and moved *noise* upward by 0.078), so the
    tolerance separates "correct formula" from "wrong formula" without pinning
    a border detail the report never claims.
    """
    pytest.importorskip("skimage.metrics", reason="scikit-image is the SSIM reference")

    from skimage.metrics import structural_similarity

    a = _textured(seed=1)
    b = _shift(a, 8, seed=2)
    ours = _calculate_ssim(
        _read_image(_write(tmp_path, "a.png", _rgb(a))),
        _read_image(_write(tmp_path, "b.png", _rgb(b))),
    )
    theirs = structural_similarity(
        a, b, gaussian_weights=True, sigma=1.5, data_range=255
    )
    assert ours == pytest.approx(theirs, abs=1e-3), (ours, theirs)


def test_ssim_ranks_a_restoration_above_unrelated_noise(tmp_path):
    """The ordering, not just the value.

    A formula that is wrong in the negative direction still preserves ordering
    often enough to pass a smoke test, so assert the ranking on a case where a
    sign error in the variance term flips it: two images at a *similar* mean
    distance from the reference, one of which is a mild restoration.
    """
    base = _textured(seed=3)
    mild = _shift(base, 3)
    noise = _shift(base, 18, seed=4)

    restored = calculate_difference_metrics(
        _write(tmp_path, "a.png", _rgb(base)), _write(tmp_path, "mild.png", _rgb(mild))
    )
    unrelated = calculate_difference_metrics(
        _write(tmp_path, "a2.png", _rgb(base)),
        _write(tmp_path, "noise.png", _rgb(noise)),
    )
    assert restored["ssim"] > unrelated["ssim"], (restored["ssim"], unrelated["ssim"])
    assert restored["ssim"] > 0.9, "a 3-level shift on a textured image scores high"


def test_ssim_of_unrelated_images_stays_low(tmp_path):
    """The negative direction of the same bug.

    The sign error raised this pair to 0.0875; the true value is 0.0316. A high
    score for two unrelated pictures is a false *positive*, and it is the half of
    the bug that "a good restoration scores well" cannot see.

    Uniform noise, not noise added to a structured image: the latter keeps the
    underlying sinusoid and legitimately scores ~0.29 at moderate amplitude
    (measured), which would make the threshold a statement about the test data
    rather than about the formula.
    """
    a = np.random.default_rng(1).integers(0, 255, (96, 96)).astype(np.uint8)
    b = np.random.default_rng(2).integers(0, 255, (96, 96)).astype(np.uint8)
    metrics = calculate_difference_metrics(
        _write(tmp_path, "a.png", _rgb(a)), _write(tmp_path, "b.png", _rgb(b))
    )
    assert metrics["ssim"] < 0.05, metrics["ssim"]


def test_tiny_images_fall_back_to_global_statistics_and_stay_bounded(tmp_path):
    """Below the 11x11 Gaussian window the whole-image form has to be used.

    Not only must it not crash: the result has to stay inside [-1, 1]. A missing
    clamp or a negative denominator shows up as a score outside the range, which
    every consumer would then have to defend against.
    """
    for side in (2, 4, 8, 10):
        metrics = calculate_difference_metrics(
            _write(tmp_path, f"a{side}.png", np.full((side, side), 90, np.uint8)),
            _write(tmp_path, f"b{side}.png", np.full((side, side), 140, np.uint8)),
        )
        assert -1.0 <= metrics["ssim"] <= 1.0, (side, metrics["ssim"])
        assert 0.0 <= metrics["mae"] <= 1.0, (side, metrics["mae"])


# ------------------------------------------------- which task types report them
def test_colorize_still_reports_no_difference_metrics():
    """`colorize` has no degraded original to compare against.

    Its input *is* the reference, so an input/output difference would be "how
    much did colourising change the picture" — not a restoration quality number.
    Comparing a scratch mask against the photo is not a difference either.
    """
    from fiximg.inference.runtime import _EVALUATED_TYPES

    assert "colorize" not in _EVALUATED_TYPES
    assert "detect_scratch" not in _EVALUATED_TYPES


def test_auto_restore_reports_the_same_difference_metrics_as_restore():
    """`auto_restore` runs the same physical pipeline, so it reports the same block.

    It used to be excluded, on the reasonable-sounding ground that an
    auto-colourised run compares a colour picture against a grayscale original.
    The correct answer is to report the numbers *and* say so — excluding the type
    threw away the restoration numbers for every non-grayscale auto_restore run
    as collateral. `options={"auto_colorize": true}` gives an ordinary `restore`
    the same shape, so the decision has to be per-plan, not per-entry-point.
    """
    from fiximg.inference.runtime import _EVALUATED_TYPES

    assert "auto_restore" in _EVALUATED_TYPES
    assert "restore" in _EVALUATED_TYPES


def test_a_colorizing_run_says_the_difference_includes_color():
    """The note is what makes a low auto_restore score interpretable."""
    from fiximg.inference.runtime import _colorization_note

    note = _colorization_note([{"stage": "global_restore"}, {"stage": "colorization"}])
    assert note is not None
    assert "coloriz" in note.lower()
    # Stated as a caveat, not as a defect: the numbers stay, and are explained.
    assert "not by itself a restoration regression" in note
    assert _colorization_note([{"stage": "global_restore"}]) is None
    # Presence is decided by the stages that actually ran, not by the entry point.
    assert _colorization_note([]) is None
    assert _colorization_note([None, {"stage": "colorization"}]) is not None


# ------------------------------------------------------- notes from the stages
@pytest.fixture()
def evaluated(tmp_path, isolated_db, monkeypatch):
    """Run `PipelineOrchestrator._evaluate` against real files and a stubbed DB.

    Returns a runner plus two views of what it produced: the notes the report was
    rendered with, and the metrics dict `_evaluate` returns. The second view is
    load-bearing — the namespaced face counts go into the returned dict, not into
    the metrics table, so a spy on `add_metric` alone cannot see a regression that
    renames them.
    """
    from fiximg.inference import runtime as rt

    image = _write(tmp_path, "img.png", _rgb(_textured(size=32, seed=9)))
    other = _write(tmp_path, "other.png", _rgb(_shift(_textured(32, 9), 6)))
    recorded: dict[str, object] = {}
    returned: dict[str, object] = {}

    real_add = rt.task_repo.add_metric
    real_format = rt.format_difference_report
    notes_seen: list = []

    def spy_add(task_id, name, value, reference_type="input_output_difference"):
        recorded[name] = value
        return real_add(task_id, name, value, reference_type)

    def spy_format(diff, notes=None):
        notes_seen.extend(notes or [])
        return real_format(diff, notes)

    monkeypatch.setattr(rt.task_repo, "add_metric", spy_add)
    monkeypatch.setattr(rt, "format_difference_report", spy_format)
    monkeypatch.setattr(rt, "compute_no_reference_metrics", lambda *a, **k: None)
    monkeypatch.setattr(rt, "compute_nr_iqa", lambda *a, **k: None)

    def run(task_id, task_type, stage_meta, identity=None):
        orch = rt.PipelineOrchestrator.__new__(rt.PipelineOrchestrator)
        orch._identity_metrics = lambda *a, **k: identity
        # `_evaluate` returns `(evaluation_text, metrics)`; the second element is
        # where the face-count collision happened, and the only place it is
        # observable — the namespaced counts are never written to the table.
        _text, metrics = orch._evaluate(task_id, task_type, image, other, stage_meta)
        returned.clear()
        returned.update(metrics or {})
        return notes_seen

    run.recorded = recorded          # type: ignore[attr-defined]
    run.returned = returned          # type: ignore[attr-defined]
    yield run, recorded, returned
    monkeypatch.undo()


def test_an_earlier_stages_reason_survives_a_later_silence(evaluated):
    """The first stage with something to say wins, not the last one inspected.

    The candidate list is ``warp_back``, ``global_restore``, ``scratch_repair``
    in plan order, and ``warp_back`` is *last* in an auto_restore plan. When it
    was skipped it reported nothing, and a "last writer wins" loop erased the
    reason ``global_restore`` had already given — a latent defect that only bites
    on the plan shape auto_restore produces.
    """
    run, _recorded, _returned = evaluated
    reason = {"report": {"degraded_count": 1, "degrade_reason": "no_face_detected"}}
    plan = [
        {"stage": "scratch_repair", "metadata": {"report": {"degraded_count": 0}}},
        {"stage": "global_restore", "metadata": reason},
        # The face chain was skipped, so warp_back ran without a report.
        {"stage": "warp_back", "metadata": {}},
    ]
    from fiximg.infrastructure.db.repositories import task_repository as repo

    repo.create_task("t-notes", "auto_restore", "u")
    notes = run("t-notes", "auto_restore", plan, identity={"skipped": True})
    assert any("No face detected" in n for n in notes), notes


def test_identity_metric_does_not_overwrite_the_face_chain_counts(evaluated):
    """The two face counts measure different things and keep different names.

    ``face_count`` is what the face *chain* processed; the identity metric's
    counts are what *its own* detector sees in the two pictures. Without dlib
    the chain reports zero while the detector still finds the face, so one shared
    key made report.json and the stage list contradict each other while both
    looked well-formed.

    Asserted on the dict `_evaluate` returns, because that is where the collision
    happened. Only the similarity score is written to the metrics table, so a spy
    on `add_metric` cannot see the three counts at all.
    """
    run, recorded, returned = evaluated
    identity = {
        "input_faces": 1, "output_faces": 1, "paired_faces": 1,
        "identity_similarity": 0.98, "backend": "gradient_fallback",
    }
    from fiximg.infrastructure.db.repositories import task_repository as repo

    repo.create_task("t-ident", "restore", "u")
    run("t-ident", "restore", [{"stage": "global_restore", "metadata": {}}],
        identity=identity)

    assert returned["identity_faces_input"] == 1
    assert returned["identity_faces_output"] == 1
    assert returned["identity_faces_paired"] == 1
    # The chain's keys are not among them — that is the whole point.
    assert "face_count" not in returned, sorted(returned)
    assert "enhanced_count" not in returned, sorted(returned)
    assert recorded["identity_similarity"] == 0.98


def test_the_face_chain_keeps_its_own_count_key(evaluated):
    """The other half: a stage's `face_count` must survive the evaluation merge."""
    run, recorded, returned = evaluated
    identity = {
        "input_faces": 3, "output_faces": 3, "paired_faces": 3,
        "identity_similarity": 0.91, "backend": "dlib_resnet",
    }
    stage_meta = [
        {"stage": "face_enhancement", "metadata": {},
         "metrics": {"face_count": 0, "enhanced_count": 0}},
    ]
    from fiximg.inference import runtime as rt
    from fiximg.infrastructure.db.repositories import task_repository as repo

    repo.create_task("t-chain", "restore", "u")
    run("t-chain", "restore", stage_meta, identity=identity)
    rt._persist_stage_metrics("t-chain", stage_meta[0]["metrics"])

    stored = repo.read_metrics("t-chain")
    assert stored.get("face_count") == 0.0, stored
    assert stored.get("identity_similarity") == 0.91, stored
