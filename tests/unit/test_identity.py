"""Unit tests for face identity preservation (plan section 17)."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.inference import identity


def _face_card(seed=5, size=(200, 200), center=(100, 100), radius=34,
               skin=(196, 168, 140)) -> Image.Image:
    """Synthetic portrait: flat background + oval face + eyes + mouth."""
    img = Image.new("RGB", size, (235, 235, 235))
    draw = ImageDraw.Draw(img)
    cx, cy = center
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=skin)
    eye_y = cy - radius // 3
    for ex in (cx - radius // 3, cx + radius // 3):
        draw.ellipse([ex - 4, eye_y - 3, ex + 4, eye_y + 3], fill=(40, 40, 40))
    draw.line([cx - radius // 2, cy + radius // 2,
               cx + radius // 2, cy + radius // 2], fill=(90, 60, 50), width=3)
    return img


@pytest.fixture(autouse=True)
def force_fallback_backend(monkeypatch):
    """Tests run on the deterministic fallback (no dlib weights locally)."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "identity_backend", "fallback", raising=False)
    identity._reset_dlib_cache()


# ----------------------------------------------------------------- embeddings
def test_gradient_embedding_self_similarity_is_one():
    crop = np.asarray(_face_card())
    a = identity._gradient_embedding(crop)
    assert identity._cosine(a, a) == pytest.approx(1.0, abs=1e-6)


def test_gradient_embedding_stable_under_mild_change():
    """Small brightness shift (what enhancement does) keeps high similarity."""
    crop = np.asarray(_face_card(), dtype=np.int16)
    shifted = np.clip(crop + 12, 0, 255).astype(np.uint8)
    sim = identity._cosine(
        identity._gradient_embedding(crop.astype(np.uint8)),
        identity._gradient_embedding(shifted),
    )
    assert sim > 0.9


def test_gradient_embedding_distinguishes_faces():
    light = identity._gradient_embedding(np.asarray(_face_card(seed=1, skin=(210, 180, 150))))
    dark = identity._gradient_embedding(np.asarray(_face_card(seed=2, skin=(90, 70, 60))))
    assert identity._cosine(light, dark) < 0.95


def test_cosine_handles_zero_vector():
    z = np.zeros(8)
    assert identity._cosine(z, z) == 0.0


# ------------------------------------------------------------------- metric
_BOX = [(60, 60, 80, 80)]  # one face centred on the card


def test_identity_similarity_same_face(monkeypatch):
    monkeypatch.setattr(identity, "detect_face_boxes", lambda rgb: list(_BOX))
    before = _face_card()
    result = identity.compute_identity_similarity(before, before.copy())
    assert result["backend"] == "gradient_fallback"
    assert result["face_count"] == 1
    assert result["enhanced_faces"] == 1
    assert result["paired_faces"] == 1
    assert result["identity_similarity"] == pytest.approx(1.0, abs=0.01)
    assert result["notice"] == identity.IDENTITY_NOTICE


def test_identity_similarity_rejects_different_faces(monkeypatch):
    monkeypatch.setattr(identity, "detect_face_boxes", lambda rgb: list(_BOX))
    sim = identity.compute_identity_similarity(
        _face_card(skin=(220, 190, 160), radius=30),
        _face_card(skin=(60, 45, 40), radius=44),
    )["identity_similarity"]
    assert sim is not None and sim < 0.99


def test_identity_no_faces(monkeypatch):
    monkeypatch.setattr(identity, "detect_face_boxes", lambda rgb: [])
    blank = Image.new("RGB", (160, 160), (230, 230, 230))
    result = identity.compute_identity_similarity(blank, blank)
    assert result["face_count"] == 0
    assert result["identity_similarity"] is None


def test_real_backend_detects_or_degrades():
    """With a real backend installed, detection runs without error.

    Synthetic cartoons may legitimately yield zero boxes; the contract under
    test is graceful behaviour, not cartoon recognition.
    """
    rgb = np.asarray(_face_card())
    boxes = identity.detect_face_boxes(rgb)
    assert isinstance(boxes, list)
    assert all(len(b) == 4 for b in boxes)


def test_identity_off_backend(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "identity_backend", "off", raising=False)
    result = identity.compute_identity_similarity(_face_card(), _face_card())
    assert result == {"skipped": True, "reason": "disabled_by_config"}


def test_format_identity_report_layout():
    text = identity.format_identity_report(
        {"face_count": 3, "enhanced_faces": 3, "identity_similarity": 0.91,
         "backend": "dlib_resnet"}
    )
    assert "Face Count            3" in text
    assert "Enhanced Faces        3" in text
    assert "Identity Similarity   0.91" in text
    assert "⚠" in text  # limitation notice (plan section 17)


def test_format_report_when_unpaired():
    text = identity.format_identity_report(
        {"face_count": 0, "enhanced_faces": 0, "identity_similarity": None,
         "backend": "gradient_fallback"}
    )
    assert "n/a" in text


# ------------------------------------------------------------------- pairing
def test_pair_boxes_nearest_centre():
    before = [(10, 10, 20, 20), (100, 100, 20, 20)]
    after = [(102, 102, 20, 20), (12, 12, 20, 20)]
    pairs = identity._pair_boxes(before, after)
    assert (0, 1) in pairs and (1, 0) in pairs
