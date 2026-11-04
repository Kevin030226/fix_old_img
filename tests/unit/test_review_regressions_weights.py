"""Regression tests for the `weights_check` fail-open finding in the v3.1.2 review.

Confirmed against the tag: `verify_weights()` iterates `manifest.get("files", [])`
and returns `len(...)` — so an empty manifest, or one that simply omits `files`,
"passes" with zero files verified. A manifest that has been truncated, replaced by
`{}`, or written by a failed generator therefore reports success, and the
tamper-evident baseline stops being tamper-evident.

The check is a startup gate (`python -m fiximg.cli.verify_weights`), so a green
answer from an empty file is the one answer it must never give.
"""
from __future__ import annotations

import json

import pytest

from fiximg.infrastructure.models import weights_check


@pytest.fixture()
def manifest(tmp_path):
    return str(tmp_path / "weights_manifest.json")


def test_an_empty_manifest_is_refused(manifest):
    """The defect: zero entries used to be reported as a pass."""
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump({"algorithm": "sha256", "file_count": 0, "files": []}, handle)

    with pytest.raises(weights_check.WeightsIntegrityError):
        weights_check.verify_weights(manifest_path=manifest)


def test_a_manifest_without_a_files_list_is_refused(manifest):
    """`{}` and `{"algorithm": "sha256"}` are both "no baseline", not "all good"."""
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump({"algorithm": "sha256"}, handle)

    with pytest.raises(weights_check.WeightsIntegrityError):
        weights_check.verify_weights(manifest_path=manifest)


def test_a_manifest_whose_file_count_contradicts_its_entries_is_refused(manifest):
    """A generator that died mid-write leaves a manifest that lies about itself."""
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump({"algorithm": "sha256", "file_count": 5, "files": []}, handle)

    with pytest.raises(weights_check.WeightsIntegrityError):
        weights_check.verify_weights(manifest_path=manifest)


def test_an_entry_without_a_digest_is_refused(manifest):
    """A path with no hash proves nothing; it must not count as verified."""
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump(
            {"algorithm": "sha256", "file_count": 1,
             "files": [{"path": "models/manifest.yaml"}]},
            handle,
        )

    with pytest.raises(weights_check.WeightsIntegrityError):
        weights_check.verify_weights(manifest_path=manifest)


def test_a_valid_manifest_still_passes(tmp_path, manifest):
    """The half that must not regress: a real entry with the right digest passes."""
    payload = tmp_path / "weight.pth"
    payload.write_bytes(b"weight-bytes")
    digest = weights_check.compute_sha256(str(payload))
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump(
            {"algorithm": "sha256", "file_count": 1,
             "files": [{"path": "weight.pth", "sha256": digest}]},
            handle,
        )

    original = weights_check.BASE_DIR
    weights_check.BASE_DIR = str(tmp_path)
    try:
        assert weights_check.verify_weights(manifest_path=manifest) == 1
        assert weights_check.verified_weights() == {"weight.pth": digest}
    finally:
        weights_check.BASE_DIR = original
        weights_check.VERIFIED.clear()
