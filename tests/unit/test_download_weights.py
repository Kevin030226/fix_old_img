"""Unit tests for the §19 weight manager (no network access needed)."""
import hashlib
import json
import os

import pytest

from fiximg.cli import download_weights as dw


def test_a_directory_anchor_does_not_count_as_a_completed_artifact(tmp_path, monkeypatch):
    """A zip artifact whose anchor is a directory must still be downloaded.

    The checkout ships ``Face_Enhancement/checkpoints`` and ``Global/checkpoints`` as
    empty directories, and ``_file_complete`` used to answer "yes, present" for any
    directory -- so ``download_all`` skipped both zip artifacts on a fresh clone and
    reported success while the entire restoration stack was missing.
    """
    monkeypatch.setattr(dw, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(dw, "MANIFEST_PATH", str(tmp_path / "config" / "weights_manifest.json"))
    monkeypatch.setattr(dw, "ARTIFACTS", {
        "zip_model": {
            "version": "1.0",
            "files": ["checkpoints"],
            "sources": [],
            "kind": "zip",
        }
    })
    (tmp_path / "checkpoints").mkdir()

    # An empty directory is not a completed artifact.
    assert dw._file_complete(str(tmp_path / "checkpoints")) is False

    # Neither is one that only holds the version-control placeholder -- that is
    # exactly what a fresh clone of this repository looks like.
    (tmp_path / "checkpoints" / ".gitkeep").write_text("", encoding="utf-8")
    assert dw._file_complete(str(tmp_path / "checkpoints")) is False

    # Real content does count, so a populated deployment still skips the download.
    (tmp_path / "checkpoints" / "latest_net_G.pth").write_bytes(b"w" * 16)
    assert dw._file_complete(str(tmp_path / "checkpoints")) is True


@pytest.fixture()
def fake_artifact(tmp_path, monkeypatch):
    """Redirect the project root + manifest to tmp; register a tiny artifact."""
    monkeypatch.setattr(dw, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(dw, "MANIFEST_PATH", str(tmp_path / "config" / "weights_manifest.json"))
    spec = {
        "test_model": {
            "version": "2.1",
            "files": ["weights/test/model.bin"],
            "sources": [],
            "kind": "raw",
        }
    }
    monkeypatch.setattr(dw, "ARTIFACTS", spec)
    weight = tmp_path / "weights" / "test" / "model.bin"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"fake-weight-bytes-12345678")
    return weight


def test_generate_manifest_structure(fake_artifact):
    m = dw.generate_manifest()
    assert m["algorithm"] == "sha256"
    assert m["file_count"] == 1
    entry = m["files"][0]
    assert entry["path"] == "weights/test/model.bin"
    assert entry["artifact"] == "test_model"
    assert entry["version"] == "2.1"
    assert entry["size_bytes"] == len(b"fake-weight-bytes-12345678")
    assert len(entry["sha256"]) == 64


def test_verify_passes_and_detects_tamper(fake_artifact):
    dw.generate_manifest()
    assert dw.verify() == 1

    # Same size but different content -> hash mismatch.
    fake_artifact.write_bytes(b"fake-weight-bytes-12345678"[::-1])
    with pytest.raises(dw.WeightsIntegrityError, match="Hash mismatch"):
        dw.verify()

    # Different size -> caught even earlier by the size check.
    fake_artifact.write_bytes(b"tampered")
    with pytest.raises(dw.WeightsIntegrityError, match="Size mismatch"):
        dw.verify()


def test_verify_detects_missing_file(fake_artifact):
    dw.generate_manifest()
    fake_artifact.unlink()
    with pytest.raises(dw.WeightsIntegrityError, match="Missing file"):
        dw.verify()


def test_verify_missing_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(dw, "MANIFEST_PATH", str(tmp_path / "none.json"))
    with pytest.raises(dw.WeightsIntegrityError, match="not found"):
        dw.verify()


def test_main_verify_exit_codes(fake_artifact, capsys):
    # verify without manifest -> exit 1
    assert dw.main(["x", "verify"]) == 1
    dw.generate_manifest()
    assert dw.main(["x", "verify"]) == 0
    assert dw.main(["x", "bogus"]) == 2


def test_directory_artifact_is_walked(tmp_path, monkeypatch):
    monkeypatch.setattr(dw, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        dw, "MANIFEST_PATH", str(tmp_path / "config" / "weights_manifest.json")
    )
    ckpt = tmp_path / "Global" / "checkpoints"
    (ckpt / "sub").mkdir(parents=True)
    (ckpt / "a.bin").write_bytes(b"A" * 10)
    (ckpt / "sub" / "b.bin").write_bytes(b"B" * 20)
    monkeypatch.setattr(
        dw,
        "ARTIFACTS",
        {"global_restore": {"version": "1.0", "files": ["Global/checkpoints"], "sources": [], "kind": "zip"}},
    )
    m = dw.generate_manifest()
    paths = {e["path"] for e in m["files"]}
    assert paths == {"Global/checkpoints/a.bin", "Global/checkpoints/sub/b.bin"}
    assert all(e["size_bytes"] > 0 for e in m["files"])


def test_resume_download_appends_part(monkeypatch, tmp_path):
    """_download_raw resumes from an existing .part file (Range header path)."""
    import io

    dest = str(tmp_path / "big.bin")
    with open(dest + ".part", "wb") as f:
        f.write(b"HEAD-")

    captured = {}

    class FakeResp(io.BytesIO):
        def __init__(self, data, headers=None):
            super().__init__(data)
            self.headers = headers or {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        captured["headers"] = dict(req.headers)
        captured["url"] = req.full_url
        return FakeResp(b"-TAIL", {"Content-Length": "5"})

    monkeypatch.setattr(dw, "urlopen", fake_urlopen)
    dw._download_raw("http://example.test/big.bin", dest, resume=True)

    assert captured["headers"].get("Range") == "bytes=5-"
    with open(dest, "rb") as f:
        assert f.read() == b"HEAD--TAIL"
    assert not os.path.exists(dest + ".part")


def test_registry_covers_legacy_layout():
    """The registry must anchor the four legacy weight groups."""
    assert set(dw.ARTIFACTS) >= {"face_detection", "face_restore", "global_restore", "ddcolor"}
    assert dw.ARTIFACTS["ddcolor"]["files"][0] == "weights/ddcolor/pytorch_model.pt"


# --------------------------------------------------- download contract tests
GOOD_PAYLOAD = b"GOOD-PAYLOAD-0123456789"
#: Same length as GOOD_PAYLOAD on purpose: only the hash can tell them apart.
WRONG_PAYLOAD = b"WRONG-PAYLOAD-012345678"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(rel: str, data: bytes, artifact: str = "repro") -> dict:
    return {
        "path": rel,
        "sha256": _sha256(data),
        "size_bytes": len(data),
        "artifact": artifact,
        "version": "1.0",
    }


def _write_manifest(manifest_path, entries: list[dict]):
    manifest = {
        "algorithm": "sha256",
        "generated_at": "2026-10-15T00:00:00+00:00",
        "file_count": len(entries),
        "files": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _fake_download(payload: bytes):
    """Stand-in for _download_raw: writes a payload without touching the network."""

    def fake_download_raw(url: str, dest: str, resume: bool = True) -> None:
        del url, resume
        with open(dest, "wb") as handle:
            handle.write(payload)

    return fake_download_raw


@pytest.fixture()
def weight_tree(tmp_path, monkeypatch):
    """Isolated deployment tree holding one raw artifact and a committed manifest."""
    monkeypatch.setattr(dw, "BASE_DIR", str(tmp_path))
    manifest_path = tmp_path / "config" / "weights_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    monkeypatch.setattr(dw, "MANIFEST_PATH", str(manifest_path))
    monkeypatch.setattr(
        dw,
        "ARTIFACTS",
        {
            "repro": {
                "version": "1.0",
                "files": ["weights/repro/model.bin"],
                "sources": ["http://example.test/model.bin"],
                "kind": "raw",
            }
        },
    )
    weight = tmp_path / "weights" / "repro" / "model.bin"
    weight.parent.mkdir(parents=True)
    _write_manifest(manifest_path, [_entry("weights/repro/model.bin", GOOD_PAYLOAD)])
    return {"weight": weight, "manifest": manifest_path}


def test_download_verifies_payload_against_committed_manifest(weight_tree, monkeypatch):
    """A correct download passes only against the manifest read before it started."""
    weight = weight_tree["weight"]
    manifest = weight_tree["manifest"]
    before = manifest.read_bytes()
    monkeypatch.setattr(dw, "_download_raw", _fake_download(GOOD_PAYLOAD))

    assert not weight.exists()
    assert dw.download_all() == {"repro": "ok"}
    assert weight.read_bytes() == GOOD_PAYLOAD
    assert manifest.read_bytes() == before
    assert dw.verify() == 1


def test_download_rejects_wrong_payload_and_never_rewrites_manifest(weight_tree, monkeypatch):
    """A corrupted/hostile payload must fail instead of certifying itself."""
    manifest = weight_tree["manifest"]
    before = manifest.read_bytes()
    monkeypatch.setattr(dw, "_download_raw", _fake_download(WRONG_PAYLOAD))

    with pytest.raises(dw.WeightsIntegrityError, match="Hash mismatch"):
        dw.download_all()
    assert manifest.read_bytes() == before
    # The failure must survive an immediate verify: no rewritten baseline exists
    # that could make the wrong bytes look expected.
    with pytest.raises(dw.WeightsIntegrityError, match="Hash mismatch"):
        dw.verify()


def test_download_rejects_size_mismatch(weight_tree, monkeypatch):
    monkeypatch.setattr(dw, "_download_raw", _fake_download(b"too-short"))
    with pytest.raises(dw.WeightsIntegrityError, match="Size mismatch"):
        dw.download_all()


def test_download_checks_every_declared_entry_not_only_downloaded_ones(weight_tree, monkeypatch):
    """A payload missing a declared file must fail, not shrink the expected set."""
    manifest = weight_tree["manifest"]
    _write_manifest(
        manifest,
        [
            _entry("weights/repro/model.bin", GOOD_PAYLOAD),
            _entry("weights/repro/extra.bin", GOOD_PAYLOAD),
        ],
    )
    monkeypatch.setattr(dw, "_download_raw", _fake_download(GOOD_PAYLOAD))

    with pytest.raises(dw.WeightsIntegrityError, match="Missing file: weights/repro/extra.bin"):
        dw.download_all()


def test_download_skips_complete_artifacts_but_still_verifies(weight_tree, monkeypatch):
    weight = weight_tree["weight"]
    weight.write_bytes(GOOD_PAYLOAD)
    monkeypatch.setattr(
        dw,
        "_fetch_from_sources",
        lambda sources, prepare: pytest.fail("a complete artifact must not be downloaded again"),
    )
    assert dw.download_all() == {"repro": "skipped (exists)"}

    # A skip decision is only about presence/size; tampering is still caught.
    weight.write_bytes(WRONG_PAYLOAD)
    with pytest.raises(dw.WeightsIntegrityError, match="Hash mismatch"):
        dw.download_all()


def test_download_requires_the_committed_manifest(weight_tree, monkeypatch):
    manifest = weight_tree["manifest"]
    manifest.unlink()
    monkeypatch.setattr(
        dw, "_fetch_from_sources", lambda sources, prepare: pytest.fail("download must not start")
    )
    with pytest.raises(dw.WeightsIntegrityError, match="not found"):
        dw.download_all()
    assert not manifest.exists()


def test_download_rejects_registry_manifest_drift(weight_tree):
    _write_manifest(
        weight_tree["manifest"],
        [_entry("weights/repro/model.bin", GOOD_PAYLOAD, artifact="not_registered")],
    )
    with pytest.raises(dw.WeightsIntegrityError, match="does not know"):
        dw.download_all()


def test_main_download_failure_exit_code_and_manifest_unchanged(weight_tree, monkeypatch, capsys):
    manifest = weight_tree["manifest"]
    before = manifest.read_bytes()
    monkeypatch.setattr(dw, "_download_raw", _fake_download(WRONG_PAYLOAD))

    assert dw.main(["fiximg-weights", "download"]) == 1
    assert "Hash mismatch" in capsys.readouterr().err
    assert manifest.read_bytes() == before


def test_main_download_success_keeps_manifest(weight_tree, monkeypatch, capsys):
    manifest = weight_tree["manifest"]
    before = manifest.read_bytes()
    monkeypatch.setattr(dw, "_download_raw", _fake_download(GOOD_PAYLOAD))

    assert dw.main(["fiximg-weights", "download"]) == 0
    assert "manifest was not modified" in capsys.readouterr().out
    assert manifest.read_bytes() == before
