"""Unit tests for the §19 weight manager (no network access needed)."""
import os

import pytest

from scripts import download_weights as dw


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
