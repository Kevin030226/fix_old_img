"""Weight management — download & verify (plan §19).

Replaces scripts/download_weights.sh with a Python implementation that supports:

  - python -m scripts.download_weights   idempotent downloads + manifest rebuild
  - python -m scripts.verify_weights     manifest verification (fail-fast)

Layout stays on the legacy roots (Global/, Face_Enhancement/, Face_Detection/,
weights/ddcolor/) because the V1 model scripts resolve those paths themselves;
the plan's `models/` rename would break the vendored code and is therefore NOT
part of this step. The manifest (config/weights_manifest.json) records
sha256 + size + version per artifact and is rebuilt after download.

Resume support: large single-file downloads (DDColor) use HTTP Range requests
with a `.part` file, so an interrupted fetch continues where it stopped.
"""
import bz2
import hashlib
import json
import os
import sys
import time
import zipfile
from datetime import datetime, UTC
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(BASE_DIR, "config", "weights_manifest.json")

#: artifact registry: manifest key -> {version, files:[path], sources:[url...]}
#: `files` lists the paths that must exist after the artifact is prepared;
#: the first entry is the download/extract anchor used for existence checks.
ARTIFACTS = {
    "face_detection": {
        "version": "1.0",
        "files": ["Face_Detection/shape_predictor_68_face_landmarks.dat"],
        "sources": ["http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"],
        "kind": "bz2",
    },
    # 128-D face embeddings for the plan §17 identity-preservation metric
    # (app/inference/identity.py). Without it the metric falls back to the
    # dependency-free gradient descriptor.
    "face_recognition": {
        "version": "1.0",
        "files": ["Face_Detection/dlib_face_recognition_resnet_model_v1.dat"],
        "sources": [
            "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
        ],
        "kind": "bz2",
    },
    "face_restore": {
        "version": "1.0",
        "files": ["Face_Enhancement/checkpoints"],
        "sources": [
            "https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/"
            "Face_Enhancement/checkpoints.zip"
        ],
        "kind": "zip",
    },
    "global_restore": {
        "version": "1.0",
        "files": ["Global/checkpoints"],
        "sources": [
            "https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/"
            "Global/checkpoints.zip"
        ],
        "kind": "zip",
    },
    "yunet_face_detector": {
        "version": "1.0",
        "files": ["weights/yunet/face_detection_yunet_2023mar.onnx"],
        "sources": [
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://gitee.com/mirrors/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
        "kind": "raw",
    },
    "ddcolor": {
        "version": "1.0",
        "files": ["weights/ddcolor/pytorch_model.pt"],
        "sources": [
            "https://modelscope.cn/models/damo/cv_ddcolor_image-colorization/"
            "resolve/master/pytorch_model.pt",
            "https://huggingface.co/piddnad/ddcolor_modelscope/resolve/main/pytorch_model.pt",
        ],
        "kind": "raw",
    },
}

CHUNK = 1 << 20


class DownloadError(RuntimeError):
    """A weight artifact could not be downloaded from any source."""


# ------------------------------------------------------------------ helpers
def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _file_complete(path: str) -> bool:
    return os.path.isdir(path) or os.path.isfile(path)


def _remote_size(url: str) -> int | None:
    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=30) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:  # noqa: BLE001
        return None


def _download_raw(url: str, dest: str, resume: bool = True) -> None:
    """Stream `url` to `dest` with progress; resume via Range when supported."""
    part = dest + ".part"
    done = os.path.getsize(part) if resume and os.path.exists(part) else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    req = Request(url, headers=headers)
    with urlopen(req, timeout=60) as resp, open(part, "ab" if done else "wb") as f:
        total = resp.headers.get("Content-Length")
        total = int(total) + done if total else None
        start = time.time()
        while True:
            block = resp.read(CHUNK)
            if not block:
                break
            f.write(block)
            done += len(block)
            _print_progress(dest, done, total, start)
    os.replace(part, dest)


def _print_progress(name: str, done: int, total: int | None, start: float) -> None:
    if total:
        pct = min(100.0, done * 100.0 / total)
        speed = done / max(time.time() - start, 1e-6) / (1 << 20)
        sys.stdout.write(f"\r    {os.path.basename(name)}: {pct:5.1f}%  ({speed:.1f} MB/s)")
    else:
        sys.stdout.write(f"\r    {os.path.basename(name)}: {done / (1 << 20):.1f} MB")
    sys.stdout.flush()


def _prepare_bz2(url: str, dest: str) -> None:
    """Download a .bz2 file and decompress it into `dest` (single file)."""
    tmp = dest + ".bz2.part"
    _download_raw(url, tmp, resume=False)
    with bz2.open(tmp, "rb") as src, open(dest + ".tmp", "wb") as out:
        while True:
            block = src.read(CHUNK)
            if not block:
                break
            out.write(block)
    os.replace(dest + ".tmp", dest)
    os.remove(tmp)


def _prepare_zip(url: str, dest_dir: str) -> None:
    """Download a zip and extract it so that <dest_dir> exists afterwards."""
    parent = os.path.dirname(os.path.abspath(dest_dir))
    os.makedirs(parent, exist_ok=True)
    tmp = os.path.join(parent, "_weights_download.zip")
    _download_raw(url, tmp, resume=False)
    with zipfile.ZipFile(tmp) as zf:
        zf.extractall(parent)
    os.remove(tmp)


# ----------------------------------------------------------------- commands
def download_all(skip_existing: bool = True) -> dict:
    """Ensure every registered artifact exists; rebuild the manifest after.

    Returns a summary dict {artifact: "ok"|"skipped"|"failed: reason"}.
    Exits non-zero via DownloadError when an artifact cannot be fetched.
    """
    summary: dict = {}
    for name, spec in ARTIFACTS.items():
        anchor = os.path.join(BASE_DIR, spec["files"][0])
        if skip_existing and _file_complete(anchor):
            summary[name] = "skipped (exists)"
            continue
        if spec["kind"] == "raw":
            dest = os.path.join(BASE_DIR, spec["files"][0])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            _fetch_from_sources(spec["sources"], lambda url, _dest=dest: _download_raw(url, _dest))
        elif spec["kind"] == "bz2":
            dest = os.path.join(BASE_DIR, spec["files"][0])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            _fetch_from_sources(spec["sources"], lambda url, _dest=dest: _prepare_bz2(url, _dest))
        else:  # zip
            dest_dir = os.path.join(BASE_DIR, spec["files"][0])
            _fetch_from_sources(spec["sources"], lambda url, _d=dest_dir: _prepare_zip(url, _d))
        summary[name] = "ok"
    generate_manifest()
    return summary


def _fetch_from_sources(sources: list, prepare) -> None:
    errors = []
    for url in sources:
        try:
            print(f"  -> {urlsplit(url).netloc} ...")
            prepare(url)
            print()
            return
        except Exception as exc:  # noqa: BLE001 — try the next mirror
            errors.append(f"{url}: {exc}")
            print(f"\n    source failed: {exc}")
    raise DownloadError("all sources failed:\n  - " + "\n  - ".join(errors))


def generate_manifest() -> dict:
    """Walk all registered artifact files and write the baseline manifest.

    Entries carry sha256, size_bytes and version (plan §19 manifest contract).
    Only files under the registered artifacts are hashed — arbitrary stray
    weights elsewhere in the repo are ignored.
    """
    entries = []
    for name, spec in ARTIFACTS.items():
        for rel in spec["files"]:
            ap = os.path.join(BASE_DIR, rel)
            if os.path.isdir(ap):
                for root, _dirs, files in os.walk(ap):
                    for f in sorted(files):
                        fp = os.path.join(root, f)
                        entries.append(
                            {
                                "path": os.path.relpath(fp, BASE_DIR).replace(os.sep, "/"),
                                "sha256": compute_sha256(fp),
                                "size_bytes": os.path.getsize(fp),
                                "artifact": name,
                                "version": spec["version"],
                            }
                        )
            elif os.path.isfile(ap):
                entries.append(
                    {
                        "path": rel,
                        "sha256": compute_sha256(ap),
                        "size_bytes": os.path.getsize(ap),
                        "artifact": name,
                        "version": spec["version"],
                    }
                )
    manifest = {
        "algorithm": "sha256",
        "generated_at": datetime.now(UTC).isoformat(),
        "file_count": len(entries),
        "files": entries,
    }
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


class WeightsIntegrityError(RuntimeError):
    """Weight verification failed (missing file, size or hash mismatch)."""


def verify(manifest_path: str | None = None) -> int:
    """Verify every manifest entry; return the number of checked files."""
    path = manifest_path or MANIFEST_PATH
    if not os.path.exists(path):
        raise WeightsIntegrityError(
            f"Weight manifest not found: {path}\n"
            "Generate it first: python -m scripts.download_weights"
        )
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)

    problems = []
    for entry in manifest.get("files", []):
        rel = entry["path"]
        ap = os.path.join(BASE_DIR, rel)
        if not os.path.exists(ap):
            problems.append(f"Missing file: {rel}")
            continue
        if entry.get("size_bytes") is not None and os.path.getsize(ap) != entry["size_bytes"]:
            problems.append(f"Size mismatch: {rel}")
            continue
        if compute_sha256(ap) != entry.get("sha256"):
            problems.append(f"Hash mismatch: {rel}")

    if problems:
        raise WeightsIntegrityError(
            "Weight verification failed:\n- " + "\n- ".join(problems)
        )
    return len(manifest.get("files", []))


def main(argv: list) -> int:
    cmd = argv[1] if len(argv) > 1 else "help"
    if cmd == "download":
        try:
            summary = download_all()
        except DownloadError as exc:
            print(f"Download failed: {exc}", file=sys.stderr)
            return 1
        for name, status in summary.items():
            print(f"  {name}: {status}")
        print(f"Manifest regenerated: {MANIFEST_PATH}")
        return 0
    if cmd == "verify":
        try:
            n = verify()
        except WeightsIntegrityError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Weight integrity check passed ({n} files)")
        return 0
    if cmd == "generate":
        m = generate_manifest()
        print(f"Manifest regenerated: {MANIFEST_PATH} ({m['file_count']} files)")
        return 0
    print("Usage: python -m scripts.download_weights [download|generate|verify]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
