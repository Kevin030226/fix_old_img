"""Weight management — download & verify (plan §19).

Replaces scripts/download_weights.sh with a Python implementation that supports:

  - python -m fiximg.cli.download_weights download  idempotent downloads, followed
    by verification against the committed manifest (which is never rewritten)
  - python -m fiximg.cli.download_weights verify     manifest verification (fail-fast)
  - python -m fiximg.cli.download_weights generate   maintainer-only: rebuild the
    baseline from the files already on disk (explicit; download never calls it)

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
from collections.abc import Sequence
from datetime import datetime, UTC
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from fiximg.paths import CONFIG_DIR, PROJECT_ROOT

BASE_DIR = PROJECT_ROOT
MANIFEST_PATH = os.path.join(CONFIG_DIR, "weights_manifest.json")

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
        # The upstream project publishes these as GitHub release assets, and
        # github.com resolves from the runners. The Azure blob host this used to
        # come from stopped resolving, which aborted the worker image build with
        # `Name or service not known`; both URLs are kept so the fetch survives
        # either one going away again. Verified: both assets answer HTTP 200 and
        # their Content-Length matches the size this entry unpacks to.
        "sources": [
            "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/"
            "releases/download/v1.0/face_checkpoints.zip",
            "https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/"
            "Face_Enhancement/checkpoints.zip",
        ],
        "kind": "zip",
    },
    "global_restore": {
        "version": "1.0",
        "files": ["Global/checkpoints"],
        "sources": [
            "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/"
            "releases/download/v1.0/global_checkpoints.zip",
            "https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/"
            "Global/checkpoints.zip",
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


class WeightsIntegrityError(RuntimeError):
    """Weight verification failed (missing file, size or hash mismatch)."""


# ------------------------------------------------------------------ helpers
def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


#: Files that exist only to keep an otherwise empty directory in version control.
#: They must not be mistaken for artifact content: the checkout ships the
#: checkpoint trees with a `.gitkeep` inside, and counting one as "the artifact is
#: here" would skip the download it is meant to trigger.
PLACEHOLDER_NAMES = (".gitkeep", ".gitignore")


def _file_complete(path: str) -> bool:
    """Whether a path listed in ``files`` is actually present.

    A directory counts only when it holds something other than a placeholder. The
    checkout ships the checkpoint directories empty (``Face_Enhancement/checkpoints``,
    ``Global/checkpoints``), and treating "the directory exists" as "the artifact
    is complete" made ``download_all`` skip both zip artifacts on a fresh clone --
    the entire restoration stack, silently, while reporting success.
    """
    if os.path.isdir(path):
        return any(name not in PLACEHOLDER_NAMES for name in os.listdir(path))
    return os.path.isfile(path)


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
    """Ensure every artifact declared in the committed manifest exists.

    The manifest read before the first download is the expected set: after the
    downloads finish, every declared path must exist with the declared byte size
    and sha256. The manifest itself is never written here -- a maintainer who
    intends to change the baseline runs the explicit generate command.

    Returns a summary dict {artifact: "ok"|"skipped (exists)"}.
    Raises DownloadError when an artifact cannot be fetched and
    WeightsIntegrityError when the result does not match the committed manifest.
    """
    manifest = _load_manifest()
    expected_by_artifact = _manifest_files_by_artifact(manifest)
    registered = set(ARTIFACTS)
    declared = set(expected_by_artifact)
    if declared - registered:
        raise WeightsIntegrityError(
            "The committed manifest declares artifacts the download registry does not "
            f"know: {sorted(declared - registered)}"
        )
    unknown = registered - declared
    if unknown:
        raise WeightsIntegrityError(
            "The download registry contains artifacts the committed manifest does not "
            f"declare: {sorted(unknown)}. The registry and the manifest are out of sync."
        )
    summary: dict = {}
    for name, spec in ARTIFACTS.items():
        # Use the manifest's complete expected file set, not just the registry's
        # anchor: a partially populated checkpoint tree re-downloads, and a complete
        # one is skipped only because the verification below still hashes every
        # declared entry.
        expected = expected_by_artifact[name]
        if skip_existing and all(_entry_complete(entry) for entry in expected):
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

    problems = _verify_manifest(manifest)
    if problems:
        raise WeightsIntegrityError(
            "Weight download does not match the committed manifest "
            "(the manifest was not modified):\n- " + "\n- ".join(problems)
        )
    return summary


def _fetch_from_sources(sources: Sequence[str], prepare) -> None:
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

    Maintainer-only and deliberately explicit: this rewrites the trust anchor,
    so download/verify never call it. Entries carry sha256, size_bytes and
    version (plan §19 manifest contract). Only files under the registered
    artifacts are hashed — arbitrary stray weights elsewhere are ignored.
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
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


# ------------------------------------------------------- manifest baseline
def _load_manifest(manifest_path: str | None = None) -> dict:
    """Read the committed baseline; fail with an actionable error when absent.

    download must never create or refresh this file. A missing manifest means the
    deployment tree lost its trust anchor; restoring it from version control is
    the answer, not silently hashing the files that happen to be on disk.
    """
    path = manifest_path or MANIFEST_PATH
    if not os.path.exists(path):
        raise WeightsIntegrityError(
            f"Weight manifest not found: {path}\n"
            "The committed manifest is the baseline for downloads and startup checks. "
            "Restore it from version control; a maintainer can rebuild it once, "
            "explicitly, with: python -m fiximg.cli.download_weights generate"
        )
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise WeightsIntegrityError(f"Cannot read weight manifest {path}: {exc}") from exc
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise WeightsIntegrityError(f"Weight manifest {path} declares no files.")
    return manifest


def _manifest_files_by_artifact(manifest: dict) -> dict[str, list[dict]]:
    """Group manifest entries by artifact, rejecting incomplete declarations."""
    grouped: dict[str, list[dict]] = {}
    for entry in manifest["files"]:
        rel = entry.get("path")
        artifact = entry.get("artifact")
        expected_sha = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if (
            not rel
            or not artifact
            or not isinstance(expected_sha, str)
            or not isinstance(expected_size, int)
        ):
            raise WeightsIntegrityError(
                "Malformed weight manifest entry: each entry needs path, artifact, "
                f"size_bytes and sha256; got {entry!r}"
            )
        grouped.setdefault(artifact, []).append(entry)
    return grouped


def _entry_complete(entry: dict) -> bool:
    """Whether the manifest entry is present with the declared byte size.

    Hash checking is left to _verify_manifest so the skip decision stays cheap
    while the post-download proof below still hashes every declared entry.
    """
    ap = os.path.join(BASE_DIR, entry["path"])
    return os.path.isfile(ap) and os.path.getsize(ap) == entry["size_bytes"]


def _verify_manifest(manifest: dict) -> list[str]:
    """Return problems for every manifest entry (missing/size/hash mismatch).

    Iterates the declared entries, not the files discovered on disk: a download
    that silently skips one declared item must fail rather than shrink coverage.
    """
    problems: list[str] = []
    for entry in manifest["files"]:
        rel = entry["path"]
        expected_size = entry["size_bytes"]
        expected_sha = entry["sha256"].lower()
        ap = os.path.join(BASE_DIR, rel)
        if not os.path.isfile(ap):
            problems.append(f"Missing file: {rel}")
            continue
        actual_size = os.path.getsize(ap)
        if actual_size != expected_size:
            problems.append(
                f"Size mismatch: {rel}\n    expected {expected_size} bytes, actual {actual_size}"
            )
            continue
        actual_sha = compute_sha256(ap)
        if actual_sha != expected_sha:
            problems.append(
                f"Hash mismatch: {rel}\n    expected {expected_sha}\n    actual   {actual_sha}"
            )
    return problems


def verify(manifest_path: str | None = None) -> int:
    """Verify every manifest entry; return the number of checked files."""
    manifest = _load_manifest(manifest_path)
    problems = _verify_manifest(manifest)
    if problems:
        raise WeightsIntegrityError(
            "Weight verification failed:\n- " + "\n- ".join(problems)
        )
    return len(manifest["files"])


def main(argv: list | None = None) -> int:
    """Entry point for both `python -m fiximg.cli.download_weights` and the
    `fiximg-weights` console script.

    `argv` defaults to `sys.argv`. pip generates `sys.exit(main())` — the target
    is called with no arguments at all — so a required positional made the
    installed command die with `TypeError: main() missing 1 required positional
    argument: 'argv'` before it could read the command name. Keeping the
    parameter optional preserves the `main(sys.argv)` call in `__main__`.
    """
    argv = list(sys.argv if argv is None else argv)
    cmd = argv[1] if len(argv) > 1 else "help"
    if cmd in {"download", "verify", "generate"}:
        from fiximg.paths import ProjectRootNotFoundError, require_project_root

        try:
            require_project_root("weight manifest and model weight tree")
        except ProjectRootNotFoundError as exc:
            # The gate exists to produce one actionable line, not a traceback:
            # this is the entry point a wheel-only install reaches first.
            print(str(exc), file=sys.stderr)
            return 1
    if cmd == "download":
        try:
            summary = download_all()
        except (DownloadError, WeightsIntegrityError) as exc:
            print(f"Download failed: {exc}", file=sys.stderr)
            return 1
        for name, status in summary.items():
            print(f"  {name}: {status}")
        checked = len(_load_manifest()["files"])
        print(
            f"Weight integrity check passed ({checked} files) against the committed "
            "manifest; the manifest was not modified."
        )
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
    print(
        "Usage: python -m fiximg.cli.download_weights [download|generate|verify]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
