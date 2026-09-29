"""Artifact store tests (plan 搂2.8 / 搂5.6).

Covers the local store, key normalisation (path traversal), the S3 adapter
against a fake client, and the `ArtifactStore` protocol boundary.
"""
import io

import pytest

from fiximg.domain.artifacts import ArtifactRef
from fiximg.domain.errors import ArtifactNotFoundError
from fiximg.infrastructure.storage import (
    LocalArtifactStore,
    get_artifact_store,
    local_path_of,
    purge_expired_published,
    reset_artifact_store,
)
from fiximg.infrastructure.storage.base import ArtifactStore
from fiximg.infrastructure.storage.s3 import S3ArtifactStore

# Cross-layer: the store talks to a real filesystem and a real S3 client API.
pytestmark = pytest.mark.integration


@pytest.fixture()
def store(tmp_path):
    return LocalArtifactStore(root=str(tmp_path / "tasks"))


# ------------------------------------------------------------------ local
def test_local_put_and_get_bytes(store):
    ref = store.put_bytes(b"hello", "2026/09/t-1/output/final.png", "image/png")
    assert ref.backend == "local"
    assert ref.key == "2026/09/t-1/output/final.png"
    assert store.get_bytes(ref) == b"hello"
    assert store.exists(ref) is True


def test_local_put_file_copies_content(store, tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"\x89PNG-fake")

    ref = store.put_file(str(source), "2026/09/t-1/input/in.png")
    assert store.get_bytes(ref) == b"\x89PNG-fake"
    assert store.open_path(ref).endswith("in.png")


def test_local_open_stream(store):
    ref = store.put_bytes(b"streamed", "a/b/c.bin")
    with store.open_stream(ref) as handle:
        assert handle.read() == b"streamed"


def test_local_delete_is_idempotent(store):
    ref = store.put_bytes(b"x", "a/b.bin")
    store.delete(ref)
    assert store.exists(ref) is False
    store.delete(ref)  # second delete must not raise


def test_local_missing_artifact_raises_domain_error(store):
    ref = ArtifactRef(key="nope/missing.png", backend="local")
    with pytest.raises(ArtifactNotFoundError):
        store.get_bytes(ref)
    with pytest.raises(ArtifactNotFoundError):
        store.open_path(ref)


def test_local_rejects_path_traversal(store):
    with pytest.raises(ValueError):
        store.put_bytes(b"x", "../../etc/passwd")
    with pytest.raises(ValueError):
        store.put_bytes(b"x", "")
    with pytest.raises(ValueError):
        store.path_for("a/../../b")


def test_local_normalises_backslashes_and_leading_slashes(store):
    ref = store.put_bytes(b"x", "\\2026\\09\\t-1\\out.png")
    assert ref.key == "2026/09/t-1/out.png"
    assert store.get_bytes(ref) == b"x"


def test_local_path_stays_under_root(store, tmp_path):
    ref = store.put_bytes(b"x", "2026/09/t-1/out.png")
    path = store.open_path(ref)
    assert path.startswith(str(tmp_path))


def test_local_satisfies_the_store_protocol(store):
    assert isinstance(store, ArtifactStore)


# --------------------------------------------------------------------- s3
class _FakeS3Client:
    """Minimal S3 surface: put/get/head/delete on an in-memory dict."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.content_types: dict[str, str] = {}

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803 - S3 API
        self.objects[(Bucket, Key)] = Body
        if ContentType:
            self.content_types[Key] = ContentType

    def get_object(self, Bucket, Key):  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise KeyError(f"NoSuchKey: {Key}")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, Bucket, Key):  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise KeyError("404")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.objects.pop((Bucket, Key), None)


@pytest.fixture()
def s3_store():
    return S3ArtifactStore(_FakeS3Client(), bucket="fiximg", prefix="artifacts")


def test_s3_put_and_get_roundtrip(s3_store):
    ref = s3_store.put_bytes(b"remote", "2026/09/t-1/out.png", "image/png")
    assert ref.backend == "s3"
    assert s3_store.get_bytes(ref) == b"remote"
    assert s3_store.exists(ref) is True
    assert s3_store.client.objects[("fiximg", "artifacts/2026/09/t-1/out.png")] == b"remote"


def test_s3_missing_key_raises_domain_error(s3_store):
    ref = ArtifactRef(key="missing.png", backend="s3")
    with pytest.raises(ArtifactNotFoundError):
        s3_store.get_bytes(ref)
    assert s3_store.exists(ref) is False


def test_s3_open_path_is_explicitly_unsupported(s3_store):
    ref = s3_store.put_bytes(b"x", "a.png")
    with pytest.raises(NotImplementedError):
        s3_store.open_path(ref)


def test_s3_delete_is_best_effort(s3_store):
    ref = s3_store.put_bytes(b"x", "a.png")
    s3_store.delete(ref)
    assert s3_store.exists(ref) is False
    s3_store.delete(ref)  # second delete must not raise


def test_s3_download_to_materialises_a_local_copy(s3_store, tmp_path):
    ref = s3_store.put_bytes(b"payload", "a.png")
    destination = tmp_path / "nested" / "out.png"
    s3_store.download_to(ref, str(destination))
    assert destination.read_bytes() == b"payload"


def test_s3_requires_a_bucket():
    with pytest.raises(ValueError):
        S3ArtifactStore(_FakeS3Client(), bucket="")


def test_s3_satisfies_the_store_protocol(s3_store):
    assert isinstance(s3_store, ArtifactStore)


# ------------------------------------------------------------------ factory
def test_get_artifact_store_defaults_to_local(isolated_storage, monkeypatch):
    reset_artifact_store()
    store = get_artifact_store()
    assert isinstance(store, LocalArtifactStore)
    assert store.backend == "local"
    reset_artifact_store()


def test_local_path_of_returns_none_for_remote_refs(s3_store):
    ref = s3_store.put_bytes(b"x", "a.png")
    assert local_path_of(s3_store, ref) is None


def test_local_path_of_returns_a_real_path(store):
    ref = store.put_bytes(b"x", "a.png")
    assert local_path_of(store, ref) is not None


# ------------------------------------------------------------- retention sweep
def _aged(path, seconds_old: float) -> None:
    """Backdate a directory's mtime, which is what the TTL reads."""
    import os
    import time

    stamp = time.time() - seconds_old
    os.utime(path, (stamp, stamp))


def _runs_root(tmp_path, monkeypatch):
    import fiximg.config as config_mod
    from fiximg.infrastructure.storage import local as artifact_service

    root = tmp_path / "tasks"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config_mod.settings, "tasks_root", str(root))
    return artifact_service, root


def test_purge_removes_expired_runs_and_keeps_recent_ones(tmp_path, monkeypatch):
    import os

    artifact_service, root = _runs_root(tmp_path, monkeypatch)
    old = root / "2026" / "09" / "gone"
    fresh = root / "2026" / "09" / "kept"
    old.mkdir(parents=True)
    fresh.mkdir(parents=True)
    (old / "final.png").write_bytes(b"x")
    _aged(old, 7200)

    artifact_service.purge_stale_runs(ttl_seconds=3600)

    assert not os.path.isdir(old), "an expired run directory must be reclaimed"
    assert os.path.isdir(fresh), "a recent one must survive"


def test_purge_never_touches_anything_outside_the_tasks_root(tmp_path, monkeypatch):
    """The allowlist check is the difference between a sweep and a disaster."""
    import os

    artifact_service, root = _runs_root(tmp_path, monkeypatch)
    outside = tmp_path / "elsewhere" / "run"
    outside.mkdir(parents=True)
    _aged(outside, 999_999)

    artifact_service.purge_stale_runs(ttl_seconds=1)

    assert os.path.isdir(outside)


def test_maybe_purge_is_probabilistic_but_probability_one_always_sweeps(
    tmp_path, monkeypatch
):
    import os

    artifact_service, root = _runs_root(tmp_path, monkeypatch)
    stale = root / "2026" / "09" / "swept"
    stale.mkdir(parents=True)
    _aged(stale, 999_999)

    artifact_service.maybe_purge(ttl_seconds=1, probability=0.0)
    assert os.path.isdir(stale), "probability 0 must not pay for the walk"

    artifact_service.maybe_purge(ttl_seconds=1, probability=1.0)
    assert not os.path.isdir(stale)


def test_the_worker_sweeps_on_its_own_cadence(monkeypatch, tmp_path):
    """Retention has to be reachable from the background process (plan 搂2.8).

    It used to hang off `PipelineOrchestrator.run()` only 鈥?the synchronous path 鈥?
    so on the documented API + GPU-worker topology the TTL was a number nothing
    read, and expired artifacts accumulated until the disk filled.
    """
    import fiximg.config as config_mod
    from fiximg.inference.worker import PipelineWorker
    from fiximg.infrastructure.storage import local as artifact_service

    monkeypatch.setattr(config_mod.settings, "artifact_sweep_seconds", 60.0)
    calls: list[float | None] = []
    monkeypatch.setattr(
        artifact_service, "purge_stale_runs",
        lambda ttl=None: calls.append(ttl),
    )
    worker = PipelineWorker(object(), worker_id="w-sweep")

    assert worker._sweep_artifacts(now=1_000.0) is True
    assert worker._sweep_artifacts(now=1_030.0) is False, "still inside the interval"
    assert worker._sweep_artifacts(now=1_061.0) is True
    assert len(calls) == 2


def test_a_sweep_failure_does_not_break_the_worker(monkeypatch):
    """Reclamation is the least important thing a running worker does."""
    import fiximg.config as config_mod
    from fiximg.inference.worker import PipelineWorker
    from fiximg.infrastructure.storage import local as artifact_service

    monkeypatch.setattr(config_mod.settings, "artifact_sweep_seconds", 0.0)
    monkeypatch.setattr(
        artifact_service, "purge_stale_runs",
        lambda ttl=None: (_ for _ in ()).throw(OSError("disk is busy")),
    )
    worker = PipelineWorker(object(), worker_id="w-boom")
    monkeypatch.setattr(config_mod.settings, "artifact_sweep_seconds", 60.0)

    assert worker._sweep_artifacts(now=500.0) is True  # ran, swallowed the error


def test_ttl_setting_is_what_the_sweep_uses(tmp_path, monkeypatch):
    """`FIXIMG_RESULT_TTL` must be the value the sweep honours, not a decoration."""
    import os

    import fiximg.config as config_mod
    from fiximg.infrastructure.storage import local as artifact_service

    artifact_service_mod, root = _runs_root(tmp_path, monkeypatch)
    monkeypatch.setattr(config_mod.settings, "result_ttl", 1)
    stale = root / "2026" / "09" / "by-ttl"
    stale.mkdir(parents=True)
    _aged(stale, 30)

    artifact_service.purge_stale_runs()  # no explicit ttl argument

    assert not os.path.isdir(stale)


# ------------------------------------------------ publish / recover (S3 leg)
#
# `moto` implements the S3 API in-process, so these exercise the *real* boto3 code
# path of S3ArtifactStore 鈥?not a hand-written client double. That distinction is
# the whole point of the section: the pipeline used to bypass the store protocol
# entirely, so `FIXIMG_STORAGE_BACKEND=s3` never had its upload path run.
def _requires_moto():
    pytest.importorskip("moto", reason="moto not installed")
    pytest.importorskip("boto3", reason="boto3 not installed")


@pytest.fixture()
def moto_s3(tmp_path, monkeypatch):
    """A real S3ArtifactStore over moto's in-memory S3, installed as the store."""
    _requires_moto()
    import boto3
    from moto import mock_aws

    from fiximg.config import settings
    from fiximg.infrastructure.storage import reset_artifact_store
    from fiximg.infrastructure.storage.s3 import S3ArtifactStore

    global _MOTO
    _MOTO = mock_aws()
    _MOTO.start()
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="fiximg")
    store = S3ArtifactStore(client, "fiximg")

    monkeypatch.setattr(settings, "tasks_root", str(tmp_path / "tasks"))
    import fiximg.infrastructure.storage as storage_package

    monkeypatch.setattr(storage_package, "get_artifact_store", lambda: store)
    yield store
    _MOTO.stop()
    reset_artifact_store()


def test_s3_store_is_selected_from_settings(monkeypatch):
    """Selection must be testable without a server: boto3's client is lazy."""
    _requires_moto()
    import fiximg.config as config_mod
    from fiximg.infrastructure.storage.s3 import build_s3_store_from_settings

    monkeypatch.setattr(config_mod.settings, "storage_backend", "s3")
    monkeypatch.setattr(config_mod.settings, "storage_bucket", "fiximg")
    monkeypatch.setattr(config_mod.settings, "s3_endpoint", "http://127.0.0.1:9")
    store = build_s3_store_from_settings(config_mod.settings)
    assert store is not None and store.backend == "s3"


def test_the_result_is_published_and_the_key_recorded(moto_s3, tmp_path):
    """A remote store is where a second node looks; the row has to say so."""
    from fiximg.domain.artifacts import ArtifactRef
    from fiximg.infrastructure.observability.metrics import metrics
    from fiximg.inference.runtime import PipelineOrchestrator

    run_dir = tmp_path / "tasks" / "2026" / "09" / "pub-1"
    (run_dir / "output").mkdir(parents=True)
    _png_at(run_dir / "output" / "final.png")

    uri = PipelineOrchestrator._publish_output(str(run_dir / "output" / "final.png"))

    assert uri == "2026/09/pub-1/output/final.png"
    ref = ArtifactRef(key=uri, backend="s3")
    assert moto_s3.exists(ref) is True
    assert moto_s3.get_bytes(ref) == (run_dir / "output" / "final.png").read_bytes()
    # 搂2.10: uploading without timing it is how a cross-node result ends up as the
    # slowest request nobody can explain.
    histograms = metrics.snapshot()["histograms"]
    assert any(k.startswith("artifact_io_seconds{op=publish") for k in histograms), list(histograms)


def test_a_local_store_publishes_nothing_and_leaves_the_key_null(isolated_db, monkeypatch, tmp_path):
    """The default deployment must be untouched by this code path.

    Storing a *relative* key in `uri` for the local backend would make every
    `row["uri"] or row["path"]` reader resolve a path against the process's
    working directory 鈥?a regression for the common single-node case, and the
    reason the column is NULL unless the bytes really went somewhere remote.
    """
    from fiximg.config import settings
    from fiximg.inference.runtime import PipelineOrchestrator

    monkeypatch.setattr(settings, "tasks_root", str(tmp_path / "tasks"))
    path = tmp_path / "tasks" / "2026" / "09" / "loc-1" / "output" / "final.png"
    path.parent.mkdir(parents=True)
    _png_at(path)

    assert PipelineOrchestrator._publish_output(str(path)) is None


def test_a_publish_failure_does_not_lose_the_result(moto_s3, monkeypatch, tmp_path):
    """The local file is still a valid outcome; a storage hiccup is not a failed run."""
    from fiximg.inference.runtime import PipelineOrchestrator

    def _boom(*_args, **_kwargs):
        raise RuntimeError("minio is unreachable")

    monkeypatch.setattr(moto_s3, "put_file", _boom)
    path = tmp_path / "tasks" / "2026" / "09" / "fail-1" / "output" / "final.png"
    path.parent.mkdir(parents=True)
    _png_at(path)

    assert PipelineOrchestrator._publish_output(str(path)) is None
    assert path.exists()


def _png_at(path) -> None:
    from PIL import Image

    Image.new("RGB", (6, 6), "green").save(str(path), format="PNG")


def test_download_recovers_a_published_result_that_is_not_on_this_disk(
    moto_s3, isolated_db, tmp_path, monkeypatch
):
    """The multi-node case: the worker wrote it, this node only has the database.

    Before this the API answered 410 Gone for a task that completed successfully,
    because "is the file here?" was the only question the download path asked.
    """
    from fiximg.application.task_service import task_service
    from fiximg.config import settings
    from fiximg.infrastructure.db.repositories import task_repository as repo

    monkeypatch.setattr(settings, "tasks_root", str(tmp_path / "tasks"))
    result = tmp_path / "tasks" / "2026" / "09" / "node-2" / "output" / "final.png"
    result.parent.mkdir(parents=True)
    _png_at(result)
    payload = result.read_bytes()

    import os

    from fiximg.inference.runtime import PipelineOrchestrator

    repo.create_task("node-2", "restore", "u")
    repo.start_task("node-2")
    repo.finish_task("node-2", str(result), None, 10)
    uri = PipelineOrchestrator._publish_output(str(result))
    assert uri, "the remote store should have taken the upload"
    repo.add_artifact("node-2", "output", str(result), "image/png", uri=uri)

    os.remove(result)                       # this node never had the bytes

    restored = task_service.task_result_path("node-2")
    assert os.path.isfile(restored), "the store should have served the download"
    with open(restored, "rb") as handle:
        assert handle.read() == payload


def test_an_unpublished_local_file_still_410s_after_the_sweeper(isolated_db, tmp_path, monkeypatch):
    """Recovery must not turn a genuinely deleted artifact into a mysterious success."""
    import fiximg.config as config_mod
    from fiximg.application.task_service import task_service
    from fiximg.domain.errors import ArtifactExpiredError
    from fiximg.infrastructure.db.repositories import task_repository as repo

    missing = tmp_path / "tasks" / "2026" / "09" / "gone" / "output" / "final.png"
    missing.parent.mkdir(parents=True)
    monkeypatch.setattr(config_mod.settings, "tasks_root", str(tmp_path / "tasks"))
    repo.create_task("gone-1", "restore", "u")
    repo.start_task("gone-1")
    repo.finish_task("gone-1", str(missing), None, 10)

    with pytest.raises(ArtifactExpiredError):
        task_service.task_result_path("gone-1")


def test_the_sweeper_reclaims_published_objects(moto_s3, isolated_db, tmp_path, monkeypatch):
    """Publishing without a remote reclaim turns a TTL into a lie (plan 搂2.8)."""
    import sqlite3

    from fiximg.config import settings
    from fiximg.domain.artifacts import ArtifactRef
    from fiximg.infrastructure.db import timestamps
    from fiximg.infrastructure.db.repositories import task_repository as repo
    from fiximg.inference.runtime import PipelineOrchestrator

    monkeypatch.setattr(settings, "tasks_root", str(tmp_path / "tasks"))
    result = tmp_path / "tasks" / "2026" / "09" / "old-1" / "output" / "final.png"
    result.parent.mkdir(parents=True)
    _png_at(result)

    repo.create_task("old-1", "restore", "u")
    repo.start_task("old-1")
    repo.finish_task("old-1", str(result), None, 10)
    uri = PipelineOrchestrator._publish_output(str(result))
    repo.add_artifact("old-1", "output", str(result), "image/png", uri=uri)
    ref = ArtifactRef(key=uri, backend="s3")
    assert moto_s3.exists(ref) is True

    # Age the row past the cutoff by rewriting its stored instant directly.
    conn = sqlite3.connect(settings.db_path)
    conn.execute("UPDATE artifacts SET created_at=? WHERE task_id='old-1'",
                 (timestamps.cutoff(999_999),))
    conn.commit()
    conn.close()

    removed = purge_expired_published(timestamps.cutoff(3600))

    assert removed == 1
    assert moto_s3.exists(ref) is False, "the object has to go, not just the pointer"
    assert repo.get_artifacts("old-1")[0]["uri"] is None
    assert repo.get_artifacts("old-1")[0]["path"] == str(result), "history survives"


def test_a_recently_published_object_survives_the_sweep(moto_s3, isolated_db, tmp_path):
    from fiximg.domain.artifacts import ArtifactRef
    from fiximg.infrastructure.db import timestamps

    key = "2026/09/fresh-1/output/final.png"
    moto_s3.put_bytes(b"png-bytes", key, "image/png")

    assert purge_expired_published(timestamps.cutoff(3600)) == 0
    assert moto_s3.exists(ArtifactRef(key=key, backend="s3")) is True


def test_the_local_backend_is_left_to_the_directory_sweep(isolated_db, monkeypatch):
    """Two reclaimers over the same files is how artifacts disappear early."""
    from fiximg.infrastructure.db import timestamps
    from fiximg.infrastructure.storage import local_store

    monkeypatch.setattr(
        "fiximg.infrastructure.storage.get_artifact_store", lambda: local_store
    )
    assert purge_expired_published(timestamps.cutoff(1)) == 0


# ------------------------------------------- the input travels too (搂2.8/搂4.2)
def _enqueue_input(moto_s3, isolated_db, monkeypatch):
    """Submit with a remote store and an inline worker off, so nothing executes."""
    from PIL import Image

    from fiximg.application.task_service import task_service
    from fiximg.config import settings

    monkeypatch.setattr(settings, "inline_worker", False, raising=False)
    monkeypatch.setattr(settings, "worker_max_queue", 0, raising=False)
    task_id = task_service.enqueue(Image.new("RGB", (12, 9), "orange"),
                                   {"username": "alice"}, "restore")
    return task_id


def _input_row(task_id):
    from fiximg.infrastructure.db.repositories import task_repository as repo

    return next((r for r in repo.get_artifacts(task_id) if r["kind"] == "input"), None)


def test_the_input_image_is_published_so_another_node_can_run_the_task(
    moto_s3, isolated_db, monkeypatch
):
    """Publishing only the result left the split topology half-supported.

    The API node holds the upload; a worker elsewhere claims the task and needs
    those bytes before stage 1 can even open them. Without this it failed as a
    missing-file error rather than as a storage question.
    """
    from fiximg.domain.artifacts import ArtifactRef

    task_id = _enqueue_input(moto_s3, isolated_db, monkeypatch)
    row = _input_row(task_id)
    assert row is not None and row["uri"], f"input was not published: {row}"
    assert row["uri"].endswith(".png")

    ref = ArtifactRef(key=row["uri"], backend="s3")
    assert moto_s3.exists(ref) is True
    with open(row["path"], "rb") as handle:
        assert moto_s3.get_bytes(ref) == handle.read()


def test_a_worker_recovers_an_input_it_never_had(moto_s3, isolated_db, monkeypatch, tmp_path):
    """The symmetric half: whoever runs the task must be able to fetch the input."""
    import os

    from fiximg.inference.runtime import PipelineOrchestrator

    task_id = _enqueue_input(moto_s3, isolated_db, monkeypatch)
    row = _input_row(task_id)
    with open(row["path"], "rb") as handle:
        original = handle.read()

    os.remove(row["path"])                      # this node never had it
    assert PipelineOrchestrator._recover_input(task_id, row["path"]) is True
    with open(row["path"], "rb") as handle:
        assert handle.read() == original


def test_recovery_is_honest_about_having_nothing_to_fetch(isolated_db, tmp_path):
    """A local store records no key, so a missing input must stay an error."""
    import os

    from fiximg.inference.runtime import PipelineOrchestrator
    from fiximg.infrastructure.db.repositories import task_repository as repo

    path = str(tmp_path / "in.png")
    from PIL import Image

    Image.new("RGB", (6, 6), "gray").save(path)
    repo.create_task("t-nouri", "restore", "alice", input_path=path)
    repo.add_artifact("t-nouri", "input", path, "image/png")
    os.remove(path)

    assert PipelineOrchestrator._recover_input("t-nouri", path) is False


def test_the_queued_path_asks_for_recovery_before_opening_the_input(
    isolated_db, monkeypatch, tmp_path
):
    """The call site is the contract: stage 1 must not be the one to discover this."""

    from fiximg.domain.errors import InvalidRequestError
    from fiximg.infrastructure.db.repositories import task_repository as repo
    from fiximg.inference.runtime import PipelineOrchestrator

    path = str(tmp_path / "input" / "gone.png")
    repo.create_task("t-call", "restore", "alice", input_path=path)
    repo.start_task("t-call")

    asked = []

    def refusing(self_task_id, input_path):
        asked.append((self_task_id, input_path))
        return False

    monkeypatch.setattr(PipelineOrchestrator, "_recover_input", staticmethod(refusing))
    orchestrator = PipelineOrchestrator()

    with pytest.raises(InvalidRequestError, match="Input image missing"):
        orchestrator.execute_queued("t-call", "restore")
    assert asked == [("t-call", path)], asked
