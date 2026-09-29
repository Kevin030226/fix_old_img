"""Error-code persistence and versioned weight layout (plan 搂6, 搂3.15).

Two schema-level promises from the report's data-model section:

* ``Task.error_code`` 鈥?a machine-readable failure kind, so clients branch on
  code instead of parsing prose and history can be aggregated by cause.
* ``models/<name>/vN/`` 鈥?the on-disk layout that makes "two versions resident"
  practical; without it a hot swap has nowhere to load a second version from.
"""
import json
import os

import pytest

from fiximg.domain.errors import (
    ErrorCode,
    ImageTooLargeError,
    ModelUnavailableError,
    PipelineFailedError,
)
from fiximg.infrastructure.db.repositories import task_repository as repo
from fiximg.inference.manifest import ModelManifest, load_manifest
from fiximg.inference.runtime import error_code_of


@pytest.fixture()
def tasks(isolated_db):
    return isolated_db


def _create(task_id="t-1", **kwargs):
    repo.create_task(task_id, "restore", "alice", **kwargs)


# ---------------------------------------------------------------- error codes
def test_error_code_of_maps_app_errors():
    assert error_code_of(PipelineFailedError("boom")) == ErrorCode.PIPELINE_FAILED
    assert error_code_of(ImageTooLargeError("big")) == ErrorCode.IMAGE_TOO_LARGE
    assert error_code_of(ModelUnavailableError("gone")) == ErrorCode.MODEL_UNAVAILABLE


def test_error_code_of_maps_unexpected_errors_to_internal():
    """Clients always get a stable code, even for a bare exception."""
    assert error_code_of(RuntimeError("boom")) == ErrorCode.INTERNAL_ERROR
    assert error_code_of(KeyError("nope")) == ErrorCode.INTERNAL_ERROR


def test_error_code_is_a_plain_string():
    """`ErrorCode` is a namespace of constants; `.value` does not exist."""
    assert isinstance(ErrorCode.INTERNAL_ERROR, str)
    assert not hasattr(ErrorCode.INTERNAL_ERROR, "value")


def test_fail_task_persists_the_code(tasks):
    _create("t-1")
    repo.start_task("t-1")  # a terminal write only lands on a running row
    repo.fail_task("t-1", "pipeline exploded", 1200, error_code=ErrorCode.PIPELINE_FAILED)

    row = repo.get_task("t-1")
    assert row["status"] == "failed"
    assert row["error_code"] == ErrorCode.PIPELINE_FAILED
    assert row["error_message"] == "pipeline exploded"
    assert row["duration_ms"] == 1200


def test_fail_task_without_a_code_leaves_it_null(tasks):
    _create("t-1")
    repo.start_task("t-1")  # a terminal write only lands on a running row
    repo.fail_task("t-1", "boom", 10)
    assert repo.get_task("t-1")["error_code"] is None


def test_retry_clears_the_previous_failure(tasks):
    """A retried task must not still advertise the old error."""
    _create("t-1", max_attempts=3)
    repo.start_task("t-1")  # a terminal write only lands on a running row
    repo.fail_task("t-1", "transient", 10, error_code=ErrorCode.INTERNAL_ERROR)
    assert repo.retry_task("t-1", delay_seconds=0) is True

    row = repo.get_task("t-1")
    assert row["status"] == "queued"
    assert row["error_code"] is None
    assert row["error_message"] is None


def test_manual_retry_clears_the_previous_failure(tasks):
    _create("t-1", max_attempts=1)
    repo.start_task("t-1")  # a terminal write only lands on a running row
    repo.fail_task("t-1", "boom", 10, error_code=ErrorCode.PIPELINE_FAILED)
    assert repo.requeue_for_manual_retry("t-1") is True
    assert repo.get_task("t-1")["error_code"] is None


def test_error_code_survives_the_task_aggregate(tasks):
    """The JSON aggregate must carry the column through to the API."""
    _create("t-1")
    repo.start_task("t-1")  # a terminal write only lands on a running row
    repo.fail_task("t-1", "boom", 10, error_code=ErrorCode.MODEL_UNAVAILABLE)
    assert repo.get_task("t-1")["error_code"] == ErrorCode.MODEL_UNAVAILABLE
    assert repo.list_tasks(limit=10)[0]["error_code"] == ErrorCode.MODEL_UNAVAILABLE


def test_stage_metrics_survive_the_task_aggregate(tasks):
    """搂5.3 stage metrics are stored and aggregated like the task metrics."""
    _create("t-1")
    repo.record_stage("t-1", 0, "global_restore", "running")
    repo.finish_stage("t-1", 0, "completed", 42, None, metrics={"tiles": 6})

    stages = repo.get_task("t-1")["stages"]
    assert json.loads(stages)[0]["metrics_json"] == json.dumps({"tiles": 6})


def test_finish_stage_without_metrics_stores_null(tasks):
    _create("t-1")
    repo.record_stage("t-1", 0, "global_restore", "running")
    repo.finish_stage("t-1", 0, "completed", 42)
    assert json.loads(repo.get_task("t-1")["stages"])[0]["metrics_json"] is None


# ------------------------------------------------------- versioned weight layout
@pytest.fixture()
def manifest():
    return load_manifest()


def test_shipped_manifest_declares_every_stage(manifest):
    for name in ("ddcolor", "global_restore", "scratch_repair",
                 "face_detection", "face_enhancement"):
        assert manifest.get(name) is not None, name


def test_resolve_weight_path_falls_back_to_the_declaration(manifest):
    """With no versioned directory, the declared path is used (V2 parity)."""
    path = manifest.resolve_weight_path("ddcolor")
    assert path is not None
    # The manifest writes POSIX separators; compare in a platform-neutral form.
    assert path.replace("\\", "/").endswith("weights/ddcolor/pytorch_model.pt")
    assert manifest.uses_versioned_layout("ddcolor") is False


def test_versioned_layout_wins_when_present(tmp_path, monkeypatch):
    """`models/<name>/<version>/` is preferred over the declared path."""
    import fiximg.inference.manifest as manifest_mod

    root = tmp_path
    versioned = root / "models" / "ddcolor" / "v2"
    versioned.mkdir(parents=True)
    (versioned / "pytorch_model.pt").write_bytes(b"weights")

    monkeypatch.setattr(manifest_mod, "PROJECT_ROOT", str(root))
    model = ModelManifest(models=load_manifest().models)

    path = model.resolve_weight_path("ddcolor")
    assert path == str(versioned / "pytorch_model.pt")
    assert model.uses_versioned_layout("ddcolor") is True
    assert model.available_versions("ddcolor") == ["v2"]


def test_version_can_be_requested_explicitly(tmp_path, monkeypatch):
    import fiximg.inference.manifest as manifest_mod

    root = tmp_path
    for version in ("v1", "v2"):
        directory = root / "models" / "ddcolor" / version
        directory.mkdir(parents=True)
        (directory / "pytorch_model.pt").write_bytes(version.encode())

    monkeypatch.setattr(manifest_mod, "PROJECT_ROOT", str(root))
    model = ModelManifest(models=load_manifest().models)

    assert model.resolve_weight_path("ddcolor", "v1").endswith(os.path.join("v1", "pytorch_model.pt"))
    assert model.resolve_weight_path("ddcolor", "v2").endswith(os.path.join("v2", "pytorch_model.pt"))
    assert model.available_versions("ddcolor") == ["v1", "v2"]


def test_numeric_version_is_normalised_to_vN(tmp_path, monkeypatch):
    import fiximg.inference.manifest as manifest_mod

    root = tmp_path
    directory = root / "models" / "ddcolor" / "v2"
    directory.mkdir(parents=True)
    (directory / "pytorch_model.pt").write_bytes(b"w")

    monkeypatch.setattr(manifest_mod, "PROJECT_ROOT", str(root))
    model = ModelManifest(models=load_manifest().models)
    assert model.resolve_weight_path("ddcolor", "2").endswith("pytorch_model.pt")
    assert model.versioned_dir("ddcolor", "2").endswith(os.path.join("ddcolor", "v2"))


def test_versioned_directory_with_a_checkpoint_subdir(tmp_path, monkeypatch):
    """Legacy models declare a checkpoint *directory*, not a file."""
    import fiximg.inference.manifest as manifest_mod

    root = tmp_path
    nested = root / "models" / "global_restore" / "v1" / "checkpoints"
    nested.mkdir(parents=True)

    monkeypatch.setattr(manifest_mod, "PROJECT_ROOT", str(root))
    model = ModelManifest(models=load_manifest().models)
    assert model.resolve_weight_path("global_restore") == str(nested)


def test_available_versions_of_an_unversioned_model(tmp_path, monkeypatch):
    import fiximg.inference.manifest as manifest_mod

    monkeypatch.setattr(manifest_mod, "PROJECT_ROOT", str(tmp_path))
    assert ModelManifest(models=load_manifest().models).available_versions("ddcolor") == []


def test_uses_versioned_layout_is_false_for_the_declared_path(manifest):
    assert manifest.uses_versioned_layout("ddcolor") is False


def test_resolve_weight_path_of_an_unknown_model(manifest):
    assert manifest.resolve_weight_path("not-a-model") is None
    assert manifest.available_versions("not-a-model") == []


def test_versioned_dir_defaults_to_the_declared_version(tmp_path, monkeypatch):
    import fiximg.inference.manifest as manifest_mod

    monkeypatch.setattr(manifest_mod, "PROJECT_ROOT", str(tmp_path))
    model = ModelManifest(models=load_manifest().models)
    # ddcolor declares version 1.0.0, which is not a vN directory name.
    assert model.versioned_dir("ddcolor").endswith(os.path.join("ddcolor", "1.0.0"))


def test_declared_semver_maps_onto_the_v_major_directory(tmp_path, monkeypatch):
    """`version: 1.0.0` + `models/ddcolor/v1/` must find each other."""
    import fiximg.inference.manifest as manifest_mod

    directory = tmp_path / "models" / "ddcolor" / "v1"
    directory.mkdir(parents=True)
    (directory / "pytorch_model.pt").write_bytes(b"w")

    monkeypatch.setattr(manifest_mod, "PROJECT_ROOT", str(tmp_path))
    model = ModelManifest(models=load_manifest().models)
    assert model.resolve_weight_path("ddcolor") == str(directory / "pytorch_model.pt")
    assert model.uses_versioned_layout("ddcolor") is True


def test_newest_available_version_wins_when_none_is_requested(tmp_path, monkeypatch):
    import fiximg.inference.manifest as manifest_mod

    for version in ("v1", "v3"):
        directory = tmp_path / "models" / "ddcolor" / version
        directory.mkdir(parents=True)
        (directory / "pytorch_model.pt").write_bytes(version.encode())

    monkeypatch.setattr(manifest_mod, "PROJECT_ROOT", str(tmp_path))
    model = ModelManifest(models=load_manifest().models)
    # The declared version is 1.0.0 -> v1 exists and is preferred over v3.
    assert model.resolve_weight_path("ddcolor").endswith(os.path.join("v1", "pytorch_model.pt"))
