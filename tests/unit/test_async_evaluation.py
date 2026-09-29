"""Deferred evaluation tests (plan 搂3.16).

The plan is explicit: *evaluation must not block the result* unless the
evaluation itself is part of the product requirement. These tests pin both
modes:

* ``inline`` (default) 鈥?V2 behaviour, metrics exist when the task completes;
* ``async`` 鈥?the task completes and the result is downloadable first, and the
  metrics land afterwards as a ``task.evaluated`` event.
"""
import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from fiximg.inference.context import StageResult
from fiximg.inference.runtime import PipelineOrchestrator, shutdown_eval_pool
from fiximg.inference.stages.base import BaseStage


class _StubRestoreStage(BaseStage):
    name = "global_restore"
    version = "stub-1.0"
    capabilities = frozenset({"restore", "scratch_repair"})

    def __init__(self, with_scratch: bool = False) -> None:
        self.with_scratch = with_scratch
        if with_scratch:
            self.name = "scratch_repair"

    def run(self, image, context):
        from PIL import ImageOps

        return StageResult(image=ImageOps.invert(image.convert("RGB")), metadata={})


class _StubPassThroughStage(BaseStage):
    """Stands in for the face chain stages, which pass the image through."""

    name = "face_detection"
    version = "stub-1.0"
    capabilities = frozenset({"face_detection"})

    def __init__(self, name="face_detection", capabilities=("face_detection",)) -> None:
        self.name = name
        self.capabilities = frozenset(capabilities)

    def run(self, image, context):
        return StageResult(image=image, metadata={"stage": self.name})


def _stub_chain(registry_module):
    """Register the full four-stage restoration chain (plan 搂3.6/搂3.7)."""
    stubbed = registry_module.build_default_registry()
    stubbed.register("global_restore", _StubRestoreStage, replace=True)
    stubbed.register(
        "scratch_repair", lambda: _StubRestoreStage(with_scratch=True), replace=True
    )
    stubbed.register("face_detection", _StubPassThroughStage, replace=True,
                     capabilities={"face_detection"})
    stubbed.register(
        "face_enhancement",
        lambda: _StubPassThroughStage("face_enhancement", ("face_restore", "face_enhance")),
        replace=True, capabilities={"face_restore", "face_enhance"},
    )
    stubbed.register(
        "warp_back",
        lambda: _StubPassThroughStage("warp_back", ("warp_back", "face_composite")),
        replace=True, capabilities={"warp_back", "face_composite"},
    )
    return stubbed


@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    import fiximg.api.security as api_security
    import fiximg.config as config_mod
    import fiximg.infrastructure.db.engine as engine
    from fiximg.infrastructure.db.repositories import task_repository as task_repo
    from fiximg.inference import registry as registry_module

    db_path = str(tmp_path / "eval.db")
    monkeypatch.setattr(config_mod.settings, "db_path", db_path)
    monkeypatch.setattr(config_mod.settings, "tasks_root", str(tmp_path / "storage" / "tasks"))
    monkeypatch.setattr(config_mod.settings, "inline_worker", True)
    monkeypatch.setattr(config_mod.settings, "worker_poll_seconds", 0.05)
    monkeypatch.setattr(engine, "DB_PATH", db_path)
    monkeypatch.setattr(engine, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(engine, "_conn", None)
    task_repo._DDL_DONE = False

    monkeypatch.setattr(registry_module, "stage_registry", _stub_chain(registry_module))

    monkeypatch.setenv("FIXIMG_API_TOKEN", "eval-token")
    api_security.reset_cache()

    from fiximg.app_factory import create_app

    app = create_app()
    with TestClient(app) as client:
        client.headers.update({"Authorization": "Bearer eval-token"})
        yield client
    shutdown_eval_pool(wait=True)


def _png(size=(48, 32), color="orange") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _submit(client):
    return client.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", _png(), "image/png")},
    ).json()["task_id"]


def _status_of(payload) -> str | None:
    """The task status, or None when the response was not a task document.

    Indexing `payload["status"]` directly turned any transient error envelope
    (a 404 from a momentary lock, a 5xx) into a `KeyError: 'status'` in this
    helper - a failure that names nothing. Callers keep polling and report the
    last payload they saw if the task never reaches a terminal state.
    """
    if not isinstance(payload, dict):
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None


def _wait(client, task_id, timeout=30.0):
    deadline = time.time() + timeout
    payload: object = {}
    while time.time() < deadline:
        payload = client.get(f"/api/v1/tasks/{task_id}").json()
        if _status_of(payload) in ("completed", "failed", "cancelled"):
            return payload
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} did not finish: {payload}")


def _wait_for_metrics(client, task_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/v1/tasks/{task_id}").json()
        if _status_of(payload) in ("completed", "failed", "cancelled") and payload.get("metrics"):
            return payload
        time.sleep(0.05)
    return client.get(f"/api/v1/tasks/{task_id}").json()


# ------------------------------------------------------------------- inline
def test_inline_mode_has_metrics_when_the_task_completes(app_client, monkeypatch):
    """The default keeps V2 behaviour: metrics are ready with the result."""
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "eval_mode", "inline")
    task_id = _submit(app_client)
    final = _wait(app_client, task_id)

    assert final["status"] == "completed"
    assert final["metrics"], "inline evaluation must produce metrics immediately"
    assert "sharpness" in final["metrics"] or "psnr" in final["metrics"]


def test_inline_mode_has_no_evaluated_event(app_client, monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "eval_mode", "inline")
    task_id = _submit(app_client)
    _wait(app_client, task_id)

    events = app_client.get(f"/api/v1/tasks/{task_id}/events?follow=false").text
    assert "task.evaluated" not in events


# -------------------------------------------------------------------- async
def test_async_mode_returns_the_result_before_the_metrics(app_client, monkeypatch):
    """The key 搂3.16 property: the result is downloadable while metrics run."""
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "eval_mode", "async")
    task_id = _submit(app_client)
    final = _wait(app_client, task_id)

    assert final["status"] == "completed"
    # The result is already there...
    result = app_client.get(f"/api/v1/tasks/{task_id}/result")
    assert result.status_code == 200

    # ...and the metrics arrive afterwards, without changing the status.
    evaluated = _wait_for_metrics(app_client, task_id)
    assert evaluated["metrics"]
    assert evaluated["status"] == "completed"


def test_async_mode_emits_a_task_evaluated_event(app_client, monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "eval_mode", "async")
    task_id = _submit(app_client)
    _wait(app_client, task_id)
    _wait_for_metrics(app_client, task_id)

    # Poll the event stream briefly: the evaluator writes it asynchronously.
    deadline = time.time() + 10
    body = ""
    while time.time() < deadline:
        body = app_client.get(f"/api/v1/tasks/{task_id}/events?follow=false").text
        if "task.evaluated" in body:
            break
        time.sleep(0.05)
    assert "task.evaluated" in body


def test_async_mode_still_reports_stage_timings(app_client, monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "eval_mode", "async")
    task_id = _submit(app_client)
    final = _wait(app_client, task_id)

    assert [s["stage_name"] for s in final["stages"]] == [
        "global_restore", "face_detection", "face_enhancement", "warp_back",
    ]
    assert final["duration_ms"] is not None


# ------------------------------------------------------------------ fallback
def test_evaluation_failure_never_fails_the_task(app_client, monkeypatch):
    """Metrics are diagnostics: a broken evaluator must not break the result."""
    import fiximg.config as config_mod
    from fiximg.inference import runtime as runtime_mod

    monkeypatch.setattr(config_mod.settings, "eval_mode", "async")

    def boom(*_args, **_kwargs):
        raise RuntimeError("evaluator exploded")

    monkeypatch.setattr(runtime_mod.PipelineOrchestrator, "_evaluate", boom)
    task_id = _submit(app_client)
    final = _wait(app_client, task_id)

    assert final["status"] == "completed"
    assert app_client.get(f"/api/v1/tasks/{task_id}/result").status_code == 200
    # Give the background job a moment to fail without taking anything down.
    time.sleep(0.3)
    assert app_client.get(f"/api/v1/tasks/{task_id}").json()["status"] == "completed"


def test_shutdown_eval_pool_is_idempotent():
    shutdown_eval_pool(wait=True)
    shutdown_eval_pool(wait=True)  # must not raise


def test_orchestrator_evaluates_inline_for_the_sync_path(monkeypatch, isolated_db, tmp_path):
    """`run()` returns the text, so it can never defer evaluation."""
    import fiximg.config as config_mod
    from fiximg.inference import registry as registry_module

    monkeypatch.setattr(config_mod.settings, "eval_mode", "async")
    monkeypatch.setattr(config_mod.settings, "tasks_root", str(tmp_path / "storage" / "tasks"))

    monkeypatch.setattr(registry_module, "stage_registry", _stub_chain(registry_module))

    image = Image.new("RGB", (32, 24), "blue")
    result_path, evaluation_text = PipelineOrchestrator().run(
        image, {"username": "alice"}, "restore"
    )
    assert result_path
    assert evaluation_text, "the synchronous path must return the evaluation text"

