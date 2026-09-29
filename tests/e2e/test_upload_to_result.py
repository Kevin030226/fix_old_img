"""End-to-end test: upload 鈫?queue 鈫?worker 鈫?result (plan 搂3.10.1 E2E).

The model chain is stubbed at the *stage registry*, so the test exercises the
whole platform 鈥?HTTP contract, task persistence, queue claim/lease, worker
loop, runtime, artifact storage, result download 鈥?without needing weights or a
GPU. That is the layer where regressions actually hurt.
"""
import io
import json
import re
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from fiximg.inference.context import StageResult
from fiximg.inference.stages.base import BaseStage


class _StubRestoreStage(BaseStage):
    """Stands in for the legacy Global pipeline: inverts the image."""

    name = "global_restore"
    version = "stub-1.0"
    capabilities = frozenset({"restore", "scratch_repair", "deblur", "denoise"})

    def __init__(self, with_scratch: bool = False) -> None:
        self.with_scratch = with_scratch
        if with_scratch:
            self.name = "scratch_repair"

    def run(self, image, context):
        from PIL import ImageOps

        context.report_progress(0.5, "stubbed")
        return StageResult(
            image=ImageOps.invert(image.convert("RGB")),
            metadata={"model": "stub", "with_scratch": self.with_scratch},
            artifacts={},
            metrics={"stub_restore": 1},
        )


class _StubFaceStage(BaseStage):
    """Stands in for face detection / enhancement: passes the image through."""

    name = "face_detection"
    version = "stub-1.0"
    capabilities = frozenset({"face_detection"})

    def __init__(self, name="face_detection", capabilities=("face_detection",),
                 metrics=None) -> None:
        self.name = name
        self.capabilities = frozenset(capabilities)
        self._metrics = metrics or {}

    def run(self, image, context):
        return StageResult(
            image=image,
            metadata={"model": "stub", "stage": self.name},
            artifacts={},
            metrics=dict(self._metrics),
        )


class _StubWarpBackStage(_StubFaceStage):
    """Stands in for warp-back: the composited result is what the user gets."""

    def __init__(self) -> None:
        super().__init__(
            name="warp_back",
            capabilities=("warp_back", "face_composite"),
            metrics={"enhanced_count": 1, "degraded_count": 0},
        )


class _StubSkippedStage(_StubFaceStage):
    """A stage that declines its own work, as the face chain does without dlib."""

    def run(self, image, context):
        return StageResult(
            image=image,
            metadata={"skipped": True, "reason": "face_dependencies_unavailable"},
            metrics={"face_count": 0},
            message="Face enhancement skipped: this installation has no dlib.",
        )


def _stubbed_registry(registry_module, restore_stage=_StubRestoreStage,
                      skipped: tuple[str, ...] = ()):
    """A registry with the whole four-stage chain stubbed (plan 搂3.6/搂3.7).

    Every test that swaps stages must stub *all* of them: a partially stubbed
    registry would let the real legacy CLI run for the face chain.

    ``skipped`` names the stages whose stub reports "I did nothing", which is how
    the face chain degrades on an install without dlib.
    """
    registry = registry_module.build_default_registry()
    registry.register("global_restore", restore_stage, replace=True,
                      capabilities={"restore", "deblur", "denoise"})
    registry.register("scratch_repair", lambda: restore_stage(with_scratch=True),
                      replace=True, capabilities={"restore", "scratch_repair"})
    registry.register("face_detection", _StubFaceStage, replace=True,
                      capabilities={"face_detection"})
    registry.register(
        "face_enhancement",
        lambda: _StubFaceStage("face_enhancement", ("face_restore", "face_enhance"),
                               {"enhanced_count": 1}),
        replace=True, capabilities={"face_restore", "face_enhance"},
    )
    registry.register("warp_back", _StubWarpBackStage, replace=True,
                      capabilities={"warp_back", "face_composite"})
    for name in skipped:
        registry.register(
            name,
            lambda name=name: _StubSkippedStage(name),
            replace=True,
            capabilities={
                "face_detection": {"face_detection"},
                "face_enhancement": {"face_restore", "face_enhance"},
                "warp_back": {"warp_back", "face_composite"},
            }[name],
        )
    return registry


def _app_client(tmp_path, monkeypatch, skipped: tuple[str, ...] = ()):
    """Build a live app with an inline worker and a stubbed stage chain."""
    import fiximg.api.security as api_security
    import fiximg.config as config_mod
    import fiximg.infrastructure.db.engine as engine
    from fiximg.infrastructure.db.repositories import task_repository as task_repo
    from fiximg.inference import registry as registry_module

    db_path = str(tmp_path / "e2e.db")
    monkeypatch.setattr(config_mod.settings, "db_path", db_path)
    monkeypatch.setattr(config_mod.settings, "tasks_root", str(tmp_path / "storage" / "tasks"))
    monkeypatch.setattr(config_mod.settings, "inline_worker", True)
    monkeypatch.setattr(config_mod.settings, "worker_poll_seconds", 0.05)
    monkeypatch.setattr(config_mod.settings, "worker_max_queue", 100)
    monkeypatch.setattr(engine, "DB_PATH", db_path)
    monkeypatch.setattr(engine, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(engine, "_conn", None)
    task_repo._DDL_DONE = False

    # Swap the real model stages for the stubs, keeping the registry contract.
    monkeypatch.setattr(
        registry_module, "stage_registry",
        _stubbed_registry(registry_module, skipped=skipped),
    )

    monkeypatch.setenv("FIXIMG_API_TOKEN", "e2e-token")
    api_security.reset_cache()

    from fiximg.app_factory import create_app

    app = create_app()
    with TestClient(app) as client:
        client.headers.update({"Authorization": "Bearer e2e-token"})
        yield client


@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    """A live app with an inline worker and a stubbed restoration stage."""
    yield from _app_client(tmp_path, monkeypatch)


@pytest.fixture()
def app_client_with_skipped_stage(tmp_path, monkeypatch):
    """Same, but the face-detection stage reports that it did nothing."""
    yield from _app_client(tmp_path, monkeypatch, skipped=("face_detection",))


def _png(size=(64, 48), color="orange") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _wait_for_terminal(client, task_id, timeout=30.0):
    """Poll the status endpoint until the task reaches a terminal state."""
    deadline = time.time() + timeout
    payload = {}
    while time.time() < deadline:
        payload = client.get(f"/api/v1/tasks/{task_id}").json()
        if payload["status"] in ("completed", "failed", "cancelled"):
            return payload
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} did not finish: {payload}")


def test_full_roundtrip_produces_a_downloadable_result(app_client):
    created = app_client.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", _png(), "image/png")},
    )
    assert created.status_code == 202
    task_id = created.json()["task_id"]

    final = _wait_for_terminal(app_client, task_id)
    assert final["status"] == "completed", final
    assert final["progress"] == 100
    assert final["duration_ms"] is not None
    assert final["attempt_count"] == 1

    # The stage actually ran and was recorded with its version.
    stages = {s["stage_name"]: s for s in final["stages"]}
    assert stages["global_restore"]["status"] == "completed"
    assert stages["global_restore"]["stage_version"] == "stub-1.0"

    # Artifacts: input (registered at enqueue) and output (registered by the run).
    artifacts = app_client.get(f"/api/v1/tasks/{task_id}/artifacts").json()["artifacts"]
    kinds = {a["kind"] for a in artifacts}
    assert {"input", "output"} <= kinds
    output = next(a for a in artifacts if a["kind"] == "output")
    assert output["width"] == 64 and output["height"] == 48
    assert output["sha256"]
    # The API tells a client where the bytes are without publishing the server's
    # filesystem: a store key when the deployment publishes, a path relative to
    # `tasks_root` when it keeps them locally.
    for row in artifacts:
        uri = row["uri"]
        assert uri and not uri.startswith(("/", "\\")), uri
        assert ":" not in uri and "\\" not in uri, uri
    assert output["uri"].endswith("output/final.png")

    # Which stage's model wrote the row, and in what position of the plan. Both
    # fields existed in the database while the API dropped them, so a client could
    # neither tell an old artifact from a re-run one nor name the weights behind it.
    # `output` is attributed to the *last* stage, because its bytes are that stage's
    # image resized back to the upload's geometry.
    assert output["model_version"] == "warp_back@stub-1.0"
    stages = app_client.get(f"/api/v1/tasks/{task_id}").json()["stages"]
    assert [s["stage_order"] for s in stages] == list(range(len(stages)))

    # The result endpoint returns a decodable image.
    result = app_client.get(f"/api/v1/tasks/{task_id}/result")
    assert result.status_code == 200
    assert result.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(result.content)) as downloaded:
        assert downloaded.size == (64, 48)

    # The report carries the stage metadata and evaluation text.
    report = app_client.get(f"/api/v1/tasks/{task_id}/report").json()
    assert report["status"] == "completed"
    assert report["task_type"] == "restore"
    assert report["stages"]


def test_a_stage_that_declines_work_is_recorded_as_skipped(app_client_with_skipped_stage):
    """Plan 搂3.9 asks for clearer status; "completed" for "did nothing" is not it.

    The face chain degrades this way on every install without dlib, so the
    distinction is not hypothetical: the task succeeds, the stage is what
    changed, and a client must be able to see which without parsing prose.
    """
    created = app_client_with_skipped_stage.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", _png(), "image/png")},
    ).json()
    task_id = created["task_id"]

    final = _wait_for_terminal(app_client_with_skipped_stage, task_id)
    assert final["status"] == "completed", "a skipped stage must not fail the task"
    by_stage = {s["stage_name"]: s for s in final["stages"]}
    assert by_stage["face_detection"]["status"] == "skipped"
    assert by_stage["global_restore"]["status"] == "completed"
    assert by_stage["face_detection"]["message"]

    # The result is still produced by the stages that did run.
    assert app_client_with_skipped_stage.get(f"/api/v1/tasks/{task_id}/result").status_code == 200

    stream = app_client_with_skipped_stage.get(
        f"/api/v1/tasks/{task_id}/events?follow=false"
    ).text
    stage_events = []
    for block in stream.strip().split("\n\n"):
        lines = block.splitlines()
        name = next((f[len("event: "):] for f in lines if f.startswith("event: ")), "")
        payload = next(
            (json.loads(f[len("data: "):]) for f in lines if f.startswith("data: ")), {}
        )
        if name.startswith("stage."):
            stage_events.append(((payload.get("data") or {}).get("stage"), name))

    assert ("face_detection", "stage.skipped") in stage_events
    # Exactly one terminal event for the stage: a client that counts stage events
    # to decide when a run is over must see neither two nor none.
    terminal = [event for stage, event in stage_events if stage == "face_detection"]
    assert terminal == ["stage.started", "stage.skipped"], terminal


def test_events_stream_records_the_whole_lifecycle(app_client):
    created = app_client.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", _png(), "image/png")},
    ).json()
    task_id = created["task_id"]
    _wait_for_terminal(app_client, task_id)

    # follow=false replays the persisted events and returns immediately.
    response = app_client.get(f"/api/v1/tasks/{task_id}/events?follow=false")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    body = response.text
    for expected in (
        "task.enqueued",
        # Documented in `docs/api.md`, declared in `EventType`, and previously
        # emitted by nobody: the queued path went straight from enqueued to
        # stage.started.
        "task.started",
        "stage.started",
        # 搂2.5 point 4: intra-stage progress. The stub stage reports 50 %, and
        # without this event an SSE client watching a single-stage plan saw
        # nothing at all until the stage finished.
        "task.progress",
        "stage.completed",
        "task.completed",
    ):
        assert expected in body, expected
    assert "event: task.completed" in body
    # The stub reports its own 50 %, and it is stage 1 of a four-stage plan, so the
    # task-level bar it produced is 12 %: the event carries the mapped value, not
    # the stage's local fraction.
    reported = [int(m) for m in re.findall(r'"progress":\s*(\d+)', body)]
    assert 12 in reported, reported


def test_task_is_attributed_to_the_api_principal(app_client, monkeypatch):
    """搂2.5 point 3: the task belongs to whoever the token resolves to.

    This test used to assert `settings.api_token_user == "api"` 鈥?a config
    default 鈥?which stays green when the route stops forwarding the principal,
    i.e. exactly the defect it claims to prevent. The principal's name is changed
    here instead, so the attribution has to follow it or the test fails.
    """
    from fiximg.config import settings
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    monkeypatch.setattr(settings, "api_token_user", "svc-pipeline")

    created = app_client.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", _png(), "image/png")},
    ).json()
    task_id = created["task_id"]
    _wait_for_terminal(app_client, task_id)

    row = task_repo.get_task(task_id)
    assert row["status"] == "completed"
    assert row["user_id"] == "svc-pipeline", (
        "the submission did not carry the authenticated principal"
    )


def test_scratch_task_uses_the_scratch_stage(app_client):
    created = app_client.post(
        "/api/v1/tasks",
        data={"type": "restore_scratch"},
        files={"image": ("in.png", _png(), "image/png")},
    ).json()
    final = _wait_for_terminal(app_client, created["task_id"])
    assert final["status"] == "completed"
    names = [s["stage_name"] for s in final["stages"]]
    assert names[0] == "scratch_repair"
    assert names == ["scratch_repair", "face_detection", "face_enhancement", "warp_back"]


def test_restoration_runs_the_face_chain_exactly_once(app_client):
    """搂3.6/搂3.7: four stages, one job each, no stage repeated."""
    created = app_client.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", _png(), "image/png")},
    ).json()
    final = _wait_for_terminal(app_client, created["task_id"])
    names = [s["stage_name"] for s in final["stages"]]

    assert names == ["global_restore", "face_detection", "face_enhancement", "warp_back"]
    assert len(names) == len(set(names)), f"a stage ran twice: {names}"


def test_face_enhance_can_be_opted_out(app_client):
    """`face_enhance: false` must really skip the chain, not be treated as absent."""
    created = app_client.post(
        "/api/v1/tasks",
        data={"type": "restore", "options": '{"face_enhance": false}'},
        files={"image": ("in.png", _png(), "image/png")},
    ).json()
    final = _wait_for_terminal(app_client, created["task_id"])

    assert final["status"] == "completed"
    assert [s["stage_name"] for s in final["stages"]] == ["global_restore"]


def test_stage_metrics_reach_the_report(app_client):
    """Stage-reported metrics (plan 搂5.3) are persisted and exposed."""
    created = app_client.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", _png(), "image/png")},
    ).json()
    task_id = created["task_id"]
    final = _wait_for_terminal(app_client, task_id)

    assert final["metrics"].get("enhanced_count") in (1, "1")
    assert final["metrics"].get("stub_restore") in (1, "1")

    report = app_client.get(f"/api/v1/tasks/{task_id}/report").json()
    by_stage = {s["stage_name"]: s for s in report["stages"]}
    assert by_stage["warp_back"]["metrics"]["enhanced_count"] == 1


def test_worker_survives_a_failing_stage_and_marks_the_task_failed(app_client, monkeypatch):
    """A stage error must fail the task, not the worker loop."""
    import fiximg.config as config_mod
    from fiximg.inference import registry as registry_module

    # max_attempts=1 makes the failure terminal. Otherwise the worker requeues
    # the task with a backoff after the first attempt (plan 搂2.6), so the
    # ``failed`` status is only visible for a few microseconds 鈥?the assertion
    # would be a race, not a test.
    monkeypatch.setattr(config_mod.settings, "task_max_attempts", 1)

    class _ExplodingStage(_StubRestoreStage):
        def run(self, image, context):
            raise RuntimeError("model exploded")

    broken = _stubbed_registry(registry_module, restore_stage=_ExplodingStage)
    monkeypatch.setattr(registry_module, "stage_registry", broken)

    created = app_client.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", _png(), "image/png")},
    ).json()
    final = _wait_for_terminal(app_client, created["task_id"])

    assert final["status"] == "failed"
    assert "model exploded" in final["error_message"]
    # The worker is still alive: a second task is accepted and processed.
    second = app_client.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", _png(), "image/png")},
    ).json()
    assert _wait_for_terminal(app_client, second["task_id"])["status"] == "failed"


def test_retry_after_failure_requeues_the_task(app_client, monkeypatch):
    import fiximg.config as config_mod
    from fiximg.inference import registry as registry_module

    # Terminal failure first (see the note above), then a manual retry.
    monkeypatch.setattr(config_mod.settings, "task_max_attempts", 1)

    class _ExplodingStage(_StubRestoreStage):
        def run(self, image, context):
            raise RuntimeError("transient failure")

    broken = _stubbed_registry(registry_module, restore_stage=_ExplodingStage)
    monkeypatch.setattr(registry_module, "stage_registry", broken)

    created = app_client.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", _png(), "image/png")},
    ).json()
    task_id = created["task_id"]
    assert _wait_for_terminal(app_client, task_id)["status"] == "failed"

    # Heal the pipeline, then ask for a manual retry.
    healthy = _stubbed_registry(registry_module)
    monkeypatch.setattr(registry_module, "stage_registry", healthy)

    retried = app_client.post(f"/api/v1/tasks/{task_id}/retry")
    assert retried.status_code == 200
    assert retried.json()["status"] == "queued"

    final = _wait_for_terminal(app_client, task_id)
    assert final["status"] == "completed"
    assert final["attempt_count"] == 1  # manual retry resets the counter
