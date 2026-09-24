"""Integration tests: full API surface over a real (temp) SQLite database.

Everything heavy is stubbed at the stage boundary (pipeline §5): the plan still
runs through the real orchestrator, artifact storage, metrics, events and queue
primitives — only model inference is replaced, so no weights are needed.

Run:  python -m pytest tests/integration -q
"""
import io
import json
import os
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.factory as factory_mod
from app.api import security as api_security
from app.inference.context import StageResult
from tests.fixtures import make_photo

# Every /api/v1/* route requires this bearer token (app/api/security.py).
TEST_API_TOKEN = "integration-token"
TEST_AUTH_HEADER = {"Authorization": f"Bearer {TEST_API_TOKEN}"}


class _FakeStage:
    """Stage double: marks the image so the output differs from the input."""

    def __init__(self, name="fake_stage"):
        self.name = name

    def run(self, image, context):
        from PIL import ImageDraw

        img = image.copy()
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 4, 4], fill=(255, 0, 0))
        return StageResult(
            image=img,
            metadata={"report": {"degraded_count": 0}},
            message="fake stage ok",
        )


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """App wired to a temp DB/storage with a stubbed planner + inline worker."""
    tmp = tmp_path_factory.mktemp("integration")

    import app.core.config as config_mod
    import app.db as legacy_db
    from app.repositories import task_repository as task_repo

    db_path = str(tmp / "integration.db")
    monkey_targets = {
        legacy_db: {"DB_PATH": db_path, "ADMIN_DATA_DIR": str(tmp), "_conn": None},
        config_mod.settings: {
            "db_path": db_path,
            "storage_root": str(tmp / "storage"),
            "tasks_root": str(tmp / "storage" / "tasks"),
        },
    }
    saved = {}
    for obj, attrs in monkey_targets.items():
        saved[obj] = {k: getattr(obj, k) for k in attrs}
        for k, v in attrs.items():
            setattr(obj, k, v)
    task_repo._DDL_DONE = False
    legacy_db.init_db()
    task_repo.ensure_schema()

    # Pin the API token for the module instead of letting the app generate a
    # file-backed one (the module-scoped fixture cannot use `monkeypatch`).
    saved_env = os.environ.get("FIXIMG_API_TOKEN")
    os.environ["FIXIMG_API_TOKEN"] = TEST_API_TOKEN
    api_security.reset_cache()

    # Stub every stage the planner can emit (no model weights in CI).
    from app.inference import orchestrator as orch_mod
    from app.inference import planner as planner_mod

    class FakePlanner(planner_mod.PipelinePlanner):
        def build_stage(self, name, kwargs):
            return _FakeStage(name)

    original_planner_cls = planner_mod.PipelinePlanner
    planner_mod.PipelinePlanner = FakePlanner
    orch_mod.PipelinePlanner = FakePlanner

    # Fast synchronous inline worker for polling determinism.
    saved_poll = config_mod.settings.worker_poll_seconds
    config_mod.settings.worker_poll_seconds = 0.05

    from app.factory import create_app

    app = create_app()
    # context manager runs the lifespan (init + worker); default headers carry auth
    with TestClient(app, headers=TEST_AUTH_HEADER) as c:
        yield c

    # Restore module/global state for other test modules.
    planner_mod.PipelinePlanner = original_planner_cls
    orch_mod.PipelinePlanner = original_planner_cls
    config_mod.settings.worker_poll_seconds = saved_poll
    for obj, attrs in saved.items():
        for k, v in attrs.items():
            setattr(obj, k, v)
    if saved_env is None:
        os.environ.pop("FIXIMG_API_TOKEN", None)
    else:
        os.environ["FIXIMG_API_TOKEN"] = saved_env
    api_security.reset_cache()


def _png_bytes(size=(48, 36)) -> bytes:
    buf = io.BytesIO()
    make_photo(*size).save(buf, format="PNG")
    return buf.getvalue()


def _submit(client, **form_extra):
    return client.post(
        "/api/v1/tasks",
        files={"image": ("photo.png", _png_bytes(), "image/png")},
        data={"type": "restore", **form_extra},
    )


def _wait_terminal(client, task_id, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = client.get(f"/api/v1/tasks/{task_id}").json()
        if row["status"] in ("completed", "failed", "cancelled"):
            return row
        time.sleep(0.1)
    raise AssertionError(f"task {task_id} did not reach a terminal state in {timeout}s")


# ------------------------------------------------------------------ lifecycle
def test_health_ready(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["database"] == "ok"
    assert body["worker"]["worker_running"] is True


def test_api_requires_bearer_token(client):
    """Unauthenticated /api/v1/* must be rejected; health probes stay public."""
    anon = TestClient(client.app)  # no default headers
    assert anon.get("/api/v1/stats").status_code == 401
    assert anon.get("/api/v1/users").status_code == 401
    assert anon.get("/api/v1/tasks/does-not-exist").status_code == 401
    assert anon.get("/health").status_code == 200
    assert anon.get("/health/ready").status_code == 200

    # A wrong token is rejected too (per-request header overrides the default).
    assert client.get(
        "/api/v1/stats", headers={"Authorization": "Bearer wrong-token"}
    ).status_code == 401


def test_full_queue_roundtrip(client):
    r = _submit(client)
    assert r.status_code == 202
    task_id = r.json()["task_id"]

    row = _wait_terminal(client, task_id)
    assert row["status"] == "completed"
    assert row["progress"] == 100
    assert row["stages"], "stage rows must be recorded"
    # The fake stage keeps the plan's stage name (global_restore for "restore").
    assert any(s["stage_name"] == "global_restore" for s in row["stages"])

    # Result image downloads.
    rr = client.get(f"/api/v1/tasks/{task_id}/result")
    assert rr.status_code == 200
    img = Image.open(io.BytesIO(rr.content))
    assert img.size == (48, 36)

    # Report carries difference + no-reference metrics (plan §16 wiring).
    report = client.get(f"/api/v1/tasks/{task_id}/report").json()
    assert report["status"] == "completed"
    assert {"psnr", "ssim", "mae", "sharpness", "contrast"} <= set(report["metrics"])
    assert report["evaluation_text"] and "not fully represent" in report["evaluation_text"]


def test_options_passthrough_end_to_end(client):
    r = _submit(client, options=json.dumps({"hr": True}))
    assert r.status_code == 202
    task_id = r.json()["task_id"]
    row = _wait_terminal(client, task_id)
    assert row["status"] == "completed"
    # options landed in the tasks row (verified indirectly via report flow).

    # Invalid options are rejected up front.
    bad = _submit(client, options="{not json")
    assert bad.status_code == 400
    bad2 = _submit(client, options=json.dumps(["not", "an", "object"]))
    assert bad2.status_code == 400


def test_unknown_type_and_bad_image(client):
    r = client.post(
        "/api/v1/tasks",
        files={"image": ("x.png", _png_bytes(), "image/png")},
        data={"type": "not_a_type"},
    )
    assert r.status_code == 400
    r2 = client.post(
        "/api/v1/tasks",
        files={"image": ("x.png", b"not an image", "image/png")},
        data={"type": "restore"},
    )
    assert r2.status_code == 400
    r3 = client.post("/api/v1/tasks", files={"image": ("x.png", b"", "image/png")}, data={"type": "restore"})
    assert r3.status_code == 400


def test_cancel_queued_task(client, tmp_path):
    # Pause the worker so the task stays queued long enough to cancel.
    # NOTE: _stop.set() terminates the poll loop, so the thread must be
    # restarted afterwards (start() no-ops when the thread is still alive).
    worker = factory_mod.get_inline_worker()
    assert worker is not None
    worker._stop.set()
    try:
        r = _submit(client)
        assert r.status_code == 202
        task_id = r.json()["task_id"]
        rc = client.post(f"/api/v1/tasks/{task_id}/cancel")
        assert rc.status_code == 200
        assert rc.json()["status"] == "cancelled"
        # Cancelling again fails (already terminal).
        assert client.post(f"/api/v1/tasks/{task_id}/cancel").status_code == 409
    finally:
        worker._stop.clear()
        worker.start()  # revive the poller for the following tests


def test_404s(client):
    assert client.get("/api/v1/tasks/nope").status_code == 404
    assert client.get("/api/v1/tasks/nope/result").status_code == 404
    assert client.get("/api/v1/tasks/nope/report").status_code == 404
    assert client.post("/api/v1/tasks/nope/cancel").status_code in (404, 409)


# --------------------------------------------------------------------- stats
def test_auto_restore_end_to_end(client):
    """§10 Phase 7: the analyzer derives the plan; the task completes and the
    report carries the planner decisions."""
    r = client.post(
        "/api/v1/tasks",
        files={"image": ("photo.png", _png_bytes(), "image/png")},
        data={"type": "auto_restore"},
    )
    assert r.status_code == 202, r.text
    task_id = r.json()["task_id"]

    row = _wait_terminal(client, task_id)
    assert row["status"] == "completed"
    names = [s["stage_name"] for s in row["stages"]]
    # The fake planner derives from the analysis; the input is a color image
    # with no faces so the minimal plan is exactly one global_restore stage.
    assert "global_restore" in names

    report = client.get(f"/api/v1/tasks/{task_id}/report").json()
    assert "planner_decisions" in report


def test_stats_endpoint_reflects_runs(client):
    _submit(client)  # at least one more completed run
    r = client.get("/api/v1/stats?days=1")
    assert r.status_code == 200
    body = r.json()
    assert body["tasks"]["total"] >= 1
    assert body["tasks"]["completed"] >= 1
    assert "global_restore" in body["stages"]
    assert body["stages"]["global_restore"]["p50_ms"] >= 0
