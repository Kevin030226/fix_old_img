"""Integration tests: full API surface over a real (temp) SQLite database.

Everything heavy is stubbed at the stage boundary (pipeline 搂5): the plan still
runs through the real orchestrator, artifact storage, metrics, events and queue
primitives 鈥?only model inference is replaced, so no weights are needed.

Run:  python -m pytest tests/integration -q
"""
import io
import json
import os
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import fiximg.app_factory as factory_mod
from fiximg.api import security as api_security
from fiximg.inference.context import StageResult
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

    import fiximg.config as config_mod
    import fiximg.infrastructure.db.engine as legacy_db
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

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
    from fiximg.inference import runtime as orch_mod
    from fiximg.inference import planner as planner_mod

    class FakePlanner(planner_mod.PipelinePlanner):
        def build_stage(self, name, kwargs):
            return _FakeStage(name)

    original_planner_cls = planner_mod.PipelinePlanner
    planner_mod.PipelinePlanner = FakePlanner
    orch_mod.PipelinePlanner = FakePlanner

    # Fast synchronous inline worker for polling determinism.
    saved_poll = config_mod.settings.worker_poll_seconds
    config_mod.settings.worker_poll_seconds = 0.05

    from fiximg.app_factory import create_app

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

    # Report carries difference + no-reference metrics (plan 搂16 wiring).
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


@pytest.mark.parametrize("switch", ["face_enhance", "auto_colorize"])
def test_an_option_the_client_never_mentioned_is_not_stored_as_a_refusal(
    client, switch
):
    """Only the keys the caller named reach the queue (plan 搂12 on the Auto path).

    The route serialises with ``exclude_defaults=True``, so a field whose default is
    ``False`` would make an explicit refusal identical to silence. On ``restore`` that is
    invisible (colorization was off anyway); on ``auto_restore``, where the analysis
    decides, silence means "colorize it if it is black and white" 鈥?so a caller that
    typed ``false`` would get the stage anyway. This reads the stored row back rather
    than the parser, because the defect lives in what the route sends on.
    """
    from fiximg.infrastructure.db.repositories import task_repository

    r = _submit(client, type="auto_restore", options=json.dumps({switch: False}))
    assert r.status_code == 202
    task_id = r.json()["task_id"]
    _wait_terminal(client, task_id)

    stored = json.loads(task_repository.get_task(task_id)["options_json"] or "{}")
    assert stored == {switch: False}, stored


def test_no_options_means_no_keys_reach_the_row(client):
    """The other half: an option-less submission must not look like a refusal."""
    from fiximg.infrastructure.db.repositories import task_repository

    r = _submit(client, type="auto_restore")
    task_id = r.json()["task_id"]
    _wait_terminal(client, task_id)

    stored = json.loads(task_repository.get_task(task_id)["options_json"] or "{}")
    assert stored == {}, stored


@pytest.mark.parametrize("switch", ["hr", "face_enhance", "auto_colorize"])
def test_every_advertised_option_is_accepted_on_the_wire(client, switch):
    """The whitelist the parser uses is the list `docs/api.md` publishes.

    Asserted through the endpoint rather than the schema so a key that is accepted but
    then dropped between the route and the row is caught here too.
    """
    from fiximg.infrastructure.db.repositories import task_repository

    r = _submit(client, options=json.dumps({switch: True}))
    assert r.status_code == 202, r.text
    _wait_terminal(client, r.json()["task_id"])

    stored = json.loads(task_repository.get_task(r.json()["task_id"])["options_json"] or "{}")
    assert stored == {switch: True}, stored


def test_an_option_nothing_reads_is_rejected_rather_than_stored(client):
    """`strength` was the option that lived here: accepted, stored, documented, unread.

    Rejecting it is the honest half of removing a switch. Keeping it in the whitelist
    would let a client send `{"strength": 0.3}`, get a 202 and a row describing the
    request, and receive the identical image it would have got without it 鈥?the failure
    mode the vocabulary gate in `tests/unit/test_option_vocabulary.py` now closes for
    every declared key.
    """
    r = _submit(client, options=json.dumps({"strength": 0.5}))
    assert r.status_code == 400, r.text
    error = r.json()["error"]
    assert error["code"] == "INVALID_OPTIONS", error
    assert "strength" in error["message"], error


# --------------------------------------------------------------------- priority
def _multipart_fields():
    """The multipart field names the create endpoint declares.

    Both kinds count: FastAPI's ``File`` derives from ``Form``, which derives from
    ``Body``, and the documentation model lists the upload parts alongside the value
    parts. Filtering on ``Form`` alone would ride on that class hierarchy by accident, and
    a renamed upload part would pass.
    """
    import inspect

    from fastapi.params import Body as BodyParam

    from fiximg.api.routes.tasks import create_task

    return {
        name
        for name, parameter in inspect.signature(create_task).parameters.items()
        if isinstance(parameter.default, BodyParam)
    }


def test_the_documented_multipart_shape_is_the_one_the_endpoint_accepts(client):
    """`TaskCreateForm` is what the OpenAPI document shows for a multipart body.

    Multipart cannot be a Pydantic body, so the shape is declared twice: once as the
    endpoint's own parameters, once as the documentation model injected through
    `openapi_extra`. When only one of them is updated, the document advertises a field the
    endpoint ignores (or hides one it honours) and no client-side test would notice;
    `priority` is exactly the kind of field that gets added to one list and not the other.
    """
    from fiximg.api.schemas.task import TaskCreateForm

    assert _multipart_fields() == {
        "type", "image", "options", "priority", "ground_truth",
    }, _multipart_fields()
    assert set(TaskCreateForm.model_fields) == _multipart_fields()

    # And the document really carries it: the create operation's request schema.
    schema = client.get("/openapi.json").json()
    content = schema["paths"]["/api/v1/tasks"]["post"]["requestBody"]["content"]
    ref = content["multipart/form-data"]["schema"].get("$ref", "").rsplit("/", 1)[-1]
    properties = schema["components"]["schemas"][ref]["properties"]
    assert set(properties) == set(TaskCreateForm.model_fields), sorted(properties)


def test_priority_is_stored_and_echoed_back(client):
    """The queue orders `priority DESC`, so the accepted value has to be visible.

    Read back from the row, not from the request: a form field that is parsed and then
    dropped on the way to `create_task` would still produce a 202, and the client's only
    evidence would be a response that repeats its own input.
    """
    from fiximg.infrastructure.db.repositories import task_repository

    r = _submit(client, priority="7")
    assert r.status_code == 202, r.text
    task_id = r.json()["task_id"]

    assert task_repository.get_task(task_id)["priority"] == 7
    assert r.json()["priority"] == 7
    assert client.get(f"/api/v1/tasks/{task_id}").json()["priority"] == 7


def test_the_priority_a_submission_omits_is_fifo(client):
    """Default 0 is the queue's normal order, not an unvalidated blank."""
    from fiximg.infrastructure.db.repositories import task_repository

    r = _submit(client)
    assert r.status_code == 202, r.text
    assert r.json()["priority"] == 0
    assert task_repository.get_task(r.json()["task_id"])["priority"] == 0


def test_a_priority_above_the_deployment_limit_is_refused_with_the_deciding_number(
    client, monkeypatch
):
    """Refused rather than clamped, and quoting the limit that actually decided.

    Silently clamping `20` to `10` would hand the caller a queue position it never asked
    for, and a `details.max` copied from a module constant could disagree with a
    deployment that set `FIXIMG_PRIORITY_MAX` 鈥?the same lesson the rate limiter was
    fixed on.
    """
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "priority_max", 3, raising=False)

    r = _submit(client, priority="9")
    assert r.status_code == 400, r.text
    error = r.json()["error"]
    assert error["code"] == "INVALID_REQUEST", error
    assert error["details"] == {"priority": 9, "min": 0, "max": 3}, error

    negative = _submit(client, priority="-1")
    assert negative.status_code == 400, negative.text
    assert negative.json()["error"]["details"]["min"] == 0


def test_an_allowed_priority_is_accepted_at_the_top_of_the_range(client, monkeypatch):
    """The bound is inclusive: `FIXIMG_PRIORITY_MAX` itself is the most urgent row."""
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "priority_max", 4, raising=False)

    r = _submit(client, priority="4")
    assert r.status_code == 202, r.text
    assert r.json()["priority"] == 4


def test_an_idempotent_replay_reports_the_priority_the_first_submission_set(client):
    """A replay returns the original task, so it must report the original queue position.

    Echoing the *requested* value here would be a lie about a task this call did not
    create 鈥?and the whole point of the field is that a client can check where its work
    stands in the queue.
    """
    key = "priority-replay"
    first = client.post(
        "/api/v1/tasks",
        files={"image": ("photo.png", _png_bytes(), "image/png")},
        headers={"Idempotency-Key": key},
        data={"type": "restore", "priority": "6"},
    )
    assert first.status_code == 202, first.text
    replay = client.post(
        "/api/v1/tasks",
        files={"image": ("photo.png", _png_bytes(), "image/png")},
        headers={"Idempotency-Key": key},
        data={"type": "restore", "priority": "1"},
    )
    assert replay.status_code == 202, replay.text

    body = replay.json()
    assert body["task_id"] == first.json()["task_id"], "the replay created a second task"
    assert body["priority"] == 6, body


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

        # The cancellation must reach the event stream: `task.cancelled` is one of
        # the events the SSE stream treats as closing, so a client that was
        # watching this task would otherwise poll until the 900-second deadline
        # with no way to tell "cancelled" from "hung" (plan 搂2.5 point 4).
        events = client.get(f"/api/v1/tasks/{task_id}/events?follow=false")
        assert events.status_code == 200
        assert "event: task.cancelled" in events.text, events.text
        assert events.text.rstrip().endswith("}")  # the stream ended, it did not stall
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
    """搂10 Phase 7: the analyzer derives the plan; the task completes and the
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


def test_a_running_task_can_be_cancelled_over_http(client):
    """搂3.8: this used to answer 409 for every task a worker already held.

    Measured on a real split-topology boot: the only cancellable window was between
    enqueue and claim, which for a 4-60 second GPU task is a few hundred milliseconds 鈥?
    so the endpoint existed and could not be used. Cancelling a running task is now
    accepted, and the event says what will really happen: stop at the next stage.
    """
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    worker = factory_mod.get_inline_worker()
    assert worker is not None
    worker._stop.set()  # keep it queued long enough to claim it by hand
    try:
        task_id = _submit(client).json()["task_id"]
        # `start_task` rather than a claim: the claim takes the oldest eligible row,
        # and earlier tests in this file leave queued rows behind, which would make
        # *their* status the thing this assertion reads.
        task_repo.start_task(task_id)
        assert task_repo.get_task(task_id)["status"] == "running"

        response = client.post(f"/api/v1/tasks/{task_id}/cancel")
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "cancelled"

        stream = client.get(f"/api/v1/tasks/{task_id}/events?follow=false")
        assert "event: task.cancelled" in stream.text, stream.text
        assert not [e for e in task_repo.list_task_events(task_id)
                    if e["event_type"] == "task.failed"], "cancel was reported as failure"
        record = next(e for e in task_repo.list_task_events(task_id)
                      if e["event_type"] == "task.cancelled")
        import json as _json

        payload = _json.loads(record["data_json"] or "{}")
        # A client watching the stream must be able to tell a stopped-while-running
        # task from one that never started: the stage rows it already saw differ.
        assert payload["effective"] == "stage boundary", payload
        assert "stage" in record["message"], record["message"]
    finally:
        worker._stop.clear()
        worker.start()
