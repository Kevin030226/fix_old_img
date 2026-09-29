"""API contract tests: the unified error envelope (plan 搂3.4.1, 搂3.8).

Every failing endpoint must return the same body shape so clients branch on
``error.code`` instead of parsing prose.
"""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A TestClient over a throwaway database and artifact root.

    Module-scoped because booting the app is the expensive part 鈥?which is exactly
    why the teardown matters: this fixture rewrites *process-global* configuration
    (`settings.db_path`, the engine's path, an env token, the schema latch), and
    without restoring it every later test in the same run inherits this module's
    database. Reproduced by running `tests/api/test_error_envelope.py` before
    `tests/unit/test_plugins_and_dialects.py`: `resolve_database_path('')` still
    answered with this temp file and two unrelated tests failed.
    """
    import os

    import fiximg.config as config_mod
    import fiximg.infrastructure.db.engine as engine
    from fiximg.app_factory import create_app
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    tmp = tmp_path_factory.mktemp("api-errors")
    db_path = str(tmp / "test.db")
    saved = {
        "db_path": config_mod.settings.db_path,
        "tasks_root": config_mod.settings.tasks_root,
        "inline_worker": config_mod.settings.inline_worker,
        "engine_db": engine.DB_PATH,
        "engine_data": engine.ADMIN_DATA_DIR,
        "ddl_done": task_repo._DDL_DONE,
        "api_token": os.environ.get("FIXIMG_API_TOKEN"),
    }
    config_mod.settings.db_path = db_path
    config_mod.settings.tasks_root = str(tmp / "storage" / "tasks")
    config_mod.settings.inline_worker = False  # no background thread in tests
    engine.DB_PATH = db_path
    engine.ADMIN_DATA_DIR = str(tmp)
    engine.close_connections()
    task_repo._DDL_DONE = False

    engine.init_db()
    task_repo.ensure_schema()

    os.environ["FIXIMG_API_TOKEN"] = "test-token"

    import fiximg.api.security as api_security

    api_security.reset_cache()

    app = create_app()
    try:
        with TestClient(app) as test_client:
            test_client.headers.update({"Authorization": "Bearer test-token"})
            yield test_client
    finally:
        engine.close_connections()
        if saved["api_token"] is None:
            os.environ.pop("FIXIMG_API_TOKEN", None)
        else:
            os.environ["FIXIMG_API_TOKEN"] = saved["api_token"]
        api_security.reset_cache()
        config_mod.settings.db_path = saved["db_path"]
        config_mod.settings.tasks_root = saved["tasks_root"]
        config_mod.settings.inline_worker = saved["inline_worker"]
        engine.DB_PATH = saved["engine_db"]
        engine.ADMIN_DATA_DIR = saved["engine_data"]
        task_repo._DDL_DONE = saved["ddl_done"]


def _png(size=(32, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "blue").save(buffer, format="PNG")
    return buffer.getvalue()


def _assert_envelope(response, code: str):
    body = response.json()
    assert set(body) == {"error"}, body
    error = body["error"]
    assert error["code"] == code, error
    assert isinstance(error["message"], str) and error["message"]
    assert "request_id" in error  # present even when empty in a bare test client
    return error


# --------------------------------------------------------------- error shapes
def test_unknown_task_returns_task_not_found(client):
    response = client.get("/api/v1/tasks/does-not-exist")
    assert response.status_code == 404
    _assert_envelope(response, "TASK_NOT_FOUND")


def test_unsupported_task_type_returns_domain_code(client):
    response = client.post(
        "/api/v1/tasks",
        data={"type": "teleport"},
        files={"image": ("in.png", _png(), "image/png")},
    )
    assert response.status_code == 400
    error = _assert_envelope(response, "UNSUPPORTED_TASK_TYPE")
    assert "teleport" in error["message"]


def test_invalid_options_json_is_rejected(client):
    response = client.post(
        "/api/v1/tasks",
        data={"type": "restore", "options": "{not json"},
        files={"image": ("in.png", _png(), "image/png")},
    )
    assert response.status_code == 400
    _assert_envelope(response, "INVALID_OPTIONS")


def test_unknown_option_switch_is_rejected(client):
    response = client.post(
        "/api/v1/tasks",
        data={"type": "restore", "options": '{"not_a_switch": true}'},
        files={"image": ("in.png", _png(), "image/png")},
    )
    assert response.status_code == 400
    error = _assert_envelope(response, "INVALID_OPTIONS")
    assert "not_a_switch" in error["message"]


def test_invalid_image_payload_is_rejected(client):
    response = client.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", b"not an image at all", "image/png")},
    )
    assert response.status_code == 400
    _assert_envelope(response, "INVALID_IMAGE")


def test_oversized_image_dimensions_are_rejected(client, monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "max_image_side", 16)
    response = client.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", _png((64, 64)), "image/png")},
    )
    assert response.status_code == 400
    _assert_envelope(response, "IMAGE_TOO_LARGE")


def test_missing_required_field_returns_validation_envelope(client):
    response = client.post("/api/v1/tasks", data={"type": "restore"})
    assert response.status_code == 422
    error = _assert_envelope(response, "INVALID_REQUEST")
    assert error["details"]["fields"]


def test_missing_model_returns_domain_code(client):
    response = client.get("/api/v1/models/not-a-model")
    assert response.status_code == 503
    _assert_envelope(response, "MODEL_UNAVAILABLE")


# ------------------------------------------------------------------- auth
def test_unauthenticated_request_is_rejected_with_envelope(client):
    anon = TestClient(client.app)
    response = anon.get("/api/v1/tasks/anything")
    assert response.status_code == 401
    _assert_envelope(response, "UNAUTHENTICATED")


def test_invalid_token_is_rejected(client):
    anon = TestClient(client.app, headers={"Authorization": "Bearer wrong"})
    response = anon.get("/api/v1/stats")
    assert response.status_code == 401
    _assert_envelope(response, "UNAUTHENTICATED")


def test_health_probes_stay_public(client):
    anon = TestClient(client.app)
    for path in ("/health", "/health/ready", "/api/v1/health/live", "/api/v1/health/ready"):
        response = anon.get(path)
        assert response.status_code == 200, path


# ----------------------------------------------------------------- success
def test_create_task_returns_a_declared_model(client):
    response = client.post(
        "/api/v1/tasks",
        data={"type": "restore", "options": '{"hr": true}'},
        files={"image": ("in.png", _png(), "image/png")},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["task_id"]
    assert "created_at" in body

    status = client.get(f"/api/v1/tasks/{body['task_id']}")
    assert status.status_code == 200
    payload = status.json()
    assert payload["task_id"] == body["task_id"]
    assert payload["status"] == "queued"
    assert payload["attempt_count"] == 0


def test_idempotency_key_returns_the_same_task(client):
    headers = {"Idempotency-Key": "idem-e2e-1"}
    first = client.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", _png(), "image/png")},
        headers=headers,
    )
    second = client.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", _png(), "image/png")},
        headers=headers,
    )
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["task_id"] == second.json()["task_id"]


def test_cancel_and_retry_roundtrip(client):
    created = client.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", _png(), "image/png")},
    ).json()

    cancelled = client.post(f"/api/v1/tasks/{created['task_id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    # Cancelling twice is a conflict, with the documented code.
    again = client.post(f"/api/v1/tasks/{created['task_id']}/cancel")
    assert again.status_code == 409
    _assert_envelope(again, "TASK_NOT_CANCELLABLE")

    retried = client.post(f"/api/v1/tasks/{created['task_id']}/retry")
    assert retried.status_code == 200
    assert retried.json()["status"] == "queued"


def test_result_not_ready_returns_documented_code(client):
    created = client.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", _png(), "image/png")},
    ).json()
    response = client.get(f"/api/v1/tasks/{created['task_id']}/result")
    assert response.status_code == 409
    _assert_envelope(response, "TASK_NOT_READY")


def test_list_tasks_is_paginated(client):
    response = client.get("/api/v1/tasks?limit=2&offset=0")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"items", "pagination"}
    assert body["pagination"]["limit"] == 2
    assert len(body["items"]) <= 2


def test_artifacts_endpoint_lists_input_artifact(client):
    created = client.post(
        "/api/v1/tasks",
        data={"type": "restore"},
        files={"image": ("in.png", _png(), "image/png")},
    ).json()
    response = client.get(f"/api/v1/tasks/{created['task_id']}/artifacts")
    assert response.status_code == 200
    kinds = [a["kind"] for a in response.json()["artifacts"]]
    assert "input" in kinds


def test_models_endpoint_lists_the_registered_backends(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    names = [m["name"] for m in response.json()["models"]]
    assert "ddcolor" in names
    assert "global_restore" in names


def test_stats_endpoint_reports_metrics_and_scheduler(client):
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    body = response.json()
    assert "tasks" in body
    assert "metrics" in body
    assert "scheduler" in body


def test_prometheus_endpoint_is_plain_text(client):
    response = client.get("/api/v1/stats/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


def test_openapi_document_declares_the_create_schema(client):
    """The multipart create endpoint must appear with a documented body."""
    schema = client.get("/openapi.json").json()
    create = schema["paths"]["/api/v1/tasks"]["post"]
    assert "requestBody" in create
    assert "multipart/form-data" in create["requestBody"]["content"]


def test_a_completed_task_whose_file_was_reclaimed_returns_410(client, tmp_path):
    """Retention and "not ready" are different answers, and clients branch on them.

    `task_result_path()` raises `ArtifactExpiredError` for a completed row whose
    file is gone; the mapping onto 410 + `ARTIFACT_EXPIRED` is what makes the
    difference visible instead of surfacing as a 500 or an empty download.
    """
    import os

    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    gone = str(tmp_path / "result-410.png")
    task_repo.create_task("exp-1", "restore", "u")
    task_repo.start_task("exp-1")
    task_repo.finish_task("exp-1", gone, None, 12)
    assert not os.path.exists(gone)

    response = client.get("/api/v1/tasks/exp-1/result")
    assert response.status_code == 410
    _assert_envelope(response, "ARTIFACT_EXPIRED")


def test_an_unfinished_task_returns_409_rather_than_410(client):
    """The two "no file yet" cases must not collapse into one status."""
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    task_repo.create_task("pending-1", "restore", "u")
    response = client.get("/api/v1/tasks/pending-1/result")
    assert response.status_code == 409
    _assert_envelope(response, "TASK_NOT_READY")
