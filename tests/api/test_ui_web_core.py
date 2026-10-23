"""The native web UI's core workflow, over a real (temp) SQLite database.

These tests are the migration's functional contract. Each one corresponds to a
behaviour the Gradio UI had and the replacement must keep: sign in, upload, submit,
poll, preview, download, and the refusals (no session, another user's task, a task
that is not ready).

Heavy inference is stubbed at the stage boundary, exactly as `tests/api` does: the
plan runs through the real orchestrator, the real artifact storage, the real queue
and the real event log — only the model call is replaced. So a green run here means
the browser's path reaches the same machinery the API uses, not that a double
answered.

The app under test is built from the UI router alone (plus the exception handlers),
not from `create_app()`. That is deliberate: it keeps the router's own contract
testable in isolation, and it means these tests still describe the UI after the
composition root changes around it.
"""
from __future__ import annotations

import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from fiximg.api.errors import register_exception_handlers
from fiximg.inference.context import StageResult
from fiximg.ui.web import routes as ui_routes
from fiximg.ui.web import session as session_mod
from tests.fixtures import make_photo


class _FakeStage:
    """Stage double: marks the image so the output differs from the input."""

    def __init__(self, name="fake_stage"):
        self.name = name

    def run(self, image, context):
        from PIL import ImageDraw

        img = image.copy()
        ImageDraw.Draw(img).rectangle([0, 0, 4, 4], fill=(255, 0, 0))
        return StageResult(
            image=img,
            metadata={"report": {"degraded_count": 0}},
            message="fake stage ok",
        )


@pytest.fixture()
def ui_app(isolated_db, isolated_storage, monkeypatch, tmp_path):
    """A TestClient over the UI router, with inference stubbed and a real worker."""
    import fiximg.config as config_mod
    import fiximg.infrastructure.db.engine as engine
    from fiximg.infrastructure.db.repositories import task_repository as task_repo
    from fiximg.infrastructure.security import passwords

    # Every stage the planner can emit, so a submitted task actually runs. The
    # planner is subclassed rather than having \`build_stage\` patched onto the class:
    # the orchestrator constructs its own planner, so the substitution has to reach
    # the class both modules resolve.
    from fiximg.inference import planner as planner_mod
    from fiximg.inference import runtime as orch_mod

    class _FakePlanner(planner_mod.PipelinePlanner):
        def build_stage(self, name, kwargs):
            return _FakeStage(name)

    # Only through monkeypatch. Assigning the attribute first and *then* registering
    # it would have monkeypatch snapshot the fake as the value to restore, so the
    # double would outlive this module and every later test that builds a real
    # planner would silently run against it.
    monkeypatch.setattr(planner_mod, "PipelinePlanner", _FakePlanner)
    monkeypatch.setattr(orch_mod, "PipelinePlanner", _FakePlanner)

    # A real inline worker: the submit path must go through the queue, not around it.
    from fiximg.inference.runtime import PipelineOrchestrator
    from fiximg.inference.worker import PipelineWorker

    monkeypatch.setattr(config_mod.settings, "worker_poll_seconds", 0.02)

    worker = PipelineWorker(PipelineOrchestrator(), poll_seconds=0.02,
                            worker_id="ui-test")
    worker.start()
    monkeypatch.setattr("fiximg.app_factory._inline_worker", worker)
    # `task_service.has_worker()` asks the composition root; point it at this worker
    # so the submission takes the queued path rather than the synchronous fallback.
    monkeypatch.setattr("fiximg.app_factory.get_inline_worker", lambda: worker)

    # Accounts: one administrator, one ordinary user, one locked bootstrap account.
    task_repo._DDL_DONE = False
    engine.init_db()
    task_repo.ensure_schema()
    from fiximg.infrastructure.db.repositories import user_repository as users

    users.add_user("admin", passwords.hash_password("admin-password"), role="admin")
    users.add_user("alice", passwords.hash_password("alice-password"), role="user")
    users.add_user("locked", passwords.hash_password("locked-password"), role="user")
    users.set_must_change_password("locked", True)

    app = FastAPI()
    register_exception_handlers(app)
    ui_routes.register_ui_routes(app)
    # The app, not a client: two fixtures need two independent cookie jars, and one
    # shared client would have the second sign-in silently replace the first
    # session's cookie (which is exactly what a role-scoping test must not do).
    try:
        yield app
    finally:
        worker.stop()


def _sign_in(app, username: str, password: str):
    """A fresh client signed in as `username`, with its CSRF token and payload."""
    client = TestClient(app)
    response = client.post(
        "/ui/login",
        data={"username": username, "password": password},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    payload = client.get("/ui/api/session").json()
    return client, payload["csrf_token"], payload


@pytest.fixture()
def app_client(ui_app):
    """A client on the app that has not signed in.

    A separate fixture from :func:`signed_in` so a test that needs to watch an
    anonymous request can have one, while the sign-in helpers below each get their
    own cookie jar.
    """
    return TestClient(ui_app)


@pytest.fixture()
def signed_in(ui_app):
    """A client with an ordinary user's session cookie, plus its CSRF token."""
    return _sign_in(ui_app, "alice", "alice-password")


@pytest.fixture()
def admin_client(ui_app):
    """A client with an administrator's session cookie, plus its CSRF token."""
    return _sign_in(ui_app, "admin", "admin-password")


def _png_bytes(width=48, height=32) -> bytes:
    buffer = io.BytesIO()
    make_photo(width=width, height=height).save(buffer, format="PNG")
    return buffer.getvalue()


def _submit(client, csrf, mode="restore", options=None, image=None):
    return client.post(
        "/ui/api/tasks",
        headers={"X-CSRF-Token": csrf},
        data={"mode": mode, "options": options or "{}"},
        files={"image": ("photo.png", image or _png_bytes(), "image/png")},
    )


def _wait_for(client, task_id, deadline=30.0):
    """Poll the UI's own frame route until the task is terminal."""
    import time

    started = time.monotonic()
    while time.monotonic() - started < deadline:
        frame = client.get(f"/ui/api/tasks/{task_id}")
        assert frame.status_code == 200, frame.text
        body = frame.json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} did not finish within {deadline}s")


# ------------------------------------------------------------------ the shell
def test_the_root_route_serves_the_sign_in_page_when_nobody_is_signed_in(app_client):
    """The Gradio mount answered `/` with a login form; so does this."""
    response = app_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert 'name="username"' in body
    assert 'name="password"' in body
    assert 'type="password"' in body
    # The registration page is still reachable, as it was before.
    assert "/register" in body


def test_the_root_route_serves_the_application_once_signed_in(signed_in):
    client, _csrf, payload = signed_in
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    # Every tab the Gradio UI had, in the document.
    for marker in ("Auto Restore", "Restore (no scratches)", "Restore (scratch",
                   "Scratch detection", "Colorize", "History"):
        assert marker in body, marker
    # The upload control, the progress bar, the download link.
    assert 'type="file"' in body
    assert "data-bar" in body
    assert "Download result" in body
    # The username is rendered, escaped, from the session.
    assert payload["username"] in body


def test_the_admin_tab_is_absent_for_a_user_and_present_for_an_admin(
    signed_in, admin_client
):
    """Role drives the layout server-side, so a user cannot unhide it with a script.

    The Gradio shell hid the Admin tab with `visible=False` plus a stylesheet that
    hid the button Gradio had already rendered — two mechanisms because the framework
    would not rebuild its button list. A server-rendered page does not need either:
    the button is simply not in a user's document, which is a stronger property and
    the one asserted here.
    """
    user_html = signed_in[0].get("/").text
    admin_html = admin_client[0].get("/").text
    assert 'id="tab-admin"' not in user_html
    assert 'id="panel-admin"' not in user_html
    assert 'id="tab-admin"' in admin_html
    assert 'id="panel-admin"' in admin_html


def test_the_page_bootstraps_the_csrf_token_and_the_upload_limit(signed_in):
    """The client gets its token from the document, not from a guess."""
    client, csrf, _payload = signed_in
    body = client.get("/").text
    assert csrf in body
    assert "fiximg-bootstrap" in body
    assert "max_upload_mb" in body


def test_the_static_script_is_served_and_contains_no_innerhtml_assignment(app_client):
    """The injection property, asserted on the shipped file rather than trusted."""
    response = app_client.get("/ui/static/ui.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    script = response.text
    assert "textContent" in script
    assert ".innerHTML" not in script


def test_an_unknown_static_asset_is_a_404_and_not_a_file_read(app_client):
    """The route matches a name against a table; it never joins a path."""
    assert app_client.get("/ui/static/nope.js").status_code == 404
    assert app_client.get("/ui/static/..%2F..%2Fconfig.py").status_code == 404


# --------------------------------------------------------------------- sign in
def test_a_wrong_password_does_not_start_a_session(app_client):
    response = app_client.post(
        "/ui/login",
        data={"username": "alice", "password": "wrong"},
        follow_redirects=False,
    )
    assert response.status_code == 401
    assert "Incorrect username or password" in response.text
    assert app_client.get("/ui/api/session").status_code == 401


def test_an_unknown_account_and_a_wrong_password_read_the_same(app_client):
    """A sign-in form that distinguishes them is an account enumerator."""
    unknown = app_client.post(
        "/ui/login", data={"username": "nobody", "password": "x"}, follow_redirects=False
    )
    wrong = app_client.post(
        "/ui/login", data={"username": "alice", "password": "x"}, follow_redirects=False
    )
    assert unknown.status_code == wrong.status_code == 401
    assert "Incorrect username or password" in unknown.text
    assert "Incorrect username or password" in wrong.text


def test_signing_out_invalidates_the_session_cookie(signed_in):
    client, csrf, _payload = signed_in
    assert client.get("/ui/api/session").status_code == 200
    response = client.post("/ui/api/session/logout", headers={"X-CSRF-Token": csrf})
    assert response.status_code == 200
    assert client.get("/ui/api/session").status_code == 401


def test_a_forged_session_cookie_is_not_a_session(app_client):
    """The cookie is signed; a value that was not issued here resolves to nobody."""
    from fiximg.ui.web.session import COOKIE_NAME

    app_client.cookies.set(COOKIE_NAME, "made-up.also-made-up")
    assert app_client.get("/ui/api/session").status_code == 401
    assert "Sign in" in app_client.get("/").text


def test_an_expired_session_is_not_a_session(app_client):
    """The store's own clock decides, so a stale cookie cannot be resurrected."""
    from fiximg.ui.web.session import COOKIE_NAME, sign_session_id

    store = session_mod.SessionStore(ttl=-1)
    session = store.create("alice")
    app_client.cookies.set(COOKIE_NAME, sign_session_id(session.session_id))
    assert app_client.get("/ui/api/session").status_code == 401


def test_a_demoted_account_loses_the_admin_routes_without_signing_in_again(admin_client):
    """The role is re-read per request, so a cookie is not a stale claim."""
    client, csrf, payload = admin_client
    assert payload["is_admin"] is True
    assert client.get("/ui/api/admin/users").status_code == 200

    from fiximg.application import auth_service

    auth_service.update_user("admin", role="user")
    try:
        assert client.get("/ui/api/admin/users").status_code == 403
        assert client.get("/ui/api/session").json()["is_admin"] is False
    finally:
        auth_service.update_user("admin", role="admin")


# ------------------------------------------------------------------- refusals
def test_every_mutating_route_refuses_a_request_without_the_csrf_token(signed_in):
    """`SameSite=Lax` does not cover a top-level form POST, so the token does."""
    client, _csrf, _payload = signed_in
    image = {"image": ("photo.png", _png_bytes(), "image/png")}
    assert client.post("/ui/api/tasks", data={"mode": "restore"},
                       files=image).status_code == 403
    assert client.post("/ui/api/session/logout").status_code == 403
    assert client.post("/ui/api/session/password",
                       json={"new_password": "x" * 10,
                             "confirm_password": "x" * 10}).status_code == 403


def test_every_read_route_refuses_an_anonymous_request(app_client):
    for path in ("/ui/api/session", "/ui/api/tasks/anything", "/ui/api/history",
                 "/ui/api/admin/users", "/ui/api/admin/history"):
        assert app_client.get(path).status_code == 401, path


def test_an_ordinary_user_is_refused_the_admin_routes(signed_in):
    client, csrf, _payload = signed_in
    for path in ("/ui/api/admin/users", "/ui/api/admin/history",
                 "/ui/api/admin/archive", "/ui/api/admin/models"):
        assert client.get(path).status_code == 403, path
    assert client.post("/ui/api/admin/users",
                       headers={"X-CSRF-Token": csrf},
                       json={"username": "x", "password": "y"}).status_code == 403


# -------------------------------------------------------------------- upload
def test_an_empty_upload_is_refused_before_anything_is_queued(signed_in):
    client, csrf, _payload = signed_in
    response = client.post(
        "/ui/api/tasks",
        headers={"X-CSRF-Token": csrf},
        data={"mode": "restore"},
        files={"image": ("photo.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_a_non_image_upload_is_refused(signed_in):
    client, csrf, _payload = signed_in
    response = client.post(
        "/ui/api/tasks",
        headers={"X-CSRF-Token": csrf},
        data={"mode": "restore"},
        files={"image": ("photo.png", b"not an image", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_an_unknown_task_type_is_refused_with_the_allowed_set(signed_in):
    client, csrf, _payload = signed_in
    response = _submit(client, csrf, mode="not-a-mode")
    assert response.status_code == 400
    body = response.json()["error"]
    assert body["code"] == "UNSUPPORTED_TASK_TYPE"
    assert "auto_restore" in body["details"]["allowed"]


def test_an_unknown_option_key_is_refused(signed_in):
    """A switch the backend cannot parse must fail at the door, not inside the plan."""
    client, csrf, _payload = signed_in
    response = _submit(client, csrf, options='{"not_a_switch": true}')
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_OPTIONS"


def test_a_priority_above_the_ceiling_is_refused_not_clamped(signed_in):
    """A clamped value would let a caller believe a queue position it did not get."""
    from fiximg.config import settings

    client, csrf, _payload = signed_in
    response = client.post(
        "/ui/api/tasks",
        headers={"X-CSRF-Token": csrf},
        data={"mode": "restore", "priority": str(settings.priority_max + 1)},
        files={"image": ("photo.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 400
    body = response.json()["error"]
    assert body["details"]["max"] == settings.priority_max


def test_a_bootstrap_account_cannot_submit_before_changing_its_password(app_client):
    """§21: the force-change gate is the service's, and the UI does not bypass it."""
    app_client.post("/ui/login", data={"username": "locked", "password": "locked-password"},
                    follow_redirects=False)
    csrf = app_client.get("/ui/api/session").json()["csrf_token"]
    response = _submit(app_client, csrf)
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "PASSWORD_CHANGE_REQUIRED"


# ------------------------------------------------------- the whole round trip
def test_upload_submit_poll_preview_and_download(signed_in):
    """The Gradio flow, end to end: choose an image, submit, watch, preview, save."""
    client, csrf, _payload = signed_in
    submitted = _submit(client, csrf)
    assert submitted.status_code == 202, submitted.text
    task_id = submitted.json()["task_id"]

    frame = _wait_for(client, task_id)
    assert frame["status"] == "completed", frame
    assert frame["progress"] == 100
    assert frame["result_ready"] is True
    # The frame carries the same three things the Gradio panel showed: a head line
    # with the duration, the stage checklist, and the metrics block.
    assert "completed in" in frame["text"]
    assert frame["stages"], frame
    assert all(stage["icon"] in "✓▶○✗⊘" for stage in frame["stages"])

    preview = client.get(f"/ui/api/tasks/{task_id}/result")
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"
    assert "attachment" not in preview.headers.get("content-disposition", "")
    with Image.open(io.BytesIO(preview.content)) as image:
        assert image.size == (48, 32)

    download = client.get(f"/ui/api/tasks/{task_id}/result?download=1")
    assert download.status_code == 200
    assert "attachment" in download.headers["content-disposition"]
    assert task_id in download.headers["content-disposition"]


def test_the_original_upload_is_served_for_the_before_after_slider(signed_in):
    client, csrf, _payload = signed_in
    task_id = _submit(client, csrf).json()["task_id"]
    _wait_for(client, task_id)
    response = client.get(f"/ui/api/tasks/{task_id}/input")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_the_auto_tab_submits_and_reports_its_analysis(signed_in):
    """The one-click tab: an analysis row before submit, a plan derived from it."""
    client, csrf, _payload = signed_in
    analysed = client.post(
        "/ui/api/analysis",
        data={"mode": "auto", "options": "{}"},
        files={"image": ("photo.png", _png_bytes(), "image/png")},
    )
    assert analysed.status_code == 200, analysed.text
    body = analysed.json()
    assert body["available"] is True
    assert "Analysis:" in body["text"]
    assert "Planned pipeline:" in body["text"]

    task_id = _submit(client, csrf, mode="auto").json()["task_id"]
    frame = _wait_for(client, task_id)
    assert frame["status"] == "completed", frame


def test_a_task_of_another_user_is_reported_as_missing_not_as_forbidden(signed_in, app_client):
    """`403` would confirm the id exists, which is what a guesser wants to learn."""
    client, csrf, _payload = signed_in
    task_id = _submit(client, csrf).json()["task_id"]
    _wait_for(client, task_id)

    from fiximg.infrastructure.db.repositories import user_repository as users
    from fiximg.infrastructure.security import passwords

    users.add_user("bob", passwords.hash_password("bob-password"), role="user")
    app_client.cookies.clear()
    app_client.post("/ui/login", data={"username": "bob", "password": "bob-password"},
                    follow_redirects=False)

    for path in (f"/ui/api/tasks/{task_id}", f"/ui/api/tasks/{task_id}/result",
                 f"/ui/api/tasks/{task_id}/input", f"/ui/api/history/{task_id}"):
        response = app_client.get(path)
        assert response.status_code == 404, (path, response.status_code)


def test_a_task_that_has_not_finished_has_no_result_yet(signed_in):
    """409, not an empty 200: "not ready" and "gone" are different answers."""
    from fiximg.infrastructure.db.repositories import task_repository as repo

    client, _csrf, _payload = signed_in
    repo.create_task("ui-pending", "restore", "alice")
    response = client.get("/ui/api/tasks/ui-pending/result")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "TASK_NOT_READY"


def test_an_unknown_task_is_a_404(app_client, signed_in):
    client, _csrf, _payload = signed_in
    response = client.get("/ui/api/tasks/no-such-task")
    assert response.status_code == 404


def test_the_poll_frame_reports_a_failure_with_its_reason(signed_in):
    """A failed run must say what happened, not just stop moving."""
    from fiximg.infrastructure.db.repositories import task_repository as repo

    client, _csrf, _payload = signed_in
    repo.create_task("ui-failed", "restore", "alice")
    repo.start_task("ui-failed")
    repo.fail_task("ui-failed", "the vendored loader raised", 1200)
    frame = client.get("/ui/api/tasks/ui-failed").json()
    assert frame["status"] == "failed"
    assert frame["failed"] is True
    assert "the vendored loader raised" in frame["text"]
    assert frame["result_ready"] is False


# --------------------------------------------------------------------- history
def test_history_lists_only_the_callers_tasks(signed_in, app_client):
    client, csrf, _payload = signed_in
    task_id = _submit(client, csrf).json()["task_id"]
    _wait_for(client, task_id)

    page = client.get("/ui/api/history").json()
    assert [row["task_id"] for row in page["rows"]] == [task_id]
    assert "user 'alice'" in page["status"]
    assert page["page_label"] == "Page 1 of 1"
    assert page["rows"][0]["status"] == "completed"
    assert page["rows"][0]["duration"] != "—"


def test_the_admin_history_sees_every_user(admin_client, signed_in):
    """Administrators see all rows; that is the rule `ui/state.py` owns."""
    user_client, csrf, _payload = signed_in
    task_id = _submit(user_client, csrf).json()["task_id"]
    _wait_for(user_client, task_id)

    admin, _acsrf, _apayload = admin_client
    page = admin.get("/ui/api/history").json()
    assert task_id in [row["task_id"] for row in page["rows"]]
    assert "all users" in page["status"]


def test_a_history_row_carries_its_previews_and_its_detail(signed_in):
    client, csrf, _payload = signed_in
    task_id = _submit(client, csrf).json()["task_id"]
    _wait_for(client, task_id)
    detail = client.get(f"/ui/api/history/{task_id}").json()
    assert detail["task_id"] in detail["detail"]
    assert detail["input_url"] and detail["result_url"]
    assert detail["download_url"]
    # No mask and no face crop for a run whose stage produced neither.
    assert detail["mask_url"] is None
    assert detail["face_url"] is None
    assert detail["can_retry"] is False


def test_rerun_submits_a_new_task_from_the_stored_input(signed_in):
    client, csrf, _payload = signed_in
    task_id = _submit(client, csrf).json()["task_id"]
    _wait_for(client, task_id)
    response = client.post(f"/ui/api/history/{task_id}/rerun",
                           headers={"X-CSRF-Token": csrf}, json={})
    assert response.status_code == 200, response.text
    new_id = response.json()["task_id"]
    assert new_id != task_id
    assert new_id in response.json()["message"]


def test_retry_refuses_a_completed_task_and_requeues_a_failed_one(signed_in):
    """Retry is the same-id operation; a completed task retried is GPU work nobody asked for."""
    from fiximg.infrastructure.db.repositories import task_repository as repo

    client, csrf, _payload = signed_in
    repo.create_task("ui-done", "restore", "alice")
    repo.start_task("ui-done")
    repo.finish_task("ui-done", "/tmp/none.png", "ok", 10)
    refused = client.post("/ui/api/history/ui-done/retry",
                          headers={"X-CSRF-Token": csrf}, json={})
    assert refused.status_code == 400
    assert "retry only applies" in refused.json()["error"]["message"]

    repo.create_task("ui-broken", "restore", "alice")
    repo.start_task("ui-broken")
    repo.fail_task("ui-broken", "boom", 10)
    accepted = client.post("/ui/api/history/ui-broken/retry",
                           headers={"X-CSRF-Token": csrf}, json={})
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["task_id"] == "ui-broken"


def test_history_pagination_clamps_and_labels_pages(signed_in):
    from fiximg.infrastructure.db.repositories import task_repository as repo

    client, _csrf, _payload = signed_in
    for index in range(3):
        repo.create_task(f"ui-page-{index}", "restore", "alice")

    first = client.get("/ui/api/history?page=1&page_size=2").json()
    assert len(first["rows"]) == 2
    assert first["page_label"] == "Page 1 of 2"

    beyond = client.get("/ui/api/history?page=99&page_size=2").json()
    assert beyond["page"] == 2
    assert beyond["page_label"] == "Page 2 of 2"

    clamped = client.get("/ui/api/history?page=1&page_size=99999").json()
    assert clamped["page_size"] <= 200
    junk = client.get("/ui/api/history?page=x&page_size=y").json()
    assert junk["page"] == 1


# ---------------------------------------------------------- account / password
def test_changing_the_password_clears_the_force_change_gate(app_client):
    """The §21 unlock: after this, submission works — so both halves are asserted."""
    app_client.post("/ui/login", data={"username": "locked", "password": "locked-password"},
                    follow_redirects=False)
    csrf = app_client.get("/ui/api/session").json()["csrf_token"]

    changed = app_client.post(
        "/ui/api/session/password",
        headers={"X-CSRF-Token": csrf},
        json={"new_password": "a-new-password", "confirm_password": "a-new-password"},
    )
    assert changed.status_code == 200, changed.text
    assert app_client.get("/ui/api/session").json()["can_submit"] is True
    assert _submit(app_client, csrf).status_code == 202


def test_a_short_or_unconfirmed_password_is_refused(app_client):
    app_client.post("/ui/login", data={"username": "locked", "password": "locked-password"},
                    follow_redirects=False)
    csrf = app_client.get("/ui/api/session").json()["csrf_token"]
    short = app_client.post(
        "/ui/api/session/password",
        headers={"X-CSRF-Token": csrf},
        json={"new_password": "short", "confirm_password": "short"},
    )
    assert short.status_code == 400
    mismatch = app_client.post(
        "/ui/api/session/password",
        headers={"X-CSRF-Token": csrf},
        json={"new_password": "a-good-password", "confirm_password": "different"},
    )
    assert mismatch.status_code == 400


# ----------------------------------------------------------------------- admin
def test_the_admin_task_table_reads_the_tasks_the_pipeline_ran(admin_client):
    """The Gradio panel read a table nothing writes; this must read the real one."""
    admin, _csrf, _payload = admin_client
    from fiximg.infrastructure.db.repositories import task_repository as repo

    repo.create_task("ui-admin-1", "colorize", "u1", input_path="/tmp/in.png")
    repo.start_task("ui-admin-1")
    repo.add_metric("ui-admin-1", "psnr", 31.5)
    repo.add_metric("ui-admin-1", "ssim", 0.91)
    repo.finish_task("ui-admin-1", "/tmp/out.png", "restored", 1234)

    body = admin.get("/ui/api/admin/history").json()
    rows = [row for row in body["rows"] if row["task_id"] == "ui-admin-1"]
    assert rows, body
    row = rows[0]
    assert row["user"] == "u1"
    assert row["type"] == "colorize"
    assert row["psnr"] == "31.5"
    assert row["ssim"] == "0.91"
    assert row["input_name"] == "in.png"
    assert "Total tasks:" in body["stats"]


def test_the_archive_lists_records_and_resolves_one(admin_client):
    admin, _csrf, _payload = admin_client
    from fiximg.infrastructure.db.repositories import task_repository as repo

    repo.create_task("ui-archive-1", "restore_scratch", "bob", input_path="/tmp/in.png")
    repo.start_task("ui-archive-1")
    repo.finish_task("ui-archive-1", "/tmp/out.png", "done", 10)

    listing = admin.get("/ui/api/admin/archive").json()
    labels = {record["task_id"]: record["label"] for record in listing["records"]}
    assert "ui-archive-1" in labels
    assert "restore_scratch" in labels["ui-archive-1"]

    entry = admin.get("/ui/api/admin/archive/ui-archive-1").json()
    assert "ui-archive-1" in entry["detail"]
    # Neither file exists on disk, so neither slot offers a preview.
    assert entry["input_url"] is None and entry["output_url"] is None


def test_user_management_creates_updates_and_deletes(admin_client):
    admin, csrf, _payload = admin_client
    headers = {"X-CSRF-Token": csrf}

    created = admin.post("/ui/api/admin/users", headers=headers,
                         json={"username": "carol", "password": "carol-password",
                               "role": "user"})
    assert created.status_code == 201, created.text
    assert created.json()["user"]["username"] == "carol"

    duplicate = admin.post("/ui/api/admin/users", headers=headers,
                           json={"username": "carol", "password": "x"})
    assert duplicate.status_code == 409

    updated = admin.patch("/ui/api/admin/users/carol", headers=headers,
                          json={"role": "admin"})
    assert updated.status_code == 200, updated.text
    assert updated.json()["user"]["role"] == "admin"

    listed = admin.get("/ui/api/admin/users").json()
    assert {user["username"] for user in listed["users"]} >= {"admin", "carol"}

    deleted = admin.delete("/ui/api/admin/users/carol", headers=headers)
    assert deleted.status_code == 204
    assert "carol" not in {user["username"]
                           for user in admin.get("/ui/api/admin/users").json()["users"]}


def test_the_built_in_admin_cannot_be_deleted(admin_client):
    """Deleting it would leave an instance nobody can administer."""
    admin, csrf, _payload = admin_client
    response = admin.delete("/ui/api/admin/users/admin", headers={"X-CSRF-Token": csrf})
    assert response.status_code == 400
    assert "built-in admin" in response.json()["error"]["message"]


def test_a_user_cannot_be_updated_into_an_unknown_role(admin_client):
    admin, csrf, _payload = admin_client
    response = admin.patch("/ui/api/admin/users/alice",
                           headers={"X-CSRF-Token": csrf},
                           json={"role": "superuser"})
    assert response.status_code == 400


def test_the_model_table_is_reachable_from_the_ui(admin_client):
    """The routes existed and the Gradio panel never reached them."""
    admin, _csrf, _payload = admin_client
    response = admin.get("/ui/api/admin/models")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["models"], list)
    assert body["models"], "the manifest declares at least one model"
    assert {"name", "version", "status"} <= set(body["models"][0])


def test_an_unknown_model_action_is_refused(admin_client):
    admin, csrf, _payload = admin_client
    response = admin.post("/ui/api/admin/models/ddcolor/explode",
                          headers={"X-CSRF-Token": csrf}, json={})
    assert response.status_code == 400
