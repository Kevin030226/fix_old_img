"""Submission rate limiting 鈥?the `RATE_LIMITED` code, which the API documented
but no JSON route could emit (plan 搂2.5's unified error format).

The queue has two distinct protections and they are asserted separately: capacity
(`QUEUE_FULL`, at enqueue time) and one caller flooding the endpoint (this file).
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from fiximg.infrastructure.security import rate_limit

TOKEN = "ratelimit-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def _png(width: int = 16, height: int = 16) -> bytes:
    handle = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(handle, format="PNG")
    return handle.getvalue()


def _submit(client, name="in.png", payload=None):
    return client.post(
        "/api/v1/tasks", headers=HEADERS,
        data={"type": "colorize", "options": "{}"},
        files={"image": (name, io.BytesIO(payload if payload is not None else _png()),
                         "image/png")},
    )


@pytest.fixture()
def rl_client(tmp_path, monkeypatch):
    """An app whose queue is a recorder, so allowance is the only thing under test."""
    import fiximg.config as config_mod
    import fiximg.infrastructure.db.engine as engine
    from fiximg.app_factory import create_app
    from fiximg.application import task_service as task_service_module
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    db_path = str(tmp_path / "rl.db")
    monkeypatch.setattr(config_mod.settings, "db_path", db_path)
    monkeypatch.setattr(config_mod.settings, "tasks_root", str(tmp_path / "tasks"))
    monkeypatch.setattr(config_mod.settings, "inline_worker", False, raising=False)
    monkeypatch.setattr(engine, "DB_PATH", db_path)
    monkeypatch.setattr(engine, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(engine, "_conn", None)
    monkeypatch.setattr(task_repo, "_DDL_DONE", False, raising=False)
    monkeypatch.setenv("FIXIMG_API_TOKEN", TOKEN)

    import fiximg.api.security as api_security

    api_security.reset_cache()

    enqueued: list[str] = []
    monkeypatch.setattr(
        task_service_module.task_service, "enqueue",
        lambda image, user_state, task_type, **kwargs: (
            enqueued.append(user_state["username"]) or f"task-{len(enqueued)}"
        ),
    )

    with TestClient(create_app()) as client:
        client.enqueued = enqueued
        yield client


def _limit_to(monkeypatch, max_count: int) -> None:
    """Replace the allowance the conftest fixture handed this test."""
    monkeypatch.setattr(
        rate_limit, "submit_limiter",
        rate_limit.SlidingWindowLimiter(max_count, rate_limit.SUBMIT_WINDOW),
    )


def test_exceeding_the_submission_limit_returns_the_documented_envelope(
    rl_client, monkeypatch
):
    """`RATE_LIMITED` is in `docs/api.md`'s common codes; this makes it reachable.

    Asserted on the whole contract: the status, the code, actionable details, and 鈥?    the part that matters operationally 鈥?that the rejected submission never reached
    the queue.
    """
    _limit_to(monkeypatch, 2)

    first = _submit(rl_client)
    second = _submit(rl_client)
    third = _submit(rl_client)

    assert first.status_code == 202, first.text
    assert second.status_code == 202, second.text
    assert third.status_code == 429, third.text

    error = third.json()["error"]
    assert error["code"] == "RATE_LIMITED", error
    assert error["request_id"], "the envelope promises a request id"
    details = error["details"]
    assert details["limit"] == 2, details
    assert details["window_seconds"] == rate_limit.SUBMIT_WINDOW
    assert details["scope"] == "principal"
    assert details["remaining"] == 0
    # The prose and the numbers must agree, and both must describe the limiter that
    # actually refused the call 鈥?not whatever the module constant said at import.
    expected = (f"Too many task submissions (2 per {rate_limit.SUBMIT_WINDOW}s); "
                "try again later")
    assert error["message"] == expected, error["message"]

    assert len(rl_client.enqueued) == 2, "the rejected call must not enqueue"


def test_the_guard_runs_before_the_image_is_decoded(rl_client, monkeypatch):
    """A flood of unreadable uploads must be answered with 429, not 400.

    The whole point of limiting at the edge is that the expensive work 鈥?decoding a
    multipart body 鈥?does not happen for a request that will be refused anyway.
    """
    _limit_to(monkeypatch, 1)
    assert _submit(rl_client).status_code == 202

    over = _submit(rl_client, payload=b"not a png at all")

    assert over.status_code == 429, over.text
    assert over.json()["error"]["code"] == "RATE_LIMITED"
    assert len(rl_client.enqueued) == 1


def test_the_allowance_is_per_key_not_per_endpoint():
    """Two principals sharing one egress address must not exhaust each other.

    Tested on the limiter and the key builder rather than through two HTTP sessions:
    the route resolves exactly one principal per token, so a route-level version would
    have to fake the identity anyway 鈥?and faking it here would prove less.
    """
    from fiximg.api.dependencies import Principal

    key_a, scope_a = rate_limit.submission_key(Principal(user_id="alice"), None)
    key_b, _scope_b = rate_limit.submission_key(Principal(user_id="bob"), None)
    assert (key_a, scope_a) == ("user:alice", "principal")
    assert key_a != key_b

    limiter = rate_limit.SlidingWindowLimiter(1, 60)
    assert limiter.hit(key_a) is True
    assert limiter.hit(key_a) is False, "the second attempt on the same key is refused"
    assert limiter.hit(key_b) is True, "a different key has its own window"
    assert limiter.remaining(key_a) == 0
    assert limiter.remaining(key_b) == 0


def test_a_request_without_an_identity_falls_back_to_the_client_address():
    """The route always has a principal, but the limiter must not key everything
    together if a future caller resolves none."""
    from types import SimpleNamespace

    class _Request:
        client = SimpleNamespace(host="203.0.113.9")

        def __init__(self):
            self.headers = {}

    key, scope = rate_limit.submission_key(None, _Request())
    assert key == "ip:203.0.113.9"
    assert scope == "client-address"


def test_registering_too_often_still_gets_429(tmp_path, monkeypatch):
    """The HTML registration limiters survived the shared-window refactor.

    They were the only producer of 429 in the whole API before the submission guard
    existed, and nothing asserted them 鈥?so a change to the limiter factory that
    silently disabled them would have gone unnoticed.
    """
    import fiximg.config as config_mod
    import fiximg.infrastructure.db.engine as engine
    from fiximg.app_factory import create_app
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    db_path = str(tmp_path / "register.db")
    monkeypatch.setattr(config_mod.settings, "db_path", db_path)
    monkeypatch.setattr(engine, "DB_PATH", db_path)
    monkeypatch.setattr(engine, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(engine, "_conn", None)
    monkeypatch.setattr(task_repo, "_DDL_DONE", False, raising=False)
    monkeypatch.setattr(rate_limit, "register_ip_limiter",
                        rate_limit.SlidingWindowLimiter(1, rate_limit.REGISTER_WINDOW))
    monkeypatch.setattr(rate_limit, "register_global_limiter",
                        rate_limit.SlidingWindowLimiter(50, rate_limit.REGISTER_WINDOW))
    monkeypatch.setattr(rate_limit, "register_username_limiter",
                        rate_limit.SlidingWindowLimiter(50, rate_limit.REGISTER_WINDOW))

    with TestClient(create_app()) as client:
        form = {"username": "someone", "password": "secret123", "confirm_password": "secret123"}
        first = client.post("/register", data=form)
        second = client.post("/register", data=form)

    assert first.status_code == 200, first.text
    assert second.status_code == 429, second.text
    assert "Too many registration attempts" in second.text
