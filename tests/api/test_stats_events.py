"""`GET /api/v1/stats/events`: the deployment-wide event view.

`GET /tasks/{id}/events` streams one task's lifecycle, so the events that carry
no `task_id` were unreachable through the API: a submission rejected because the
queue was saturated, a worker that started or died, a task whose row was never
created. The list_events reader existed and had tests, but no production caller —
so the whole question "why is my submission not in the queue" had no answer on the
wire even though the data was in the table.

These tests pin the two things that make the endpoint useful rather than merely
present: the task-less events are actually in it, and the per-task stream is not a
substitute for it.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fiximg.api import security as api_security
from fiximg.domain.events import EventType
from fiximg.infrastructure.db.repositories import task_repository as repo


@pytest.fixture()
def client(isolated_db, monkeypatch):
    api_security.reset_cache()
    monkeypatch.setenv("FIXIMG_API_TOKEN", "test-token")
    api_security.reset_cache()
    from fiximg.app_factory import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
    api_security.reset_cache()


def _headers():
    return {"Authorization": "Bearer test-token"}


def test_it_returns_the_events_a_task_scoped_stream_cannot(client):
    """A rejected submission has no task row, so `/tasks/{id}/events` cannot show it.

    This is the reason the endpoint exists at all, so it is asserted on the
    *absence* of a task id rather than on the presence of a type.
    """
    repo.create_task("t-1", "restore", "u")
    repo.start_task("t-1")
    repo.add_event(EventType.TASK_ENQUEUED, task_id="t-1", message="queued")
    repo.add_event(
        EventType.TASK_REJECTED, level="warning",
        message="queue is full", data={"depth": 500},
    )

    body = client.get("/api/v1/stats/events", headers=_headers()).json()

    taskless = [e for e in body["events"] if e["task_id"] is None]
    assert taskless, body
    assert taskless[0]["event"] == EventType.TASK_REJECTED
    assert taskless[0]["level"] == "warning"
    assert taskless[0]["data"] == {"depth": 500}
    # ...and the per-task reader confirms the other one is invisible there.
    scoped = repo.list_task_events("t-1")
    assert all(row["task_id"] == "t-1" for row in scoped)


def test_it_is_newest_first_and_honours_the_limit(client):
    for n in range(5):
        repo.add_event("task.progress", task_id=f"t-{n}", message=str(n))

    body = client.get("/api/v1/stats/events?limit=3", headers=_headers()).json()
    seqs = [e["seq"] for e in body["events"]]
    assert len(seqs) == 3
    assert seqs == sorted(seqs, reverse=True), seqs


def test_the_level_filter_is_the_only_filter_it_offers(client):
    repo.add_event("task.progress", level="info", message="i")
    repo.add_event("task.failed", level="error", message="e")

    body = client.get("/api/v1/stats/events?level=error", headers=_headers()).json()
    assert [e["message"] for e in body["events"]] == ["e"]


def test_it_requires_the_api_token(client):
    """The deployment view is as sensitive as the task view.

    It carries usernames in `user_id` and the failure prose of every task, so an
    unauthenticated read of it would be a wider disclosure than
    `/tasks/{id}/events`, not a narrower one.
    """
    assert client.get("/api/v1/stats/events").status_code == 401


def test_an_empty_table_reads_as_an_empty_list(client):
    body = client.get("/api/v1/stats/events", headers=_headers()).json()
    assert body == {"events": []}


def test_it_is_declared_in_openapi_with_a_schema(client):
    """Not a free-form object, like the two endpoints the schema gate once missed."""
    spec = client.get("/openapi.json").json()
    ok = spec["paths"]["/api/v1/stats/events"]["get"]["responses"]["200"]
    assert "application/json" in ok["content"], ok
    ref = ok["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith("RecentEventsResponse"), ref
