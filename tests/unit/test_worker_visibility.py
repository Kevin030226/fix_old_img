"""The UI must not run inference in the API process while a worker is running.

Found by doing it, and the cost of not noticing was a CUDA OOM rather than a
failed request. The deployment was right: `fiximg.cli.api` with
`FIXIMG_INLINE_WORKER=0` and a separate `fiximg.cli.worker`, which is the
documented split topology. Tasks completed and `/health/ready` reported the worker
topology, so nothing looked wrong.

What was wrong: the UI cannot see that worker. `has_worker()` reads
`settings.external_worker` — `FIXIMG_HAS_EXTERNAL_WORKER`, default false — because
a separate process is not observable from here. With it unset, every submission
took the synchronous fallback and ran inference *inside the API process* while the
worker held its own copy of the models. Two model sets on one 8 GB card, and the
result was

    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.22 GiB.
    GPU 0 has a total capacity of 7.93 GiB of which 0 bytes is free.

raised from inside a vendored script. The API log carried one line per submission,
`ui submit sync fallback`, naming neither the remedy nor the consequence.

These pin the two things that must be true of any deployment in this shape: the
configuration is caught at boot with a remedy, and the per-submission line is
actionable.
"""
from __future__ import annotations

import pytest

from fiximg import app_factory
from fiximg.application.pipeline_modes import SWITCHES
from fiximg.config import settings


@pytest.fixture()
def boot_check(monkeypatch):
    """Run the boot check with CUDA availability under our control."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _events(monkey_settings):
        recorded: list[dict] = []

        monkeypatch.setattr(
            "fiximg.infrastructure.observability.logging.get_logger",
            lambda *_a, **_k: object(),
        )
        # `log_event` is imported inside the function under test, so the patch
        # target is the module it comes from, not `app_factory`.
        monkeypatch.setattr(
            "fiximg.infrastructure.observability.logging.log_event",
            lambda _logger, level, message, **fields: recorded.append(
                {"level": level, "message": message, **fields}),
            raising=False,
        )
        for key, value in monkey_settings.items():
            monkeypatch.setattr(settings, key, value, raising=False)
        app_factory._check_worker_visibility()
        return recorded

    return _events


def test_a_split_topology_without_the_flag_is_caught_at_boot(boot_check, monkeypatch):
    """`api` + a separate worker, `external_worker` left at its default."""
    events = boot_check({"inline_worker": False, "external_worker": False})

    warnings = [e for e in events if e["level"] == "WARNING"]
    assert warnings, f"nothing warned at boot: {events}"
    note = warnings[0]
    # The remedy, by name. A warning that does not say what to change is the
    # failure being guarded against: the existing per-submission line named neither.
    assert "FIXIMG_HAS_EXTERNAL_WORKER" in note["hint"], note
    # And the consequence, so an operator can connect it to the OOM they will meet.
    assert "OOM" in note["consequence"] or "CUDA OOM" in note["consequence"], note


def test_a_correctly_configured_split_topology_stays_quiet(boot_check, monkeypatch):
    """The documented compose setup sets the flag; it must not cry wolf."""
    events = boot_check({"inline_worker": False, "external_worker": True})
    assert not [e for e in events if e["level"] == "WARNING"], events


def test_a_single_process_deployment_stays_quiet(boot_check, monkeypatch):
    """`compose.local.yaml` runs one process on purpose; the fallback is intended."""
    events = boot_check({"inline_worker": True, "external_worker": False})
    assert not [e for e in events if e["level"] == "WARNING"], events


def test_no_gpu_means_no_warning(boot_check, monkeypatch):
    """On CPU the synchronous path costs nothing; warning there is noise."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    events = boot_check({"inline_worker": False, "external_worker": False})
    assert not [e for e in events if e["level"] == "WARNING"], events


def test_the_check_is_actually_wired_into_startup():
    """A function nothing calls passes every test above.

    This is the same shape as the `append_history` removal in an earlier batch: a
    well-tested helper that no longer runs. Every other test here calls
    `_check_worker_visibility` by hand, so without this one the check could be
    deleted from the lifespan and the suite would stay green — which is precisely
    mutation A.
    """
    import inspect

    source = inspect.getsource(app_factory._app_lifespan)
    assert "_check_worker_visibility()" in source, (
        "the boot check is not called from the application lifespan, so it never "
        "runs: a correct helper wired to nothing"
    )


def test_the_boot_check_runs_when_the_app_starts(isolated_db, monkeypatch):
    """End to end: starting the app emits the warning for a misconfigured split.

    Driven through the real lifespan rather than by calling the helper, so the call
    site is exercised as well as the function.
    """
    import torch
    from fastapi.testclient import TestClient

    from fiximg.api import security as api_security

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(settings, "inline_worker", False, raising=False)
    monkeypatch.setattr(settings, "external_worker", False, raising=False)
    monkeypatch.setenv("FIXIMG_API_TOKEN", "visibility-token")
    api_security.reset_cache()

    recorded: list[dict] = []
    monkeypatch.setattr(
        "fiximg.infrastructure.observability.logging.log_event",
        lambda _logger, level, message, **fields: recorded.append(
            {"level": level, "message": message, **fields}),
        raising=False,
    )

    from fiximg.app_factory import create_app

    with TestClient(create_app()):
        pass

    warnings = [e for e in recorded if e["level"] == "WARNING"
                and "cannot see a worker" in e["message"]]
    assert warnings, [e["message"] for e in recorded if e["level"] == "WARNING"]
    assert "FIXIMG_HAS_EXTERNAL_WORKER" in warnings[0]["hint"], warnings[0]


def test_the_per_submission_line_names_the_remedy(boot_check, monkeypatch):
    """The boot line can be missed; the line that repeats on every submit cannot.

    This is the one that would have saved the debugging session: it appears at the
    moment the work is redirected, so it is the natural place to read the hint.
    """
    from fiximg.ui import task_progress as tp

    seen: list[dict] = []

    def _capture(logger, level, message, **fields):
        seen.append({"level": level, "message": message, **fields})

    monkeypatch.setattr(tp, "log_event", _capture, raising=False)
    monkeypatch.setattr(tp, "_check_password_change", lambda _s: None)
    monkeypatch.setattr(tp.task_service, "has_worker", lambda: False)
    monkeypatch.setattr(tp.task_service, "submit",
                        lambda *a, **k: ("/tmp/out.png", "done"))

    names = list(SWITCHES.get("auto", {}))
    handler = tp.make_submit_handler("auto")
    for _frame in handler(object(), {"username": "tester"}, *([False] * len(names))):
        pass

    fallbacks = [e for e in seen if e["message"] == "ui submit sync fallback"]
    assert fallbacks, seen
    assert "FIXIMG_HAS_EXTERNAL_WORKER" in fallbacks[0]["hint"], fallbacks[0]
    assert "OOM" in fallbacks[0]["consequence"], fallbacks[0]


def test_the_queued_path_does_not_log_the_fallback_at_all(boot_check, monkeypatch):
    """The warning must mean something: a healthy deployment never emits it."""
    from fiximg.ui import task_progress as tp

    seen: list[dict] = []
    monkeypatch.setattr(tp, "log_event",
                        lambda _logger, _level, message, **_f: seen.append(
                            {"message": message}), raising=False)
    monkeypatch.setattr(tp, "_check_password_change", lambda _s: None)
    monkeypatch.setattr(tp.task_service, "has_worker", lambda: True)
    monkeypatch.setattr(tp.task_service, "enqueue", lambda *a, **k: "t-1")
    monkeypatch.setattr(tp.task_service, "get_task", lambda _t: {
        "status": "failed", "error_code": "MODEL_UNAVAILABLE",
        "error_message": "Native scratch repair does not implement the HR branch",
    })

    names = list(SWITCHES.get("auto", {}))
    handler = tp.make_submit_handler("auto")
    for _frame in handler(object(), {"username": "tester"}, *([True] * len(names))):
        pass

    assert not [e for e in seen if e["message"] == "ui submit sync fallback"], seen
    assert [e for e in seen if e["message"] == "ui submit queued"], seen
