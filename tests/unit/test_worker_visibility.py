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


def test_the_web_ui_always_enqueues_and_never_runs_inference_in_process():
    """The sync fallback is gone, which is a stronger property than a warning.

    The Gradio callback checked `has_worker()` and, when the answer was false,
    executed the pipeline inside the API process — the CUDA-OOM path this module
    exists for. Its only mitigation was a log line naming the remedy.

    The native UI has no such branch: a browser submission goes through
    `application.submission.submit_upload` to `TaskService.enqueue` and nowhere else,
    so there is no configuration in which a request can start inference in the web
    process. That is asserted here rather than described, because "we removed the
    branch" is exactly the kind of claim that comes back.
    """
    from fiximg.api.dependencies import Principal
    from fiximg.application import submission

    calls: list[tuple] = []

    def _enqueue(*args, **kwargs):
        calls.append((args, kwargs))
        return "t-1"

    # If the route could reach the synchronous orchestrator, this would be the
    # method it used. It must not be reachable from the submission path at all.
    monkeypatch_target = submission.task_service
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(monkeypatch_target, "enqueue", _enqueue)
        patch.setattr(monkeypatch_target, "submit",
                      lambda *a, **k: pytest.fail("the web UI ran inference in-process"))
        patch.setattr(monkeypatch_target, "has_worker", lambda: False)

        principal = Principal(user_id="tester", role="user", via="session")
        result = submission.submit_upload(
            principal=principal,
            task_type="restore",
            image_bytes=_png_bytes(),
            options_raw="{}",
            request=None,
        )

    assert result.task_id == "t-1"
    assert calls, "the submission never reached the queue"
    assert not [call for call in calls if call[1].get("_synchronous")], calls


def test_the_ui_module_has_no_synchronous_execution_path():
    """Read the source: the removed branch must not reappear under a new name.

    A behavioural test can only prove the path taken by the input it was given.
    This one is about the code: the UI must not import or call the orchestrator's
    synchronous `run`, so no future edit can quietly restore the fallback.
    """
    import ast
    from pathlib import Path

    ui_dir = Path(__file__).resolve().parents[2] / "src" / "fiximg" / "ui"
    offenders: list[str] = []
    for path in sorted(ui_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "submit":
                offenders.append(f"{path.name}: .submit(")
            if isinstance(node, ast.Name) and node.id == "has_worker":
                offenders.append(f"{path.name}: has_worker")
    assert not offenders, (
        "the web UI can still reach the synchronous in-process path, which is the "
        f"CUDA-OOM shape this module exists for: {offenders}"
    )


def _png_bytes(width: int = 48, height: int = 32) -> bytes:
    import io

    from tests.fixtures import make_photo

    buffer = io.BytesIO()
    make_photo(width=width, height=height).save(buffer, format="PNG")
    return buffer.getvalue()
