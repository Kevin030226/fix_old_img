"""The HR-on-a-scratched-photo refusal must reach the person as a sentence.

Found by driving the real UI handler, not by reading it. Ticking "High-resolution
face path (needs the HR weights)" on a photo the analysis marks as scratched plans
`scratch_repair`, whose native backend refuses the HR branch — it selects
`mapping_Patch_Attention`, whose weights upstream does not publish
(`docs/deployment.md`: "a directory not shipped upstream"), and the vendored loader
would run that network randomly initialised. The refusal itself is right, and three
existing tests pin it.

What was wrong is what the person saw. The synchronous path re-raised the
`ModelUnavailableError` and Gradio rendered a Python traceback, at the exact moment
the system needed to say "untick that box". The queued path did better — it raised a
`gr.Error` — but quoted the raw internal prose, naming a file the person never chose
and offering no way forward.

These tests pin that both paths say the same actionable thing, and that neither
mentions a stack.
"""
from __future__ import annotations

import pytest

from fiximg.domain.errors import AppError, ModelUnavailableError
from fiximg.ui import task_progress as tp

#: What the native backend raises, quoted from global_restore.run_folder.
HR_REFUSAL = ModelUnavailableError(
    "Native scratch repair does not implement the HR branch: it selects "
    "mapping_Patch_Attention, whose weights are not shipped, and the vendored "
    "loader would silently run it uninitialised",
    details={"backend": "scratch_repair", "hr": True},
)


def _drive(mode, monkeypatch, *switch_values):
    """Run the real handler to completion, collecting what it yields and what it raises.

    The yielded tuple's second element is the progress textbox — the component the
    user reads. Asserting on a raised `gr.Error` instead is what let this defect
    through twice: the exception was composed correctly every time and never
    rendered.
    """
    from fiximg.application.pipeline_modes import SWITCHES

    monkeypatch.setattr(tp, "_check_password_change", lambda _s: None)
    names = list(SWITCHES.get(mode, {}))
    handler = tp.make_submit_handler(mode)
    gen = handler(object(), {"username": "tester"},
                  *(switch_values if switch_values else [True] * len(names)))

    yields, raised = [], None
    try:
        for item in gen:
            yields.append(item)
    except Exception as exc:  # noqa: BLE001
        raised = f"{type(exc).__name__}: {exc}"
    return yields, raised


def test_a_run_failure_is_reported_in_the_textbox_and_does_not_raise(monkeypatch):
    """The browser check, twice failed and now the reason it is written this way.

    Verified as `tester` in the browser on the Auto Restore tab with the HR box
    ticked. The task recorded the refusal correctly (`error_code=MODEL_UNAVAILABLE`,
    `options={"hr": true}`) and the sentence was composed, yet nothing readable
    reached the page: raising put the three result components into the generic
    "error" badge state with no text, and yielding the sentence *then* raising still
    left the textbox blank because the error state supersedes its content.

    So a run failure must not raise. Gradio's error UI is not relied on at all.
    """
    monkeypatch.setattr(tp.task_service, "has_worker", lambda: False)
    monkeypatch.setattr(tp.task_service, "submit",
                        lambda *a, **k: (_ for _ in ()).throw(HR_REFUSAL))

    yields, raised = _drive("auto", monkeypatch)
    assert raised is None, (
        f"the callback raised, so Gradio showed the generic error badge instead of "
        f"the reason: {raised}"
    )
    assert yields, "the handler produced no frame at all"
    text = yields[-1][1]
    assert "High-resolution face path" in text, text
    assert "Untick" in text, text
    assert "Traceback" not in text, text
    # The download control must not be offered for a run that produced nothing.
    assert yields[-1][3].get("visible") is False, yields[-1][3]


def test_the_queued_path_writes_the_reason_into_the_textbox(monkeypatch):
    """Same for a worker failure, which arrives as a row rather than an exception."""
    monkeypatch.setattr(tp.task_service, "has_worker", lambda: True)
    monkeypatch.setattr(tp.task_service, "enqueue", lambda *a, **k: "t-queued")
    monkeypatch.setattr(
        tp.task_service, "get_task",
        lambda _tid: {
            "status": "failed",
            "error_code": "MODEL_UNAVAILABLE",
            "error_message": str(HR_REFUSAL),
        },
    )

    yields, raised = _drive("auto", monkeypatch)
    assert raised is None, raised
    assert yields, "the queued path produced no frame at all"
    text = yields[-1][1]
    assert "High-resolution face path" in text, text
    assert "Untick" in text, text


def test_a_successful_run_is_untouched(monkeypatch):
    """The failure frame must not appear on the happy path."""
    monkeypatch.setattr(tp.task_service, "has_worker", lambda: False)
    monkeypatch.setattr(tp.task_service, "submit",
                        lambda *a, **k: ("/tmp/out.png", "done"))

    yields, raised = _drive("auto", monkeypatch)
    assert raised is None, raised
    assert yields and "done" in str(yields[-1][1]), yields[-1][1] if yields else None
    assert not any("Untick" in str(y[1]) for y in yields), yields
    assert not any(str(y[1]).startswith("⚠") for y in yields), yields


def test_both_paths_produce_the_same_sentence(monkeypatch):
    """A queued worker and an in-process run are the same refusal to the user."""
    monkeypatch.setattr(tp.task_service, "has_worker", lambda: False)
    monkeypatch.setattr(tp.task_service, "submit",
                        lambda *a, **k: (_ for _ in ()).throw(HR_REFUSAL))
    yields, _raised = _drive("auto", monkeypatch)

    # One wording for both paths, compared as text: `gr.Error` stringifies with its
    # own quoting, so the exception objects are not directly comparable.
    assert (tp._describe_failure(HR_REFUSAL)
            == tp._hr_scratch_message()), "the two paths must share one wording"
    # The textbox carries the same sentence behind a failure marker.
    assert yields[-1][1].lstrip("⚠ ") == tp._hr_scratch_message(), yields[-1][1]


def test_the_synchronous_helper_lets_the_domain_error_through(monkeypatch):
    """It no longer translates; the handler does, once, for both routes.

    Translating in both places is how two sentences drift apart — which is how the
    original defect happened, one of the two routes simply having none.
    """
    monkeypatch.setattr(tp.task_service, "submit",
                        lambda *a, **k: (_ for _ in ()).throw(HR_REFUSAL))
    with pytest.raises(ModelUnavailableError):
        tp._sync_fallback("auto", object(), {"username": "tester"},
                          options={"hr": True})


    # Same error code, HR was requested, but a *different* model is missing. The
    # HR sentence would be a lie here, so the backend's own text is passed through.
    other = tp._failure_text(
        {"status": "failed", "error_code": "MODEL_UNAVAILABLE",
         "error_message": "ddcolor weights are absent"},
        {"hr": True},
    )
    assert "Untick" not in other, other
    assert "ddcolor weights are absent" in other, other

    # And with the marker present, the sentence is substituted as intended.
    assert tp._failure_text(
        {"status": "failed", "error_code": "MODEL_UNAVAILABLE",
         "error_message": str(HR_REFUSAL)},
        {"hr": True},
    ) == tp._describe_failure(HR_REFUSAL)


def test_the_queued_marker_is_the_backends_own_word():
    """The queued path matches on a fragment of prose; hold it to the real thing.

    `_failure_text` cannot see the exception's `details`, so it recognises the HR
    refusal by a marker in the message. If the backend's wording changes, this
    fails instead of the substitution silently ceasing to happen — which would
    leave the queued path quoting internal prose while the synchronous path did
    not, the exact split this batch removed.
    """
    import inspect as _inspect

    from fiximg.inference.backends import global_restore

    raiseers = [
        (name, obj) for name, obj in vars(global_restore).items()
        if isinstance(obj, type) and _inspect.isclass(obj)
        and obj.__module__ == global_restore.__name__
        and "does not implement the HR branch" in _inspect.getsource(obj)
    ]
    assert raiseers, (
        "no class in global_restore raises the HR refusal any more, so the queued "
        "path can no longer recognise it"
    )
    for name, _obj in raiseers:
        assert tp._HR_REFUSAL_MARKER in _inspect.getsource(
            next(obj for n, obj in raiseers if n == name)
        ), (
            f"{name} raises the HR refusal but the marker "
            f"{tp._HR_REFUSAL_MARKER!r} is absent from its message, so the queued "
            "path would quote internal prose again"
        )


def test_the_sentence_does_not_appear_when_hr_was_not_asked_for():
    """The hint must not follow the user around on unrelated failures."""
    other = ModelUnavailableError("ddcolor weights are absent",
                                  details={"backend": "ddcolor"})
    assert "High-resolution face path" not in tp._describe_failure(other)

    row = {"status": "failed", "error_code": "MODEL_UNAVAILABLE",
           "error_message": "ddcolor weights are absent"}
    assert "High-resolution face path" not in tp._failure_text(row, {"hr": False})
    assert "High-resolution face path" not in tp._failure_text(row, None)


def test_an_unrelated_failure_still_reaches_the_user(monkeypatch):
    """The translation must not swallow errors it has nothing to say about."""
    boom = AppError("the image is larger than the size policy allows")
    monkeypatch.setattr(tp.task_service, "has_worker", lambda: False)
    monkeypatch.setattr(tp.task_service, "submit",
                        lambda *a, **k: (_ for _ in ()).throw(boom))

    yields, raised = _drive("restore", monkeypatch)
    assert raised is None, f"a run failure must not raise: {raised}"
    assert "larger than the size policy" in yields[-1][1], yields[-1][1]
    # It must not be dressed up as the HR refusal.
    assert "High-resolution face path" not in yields[-1][1], yields[-1][1]


def test_a_failure_with_no_message_still_names_something():
    assert tp._describe_failure(AppError("")).strip(), "an empty error must not render blank"


@pytest.mark.parametrize("mode", ["auto", "restore", "restore_scratch"])
def test_no_submission_path_can_leak_a_bare_exception(monkeypatch, mode):
    """Every mode's handler: a domain failure becomes readable text, never a stack.

    Asserted on the handler rather than on `_sync_fallback` alone, because the
    handler is what Gradio calls, and the queued branch is chosen at runtime by
    `has_worker()` — which in a test process is False, so without forcing the
    branch the queued path would never be exercised here.
    """
    monkeypatch.setattr(tp.task_service, "has_worker", lambda: False)
    monkeypatch.setattr(tp.task_service, "submit",
                        lambda *a, **k: (_ for _ in ()).throw(HR_REFUSAL))

    yields, raised = _drive(mode, monkeypatch)
    assert raised is None, f"{mode}: the callback raised: {raised}"
    assert yields, f"{mode}: the handler produced no frame"
    text = yields[-1][1]
    assert "High-resolution face path" in text, text
    assert "Untick" in text, text
    # Nothing a Python reader would recognise: no stack, no exception class, no path.
    assert "Traceback" not in text, text
    assert "ModelUnavailableError" not in text, text
    assert ".py" not in text, text
