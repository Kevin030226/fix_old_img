"""Domain model tests (plan 搂3.1 Step 1, 搂6).

The V3 contract is that services exchange typed objects instead of anonymous
dicts. These tests pin that contract: row鈫抦odel conversion, option parsing, and
the status helpers the worker/runtime rely on.
"""
import json

import pytest

from fiximg.domain.artifacts import Artifact, ArtifactRef, build_artifact
from fiximg.domain.enums import ArtifactKind, EventType, TaskStatus, TaskType
from fiximg.domain.errors import (
    AppError,
    ErrorCode,
    ImageTooLargeError,
    InvalidRequestError,
    PipelineFailedError,
    TaskError,
    TaskNotFoundError,
)
from fiximg.domain.events import STREAM_CLOSING_EVENTS, TaskEvent
from fiximg.domain.models import ModelVersion
from fiximg.domain.tasks import KNOWN_OPTIONS, Task, TaskOptions, TaskStage


# ------------------------------------------------------------------ status
def test_task_status_helpers():
    assert TaskStatus.QUEUED.is_active
    assert TaskStatus.RUNNING.is_active
    assert not TaskStatus.QUEUED.is_terminal
    assert TaskStatus.COMPLETED.is_terminal
    assert TaskStatus.FAILED.is_terminal
    assert TaskStatus.CANCELLED.is_terminal
    # String-valued enums serialise straight into SQLite/JSON.
    assert str(TaskStatus.QUEUED) == "queued"
    assert TaskStatus("completed") is TaskStatus.COMPLETED


def test_task_type_and_event_names():
    assert TaskType.AUTO_RESTORE == "auto_restore"
    assert EventType.TASK_COMPLETED == "task.completed"
    assert ArtifactKind.GROUND_TRUTH == "ground_truth"


# ------------------------------------------------------------------ options
def test_task_options_roundtrip_preserves_extra_keys():
    options = TaskOptions.from_dict(
        {"hr": True, "auto_colorize": True, "_ground_truth_path": "/tmp/gt.png"}
    )
    assert options.hr is True
    assert options.auto_colorize is True
    assert options.face_enhance is False
    assert options.extra["_ground_truth_path"] == "/tmp/gt.png"

    restored = options.to_dict()
    assert restored["hr"] is True
    assert restored["_ground_truth_path"] == "/tmp/gt.png"


def test_a_key_this_build_does_not_declare_survives_the_round_trip():
    """Dropping a declared switch must not destroy rows that still carry it.

    `strength` used to be declared and read by nothing; removing it from the vocabulary
    is only safe because an undeclared key lands in `extra` and is written back untouched.
    A row submitted by an older build therefore still round-trips, and `to_dict()` stays
    an accurate description of the stored column.
    """
    options = TaskOptions.from_dict({"strength": 0.5, "hr": True})

    assert "strength" not in KNOWN_OPTIONS
    assert options.extra == {"strength": 0.5}
    assert options.to_dict() == {"hr": True, "face_enhance": False,
                                 "auto_colorize": False, "strength": 0.5}


# --------------------------------------------------------------------- task
def _row(**overrides):
    base = {
        "id": "t-1",
        "task_type": "restore",
        "user_id": "alice",
        "status": "running",
        "progress": 42,
        "current_stage": "global_restore",
        "input_path": "/tmp/in.png",
        "result_path": None,
        "error_message": None,
        "evaluation_text": None,
        "duration_ms": None,
        "created_at": "2026-09-25 10:00:00",
        "started_at": "2026-09-25 10:00:01",
        "finished_at": None,
        "options_json": json.dumps({"hr": True}),
        "priority": 5,
        "attempt_count": 2,
        "max_attempts": 3,
        "worker_id": "w-1",
        "lease_until": "2026-09-25 11:00:00",
        "last_heartbeat": "2026-09-25 10:05:00",
        "retry_at": None,
        "idempotency_key": "idem-1",
    }
    base.update(overrides)
    return base


def test_task_from_row_maps_every_field():
    task = Task.from_row(_row())
    assert task.id == "t-1"
    assert task.user_id == "alice"
    assert task.status_enum is TaskStatus.RUNNING
    assert task.type_enum is TaskType.RESTORE
    assert task.options.hr is True
    assert task.priority == 5
    assert task.attempt_count == 2
    assert task.worker_id == "w-1"
    assert task.idempotency_key == "idem-1"


def test_task_can_retry_only_while_attempts_remain():
    assert Task.from_row(_row(status="failed", attempt_count=1)).can_retry
    assert not Task.from_row(_row(status="failed", attempt_count=3)).can_retry
    assert not Task.from_row(_row(status="running", attempt_count=0)).can_retry


def test_task_from_row_survives_broken_options_json():
    task = Task.from_row(_row(options_json="{not json"))
    assert task.options.hr is False


def test_task_stage_from_row():
    stage = TaskStage.from_row(
        {
            "task_id": "t-1",
            "stage_name": "colorization",
            "stage_order": 1,
            "status": "completed",
            "duration_ms": 120,
            "message": None,
            "stage_version": "1.0",
        }
    )
    assert stage.stage_name == "colorization"
    assert stage.stage_order == 1
    assert stage.stage_version == "1.0"


# ----------------------------------------------------------------- artifact
def test_artifact_from_row_accepts_legacy_path_column():
    """V2 stored a filesystem path in `path`; the domain exposes it as `uri`."""
    artifact = Artifact.from_row(
        {"task_id": "t-1", "kind": "output", "path": "/tmp/out.png", "size_bytes": 10}
    )
    assert artifact.uri == "/tmp/out.png"
    assert artifact.kind == "output"


def test_build_artifact_collects_metadata(tmp_path):
    from PIL import Image

    path = tmp_path / "out.png"
    Image.new("RGB", (32, 16), "red").save(path)

    artifact = build_artifact("t-1", ArtifactKind.OUTPUT, str(path), "image/png")
    assert artifact.width == 32
    assert artifact.height == 16
    assert artifact.size_bytes and artifact.size_bytes > 0
    assert artifact.sha256 and len(artifact.sha256) == 64


def test_build_artifact_survives_missing_file():
    artifact = build_artifact("t-1", "output", "/nonexistent/nope.png")
    assert artifact.size_bytes is None
    assert artifact.sha256 is None


def test_artifact_ref_is_location_independent():
    ref = ArtifactRef(key="2026/09/t-1/output/final.png", backend="local")
    assert str(ref) == "local:2026/09/t-1/output/final.png"


# -------------------------------------------------------------------- event
def test_only_the_closing_events_end_the_stream():
    """`task.completed` does NOT close the stream; `task.evaluated` does.

    With ``FIXIMG_EVAL_MODE=async`` the metrics arrive after completion (plan
    搂3.16), so a transport that stopped at "completed" would silently drop them -
    which is what the previous `is_terminal` definition did.
    """
    assert not TaskEvent(event_type=EventType.TASK_COMPLETED).closes_stream
    assert TaskEvent(event_type=EventType.TASK_EVALUATED).closes_stream
    assert TaskEvent(event_type=EventType.TASK_FAILED).closes_stream
    assert TaskEvent(event_type=EventType.TASK_CANCELLED).closes_stream
    assert not TaskEvent(event_type=EventType.STAGE_STARTED).closes_stream
    assert STREAM_CLOSING_EVENTS <= set(EventType), "the set must stay inside the vocabulary"


def test_the_stream_vocabulary_is_defined_once():
    """The transport may not keep its own list of closing events.

    Two lists drifted once already (`is_terminal` said completed was terminal,
    the route's tuple said it was not), so the names must come from the domain.
    """
    import inspect

    from fiximg.api.routes import tasks as tasks_route

    source = inspect.getsource(tasks_route.stream_events)
    executable = "\n".join(ln for ln in source.splitlines() if not ln.strip().startswith("#"))
    for closing in STREAM_CLOSING_EVENTS:
        assert f'"{closing}"' not in executable, (
            f"{closing!r} is spelled out in the route instead of coming from EventType"
        )


def test_task_event_from_row_decodes_data():
    event = TaskEvent.from_row(
        {
            "id": 7,
            "event_type": "stage.completed",
            "task_id": "t-1",
            "level": "info",
            "message": "global_restore",
            "data_json": json.dumps({"stage": "global_restore"}),
            "created_at": "2026-09-25 10:00:00",
        }
    )
    assert event.seq == 7
    assert event.task_id == "t-1"
    assert event.data["stage"] == "global_restore"
    assert not event.closes_stream


# ------------------------------------------------------------------- errors
def test_app_error_carries_code_and_status():
    error = ImageTooLargeError("too big", details={"max_side": 4096})
    assert isinstance(error, InvalidRequestError)
    assert isinstance(error, AppError)
    assert error.code == ErrorCode.IMAGE_TOO_LARGE
    assert error.status_code == 400
    assert error.to_dict() == {
        "code": ErrorCode.IMAGE_TOO_LARGE,
        "message": "too big",
        "details": {"max_side": 4096},
    }


def test_error_hierarchy_preserves_v2_names():
    """V2 raise sites keep working: the old names are still the right classes."""
    assert issubclass(PipelineFailedError, TaskError)
    assert issubclass(TaskNotFoundError, TaskError)
    assert PipelineFailedError("x").code == ErrorCode.PIPELINE_FAILED
    assert TaskNotFoundError("x").status_code == 404


def test_app_error_defaults_to_internal():
    assert AppError("boom").code == ErrorCode.INTERNAL_ERROR
    assert AppError("boom").status_code == 500


def _reachable_error_codes() -> tuple[set[str], list[str]]:
    """``({codes a request can be answered with}, [error classes nobody produces])``.

    Resolved through the real classes, not the text of their bodies:
    `ArtifactNotFoundError` declares no `code` of its own and inherits
    `ARTIFACT_NOT_FOUND` from `ArtifactError`, which a regex over the class body
    reported as an unreachable code.

    Lenient about *how* a class is used 鈥?a construction inside a helper counts, and
    so would a docstring mention. The target is vocabulary wired to nothing, not
    reference style; a rule nobody satisfies converts into a permanent waiver.
    """
    import pathlib
    import re

    from fiximg.domain import errors as errors_module
    from fiximg.domain.errors import AppError

    declared = [
        value for value in vars(errors_module).values()
        if isinstance(value, type) and issubclass(value, AppError) and value is not AppError
    ]
    used_as_base = {base for cls in declared for base in cls.__bases__}

    root = pathlib.Path(__file__).resolve().parents[2] / "src"
    others = [
        path.read_text(encoding="utf-8", errors="replace")
        for path in root.rglob("*.py")
        if "__pycache__" not in str(path) and path != pathlib.Path(errors_module.__file__)
    ]

    produced, unproduced = [], []
    for cls in declared:
        if cls in used_as_base or any(
            re.search(rf"\b{cls.__name__}\(", text) for text in others
        ):
            produced.append(cls)
        else:
            unproduced.append(cls.__name__)

    # The status map is how codes reach routes that raise the framework's own
    # HTTPException (401 from the token check, 404 from a FastAPI router).
    handler = (root / "fiximg/api/errors.py").read_text(encoding="utf-8")
    mapped = set(re.findall(r"ErrorCode\.(\w+)", handler))
    return {cls.code for cls in produced} | mapped, unproduced


def test_every_declared_error_is_something_a_request_can_produce():
    """An error class nobody constructs is an error code no client can ever receive.

    This is the hole `RATE_LIMITED` lived in: the class existed, `docs/api.md`
    advertised the code, and no JSON route could produce it 鈥?while the HTML register
    page answered 429 with its own body, outside the envelope entirely. A green suite
    said nothing, because nothing compared the vocabulary with the code.
    """
    _reachable, unproduced = _reachable_error_codes()
    assert not unproduced, f"declared but produced nowhere in src/: {unproduced}"

    declared = {
        value for key, value in vars(ErrorCode).items()
        if not key.startswith("_") and isinstance(value, str)
    }
    unreachable = sorted(declared - _reachable)
    assert not unreachable, f"ErrorCode nobody can emit: {unreachable}"


def test_every_error_code_the_api_documents_is_reachable():
    """`docs/api.md` lists the codes a client should handle 鈥?each must be producible.

    The expectation is read out of the document instead of copied into this test, so
    the gate is "the contract matches the code", not two lists that drift together.
    """
    import pathlib
    import re

    document = (
        pathlib.Path(__file__).resolve().parents[2] / "docs/api.md"
    ).read_text(encoding="utf-8", errors="replace")
    # The list is its own paragraph: taking everything up to the next heading also
    # swept in the prose below it, whose `FIXIMG_SUBMIT_MAX` and friends are knobs,
    # not error codes.
    block = document.split("Common codes:", 1)[1].split("\n\n", 1)[0]
    documented = set(re.findall(r"`([A-Z][A-Z_]{3,})`", block))
    assert documented, "the common-codes paragraph disappeared from docs/api.md"

    reachable, _unproduced = _reachable_error_codes()
    missing = sorted(documented - reachable)
    assert not missing, f"documented but unreachable from any route: {missing}"
    assert AppError("boom").status_code == 500


# -------------------------------------------------------------- model version
def test_model_version_from_dict_maps_weight_aliases():
    declared = ModelVersion.from_dict(
        "ddcolor",
        {
            "version": "2.1.0",
            "weight": "weights/ddcolor/pytorch_model.pt",
            "capabilities": ["colorize"],
            "input_size": 512,
        },
    )
    assert declared.version == "2.1.0"
    assert declared.weight_uri == "weights/ddcolor/pytorch_model.pt"
    assert declared.supports("colorize")
    assert not declared.supports("face_restore")
    assert declared.to_dict()["capabilities"] == ["colorize"]


def test_model_version_accepts_checkpoint_key():
    declared = ModelVersion.from_dict("global_restore", {"checkpoint": "Global/ckpt"})
    assert declared.weight_uri == "Global/ckpt"


@pytest.mark.parametrize("value", ["queued", "running", "completed", "failed", "cancelled"])
def test_status_enum_covers_every_persisted_value(value):
    assert TaskStatus(value).value == value


def test_no_domain_enum_member_is_unreachable():
    """A value nothing can produce is a value nothing can be written against.

    This is the same gate `test_observability` runs over `MetricName`, and it
    caught the same shape three times in the domain layer: `StageStatus.SKIPPED`
    (declared, never written, so a skipped face chain reported "completed"),
    `EventType.TASK_PROGRESS` and `EventType.TASK_STARTED` (both listed in
    `docs/api.md` as part of the SSE contract, neither ever emitted), and two
    `ModelStatus` states that no code path could reach.

    A member counts as live when `src/` mentions it by name (`StageStatus.SKIPPED`)
    or by its persisted value (`"skipped"`, which is how the repositories write
    them). That is deliberately lenient: the point is to catch vocabulary that is
    wired to *nothing*, not to police how it is referenced.
    """
    import enum
    import pathlib
    import re

    from fiximg.domain import enums as enums_module

    sources = [
        path.read_text(encoding="utf-8", errors="replace")
        for path in (pathlib.Path(__file__).resolve().parents[2] / "src").rglob("*.py")
        if "__pycache__" not in str(path) and path.name != "enums.py"
    ]

    unreachable = []
    for class_name, member_class in vars(enums_module).items():
        if not (isinstance(member_class, type)
                and issubclass(member_class, enum.Enum)
                and member_class.__module__ == enums_module.__name__):
            continue
        for member in member_class:
            referenced = any(
                re.search(rf"{class_name}\.{member.name}\b", text)
                or f'"{member.value}"' in text
                or f"'{member.value}'" in text
                for text in sources
            )
            if not referenced:
                unreachable.append(f"{class_name}.{member.name} = {member.value!r}")

    assert not unreachable, "declared but produced nowhere in src/: " + ", ".join(unreachable)


# ------------------------------------------------------- event vocabulary, both ways
def _src_files():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src"
    return [
        path.read_text(encoding="utf-8", errors="replace")
        for path in root.rglob("*.py")
        if "__pycache__" not in str(path)
    ]


def _emit_sites():
    """``{name: emits_with_a_task_id}`` for every ``add_event(...)`` call in src/.

    Parsed from the AST rather than grepped, so a name built by a conditional
    expression (``EventType.STAGE_SKIPPED.value if ... else ...``) still counts, and
    so a *span* name 鈥?``tracer.span("task.execute", ...)``, which is not an event
    at all 鈥?cannot leak in.
    """
    import ast

    found: dict[str, bool] = {}
    for text in _src_files():
        for call in ast.walk(ast.parse(text)):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not (isinstance(func, ast.Attribute) and func.attr == "add_event"):
                continue
            if not call.args:
                continue
            names = _names_in(call.args[0])
            carries_task_id = any(kw.arg == "task_id" for kw in call.keywords)
            for name in names:
                found[name] = found.get(name, False) or carries_task_id
    return found


def _names_in(node):
    """Event names an expression can produce: string constants and ``EventType.X``."""
    import ast

    from fiximg.domain.enums import EventType

    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            names.add(sub.value)
        elif isinstance(sub, ast.Attribute):
            root = sub
            chain = []
            while isinstance(root, ast.Attribute):
                chain.append(root.attr)
                root = root.value
            if isinstance(root, ast.Name) and root.id == "EventType":
                # chain is innermost-first: TASK_STARTED for `EventType.TASK_STARTED`,
                # and also for `EventType.TASK_STARTED.value`.
                member = chain[0]
                if hasattr(EventType, member):
                    names.add(getattr(EventType, member).value)
    return names


def test_every_event_the_code_emits_is_declared():
    """An emitted name with no member is a name a client cannot type-check against.

    The mirror of `test_no_domain_enum_member_is_unreachable`, which only hunts
    declared-but-produced-nowhere. Together they close the class: `stage.retried`
    and `task.rejected` were written to `system_events` from production code while
    `EventType` had no such member and `docs/api.md` did not list them, so the two
    halves of the vocabulary 鈥?the enum and the wire 鈥?had no way to disagree loudly.
    """
    undeclared = sorted(set(_emit_sites()) - set(EventType))
    assert not undeclared, "emitted but not in EventType: " + ", ".join(undeclared)


def test_the_documented_stream_names_are_the_names_a_client_can_receive():
    """`docs/api.md` lists exactly the events the SSE route can deliver.

    Split on `task_id`, because that column is what makes an event deliverable:
    `GET /tasks/{id}/events` selects `WHERE task_id=?`, so a row written without one
    is invisible to every client no matter what the documentation promises.
    """
    import re
    from pathlib import Path

    doc = Path(__file__).resolve().parents[2] / "docs" / "api.md"
    section = doc.read_text(encoding="utf-8").split("### Events (SSE)", 1)[1]
    section = section.split("\n### ", 1)[0]
    paragraph = section.split("Event names:", 1)[1].split("\n\n", 1)[0]
    documented = set(re.findall(r"`([a-z_]+\.[a-z_]+)`", paragraph))

    sites = _emit_sites()
    on_stream = {name for name, with_id in sites.items() if with_id}
    off_stream = {name for name, with_id in sites.items() if not with_id}

    assert documented == on_stream, (
        "documented but never delivered: "
        + ", ".join(sorted(documented - on_stream))
        + " | delivered but undocumented: "
        + ", ".join(sorted(on_stream - documented))
    )
    # A row-less event is still part of the contract story: the doc has to say why
    # the client sees an HTTP error instead of a stream frame for it.
    for name in off_stream:
        assert f"`{name}`" in section, f"{name} is written without a task_id and the doc never explains it"


# ------------------------------------------------- artifact kinds, both directions
def _emitted_artifact_kinds() -> set[str]:
    """Every artifact role src/ can persist, read from the AST.

    Two shapes produce a row: a literal second argument to `add_artifact(...)`, and
    a key of any `artifacts={...}` dict literal 鈥?a stage returns one in its
    `StageResult`, and a backend returns one in its own result object, which the
    stage then forwards. The runtime registers those keys verbatim, so a typo there
    is a persisted value nothing can name.
    """
    import ast

    found: set[str] = set()
    for text in _src_files():
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "add_artifact":
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    if isinstance(node.args[1].value, str):
                        found.add(node.args[1].value)
            for kw in node.keywords:
                if kw.arg == "artifacts" and isinstance(kw.value, ast.Dict):
                    for key in kw.value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            found.add(key.value)
    return found


def test_the_artifact_vocabulary_is_exactly_what_the_code_writes():
    """`ArtifactKind` was documentation, not a contract.

    The real task run put five kinds in the table; `faces_dir`, `final`, `result`
    and `each_img_dir` were not among the declared seven, while `colorized` and
    `report` had no producer at all 鈥?the lenient member gate above had counted them
    live on an unrelated metadata key and a progress label. Checked in both
    directions, so neither half can drift again.
    """
    emitted = _emitted_artifact_kinds()
    declared = set(ArtifactKind)

    assert emitted, "the scan found no artifact writes: it is not measuring anything"
    undeclared = sorted(emitted - declared)
    unproduced = sorted(declared - emitted)
    assert not undeclared, "persisted but not in ArtifactKind: " + ", ".join(undeclared)
    assert not unproduced, "declared but written by nothing: " + ", ".join(unproduced)
