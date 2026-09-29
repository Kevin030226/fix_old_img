"""Every option a client may send has to be read by something that runs.

This is the class that `strength` belonged to: declared in the API whitelist, in the
schema, in the domain vocabulary and in `docs/api.md`, stored in
``tasks.options_json`` 鈥?and read by nothing. A client that sent
``{"strength": 0.5}`` got a 202, a row describing the switch, and the identical image it
would have got without it, so the switch was decoration the whole way down and no test
could say so (deleting the field only failed a test that asserted the *field's own*
parse tolerance).

The gates below are therefore about the vocabulary as a layer: one list of accepted keys
(a second one had already drifted into `TaskOptions.planner_switches`, which claimed `hr`
was a planner switch when the planner never reads it), and a production reader for each
key derived from the sources rather than from a list kept next to this test.
"""
from __future__ import annotations

import re
from pathlib import Path

from fiximg.api.schemas.task import _ALLOWED_OPTIONS
from fiximg.application.pipeline_modes import SWITCHES
from fiximg.domain.tasks import INTERNAL_OPTION_KEYS, KNOWN_OPTIONS
from fiximg.inference.planner import PipelinePlanner

SRC = Path(__file__).resolve().parents[2] / "src" / "fiximg"
RUNTIME_LAYERS = SRC / "inference"


def _readers_of(key: str) -> list[str]:
    """Files under `src/fiximg/inference` that consult this option key.

    Matches only the shape of a runtime *asking* for it 鈥?``options.get("hr")`` or
    ``(context.options or {}).get("hr")`` 鈥?because a capability list or a comment that
    happens to contain the same word is not a reader.
    """
    pattern = re.compile(
        r"""options(?:\s+or\s+\{\})?\)?\.get\(\s*['"]""" + re.escape(key) + r"""['"]"""
        r"""|options\[\s*['"]""" + re.escape(key) + r"""['"]\s*\]"""
    )
    hits = []
    for path in sorted(RUNTIME_LAYERS.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if pattern.search(source):
            hits.append(str(path.relative_to(SRC.parent)))
    return hits


def _declared() -> set[str]:
    return set(_ALLOWED_OPTIONS)


def test_the_option_vocabulary_is_one_list_not_three():
    """Schema whitelist, domain vocabulary and schema fields must agree.

    `KNOWN_OPTIONS` and `_ALLOWED_OPTIONS` are separate declarations in separate layers
    (domain must not import the API), so the only thing keeping them honest is this check.
    A drift in either direction is invisible at runtime: a key missing from the schema is
    simply never interpreted, and a key missing from the whitelist is rejected before it
    reaches the planner that would have read it.
    """
    from fiximg.api.schemas.task import TaskOptionsSchema

    declared = _declared()
    assert declared == set(KNOWN_OPTIONS), (
        f"schema accepts {sorted(declared ^ set(KNOWN_OPTIONS))} against the domain list"
    )
    assert declared == set(TaskOptionsSchema.model_fields), (
        f"schema fields {sorted(declared ^ set(TaskOptionsSchema.model_fields))}"
    )
    assert declared >= {"hr", "face_enhance", "auto_colorize"}, sorted(declared)


def test_every_declared_option_is_read_by_something_that_runs():
    """The gate that `strength` would have failed.

    A plan switch counts as read only if the planner declares it, because the planner is
    what turns the switch into stages; anything else needs an actual read site in the
    inference layer.
    """
    plan_switches = set(PipelinePlanner.PLAN_SWITCHES)
    unread = []
    for key in sorted(_declared()):
        if key in plan_switches:
            # Declared as a plan switch *and* read: check the planner really consults it,
            # so the declaration cannot be the only evidence.
            assert _readers_of(key), f"planner declares {key} but never reads it"
            continue
        if not _readers_of(key):
            unread.append(key)

    assert not unread, f"options accepted from clients and read by nobody: {unread}"


def test_nothing_in_the_inference_layer_reads_a_task_option_the_api_refuses():
    """The other direction: a key the pipeline consults must be sendable.

    An undocumented-but-read key is the same defect seen from the other side 鈥?the client
    has no way to reach behaviour that exists, and `TaskOptionsSchema.parse` would reject
    the attempt as an unknown option.

    Only `context.options` is scanned, because that bag is the client's task options by
    construction. `request.options` at the backend layer is a *merged* bag 鈥?stages put
    the device and their own kwargs in it (`gpu=鈥, `hr=鈥) 鈥?so requiring every key there
    to be client-declared would be a false rule. Those keys are still checked in the
    other direction: `hr` has to be declared, and it is read in both bags.

    Internal keys the service injects itself (`_ground_truth_path`) are exempt because they
    are deliberately never accepted from a caller.
    """
    read = set()
    pattern = re.compile(
        r"""context\.options(?:\s+or\s+\{\})?\)?\.get\(\s*['"]([a-z_]+)['"]"""
        r"""|context\.options\[\s*['"]([a-z_]+)['"]\s*\]"""
    )
    for path in sorted(RUNTIME_LAYERS.rglob("*.py")):
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            read.add(match.group(1) or match.group(2))

    assert read, "the context.options scan found no read sites at all"
    unreachable = sorted(read - _declared() - set(INTERNAL_OPTION_KEYS))
    assert not unreachable, f"read by the pipeline but not accepted from clients: {unreachable}"


def test_the_ui_only_offers_switches_the_pipeline_reads():
    """A checkbox for an option with no reader is the decoration this file exists to stop."""
    offered = {name for names in SWITCHES.values() for name in names}
    assert offered, "no tab offers any switch"
    assert offered <= _declared(), f"UI offers unknown options: {sorted(offered - _declared())}"
    assert offered <= {k for k in _declared() if k in set(PipelinePlanner.PLAN_SWITCHES)
                       or _readers_of(k)}


def test_a_declared_plan_switch_changes_the_plan_it_claims_to_control():
    """`PLAN_SWITCHES` is a claim about the plan, so the plan has to prove it.

    A key can be *read* by the runtime and still not steer the pipeline 鈥?`hr` is exactly
    that case (the face stages read it to pick weights). Reading the declaration alone
    cannot tell the two apart, so each declared switch is toggled through both plan builders
    and the resulting stage lists must differ. A switch added to the declaration without
    wiring, or one whose effect was silently removed, fails here rather than becoming a
    checkbox the user can click for no reason.
    """
    from fiximg.inference.planner import PipelinePlanner

    planner = PipelinePlanner()
    analysis = {"width": 800, "height": 600, "is_grayscale": False, "face_count": 0,
                "blur_score": 0.2, "scratch_score": 0.9}

    for key in PipelinePlanner.PLAN_SWITCHES:
        static_on = [n for n, _ in planner.plan("restore", {key: True}).stages]
        static_off = [n for n, _ in planner.plan("restore", {key: False}).stages]
        assert static_on != static_off, f"{key}: no effect on plan('restore')"

        dynamic_on = [n for n, _ in planner.plan_auto_restore(analysis, {key: True}).stages]
        dynamic_off = [n for n, _ in planner.plan_auto_restore(analysis, {key: False}).stages]
        assert dynamic_on != dynamic_off, f"{key}: no effect on plan_auto_restore"


def test_the_two_plan_builders_apply_the_same_switches():
    """One vocabulary, one behaviour: refusing a switch removes the same stage twice.

    The UI's decline-only mapping is written against the plan switches, so a key the
    static plan honours but the dynamic one ignores would be a checkbox that works on one
    tab and not the other 鈥?the asymmetry this batch's predecessor started out to remove.
    The analysis is deliberately "busy" (faces *and* grayscale) so every optional stage
    wants to run, and each refusal therefore has somewhere visibly to land.
    """
    from fiximg.inference.planner import PipelinePlanner

    planner = PipelinePlanner()
    analysis = {"is_grayscale": True, "face_count": 2, "scratch_score": 0.9,
                "blur_score": 0.6}
    #: Which stage each switch is responsible for scheduling.
    stage_of = {"face_enhance": "face_detection", "auto_colorize": "colorization"}

    assert set(PipelinePlanner.PLAN_SWITCHES) == set(stage_of), PipelinePlanner.PLAN_SWITCHES

    for key, stage in stage_of.items():
        refused_static = [n for n, _ in planner.plan("restore", {key: False}).stages]
        refused_dynamic = [
            n for n, _ in planner.plan_auto_restore(analysis, {key: False}).stages
        ]
        assert stage not in refused_static, (key, refused_static)
        assert stage not in refused_dynamic, (key, refused_dynamic)
        # A refusal is only about its own stage: the other optional stage is still
        # scheduled on the analysis-driven plan, so the switches cannot leak into
        # each other's meaning.
        others = [other for name, other in stage_of.items() if name != key]
        assert all(other in refused_dynamic for other in others), (key, refused_dynamic)
        assert planner.plan("restore", {key: False}).decisions["options"][key] is False
        assert planner.plan_auto_restore(analysis, {key: False}).decisions["options"][key] is False


def test_the_reader_scan_is_not_vacuous():
    """A gate that scans nothing looks identical to a gate that passes.

    `hr` must be found in at least two runtime layers (a stage and a backend), and the
    scan must see the plan switches too 鈥?if the regex ever stops matching, these
    assertions fail instead of the gates above silently approving anything.
    """
    readers = {key: _readers_of(key) for key in sorted(_declared())}
    assert len(readers["hr"]) >= 2, readers
    for key in PipelinePlanner.PLAN_SWITCHES:
        assert readers[key], f"{key} has no read site"
    assert len(readers) >= 3, readers
