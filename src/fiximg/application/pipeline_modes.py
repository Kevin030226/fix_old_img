"""Pipeline mode labels and option switches, shared by the UI and the task service.

``SWITCHES`` is plan §3.9.1's input-area option list expressed in the vocabulary the
backend already accepts: every key is a field of ``TaskOptionsSchema``, and the value
is the checkbox's initial state. A mode mapped to an empty dict shows no checkboxes.

It is a separate mapping rather than another key inside ``PIPELINE_MODES`` on purpose:
mixing ``str`` values with a nested dict makes every consumer of the table (task-type
resolution, history re-run, the UI) argue with the type checker about a value that is
"either a string or a collection of strings", which is a worse answer than two tables
that each have one shape.
"""

PIPELINE_MODES = {
    "restore": {"label": "Restore (no scratches)", "task_type": "restore"},
    "restore_scratch": {
        "label": "Restore (with scratches)", "task_type": "restore_scratch",
    },
    "detect": {"label": "Scratch detection", "task_type": "detect_scratch"},
    "colorize": {"label": "Photo colorization", "task_type": "colorize"},
    # §10/§11 Phase 7: one-click entry; the analyzer derives the pipeline.
    "auto": {"label": "Auto Restore (analyze + repair)", "task_type": "auto_restore"},
}

#: Options whose effect is *which stages the planner schedules*, as opposed to what a
#: scheduled stage does. Mirrors ``PipelinePlanner.PLAN_SWITCHES``; a test compares the
#: two so this cannot drift into promising a switch the planner ignores.
PLAN_SWITCHES = frozenset({"face_enhance", "auto_colorize"})

#: The switches each tab offers, in the order the checkboxes are built and read back.
#:
#: Auto Restore offers the same two pipeline switches as the restoration tabs because
#: ``plan_auto_restore(analysis, options)`` honours them, but what a ticked box *sends*
#: differs — see ``ANALYSIS_DECIDED_MODES``. The colourisation and detection tabs have
#: no face chain and no grayscale branch to turn down, so they offer nothing.
SWITCHES: dict[str, dict[str, bool]] = {
    "restore": {"face_enhance": True, "auto_colorize": False, "hr": False},
    "restore_scratch": {"face_enhance": True, "auto_colorize": False, "hr": False},
    "auto": {"face_enhance": True, "auto_colorize": True, "hr": False},
    "colorize": {},
    "detect": {},
}

#: Modes whose plan is derived from the image analysis rather than the task type.
#: There, a checked box must mean "let the analysis decide", so the UI omits the key;
#: sending ``true`` instead would tell the planner to run the stage *even when the
#: analysis says it is not needed*, which is not what the user ticked. Unchecked is a
#: refusal in both kinds of mode, and sends an explicit ``false``.
ANALYSIS_DECIDED_MODES = frozenset({"auto"})

#: Checkbox captions, keyed by the same option names.
SWITCH_LABELS = {
    "face_enhance": "Face enhancement (detect, enhance, warp back)",
    "auto_colorize": "Auto colorize the result",
    "hr": "High-resolution face path (needs the HR weights)",
}

#: Captions that differ on one tab because the same key means something else there.
#: Reusing the restoration wording on Auto would promise "always run the chain" for a
#: box that only permits what the analysis decided.
SWITCH_LABELS_BY_MODE = {
    "auto": {
        "face_enhance": "Face enhancement (only when the analysis finds a face)",
        "auto_colorize": "Auto colorize (only when the photo is black and white)",
    },
}


def switch_label(mode: str, name: str) -> str:
    """The caption for one checkbox on one tab."""
    return SWITCH_LABELS_BY_MODE.get(mode, {}).get(name, SWITCH_LABELS.get(name, name))
