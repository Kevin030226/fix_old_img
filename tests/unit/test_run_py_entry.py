"""`run.py` must stay a shim over `fiximg.cli.batch`, and both must stay in step.

This is the third place the project has been bitten by the same thing, and the
first where the *whole* CPU suite stayed green while the product was broken.

`LegacyCliBackend` spawns `run.py` with a fixed argument list, and the pipeline
was split into four separately runnable stages with `--stages 1,2,3,4` (plan
§3.7/§3.6). `run.py` was then edited to hold a self-contained copy of the old
four-path implementation — the version from before the split — so:

* it no longer accepted `--stages`, and **every** subprocess the backend spawns
  died with `unrecognized arguments: --stages 1`;
* the copy could not have been produced by the split, and nothing said so;
* the only thing that noticed was `pytest tests/gpu -m gpu`, because that is the
  tier that actually spawns it. `test_legacy_chain.py` asserts the *backend*
  builds the right argv — against a recorder, never a process.

So this file pins the two things the GPU tier was doing implicitly: that the
entry point is a delegation rather than a copy, and that the flags the backend
sends are flags the entry point's parser accepts.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN_PY = ROOT / "run.py"


def _run_py_tree() -> ast.Module:
    return ast.parse(RUN_PY.read_text(encoding="utf-8-sig"))


def test_run_py_exists_and_is_importable_as_a_file():
    """The backend resolves this exact path; a move breaks every legacy run."""
    assert RUN_PY.is_file()
    from fiximg.infrastructure.db import engine  # noqa: F401  (sets up src path)

    from fiximg.inference.backends import legacy_cli

    default = legacy_cli.LegacyCliBackend({}).cli_path
    assert Path(default).resolve() == RUN_PY.resolve(), default


def test_run_py_delegates_instead_of_reimplementing():
    """No pipeline logic in the entry point.

    Asserted structurally: the module must import `main`, `StageError` and the
    progress marker from `fiximg.cli.batch`, and define no function of its own.
    A re-implementation — which is what actually happened — re-adds `def main`,
    and this fails.
    """
    tree = _run_py_tree()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "fiximg.cli.batch":
            imported.update(alias.name for alias in node.names)
    assert {"main", "StageError", "PROGRESS_PREFIX"} <= imported, imported

    own = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert own == [], (
        f"run.py defines {own}: the pipeline belongs in fiximg/cli/batch.py, and a "
        "second copy is exactly how it lost --stages"
    )


def test_the_progress_marker_is_the_same_string_on_both_sides():
    """The CLI writes it and the backend parses it, so there is one definition.

    A re-implementation of `run.py` carried its own copy of the constant while
    the backend kept the other; the mismatch would have shown up as a progress bar
    stuck at its first tick with the work still running.
    """
    import run as run_module
    from fiximg.inference.backends.legacy_cli import PROGRESS_PREFIX, parse_marker

    assert run_module.PROGRESS_PREFIX == PROGRESS_PREFIX
    assert parse_marker(f"{PROGRESS_PREFIX} 2/4 face detection") == (
        2, 4, "face detection"
    )


@pytest.mark.parametrize("argv", [
    ["--input_folder", "in", "--output_folder", "out", "--GPU", "0",
     "--stages", "1"],
    ["--input_folder", "in", "--output_folder", "out", "--GPU", "0",
     "--stages", "1,2,3,4", "--with_scratch", "--HR"],
])
def test_the_backend_argv_is_accepted_by_the_entry_point(argv):
    """The contract the backend depends on, checked against the real parser.

    `--stages` was the field that broke: the four-stage split added it to
    `cli/batch.py` and the entry point kept the pre-split parser, so the process
    exited 2 before loading a single weight.
    """
    from fiximg.cli.batch import build_parser

    parsed = build_parser().parse_args(argv)
    assert parsed.stages, parsed
    # ...and `run.py` really does forward to that same parser.
    probe = subprocess.run(
        [sys.executable, str(RUN_PY), "--help"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert probe.returncode == 0, probe.stderr
    for flag in ("--stages", "--with_scratch", "--HR", "--GPU",
                 "--input_folder", "--output_folder"):
        assert flag in probe.stdout, flag


def test_parse_stages_rejects_an_unknown_stage():
    """A typo in `--stages` must fail loudly, not silently restore nothing.

    `StageError` rather than `SystemExit`: the entry point converts it into exit
    code 2, and the backend surfaces the non-zero code as a stage failure. A
    `SystemExit` here would bypass the `except StageError` in `__main__` and
    lose the message.
    """
    from fiximg.cli.batch import ALL_STAGES, StageError, parse_stages

    assert parse_stages("") == list(ALL_STAGES)
    assert parse_stages("1") == ["1"]
    assert parse_stages("1,3") == ["1", "3"]
    with pytest.raises(StageError, match="9"):
        parse_stages("9")
