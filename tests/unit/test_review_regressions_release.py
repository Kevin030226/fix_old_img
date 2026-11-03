"""Regression tests for two P1 release defects confirmed against v3.1.2.

1. `pyproject.toml` declares `fiximg-weights = "fiximg.cli.download_weights:main"`
   and `fiximg-verify-weights = "fiximg.cli.verify_weights:main"`, but pip
   console scripts call the target with **no arguments**. Both `main` functions
   take a required `argv` positional, so both installed entry points raised
   `TypeError: main() missing 1 required positional argument: 'argv'` on every
   invocation — the entry points were broken by construction, and nothing in the
   suite invoked them.

2. `fiximg.paths.PROJECT_ROOT` climbs three levels from `<root>/src/fiximg/paths.py`.
   Under a standard wheel install the file is at
   `<env>/Lib/site-packages/fiximg/paths.py`, so the root resolves to the
   environment prefix and every runtime resource (`config/`, `models/`,
   `Global/`) is looked for in a directory that does not contain them.
"""
from __future__ import annotations

import inspect
import os

import pytest


# --------------------------------------------------- console script entry points
ENTRY_POINTS = [
    ("fiximg.cli.download_weights", "main"),
    ("fiximg.cli.verify_weights", "main"),
    ("fiximg.cli.batch", "main"),
    ("fiximg.cli.worker", "main"),
    ("fiximg.cli.api", "run"),
]


@pytest.mark.parametrize("module_name,attribute", ENTRY_POINTS)
def test_a_console_script_target_is_callable_without_arguments(module_name, attribute):
    """What pip actually does: `sys.exit(target())` with no arguments.

    A required positional parameter makes the installed command unusable, and
    the failure is invisible to any test that only imports the module.
    """
    import importlib

    module = importlib.import_module(module_name)
    target = getattr(module, attribute)
    signature = inspect.signature(target)

    required = [
        name for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    ]
    assert not required, (
        f"{module_name}:{attribute} needs {required} from a caller, but the "
        f"console script invokes it with no arguments: {signature}"
    )


def test_the_declared_entry_points_are_the_ones_this_package_ships():
    """The mapping in pyproject is what pip generates; check it against the code."""
    import tomllib
    from pathlib import Path

    import fiximg

    root = Path(fiximg.__file__).resolve().parents[2]
    with open(root / "pyproject.toml", "rb") as handle:
        project = tomllib.load(handle)
    scripts = project["project"]["scripts"]

    import importlib

    for name, spec in scripts.items():
        module_name, _, attribute = spec.partition(":")
        module = importlib.import_module(module_name)
        target = getattr(module, attribute)
        signature = inspect.signature(target)
        required = [
            param_name for param_name, param in signature.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
        assert not required, f"{name} -> {spec} requires {required}: {signature}"


def test_verify_weights_defaults_to_the_verify_subcommand(monkeypatch):
    """`python -m fiximg.cli.verify_weights` with no arguments means "verify".

    The alias used to be only an `if __name__ == "__main__"` block, so the
    console script that points at `fiximg.cli.verify_weights:main` had no
    `main` of its own to call. Asserted on the wrapper rather than by running a
    real verification, which would hash the weight tree.
    """
    from fiximg.cli import verify_weights as vw

    calls: list[list[str]] = []

    def recorder(argv):
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(vw, "_download_main", recorder)
    assert vw.main(["fiximg-verify-weights"]) == 0
    assert calls == [["fiximg-verify-weights", "verify"]], calls

    # An explicit sub-command is forwarded untouched.
    assert vw.main(["fiximg-verify-weights", "generate"]) == 0
    assert calls[-1] == ["fiximg-verify-weights", "generate"]


def test_the_download_weights_usage_names_the_verify_command():
    """The shared parser still recognises the sub-command the alias defaults to."""
    from fiximg.cli import download_weights as dw

    assert dw.main(["fiximg-weights", "help"]) == 2
    assert dw.main([]) == 2, "a bare call must print usage rather than raise"

# ------------------------------------------------------------- wheel path layout
def test_project_root_is_overridable_from_the_environment(monkeypatch, tmp_path):
    """`FIXIMG_PROJECT_ROOT` is the supported way to point an install at assets.

    A wheel ships code only, so a non-editable install has no other way to name
    the deployment tree. The override must win before any layout guess.
    """
    from fiximg import paths

    (tmp_path / "config").mkdir()
    (tmp_path / "models").mkdir()
    monkeypatch.setenv("FIXIMG_PROJECT_ROOT", str(tmp_path))
    reloaded = importlib_reload(paths)
    assert reloaded.PROJECT_ROOT == str(tmp_path), reloaded.PROJECT_ROOT
    assert reloaded.CONFIG_DIR == os.path.join(str(tmp_path), "config")
    assert reloaded.require_project_root() == str(tmp_path)


def importlib_reload(module):
    import importlib

    return importlib.reload(module)


def test_a_src_layout_checkout_still_resolves_its_own_root(monkeypatch):
    """The override must not disturb the normal checkout, which needs no variable."""
    from fiximg import paths

    monkeypatch.delenv("FIXIMG_PROJECT_ROOT", raising=False)
    reloaded = importlib_reload(paths)
    assert os.path.isdir(os.path.join(reloaded.PROJECT_ROOT, "config"))
    assert reloaded.require_project_root() == reloaded.PROJECT_ROOT


def test_a_site_packages_install_without_resources_fails_with_an_actionable_error(tmp_path):
    """The real wheel layout: three levels up is the env prefix, not a project.

    The resolver may return that guess (so path constants are strings), but
    `require_project_root` must refuse it and name FIXIMG_PROJECT_ROOT — the
    difference between a clear deployment error and a missing-manifest traceback
    from inside a model loader.
    """
    from fiximg import paths

    package = tmp_path / "Lib" / "site-packages" / "fiximg"
    package.mkdir(parents=True)
    resolved = paths._resolve_project_root(str(package / "paths.py"))
    assert not paths._looks_like_project_root(resolved)

    original = paths.PROJECT_ROOT
    paths.PROJECT_ROOT = resolved
    try:
        with pytest.raises(paths.ProjectRootNotFoundError) as excinfo:
            paths.require_project_root()
    finally:
        paths.PROJECT_ROOT = original
    message = str(excinfo.value)
    assert "FIXIMG_PROJECT_ROOT" in message
    assert "config" in message and "models" in message


def test_require_project_root_accepts_a_complete_tree(tmp_path):
    from fiximg import paths

    (tmp_path / "config").mkdir()
    (tmp_path / "models").mkdir()
    original = paths.PROJECT_ROOT
    paths.PROJECT_ROOT = str(tmp_path)
    try:
        assert paths.require_project_root() == str(tmp_path)
    finally:
        paths.PROJECT_ROOT = original


def test_the_override_beats_every_guess(monkeypatch, tmp_path):
    """An explicit answer is never second-guessed by the layout heuristics."""
    from fiximg import paths

    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv("FIXIMG_PROJECT_ROOT", str(other))
    assert paths._resolve_project_root() == str(other)


def test_the_resolver_always_returns_an_absolute_path(tmp_path):
    """Whatever branch is taken, callers can join it to a filename safely."""
    from fiximg import paths

    assert os.path.isabs(paths._resolve_project_root())
    layout = tmp_path / "nowhere" / "fiximg"
    layout.mkdir(parents=True)
    assert os.path.isabs(paths._resolve_project_root(str(layout / "paths.py")))
