"""The vendored scripts V3 runs must need nothing that is not declared.

CI's 3.11 job failed on this, locally-invisible bug: ``Global/detection.py`` imports
``detection_util.util``, which imported ``matplotlib`` at module level. Nothing declares
matplotlib as a dependency of the install CI (and ``docker/api.Dockerfile``) runs, so the
scratch detector could not load in a clean environment -- while passing on every
developer machine that happened to have matplotlib installed. ``show_detection`` /
``imshow``, the only users of that import, are upstream debug helpers no pipeline path
calls.

The check is static on purpose. A runtime "import it with matplotlib blocked" test would
need torch, dlib and opencv present to get as far as the assertion, so it would skip on
exactly the environments whose import closure is in question. What is verified here needs
no packages installed at all, and the suite's own native-backend tests remain the runtime
proof in the environment CI actually builds.
"""
from __future__ import annotations

import ast
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
STDLIB = set(sys.stdlib_module_names)

#: Directories holding upstream code this project runs (plan §3.5.1: reused, not forked).
VENDORED_ROOTS = ("Global", "Face_Detection", "Face_Enhancement", "ddcolor", "basicsr")

#: Import roots whose distribution is named differently from the module.
IMPORT_ALIASES = {
    "cv2": ("opencv-python", "opencv-python-headless", "opencv-contrib-python"),
    "PIL": ("pillow",),
    "yaml": ("pyyaml",),
    "skimage": ("scikit-image",),
}


def _is_vendored(path: str) -> bool:
    relative = os.path.relpath(path, ROOT)
    return relative.split(os.sep)[0] in VENDORED_ROOTS


def _tree_root(path: str) -> str:
    return os.path.join(ROOT, os.path.relpath(path, ROOT).split(os.sep)[0])


def _local_module(base: str, module: str) -> str | None:
    """The file a dotted name resolves to inside one vendored tree, if any."""
    if not module:
        return None
    parts = module.split(".")
    as_file = os.path.join(base, *parts) + ".py"
    if os.path.isfile(as_file):
        return as_file
    as_package = os.path.join(base, *parts, "__init__.py")
    return as_package if os.path.isfile(as_package) else None


def _string_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "string"`` pairs, to resolve ``os.path.join(GLOBAL_DIR, x)``."""
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        out[target.id] = node.value.value
    return out


def executed_vendored_files() -> list[str]:
    """Every vendored ``.py`` that ``src`` loads in-process or runs as a subprocess.

    Derived, not listed: a new ``spec_from_file_location(..., "somthing.py")`` or a new
    ``[sys.executable, "script.py"]`` in the CLI joins the set automatically, which is the
    point -- the bug this file guards was one import three files deep.
    """
    found: set[str] = set()
    for dirpath, _dirs, files in os.walk(SRC):
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            tree = ast.parse(_read(path), filename=path)
            consts = _string_constants(tree)
            bases = [value for value in consts.values() if os.path.isdir(value)]
            bases += [os.path.join(ROOT, top) for top in VENDORED_ROOTS]
            bases.append(os.path.dirname(path))
            for candidate_names in _vendored_script_names(tree):
                for base in bases:
                    candidate = os.path.normpath(os.path.join(base, candidate_names))
                    if os.path.isfile(candidate) and _is_vendored(candidate):
                        found.add(candidate)
    return sorted(found)


def _vendored_script_names(tree: ast.Module) -> set[str]:
    """.py file names a module refers to, including the ones inside a path join."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_file_location_call(node):
            for argument in node.args[1:]:
                names |= _string_leaves(argument)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.endswith(".py"):
                names.add(node.value)
    return {os.path.basename(name) for name in names if name.endswith(".py")}


def _is_file_location_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and node.func.attr == "spec_from_file_location"


def _string_leaves(node: ast.AST) -> set[str]:
    return {
        sub.value
        for sub in ast.walk(node)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    }


def closure_roots(entry: str) -> tuple[set[str], set[str]]:
    """Third-party module-level import roots of one vendored file, and files walked.

    Sibling modules inside the same tree are followed rather than counted, because the
    undeclared import that broke CI lived in one (``detection.py`` -> ``detection_util``).
    """
    roots: set[str] = set()
    seen: set[str] = set()
    stack = [entry]
    while stack:
        path = stack.pop()
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        tree = ast.parse(_read(path), filename=path)
        base = _tree_root(path)
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in STDLIB:
                        continue
                    local = _local_module(base, alias.name)
                    if local:
                        stack.append(local)
                    else:
                        roots.add(top)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    sibling = os.path.join(base, *(node.module or "").split(".")) + ".py"
                    stack.append(os.path.normpath(sibling))
                    continue
                module = node.module or ""
                top = module.split(".")[0]
                if top in STDLIB:
                    continue
                local = _local_module(base, module)
                if local:
                    stack.append(local)
                else:
                    roots.add(top)
    roots.discard("__future__")
    return roots, seen


def declared_dependencies() -> set[str]:
    """Every distribution ``pyproject.toml`` declares: base plus every extra."""
    return _declared_groups()[0]


#: Distributions the CI test jobs install by command rather than through an
#: extra, so the `.[test]` extra alone is not the whole environment. Read off
#: `.github/workflows/ci.yml`:
#:
#:     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#:
#: The CPU wheel index is deliberate -- the tests stub inference at the
#: stage/backend boundary, so a 2 GB CUDA build would buy nothing. Listed here
#: rather than added to the `test` extra because an extra that pulls torch by
#: default would make `pip install -e ".[test]"` a multi-gigabyte operation for
#: anyone running the suite locally.
INSTALL_OUTSIDE_EXTRAS = {"torch", "torchvision"}


def _declared_groups() -> tuple[set[str], set[str]]:
    """(everything declared, what the *test* install can pull).

    The two are not the same, and the difference is a defect class of its own.
    CI installs ``.[test]``, which resolves to the base dependencies plus the
    ``test`` extra. ``easydict`` was declared only in the ``gpu`` extra, so a
    name being *declared* proved nothing about the environment these tests
    actually run in: the closure check above was satisfied while the import
    still failed, and the next CI run failed with "No module named 'easydict'".
    """
    import tomllib

    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
        data = tomllib.load(handle)
    project = data["project"]
    base = list(project.get("dependencies", []))
    extras = project.get("optional-dependencies") or {}

    everything = {_requirement_name(s) for s in base}
    for extra in extras.values():
        everything |= {_requirement_name(s) for s in extra}
    everything.discard("")
    assert None not in everything
    assert len(everything) >= 15, f"only {len(everything)} requirements parsed from pyproject.toml"

    test_env = {_requirement_name(s) for s in base}
    test_env |= {_requirement_name(s) for s in (extras.get("test") or [])}
    test_env.discard("")
    return everything, test_env | set(INSTALL_OUTSIDE_EXTRAS)


def _requirement_name(specifier: str) -> str:
    head = re.split(r"[\[<>=!~;]", specifier.strip(), maxsplit=1)[0].strip()
    return head.lower().replace("_", "-")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


# --------------------------------------------------------------------------- tests
def test_the_derivation_finds_the_scripts_the_pipeline_and_cli_actually_run():
    """A guard against the check passing because it looked at nothing."""
    entries = [os.path.relpath(p, ROOT).replace(os.sep, "/") for p in executed_vendored_files()]
    assert len(entries) >= 5, f"the derivation regressed to {entries}"
    # The two chains the native backends load in-process, and one CLI subprocess script.
    assert "Global/detection.py" in entries
    assert "Face_Detection/detect_all_dlib.py" in entries
    assert "Face_Detection/align_warp_back_multiple_dlib.py" in entries


def test_every_module_the_executed_scripts_import_is_declared_by_the_package():
    """No vendored script this project runs may import an undeclared distribution.

    This is the assertion that would have failed before the fix: the closure reached
    ``detection_util.util`` and reported ``matplotlib``, which no install path declares.
    """
    deps = declared_dependencies()
    entries = executed_vendored_files()
    assert entries, "the derivation found no vendored scripts"

    undeclared: dict[str, str] = {}
    exercised_alias = False
    for entry in entries:
        roots, walked = closure_roots(entry)
        assert len(walked) > 1 or os.path.basename(entry) != "detection.py", (
            "the sibling walk stopped at one file, so a nested import cannot be found"
        )
        for root in sorted(roots):
            normalised = root.lower().replace("_", "-")
            if normalised in deps:
                continue
            aliases = IMPORT_ALIASES.get(root, ())
            if any(a.lower().replace("_", "-") in deps for a in aliases):
                exercised_alias = True
                continue
            undeclared[root] = os.path.relpath(entry, ROOT)

    assert not undeclared, (
        "vendored scripts import packages no install path declares "
        f"(a clean `pip install` cannot run them): {undeclared}"
    )
    assert exercised_alias, (
        "no import root needed an alias, so IMPORT_ALIASES is a waiver nobody uses -- "
        "delete the entries this run did not exercise"
    )


def test_every_module_the_detection_chain_imports_is_installable_by_the_test_extra():
    """The stronger claim for the chain that actually broke: installable, not declared.

    ``test_every_module_the_executed_scripts_import_is_declared_by_the_package``
    unions every extra, so a package in ``gpu`` alone satisfies it. CI installs
    ``.[test]`` and never ``gpu``, so the union answers the wrong question for the
    one vendored chain the test suite really imports in-process: the scratch
    detector, loaded by ``global_restore._detection_module()``. ``easydict`` was
    declared only in ``gpu``, the check was satisfied, and the next CI run failed
    with "No module named 'easydict'".

    Scoped to that chain deliberately. ``executed_vendored_files()`` also returns
    the training entry points -- ``Global/test.py`` and
    ``Face_Enhancement/test_face.py`` -- which import torch, dlib and tensorboardX
    at module level. Demanding the CPU-only test environment be able to *run* a
    training script would mean installing the whole GPU stack to run unit tests,
    and no test here executes one. The vendored scripts the pipeline loads through
    the subprocess adapter need no such thing, because a subprocess inherits the
    worker image, not this one.
    """
    _everything, test_env = _declared_groups()
    entry = os.path.join(ROOT, "Global", "detection.py")
    assert os.path.isfile(entry), f"the scratch detector moved: {entry}"

    roots, walked = closure_roots(entry)
    assert len(walked) > 1, "the sibling walk stopped at one file"

    missing = sorted(
        root for root in roots
        if not ({root.lower().replace("_", "-")}
                | {a.lower().replace("_", "-")
                   for a in IMPORT_ALIASES.get(root, ())}) & test_env
    )
    assert not missing, (
        "the scratch detector's import chain needs packages that "
        f"`pip install -e \".[test]\"` does not install, so it fails in CI while "
        f"looking declared: {missing}"
    )


@pytest.mark.parametrize(
    "alias,distributions", sorted(IMPORT_ALIASES.items()), ids=lambda v: str(v)
)
def test_each_import_alias_points_at_a_declared_distribution(alias, distributions):
    """Keeps the alias table from accumulating entries that save nothing.

    An alias whose candidates are all undeclared is not a mapping, it is a hole in the
    check above -- and an alias for a root no vendored script imports is dead weight that
    reads like coverage.
    """
    deps = declared_dependencies()
    assert any(d.lower().replace("_", "-") in deps for d in distributions), (
        f"alias {alias!r} maps to {distributions}, none of which pyproject declares"
    )


def test_the_flat_requirements_file_declares_nothing_the_package_does_not():
    """``requirements.txt`` and ``pyproject.toml`` are two names for one dependency set.

    They had drifted: the flat file pinned ``matplotlib`` while the package never
    declared it, so whichever install path a machine took decided whether the scratch
    detector could import. Every name in the flat list must therefore be a dependency or
    an extra of the package -- including the test/dev tools it deliberately carries.
    """
    deps = declared_dependencies()
    text = _read(os.path.join(ROOT, "requirements.txt"))
    pinned = set()
    for line in text.splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry and entry[0].isalpha():
            pinned.add(_requirement_name(entry))
    assert len(pinned) >= 20, f"only {len(pinned)} requirements parsed"
    undocumented = sorted(pinned - deps)
    assert not undocumented, (
        f"requirements.txt pins what pyproject.toml does not declare: {undocumented}"
    )
