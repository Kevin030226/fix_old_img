"""The test layout must not be able to hide tests from the project's own command.

Three separate ways this repository could add tests that never run, and one
mechanism hid all three at once.

**A test directory missing from `testpaths`.** `tests/ui` was created and not
registered. `make test`, `make test-all`, `make test-gpu`, `make cov` and
`.github/workflows/ci.yml` all invoke a bare `pytest`, which resolves `testpaths`
and ignores anything not listed — so all 32 of its tests, including every gate
added alongside three fixes, were invisible to the gate. Measured: bare `pytest`
collected 1483 while `pytest tests` collected 1515.

**A test package without `__init__.py`.** Under pytest's default `prepend` import
mode a file in an unmarked directory is imported by its bare basename, so two
same-named files in two unmarked directories collide and one is silently dropped.
Six test packages had the marker and two did not.

**A test file with a name pytest does not collect.** `python_files = "test_*.py"`
is explicit, so `foo_test.py` or `tests_check.py` would be skipped with no warning.

These are static checks over the tree, so they cost nothing and cannot rot as
directories are added.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
PYPROJECT = ROOT / "pyproject.toml"


def _pytest_config() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["tool"]["pytest"]["ini_options"]


def _test_directories() -> set[str]:
    """Every directory under tests/ that directly holds a `test_*.py`."""
    out = set()
    for path in TESTS.rglob("test_*.py"):
        rel = path.parent.relative_to(ROOT).as_posix()
        out.add(rel)
    return out


def test_the_configured_testpaths_exist_on_disk():
    for entry in _pytest_config()["testpaths"]:
        assert (ROOT / entry).is_dir(), f"testpaths names a missing directory: {entry}"


def test_every_test_directory_is_registered_in_testpaths():
    """The gate that would have caught `tests/ui`."""
    configured = set(_pytest_config()["testpaths"])
    on_disk = _test_directories()
    missing = sorted(on_disk - configured)
    assert not missing, (
        "these directories hold tests but are absent from `testpaths`, so a bare "
        "`pytest` — which is what `make test`, `make test-all`, `make test-gpu`, "
        f"`make cov` and CI all run — never collects them: {missing}"
    )


def test_testpaths_names_nothing_superfluous():
    """The other direction: a stale entry hides nothing but misleads a reader."""
    configured = set(_pytest_config()["testpaths"])
    stale = sorted(configured - _test_directories())
    assert not stale, (
        f"testpaths names directories that hold no tests: {stale}. A stale entry "
        "usually means a directory was renamed or emptied."
    )


def test_every_test_package_carries_an_init_file():
    """Uniformity, so a future directory cannot silently opt out of the package.

    Under `prepend` import mode the marker is what distinguishes two same-named
    files; without a uniform convention, whether a collision is caught depends on
    which directory a file happens to land in.
    """
    without = sorted(
        d for d in _test_directories()
        if not (ROOT / d / "__init__.py").is_file()
    )
    assert not without, (
        "these test directories have no __init__.py while the others do; a "
        f"same-named file in two of them would collide and one would be dropped: "
        f"{without}"
    )


def test_test_files_are_named_so_pytest_collects_them():
    """`python_files` is explicit, so a conventional `*_test.py` is skipped."""
    config = _pytest_config()
    pattern = config.get("python_files", "test_*.py")
    assert pattern == "test_*.py", (
        f"python_files changed to {pattern!r}; this test assumes the test_ prefix "
        "and should be rechecked against the new pattern"
    )
    misnamed = sorted(
        p.relative_to(ROOT).as_posix()
        for p in TESTS.rglob("*_test.py")
    )
    assert not misnamed, (
        f"these end in _test.py and would not be collected under {pattern!r}: "
        f"{misnamed}"
    )


def test_no_two_test_files_share_a_basename():
    """The collision `__init__.py` exists to prevent, asserted directly.

    Currently harmless, and cheap to keep harmless: a collision is silent — the
    suite still reports green, with fewer tests than the tree holds.
    """
    seen: dict[str, list[str]] = {}
    for path in TESTS.rglob("test_*.py"):
        seen.setdefault(path.name, []).append(path.relative_to(ROOT).as_posix())
    collisions = {name: paths for name, paths in seen.items() if len(paths) > 1}
    assert not collisions, (
        "two test files share a basename, so pytest can import only one of them: "
        f"{collisions}"
    )


def test_the_marker_files_explain_themselves():
    """A bare `__init__.py` gets deleted by the next person who tidies up."""
    for directory in sorted(_test_directories()):
        marker = ROOT / directory / "__init__.py"
        text = marker.read_text(encoding="utf-8", errors="replace").strip()
        assert text, f"{marker.relative_to(ROOT).as_posix()} is empty and will look like a mistake"
        assert "prepend" in text or "collide" in text, (
            f"{marker.relative_to(ROOT).as_posix()} does not say why it exists"
        )


@pytest.mark.parametrize("entry", [
    "tests/unit", "tests/api", "tests/ui", "tests/inference",
    "tests/integration", "tests/e2e", "tests/gpu",
])
def test_each_registered_directory_holds_tests(entry):
    assert any((ROOT / entry).glob("test_*.py")), f"{entry} is registered but holds no tests"
