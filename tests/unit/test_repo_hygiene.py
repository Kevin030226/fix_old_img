"""Repository hygiene gates that no existing CI step covered.

The RC2.1 audit ran `git diff --check v3.1.2..HEAD` by hand and found a new
blank line at EOF in a test file. The lint job runs ruff and compileall, not
git whitespace checks, so nothing would have caught a repeat. These tests make
the manual command a gate and keep it wired into CI.
"""
import subprocess

import pytest

BASE = "v3.1.2"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, timeout=120)


def test_the_branch_diff_has_no_whitespace_errors():
    """`git diff --check v3.1.2` (working tree included) must stay clean.

    Uses the tag directly instead of a range so uncommitted edits are covered
    locally; in CI the tree is clean, so this is exactly the committed branch.
    """
    if _git("rev-parse", "--is-inside-work-tree").returncode != 0:
        pytest.skip("not a git checkout")
    if _git("rev-parse", "-q", "--verify", BASE).returncode != 0:
        pytest.skip(f"release base {BASE} is not available (shallow clone)")
    result = _git("diff", "--check", BASE)
    assert result.returncode == 0, (
        "git diff --check reports whitespace errors:\n"
        + (result.stdout or "") + (result.stderr or "")
    )


def test_ci_lint_runs_the_whitespace_gate():
    """The gate must run in CI, not only on the machine that remembered it."""
    with open(".github/workflows/ci.yml", encoding="utf-8") as handle:
        ci = handle.read()
    assert "git diff --check v3.1.2..HEAD" in ci, (
        "the lint job no longer runs the whitespace gate; a new blank line at EOF"
        " would reach a release branch silently, which is the RC2.1 finding"
    )
