"""Behavioural tests for scripts/ci/release_tag_guard.sh.

The guard exists because actions/checkout@v5 checks a tag push out by fetching
the tag object and then force-updating the checkout-owned refs/tags/<tag> ref to
the peeled commit. Code that asks that ref what type it is therefore sees a
lightweight ref and concludes that a perfectly good annotated tag is
unannotated -- which is exactly how the v3.1.1 release failed after passing the
workflow-structure tests.

These tests reproduce that state with real git repositories, so the guard is
judged by behaviour instead of by grepping the workflow for an error message.
"""
import os
import shutil
import subprocess

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HELPER = os.path.join(PROJECT_ROOT, "scripts", "ci", "release_tag_guard.sh").replace(os.sep, "/")

TAG = "v1.2.3"
#: The dash below is U+2014 EM DASH, the character the real v3.1.1 annotation uses.
TITLE = "v1.2.3 \u2014 eval-pool race fix and hardened release delivery"


def _find_bash():
    """Return a bash that can reach this checkout, or None.

    On Windows a bare ``bash`` is usually the WSL stub in system32, whose view of
    drive-letter paths is a different filesystem, so Git-for-Windows candidates are
    tried first and every candidate must prove it can see a real file here.
    """
    candidates = []
    if os.name == "nt":
        for env in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData"):
            base = os.environ.get(env)
            if not base:
                continue
            candidates.append(os.path.join(base, "Git", "bin", "bash.exe"))
            candidates.append(os.path.join(base, "Programs", "Git", "bin", "bash.exe"))
    found = shutil.which("bash") or shutil.which("bash.exe")
    if found and "Windows" not in found:
        candidates.append(found)

    probe_path = PROJECT_ROOT.replace(os.sep, "/")
    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        try:
            probe = subprocess.run(
                [candidate, "-c", 'test -f "$1/pyproject.toml" && git -C "$1" rev-parse --git-dir',
                 "_", probe_path],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            return candidate
    return None


@pytest.fixture(scope="session")
def bash():
    path = _find_bash()
    if path is None:
        pytest.skip("no bash that can reach this checkout is available")
    return path


def _git_env(home):
    env = dict(os.environ)
    env.update({
        "GIT_AUTHOR_NAME": "release-test",
        "GIT_AUTHOR_EMAIL": "release-test@example.invalid",
        "GIT_COMMITTER_NAME": "release-test",
        "GIT_COMMITTER_EMAIL": "release-test@example.invalid",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.path.join(home, "gitconfig"),
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": home,
        "USERPROFILE": home,
    })
    return env


def _git(env, cwd, *args, check=True):
    result = subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {args} failed: {result.stdout}{result.stderr}")
    return result


class Fixture:
    """A bare origin plus a working clone that looks like a checked-out tag push."""

    def __init__(self, env, work, main_sha):
        self.env = env
        self.work = work
        self.main_sha = main_sha

    def git(self, *args, check=True):
        return _git(self.env, self.work, *args, check=check)

    def object_type(self, ref):
        return self.git("cat-file", "-t", ref).stdout.strip()

    def private_refs(self):
        listing = self.git("for-each-ref", "--format=%(refname)", "refs/release-tags")
        return [line for line in listing.stdout.splitlines() if line.strip()]

    def guard(self, bash, tag=TAG, expected=None, remote="origin", output_path=None):
        """Source the helper, run the guard, and report rc plus its validated globals."""
        env = dict(self.env)
        env.pop("GITHUB_OUTPUT", None)
        if output_path is not None:
            env["GITHUB_OUTPUT"] = output_path
        if expected is None:
            expected = self.main_sha
        script = (
            'source "$1" || exit 90; '
            'release_tag_guard "$2" "$3" "$4" || exit $?; '
            'if [ -n "${GITHUB_OUTPUT:-}" ]; then release_tag_write_outputs || exit $?; fi; '
            'printf "SHA=%s\\n" "$RELEASE_TAG_SHA"; '
            'printf "TITLE=%s\\n" "$RELEASE_TAG_TITLE"'
        )
        return subprocess.run(
            [bash, "-c", script, "_", HELPER, tag, expected, remote],
            cwd=self.work, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )


def _build(tmp_path, tag=TAG, message=TITLE, annotated=True, tag_target="main"):
    """Create origin, a seed clone with two commits, and a work clone.

    The work clone ends in the state actions/checkout leaves behind:
    refs/remotes/origin/main exists and refs/tags/<tag>, if present, has been
    replaced by the peeled commit.
    """
    env = _git_env(str(tmp_path))
    origin = os.path.join(str(tmp_path), "origin.git")
    seed = os.path.join(str(tmp_path), "seed")
    work = os.path.join(str(tmp_path), "work")

    _git(env, str(tmp_path), "init", "--bare", "-q", origin)
    _git(env, str(tmp_path), "clone", "-q", origin, seed)
    _git(env, seed, "checkout", "-q", "-B", "main")
    _git(env, seed, "commit", "-q", "--allow-empty", "-m", "one")
    _git(env, seed, "commit", "-q", "--allow-empty", "-m", "two")
    older = _git(env, seed, "rev-parse", "HEAD~1").stdout.strip()
    main_sha = _git(env, seed, "rev-parse", "HEAD").stdout.strip()

    if tag is not None:
        target = older if tag_target == "older" else main_sha
        if annotated:
            message_file = os.path.join(str(tmp_path), "tag-message.txt")
            payload = message if isinstance(message, bytes) else message.encode("utf-8")
            with open(message_file, "wb") as handle:
                handle.write(payload)
            _git(env, seed, "-c", "tag.gpgsign=false", "tag", "-a",
                 "--cleanup=verbatim", "-F", message_file, tag, target)
        else:
            _git(env, seed, "tag", tag, target)

    _git(env, seed, "push", "-q", "origin", "main")
    if tag is not None:
        _git(env, seed, "push", "-q", "origin", "refs/tags/" + tag)

    _git(env, str(tmp_path), "clone", "-q", origin, work)
    _git(env, work, "fetch", "-q", "--no-tags", "origin",
         "+refs/heads/main:refs/remotes/origin/main")
    _git(env, work, "checkout", "-q", "-B", "main", "refs/remotes/origin/main")

    if tag is not None:
        # The runner state: the public tag ref now points at the commit itself.
        _git(env, work, "update-ref", "refs/tags/" + tag, main_sha)

    return Fixture(env, work, main_sha)

# ---------------------------------------------------------------------------
# The regression these tests exist for
# ---------------------------------------------------------------------------


def test_guard_accepts_an_annotated_tag_whose_public_ref_was_rewritten(bash, tmp_path):
    """The v3.1.1 failure: the public ref is a commit, the remote tag is annotated."""
    fixture = _build(tmp_path)
    # Precondition -- without this the test would pass for the wrong reason.
    assert fixture.object_type("refs/tags/" + TAG) == "commit"

    result = fixture.guard(bash)

    assert result.returncode == 0, result.stderr
    assert "SHA=" + fixture.main_sha in result.stdout
    assert "TITLE=" + TITLE in result.stdout
    # Proved from a private copy, and the checkout-owned ref is left alone.
    assert fixture.object_type("refs/release-tags/" + TAG) == "tag"
    assert fixture.object_type("refs/tags/" + TAG) == "commit"


def test_guard_writes_the_validated_values_to_github_output(bash, tmp_path):
    fixture = _build(tmp_path)
    output = os.path.join(str(tmp_path), "github-output")
    with open(output, "w", encoding="utf-8") as handle:
        handle.write("version=3.1.2\n")

    result = fixture.guard(bash, output_path=output)

    assert result.returncode == 0, result.stderr
    with open(output, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    assert lines[0] == "version=3.1.2", "the guard must not disturb earlier outputs"
    assert lines[1:] == ["tag_sha=" + fixture.main_sha, "release_title=" + TITLE]


def test_guard_uses_the_first_paragraph_of_a_multiline_annotation(bash, tmp_path):
    fixture = _build(tmp_path, message=TITLE + "\n\nBody text that must not become the title.\n")
    result = fixture.guard(bash)
    assert result.returncode == 0, result.stderr
    assert "TITLE=" + TITLE in result.stdout


def test_guard_keeps_shell_metacharacters_literal(bash, tmp_path):
    """Annotation text is data: it must never be evaluated by the shell."""
    payload = "v1.2.3 $(touch PWNED_SUBST) `touch PWNED_TICK` a=b ; echo hi 'q' \"dq\""
    fixture = _build(tmp_path, message=payload)

    result = fixture.guard(bash)

    assert result.returncode == 0, result.stderr
    assert "TITLE=" + payload in result.stdout, result.stdout
    assert not os.path.exists(os.path.join(fixture.work, "PWNED_SUBST"))
    assert not os.path.exists(os.path.join(fixture.work, "PWNED_TICK"))


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_guard_rejects_a_lightweight_tag(bash, tmp_path):
    fixture = _build(tmp_path, annotated=False)
    result = fixture.guard(bash)
    assert result.returncode != 0
    assert "is not an annotated tag" in result.stderr


def test_guard_rejects_a_tag_pointing_at_an_older_commit(bash, tmp_path):
    fixture = _build(tmp_path, tag_target="older")
    result = fixture.guard(bash)  # expected commit is main
    assert result.returncode != 0
    assert "but the expected commit is" in result.stderr


def test_guard_reports_a_missing_remote_tag(bash, tmp_path):
    fixture = _build(tmp_path, tag=None)
    result = fixture.guard(bash)
    assert result.returncode != 0
    assert "could not fetch tag" in result.stderr


def test_guard_failures_speak_only_the_guard_message(bash, tmp_path):
    """No git transport, path or credential detail may reach the log."""
    fixture = _build(tmp_path, tag=None)
    result = fixture.guard(bash)
    assert result.returncode != 0
    for line in result.stderr.splitlines():
        if line.strip():
            assert line.startswith("::error::"), line


@pytest.mark.parametrize(
    "bad_tag",
    ["v1.2", "1.2.3", "V1.2.3", "release-1.2.3", "v1.2.3;rm -rf /", "v1.2.3/../x", "v1.2.3 "],
)
def test_guard_rejects_tag_names_it_would_have_to_quote(bash, tmp_path, bad_tag):
    fixture = _build(tmp_path)
    result = fixture.guard(bash, tag=bad_tag)
    assert result.returncode != 0
    assert "is not a vMAJOR.MINOR.PATCH tag name" in result.stderr
    assert fixture.private_refs() == [], "no refspec may be attempted for a bad name"


@pytest.mark.parametrize("bad_sha", ["", "abc", "da11a30", "A" * 40, "z" * 40])
def test_guard_rejects_an_expected_commit_that_is_not_a_full_lowercase_sha(bash, tmp_path, bad_sha):
    fixture = _build(tmp_path)
    result = fixture.guard(bash, expected=bad_sha)
    assert result.returncode != 0
    assert "is not a 40-character lowercase hex SHA" in result.stderr


def test_guard_rejects_a_whitespace_only_annotation(bash, tmp_path):
    fixture = _build(tmp_path, message=b"   \n")
    result = fixture.guard(bash)
    assert result.returncode != 0
    assert "empty annotation" in result.stderr


def test_guard_rejects_an_overlength_title(bash, tmp_path):
    fixture = _build(tmp_path, message="v1.2.3 " + ("x" * 260))
    result = fixture.guard(bash)
    assert result.returncode != 0
    assert "release title budget is 200" in result.stderr


@pytest.mark.parametrize(
    "payload",
    [
        b"v1.2.3 bad\rrest",
        b"v1.2.3 bad\x07rest",
        b"v1.2.3 bad\x1b[31mred",
    ],
)
def test_guard_rejects_control_characters_in_the_title(bash, tmp_path, payload):
    fixture = _build(tmp_path, message=payload)
    result = fixture.guard(bash)
    assert result.returncode != 0, "a control character must not reach GITHUB_OUTPUT"
    assert "control character" in result.stderr


def test_guard_refuses_to_reuse_a_stale_private_ref(bash, tmp_path):
    fixture = _build(tmp_path)
    fixture.git("update-ref", "refs/release-tags/" + TAG, fixture.main_sha)

    result = fixture.guard(bash)

    assert result.returncode != 0
    assert "already exists" in result.stderr
    # The stale ref was neither reused nor force-updated into the tag object.
    assert fixture.object_type("refs/release-tags/" + TAG) == "commit"


def test_guard_rejects_a_tag_that_is_not_the_commit_on_main(bash, tmp_path):
    """A tag on some other commit must not publish: expected alone is not enough."""
    fixture = _build(tmp_path)
    older = fixture.git("rev-parse", "HEAD~1").stdout.strip()
    fixture.git("update-ref", "refs/remotes/origin/main", older)

    # The caller passes the tag's own target, so only the origin/main comparison
    # can reject this.
    result = fixture.guard(bash, expected=fixture.main_sha)

    assert result.returncode != 0
    assert "but origin/main is" in result.stderr


def test_guard_requires_origin_main(bash, tmp_path):
    fixture = _build(tmp_path)
    fixture.git("update-ref", "-d", "refs/remotes/origin/main")
    result = fixture.guard(bash)
    assert result.returncode != 0
    assert "cannot prove the tag is on main" in result.stderr
