"""The two ignore lists must agree on what must never leave the machine.

Found by auditing the tree, not by a failing build. Two entries had drifted:

* `.mypy_cache/` was in neither list. `.gitignore` enumerates the tool caches
  explicitly — `.pytest_cache/`, `.ruff_cache/` — and mypy was simply missed, so
  123 MB of cache was stageable by a plain `git add -A`.
* `.gates/` was in `.gitignore` but not `.dockerignore`. The two are independent
  lists and the Dockerfile's only copy instruction is `COPY . /app`, so a build
  would have embedded a PostgreSQL data directory, live SQLite databases holding
  the `admin` and `tester` accounts, the bearer token, both test-account passwords
  and the run logs. That directly contradicts the Dockerfile's own comment:
  "Admin credentials are NOT baked into the image (plan §21)".

Nothing here is a *content* problem — the credentials in those files are
deliberate local test values, and every credential-shaped string in a tracked file
is a placeholder. The defect is that the packaging rules let local material reach a
published artifact, and nothing checked the two lists against each other.

These tests are the check. They are about the rules, not the files, so they stay
cheap and cannot rot as scratch files are added and removed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GITIGNORE = ROOT / ".gitignore"
DOCKERIGNORE = ROOT / ".dockerignore"
DOCKERFILE = ROOT / "Dockerfile"

#: Never committed, and never in an image. Each entry is here because putting it in
#: one list and not the other is the exact failure that produced this file.
MUST_BE_EXCLUDED_EVERYWHERE = [
    ".gates",            # local scratch: databases, credentials, logs, a PG data dir
    ".mypy_cache",       # 123 MB of type-check cache
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "admin_data",        # api_token.txt and initial_admin_password.txt
    "storage",           # user-uploaded photos and results
    ".workbuddy-ai",     # agent memory; contains local paths
]


def _entries(path: Path) -> set[str]:
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line.rstrip("/"))
    return out


def _names(path: Path, names: list[str]) -> set[str]:
    entries = _entries(path)
    found = set()
    for name in names:
        bare = name.rstrip("/")
        if bare in entries or name in entries:
            found.add(name)
    return found


def test_both_files_exist_and_are_readable():
    assert GITIGNORE.is_file(), GITIGNORE
    assert DOCKERIGNORE.is_file(), DOCKERIGNORE


@pytest.mark.parametrize("name", MUST_BE_EXCLUDED_EVERYWHERE)
def test_sensitive_material_is_never_stageable(name):
    missing = [p.name for p in (GITIGNORE, DOCKERIGNORE) if name.rstrip("/") not in _entries(p)]
    assert not missing, f"{name} is missing from: {missing}"


def test_the_two_lists_do_not_drift_on_the_sensitive_entries():
    """`.gates` was in one list and not the other. That asymmetry is the bug.

    Reported as a set difference in both directions, because the failure is not
    "one list is missing an entry" but "the lists disagree".
    """
    git = _entries(GITIGNORE)
    docker = _entries(DOCKERIGNORE)
    only_git = {n.rstrip("/") for n in MUST_BE_EXCLUDED_EVERYWHERE} - docker
    only_docker = {n.rstrip("/") for n in MUST_BE_EXCLUDED_EVERYWHERE} - git
    assert not (only_git or only_docker), (
        f"excluded from .dockerignore only: {sorted(only_git)}; "
        f"excluded from .gitignore only: {sorted(only_docker)}"
    )
    # And the files actually differ, so the test is reading two real lists.
    assert git != docker, (
        "the two ignore files are identical; if that is intended, this module's "
        "reason for existing needs rechecking"
    )


def test_the_dockerfile_copies_the_whole_context():
    """Why `.dockerignore` is load-bearing at all.

    If the copy instruction were narrowed to specific paths, none of the above
    would matter. Asserting the opposite guards against someone "fixing" the
    symptom by believing the ignore file is decorative.
    """
    dockerfile = DOCKERFILE.read_text(encoding="utf-8", errors="replace")
    copies_all = any(
        line.strip().upper().startswith(("COPY . ", "ADD . "))
        for line in dockerfile.splitlines()
    )
    assert copies_all, (
        "the Dockerfile no longer copies the whole context, so the exact set of "
        "files in `.dockerignore` may no longer be what reaches the image; "
        "re-check these expectations before relaxing them"
    )


def test_no_credential_bearing_file_is_ignored_from_both_lists():
    """A file that holds a real secret must be excluded from *both* lists.

    Named explicitly because the failure is silent: an excluded file is invisible
    to a content scan, so nothing else in this repository would notice one that
    stopped being ignored.
    """
    git = _entries(GITIGNORE)
    docker = _entries(DOCKERIGNORE)
    for name in ("admin_data", "storage", "output_img", "user_upload_images"):
        bare = name.rstrip("/")
        assert bare in git, f"{name} is stageable"
        assert bare in docker, f"{name} would reach the image"


def test_local_scratch_in_this_checkout_is_actually_ignored_now():
    """The rule exists for files that exist; prove it against the real tree.

    The static checks above would pass with `.gates` never created. This one runs
    against what is on disk, so the moment someone creates a `.gates` directory
    again the behaviour is verified rather than assumed.
    """
    for name in (".gates", ".mypy_cache"):
        if not (ROOT / name).is_dir():
            continue
        bare = name
        assert bare in _entries(GITIGNORE), f"{name} exists but is not gitignored"
        assert bare in _entries(DOCKERIGNORE), f"{name} exists but is not dockerignored"


# ------------------------------------------------- tracked, but not in the image
#: Excluded from the *image* and tracked in *git*. These are a third category,
#: distinct from the entries above: those are local scratch that must never be
#: committed, these are project source that must never be shipped but must never
#: be deleted. Conflating the two is how "excluded from the image" turns into
#: "gone from the project".
TRACKED_BUT_NOT_IN_IMAGE = [
    "tests",
    "docs",
]

#: CI runs these against `tests/`, so if the directory stops being the project's
#: subject, that is a different decision and not one this file may make silently.
CI_STILL_COVERS_TESTS = [
    "ruff check src tests",
    "compileall -q src tests",
]


@pytest.mark.parametrize("name", TRACKED_BUT_NOT_IN_IMAGE)
def test_tracked_material_is_kept_out_of_the_image(name):
    assert name in _entries(DOCKERIGNORE), f"{name} would be baked into the image"


@pytest.mark.parametrize("name", TRACKED_BUT_NOT_IN_IMAGE)
def test_it_is_not_simultaneously_untracked(name):
    """The failure this guards: someone reads the .dockerignore entry and
    'tidies up' .gitignore too, which would delete 1517 tests and the API
    contract from the repository rather than from the image."""
    assert name not in _entries(GITIGNORE), (
        f"{name} is gitignored; it is tracked source that belongs in the repo and "
        f"only needs to stay out of the image"
    )


@pytest.mark.parametrize("name", TRACKED_BUT_NOT_IN_IMAGE)
def test_the_exclusion_is_not_vacuous(name):
    """An exclusion for a directory that does not exist excludes nothing.

    Both of these exist and are non-trivial in size, so a rule naming them is a
    real saving rather than a line that looks like tidiness.
    """
    directory = ROOT / name
    assert directory.is_dir(), f".dockerignore excludes {name} but it is not here"
    assert any(directory.rglob("*")), f"{name} is empty"


def test_ci_still_lints_and_tests_the_directory_the_image_excludes():
    """"Not in the image" is a packaging decision, not a retirement.

    If CI stopped running the tests, the exclusion would be the only thing left
    saying they are wanted, and the next reader would have no way to tell a
    deliberate omission from a stale one.
    """
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8", errors="replace")
    for fragment in CI_STILL_COVERS_TESTS:
        assert fragment in workflow, (
            f"CI no longer contains {fragment!r}; tests/ is excluded from the image, "
            "so CI is the only remaining statement that the suite is wanted"
        )
    assert "pytest" in workflow, "CI no longer runs pytest"


def test_the_examples_the_ui_serves_are_not_the_documentation_images():
    """The Gradio Examples tab is the one image consumer in the running system.

    `gr.Examples` reads `./examples/...` from the repository root, not
    `docs/examples/`. The two are easy to confuse because the names match, and
    confusing them here would ship an image whose Auto tab shows nothing.

    Half of these are built by an f-string loop (`f"./examples/old/{name}.png"`),
    so the paths are checked by directory rather than by file: a template has no
    file to open, but the directory it is built from has to hold images.
    """
    source = (ROOT / "src" / "fiximg" / "ui" / "gradio_app.py").read_text(
        encoding="utf-8", errors="replace")
    # Match every image path the UI offers, not just ones under `examples/`.
    # A pattern anchored on `examples/` cannot see a path that has been moved
    # *out* of examples/ -- which is exactly the move this test exists to catch,
    # so anchoring here would make the check blind to its own subject.
    referenced = re.findall(r'"\./([^"]+\.(?:png|jpg|jpeg|webp))"', source)
    assert referenced, "no example image paths found -- has the UI moved?"

    directories = set()
    for path in referenced:
        assert not path.startswith("docs/"), (
            f"the UI now reads {path} from docs/, which .dockerignore excludes"
        )
        directories.add(path.split("/")[0] + "/" + path.split("/")[1])
        if "{" not in path:
            assert (ROOT / path).is_file(), f"the UI serves {path} but it is not here"

    assert len(directories) == 3, (
        f"the UI offers images from {sorted(directories)}; expected the three "
        "example directories, so recheck this test if the tabs were reorganised"
    )
    for directory in sorted(directories):
        images = [p for p in (ROOT / directory).iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
        assert images, f"{directory} holds no images; the Examples tabs render empty"

    # The literal tab definitions name these three directories directly.
    for name in ("old", "old_w_scratch", "color"):
        assert f"./examples/{name}/" in source, (
            f"no tab references examples/{name}/ any more -- recheck the "
            "directory list above, they may have moved"
        )
