"""The CI configuration itself is under test (plan §3.11).

A workflow step that cannot fail is not a gate, and the only durable way to keep
one from coming back is to check the workflows the way the code is checked. Two of
these existed in this repository for releases: the type-check job ran with
`continue-on-error` (44 findings, read by nobody) and the dependency audit ended
with `|| true` next to a placeholder `--ignore-vuln GHSA-0000-0000-0000`.

The rules encode what the project already satisfies. A gate the configuration
cannot meet would just earn a permanent waiver, which is the failure mode this file
exists to prevent.
"""
import os
import re

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKFLOWS = os.path.join(PROJECT_ROOT, ".github", "workflows")
AUDIT_IGNORE = os.path.join(PROJECT_ROOT, "config", "audit_ignore.txt")

#: `a11y` and `no-qa` are real job names that would otherwise read as waivers.
_EXEMPT_TOKENS = ("a11y", "no-qa", "s01e01")
#: A step whose last command cannot fail reports success no matter what happened.
SWALLOW = re.compile(r"\|\|\s*(true|:)\b|&&\s*true\b|\bexit\s+0\b")


def _steps():
    import yaml

    for name in sorted(os.listdir(WORKFLOWS)):
        if not name.endswith(".yml"):
            continue
        with open(os.path.join(WORKFLOWS, name), encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        for job_id, job in (document.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                yield name, job_id, step


@pytest.mark.parametrize("name", sorted(f for f in os.listdir(WORKFLOWS) if f.endswith(".yml")))
def test_no_step_declares_continue_on_error(name):
    offenders = [
        f"{job_id}/{step.get('name') or step.get('id') or '?'}"
        for _file, job_id, step in _steps()
        if step.get("continue-on-error") is True
    ]
    assert not offenders, (
        f"{name}: a step with continue-on-error is a report, not a gate; fix the "
        f"findings or drop the job: {offenders}"
    )


def test_no_step_ends_with_a_swallowed_failure():
    """Only the *last* command of a script counts.

    `cmd || true` mid-script is a legitimate probe (the release job reads the
    previous tag that way and does not care when there is none). As the final
    line, it decides the step's exit status — and then nothing downstream can
    read the result.
    """
    offenders = []
    for file_name, job_id, step in _steps():
        script = (step.get("run") or "").strip()
        if not script:
            continue
        last = [ln for ln in script.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if not last:
            continue
        tail = last[-1]
        if SWALLOW.search(tail) and not any(token in tail for token in _EXEMPT_TOKENS):
            offenders.append(f"{file_name}:{job_id}: {tail.strip()}")
    assert not offenders, "steps whose final command cannot fail: " + "; ".join(offenders)


def test_the_local_lint_target_checks_what_ci_checks():
    """`make lint` green while the workflow is red is not a gate either.

    The local target used to lint `src tests` only, so an entry-point script or
    `scripts/` could break CI without any local command noticing. Both lists are
    compared as token sets, so reordering either file does not fail this test.
    """
    makefile = os.path.join(PROJECT_ROOT, "Makefile")
    with open(makefile, encoding="utf-8") as handle:
        make_text = handle.read()
    lint_line = next(
        line for line in make_text.splitlines() if line.startswith("LINT_PATHS :=")
    )
    local = set(lint_line.split(":=", 1)[1].split())

    ci_paths: set[str] = set()
    for _job, step in ((j, s) for _f, j, s in _steps() if "ruff check" in (s.get("run") or "")):
        for line in step["run"].splitlines():
            if "ruff check" in line:
                ci_paths = set(line.split("ruff check", 1)[1].split())
    assert ci_paths, "CI no longer lints anything"
    assert local == ci_paths, f"local lints {sorted(local)}, CI lints {sorted(ci_paths)}"


def test_the_target_interpreters_grammar_is_checked_by_a_gate():
    """3.12+ syntax passes a newer interpreter's tools and dies at import in 3.11.

    A nested-quote f-string (PEP 701, Python 3.12) is a *syntax* difference, so it
    sails through ruff configured for py311 and through mypy on a 3.13/3.14 box, and
    only shows up as `SyntaxError` while pytest imports the file — which here
    happened in the deployment environment, after the dev environment had reported
    the whole suite green. `compileall` run by the target interpreter is the check
    that closes the class, over the same path list as `make lint` so nothing can be
    quietly excluded from it.
    """
    with open(os.path.join(PROJECT_ROOT, "Makefile"), encoding="utf-8") as handle:
        make_text = handle.read()
    lint_line = next(line for line in make_text.splitlines() if line.startswith("LINT_PATHS :="))
    expected = set(lint_line.split(":=", 1)[1].split())
    assert expected, "LINT_PATHS is empty"

    compile_steps = [
        (job_id, step["run"])
        for _file, job_id, step in _steps()
        if "compileall" in (step.get("run") or "")
    ]
    assert compile_steps, "no CI job parses the tree with the target grammar"

    for job_id, script in compile_steps:
        line = next(ln for ln in script.splitlines() if "compileall" in ln)
        paths = {token for token in line.split("compileall", 1)[1].split() if not token.startswith("-")}
        assert paths == expected, f"{job_id} compiles {sorted(paths)}, expected {sorted(expected)}"

        # A grammar check only means something if the interpreter is the supported
        # one; a 3.14 job would accept exactly the code this test exists to catch.
        versions = [
            str((step.get("with") or {}).get("python-version", ""))
            for _file, other, step in _steps()
            if other == job_id and "setup-python" in (step.get("uses") or "")
        ]
        assert "3.11" in versions, f"job {job_id} parses with {versions}, not with the target 3.11"


def _jobs():
    import yaml

    for name in sorted(os.listdir(WORKFLOWS)):
        if not name.endswith(".yml"):
            continue
        with open(os.path.join(WORKFLOWS, name), encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        for job_id, job in (document.get("jobs") or {}).items():
            yield name, job_id, job


def test_the_postgres_tier_runs_against_a_real_server():
    """A tier that can only run on one laptop is not a gate (plan §4.3 Step 2).

    The PostgreSQL path was developed against a fake driver that answered exactly
    what the code asked; pointing it at a real server produced four defects the fake
    could not see (SQLite-only DDL, a reserved column name, positional row access, a
    cross-type COALESCE). So the job must have a real server as a service, must test
    the rollback as well as the upgrade — a one-way "upgrade works" check says
    nothing about `downgrade` — and must not swallow a failure.
    """
    candidates = [
        (job_id, job)
        for _file, job_id, job in _jobs()
        if any("test_postgres_server.py" in (step.get("run") or "")
               for step in job.get("steps") or [])
    ]
    assert candidates, "no CI job runs the PostgreSQL server-backed tier"

    for job_id, job in candidates:
        services = job.get("services") or {}
        assert any("postgres" in name for name in services), (
            f"job {job_id} runs the PostgreSQL tier without a PostgreSQL service "
            f"(services: {sorted(services) or 'none'})"
        )
        script = "\n".join((step.get("run") or "") for step in job.get("steps") or [])
        assert not SWALLOW.search(script), f"job {job_id} swallows the PostgreSQL tier"
        assert "downgrade base" in script, (
            f"job {job_id} only checks the upgrade direction on PostgreSQL"
        )
        assert "runner check" in script, (
            f"job {job_id} never asks whether the migrated database matches the code"
        )


def test_audit_exemptions_are_well_formed_and_reviewed():
    """A typo in an exemption is a silently re-opened hole, not a failed build."""
    if not os.path.exists(AUDIT_IGNORE):
        pytest.skip("no exemption file yet")
    pattern = re.compile(r"^(GHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}|CVE-\d{4}-\d{4,})$")
    with open(AUDIT_IGNORE, encoding="utf-8") as handle:
        ids = [ln.strip() for ln in handle if ln.strip() and not ln.startswith("#")]
    for entry in ids:
        assert pattern.match(entry), f"not a GHSA/CVE id: {entry!r}"
        # `pip-audit` rejects an unknown id outright, so a stale entry fails the
        # audit step rather than rotting into a permanent silent waiver.
    assert len(ids) == len(set(ids)), f"duplicate exemption ids: {ids}"


def test_benchmark_job_runs_the_gate_not_just_the_report():
    """`benchmark.latency` exits 1 on overhead regression; CI must not hide it."""
    scripts = [
        (step.get("run") or "")
        for _file, _job, step in _steps()
        if "benchmark.latency" in (step.get("run") or "")
    ]
    assert scripts, "CI no longer runs the latency benchmark at all"
    for script in scripts:
        assert "benchmark.latency" in script
        assert not SWALLOW.search(script), "the latency gate is being swallowed"


#: Parameters that turn an action step into a report no matter what it found.
SOFT_ACTION_PARAMS = ("exit-code", "exit_code", "fail-build", "fail_on_error")


def test_no_action_step_is_parametrised_to_pass_anyway():
    """`continue-on-error` is not the only way to defang a check.

    The image scan used to run with `exit-code: "0"`, which trivy documents as
    "always exit successfully" — a vulnerability report that can never be read as
    a failure. The regex above only inspects `run:` scripts, so that form was
    invisible to the workflow's own self-check; this reads action parameters.
    """
    offenders = []
    for _file, job_id, step in _steps():
        with_block = step.get("with") or {}
        if not isinstance(with_block, dict) or not step.get("uses"):
            continue
        for key, value in with_block.items():
            if str(key).lower() in SOFT_ACTION_PARAMS and str(value).strip() in ("0", "false"):
                offenders.append(
                    f"{job_id}/{step.get('name') or '?'}: {key}={value}"
                )
    assert not offenders, (
        "an action step is configured to succeed regardless of its findings: "
        f"{offenders}. Record the findings in the project's ignore file instead."
    )


#: buildx spells its provenance levels `mode=min` / `mode=max`. A dash is not a
#: value it accepts, and it rejects the build before any Dockerfile is read.
PROVENANCE_OK = re.compile(r"^(true|false|mode=(min|max))$")


def test_provenance_input_uses_the_syntax_buildx_accepts():
    """`provenance: mode-max` failed both release images without building either.

    The message it produced -- "buildx failed with: invalid value mode-max" -- was
    read as a complaint about the GHA cache exporter, which was removed; the next
    tagged run failed identically with no exporter left to blame. The value was the
    fault the whole time, and unlike the exporter it is checkable from here.
    """
    offenders = []
    for file_name, job_id, step in _steps():
        with_block = step.get("with") or {}
        if not isinstance(with_block, dict):
            continue
        value = with_block.get("provenance")
        if value is None:
            continue
        if not PROVENANCE_OK.match(str(value).strip()):
            offenders.append(f"{file_name}:{job_id}/{step.get('name') or '?'}: provenance={value}")
    assert not offenders, (
        "buildx takes `mode=min`/`mode=max` for provenance, never a dash: "
        + "; ".join(offenders)
    )


def test_container_paths_built_from_the_owner_are_lowercased():
    """GHCR paths are lowercase; `github.repository_owner` is not.

    Both v3.0.0 image jobs failed with

        invalid tag "ghcr.io/Kevin030226/fiximg-api:3.0.0":
        repository name must be lowercase

    before buildx read a Dockerfile. A workflow that builds a registry path out of
    the owner has to fold the case in the shell -- Actions expressions have no
    `lower()` -- so this checks the fold is still written down beside the
    reference, not that the resulting build works.
    """
    offenders = []
    for name in sorted(os.listdir(WORKFLOWS)):
        if not name.endswith(".yml"):
            continue
        with open(os.path.join(WORKFLOWS, name), encoding="utf-8") as handle:
            text = handle.read()
        if "repository_owner" in text and "[:lower:]" not in text:
            offenders.append(name)
    assert not offenders, (
        "a workflow builds a container path from github.repository_owner without "
        "lowercasing it: " + ", ".join(offenders)
    )


def test_referenced_ignore_files_exist():
    """An ignore list that is not in the repository means an unbounded exemption."""
    import os

    missing = []
    for _file, job_id, step in _steps():
        with_block = step.get("with") or {}
        if not isinstance(with_block, dict):
            continue
        for key, value in with_block.items():
            # `trivyignores` is the name the scan action actually reads. The
            # spelling CI used before that (`ignore-file`) stays in the list: a
            # workflow that goes back to it is still referencing a file.
            if any(
                token in str(key).lower()
                for token in ("ignore-file", "ignorelist", "trivyignores")
            ):
                if not os.path.exists(os.path.join(PROJECT_ROOT, str(value))):
                    missing.append(f"{job_id}/{step.get('name') or '?'}: {value}")
    assert not missing, f"CI references ignore files that are not committed: {missing}"


# --------------------------------------------------------------- image matrix
#: Every container image CI is required to build, smoke-test and scan.
REQUIRED_IMAGES = {"api", "worker", "root"}


def _docker_job():
    """The `docker` job document, or None when the workflow moved it elsewhere."""
    import yaml

    with open(os.path.join(WORKFLOWS, "ci.yml"), encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    return (document.get("jobs") or {}).get("docker")


def test_the_docker_job_is_a_matrix_over_every_image():
    """One runner cannot hold three images, so each gets its own shard.

    A standard `ubuntu-latest` runner has 14 GB of SSD and one worker/root image
    needs ~19 GB transient to build (11.4 GB rootfs + 7.4 GB cuda base). Building
    all three in one job therefore cannot succeed; it fails at the second build
    after paying for the first. This asserts the matrix that replaced it still
    names every image exactly once, and that the job keeps its id so nothing
    downstream has to know the topology changed.
    """
    job = _docker_job()
    assert job is not None, "the docker job is gone; image verification disappeared with it"
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    include = matrix.get("include") or []
    assert include, "the docker job is no longer a matrix; three images do not fit in one runner"

    images = [entry.get("image") for entry in include]
    assert set(images) == REQUIRED_IMAGES, (
        f"the matrix covers {sorted(images)}, expected {sorted(REQUIRED_IMAGES)}"
    )
    assert len(images) == len(REQUIRED_IMAGES), f"duplicate matrix entries: {images}"
    for entry in include:
        for key in ("image", "dockerfile", "trivy_timeout"):
            assert entry.get(key), f"matrix entry {entry} is missing {key!r}"


def test_every_image_is_guarded_by_its_own_smoke_step():
    """A matrix entry without a guard is an image that is built and never run."""
    job = _docker_job()
    assert job is not None
    steps = job.get("steps") or []
    for image in sorted(REQUIRED_IMAGES):
        guards = [
            step for step in steps
            if f"matrix.image == '{image}'" in str(step.get("if", ""))
        ]
        assert guards, f"no step is guarded for image {image!r}; it is built but never exercised"


def test_the_scan_is_parameterised_over_the_matrix():
    """A hard-coded `image-ref` would scan one image three times."""
    job = _docker_job()
    assert job is not None
    scans = [
        step for step in (job.get("steps") or [])
        if "trivy" in str(step.get("uses", "")).lower()
    ]
    assert scans, "the docker job no longer scans anything"
    for scan in scans:
        ref = str((scan.get("with") or {}).get("image-ref", ""))
        assert "matrix.image" in ref, (
            f"the scan targets {ref!r}; a fixed reference means the other images are never scanned"
        )


def test_compose_validation_runs_once_outside_the_matrix():
    """Topology is not per-image; validating it three times hides nothing."""
    import yaml

    with open(os.path.join(WORKFLOWS, "ci.yml"), encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    jobs = document.get("jobs") or {}
    compose_jobs = [
        (job_id, job) for job_id, job in jobs.items()
        if any("compose" in (step.get("run") or "") and "config" in (step.get("run") or "")
               for step in job.get("steps") or [])
    ]
    assert compose_jobs, "no job validates the compose files"
    for job_id, job in compose_jobs:
        assert "strategy" not in job, (
            f"job {job_id} runs compose validation inside a matrix; it is not a per-image check"
        )


# ------------------------------------------------------- root smoke lifecycle

#: The step that boots the root image's ASGI app and asserts how it ends.
SMOKE_STEP_NAME = "Boot the ASGI app and probe liveness"
#: The message the app prints, and the step asserts, when weights are absent.
EXPECTED_EXIT_MESSAGE = "Weight verification failed"


def _smoke_step() -> dict:
    """The root-image boot step, found by name so line moves do not break this."""
    job = _docker_job()
    assert job is not None, "the docker job is gone; image verification disappeared with it"
    for step in job.get("steps") or []:
        if SMOKE_STEP_NAME in str(step.get("name") or ""):
            return step
    raise AssertionError(
        f"the docker job no longer has a {SMOKE_STEP_NAME!r} step; "
        "the root image would be built and never run"
    )


def _smoke_script() -> str:
    script = _smoke_step().get("run")
    assert script, f"the {SMOKE_STEP_NAME!r} step has no script"
    return str(script)


def _smoke_commands() -> list[str]:
    """The script's command lines, comments and blank lines dropped."""
    return [
        line.strip()
        for line in _smoke_script().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_the_root_smoke_step_is_guarded_to_the_root_image():
    """A boot probe that also runs on api/worker would fail on their missing app."""
    assert "root" in str(_smoke_step().get("if", "")), (
        "the root boot step is not guarded to the root image"
    )


def test_a_container_whose_state_is_read_after_exit_is_never_run_with_rm():
    """`--rm` deletes the container the moment it stops.

    The step's whole point is to read the *final* state and the logs of a
    container that exits by design, so `--rm` on that run would delete the
    evidence before it is read -- which is exactly how this step failed.
    """
    commands = _smoke_commands()
    offenders = [
        command for command in commands
        if re.search(r"\bdocker\s+run\b", command) and "--rm" in command
    ]
    assert not offenders, (
        "the root smoke container is started with --rm, so it is deleted on exit "
        f"and cannot be inspected afterwards: {offenders}"
    )


def test_the_smoke_container_is_started_detached_with_a_name():
    """Detached and named: attached would block, unnamed cannot be inspected."""
    runs = [c for c in _smoke_commands() if re.search(r"\bdocker\s+run\b", c)]
    assert runs, "the root smoke step no longer starts a container"
    for command in runs:
        assert re.search(r"\s-d\b|\s--detach\b", command), (
            f"the smoke container is not detached: {command}"
        )
        assert re.search(r"\s--name\s+\S+", command), (
            f"the smoke container has no fixed name to inspect: {command}"
        )


def test_the_smoke_step_cleans_up_unconditionally():
    """Cleanup must not depend on the assertions above it succeeding."""
    script = _smoke_script()
    traps = [line.strip() for line in script.splitlines() if re.search(r"\btrap\b", line)]
    assert traps, (
        "the root smoke step has no trap, so a failing assertion leaves the "
        "container behind (and the port bound) for the next step"
    )
    assert any("EXIT" in trap for trap in traps), (
        f"the smoke trap does not run on EXIT, so it is skipped on failure: {traps}"
    )
    assert any("rm" in trap for trap in traps), (
        f"the smoke trap does not remove the container: {traps}"
    )


def test_the_smoke_step_reads_the_state_once_and_branches_on_it():
    """One read, then both outcomes handled: a second read can disagree."""
    commands = _smoke_commands()
    inspects = [
        c for c in commands
        if re.search(r"\bdocker\s+inspect\b", c) and "State.Status" in c
    ]
    assert inspects, "the smoke step never reads the container's final state"
    assert len(inspects) == 1, (
        "the container state is read more than once; the branches can then "
        f"disagree about what happened: {inspects}"
    )
    script = _smoke_script()
    assert "running" in script, "the smoke step has no path for a still-running container"
    assert "exited" in script, "the smoke step has no path for a container that exited"


def test_a_running_smoke_container_is_probed_for_liveness():
    """If the app stayed up, that has to be proven, not assumed."""
    script = _smoke_script()
    assert "health/live" in script, (
        "the running branch does not probe the liveness endpoint"
    )
    commands = _smoke_commands()
    assert any("curl" in c for c in commands), "the liveness probe does not use curl"


def test_an_exited_smoke_container_must_show_the_expected_message():
    """Exiting is the designed outcome, so the *reason* is what gets asserted."""
    script = _smoke_script()
    assert EXPECTED_EXIT_MESSAGE in script, (
        f"the smoke step does not assert {EXPECTED_EXIT_MESSAGE!r}; an app that "
        "exits for any other reason would pass"
    )
    # The assertion must be able to fail: a grep whose result is discarded, or an
    # exit code swallowed afterwards, is not an assertion.
    assertion = [
        line for line in script.splitlines()
        if EXPECTED_EXIT_MESSAGE in line and "grep" in line
    ]
    assert assertion, "the expected message is mentioned but never matched against the logs"
    for line in assertion:
        assert "if " in line or "&&" in line, (
            f"the log assertion's result is not used to decide the step's outcome: {line.strip()}"
        )


def test_an_unexpected_container_state_fails_the_step():
    """Neither running nor exited means the probe proved nothing."""
    script = _smoke_script()
    branches = re.findall(r"\belif\b|\belse\b", script)
    assert len(branches) >= 2, (
        "the smoke step does not have a final else branch, so an unexpected "
        "container state would fall through as success"
    )
    # The final branch must end the step with a failure.
    tail = script.split("else", 1)[-1]
    assert re.search(r"\bexit\s+[1-9]", tail), (
        "the unexpected-state branch does not exit non-zero"
    )


def test_the_smoke_assertions_cannot_be_waived_away():
    """The gate is only a gate while it can fail."""
    step = _smoke_step()
    assert not step.get("continue-on-error"), (
        "the root smoke step is allowed to fail; it no longer gates anything"
    )
    commands = _smoke_commands()
    offenders = [
        command for command in commands
        if re.search(r"\|\|\s*(true|:)\b", command)
        and not re.search(r"\b(rm|trap)\b", command)
    ]
    assert not offenders, (
        "a smoke assertion swallows its own failure with `|| true`: " f"{offenders}"
    )


# ------------------------------------------------------------- pillow pinning

#: The advisory block that forces this pin; keep the four files in step.
PILLOW_PIN = "13.0.0"


def _declared_requirements() -> dict[str, list[str]]:
    """`pyproject.toml`'s base dependencies and every extra, as raw requirement strings.

    Read with `tomllib` rather than by scanning lines: a gate that greps text is
    checking the file's spelling, not what an installer would resolve.
    """
    import tomllib

    with open(os.path.join(PROJECT_ROOT, "pyproject.toml"), "rb") as handle:
        document = tomllib.load(handle)
    project = document["project"]
    declared = {"(base)": list(project["dependencies"])}
    for extra, requirements in (project.get("optional-dependencies") or {}).items():
        declared[extra] = [str(requirement) for requirement in requirements]
    return declared


def _requirement_for(requirements: list[str], name: str) -> str | None:
    """The requirement string for `name`, matched on the distribution name."""
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*([<>=!~].*)?$", re.IGNORECASE)
    for requirement in requirements:
        if pattern.match(requirement.strip()):
            return requirement.strip()
    return None


def test_pyproject_pins_pillow_exactly_rather_than_as_a_floor():
    """A floor cannot clear the advisories on the worker image, and this is why.

    The worker image installs the torch stack first. `torchvision==0.22.1` declares
    `pillow!=8.3.*,>=5.3.0` with no upper bound, so pip satisfies it with whatever
    Pillow the index serves -- 12.3.0 when this was measured. The project is
    installed *afterwards*, and `Pillow>=10.0` then finds 12.3.0 already satisfied
    and never upgrades it: the image builds, ships the vulnerable Pillow, and only
    the Trivy scan notices (P18 run 39777646356, job `image (worker)`).

    An exact pin is what makes the second install converge to 13.0.0 regardless of
    what the first one left behind. Measured both ways in
    `reports/p19_worker_pillow_remediation.md`: exact pin -> 13.0.0, floor ->
    "Requirement already satisfied" at 12.3.0.
    """
    declared = _declared_requirements()
    requirement = _requirement_for(declared["(base)"], "pillow")
    assert requirement is not None, (
        "pyproject.toml no longer declares Pillow at all; the pin moved somewhere "
        "this gate does not see"
    )
    assert requirement == f"Pillow=={PILLOW_PIN}", (
        f"pyproject.toml declares {requirement!r}. It must be the exact pin "
        f"'Pillow=={PILLOW_PIN}', not a floor: a floor is already satisfied by the "
        "12.3.0 the torch stack leaves behind, so the worker image would keep "
        "shipping the advisories while every other path looks clean."
    )


def test_no_extra_relaxes_the_pillow_pin():
    """An extra that re-declares Pillow as a floor would reopen the same hole."""
    for extra, requirements in _declared_requirements().items():
        requirement = _requirement_for(requirements, "pillow")
        if requirement is None:
            continue
        assert requirement == f"Pillow=={PILLOW_PIN}", (
            f"extra {extra!r} declares {requirement!r}; Pillow may only be declared "
            f"as the exact pin 'Pillow=={PILLOW_PIN}'"
        )
#: Every file that hard-pins Pillow for an installable surface.
PILLOW_PIN_FILES = (
    "requirements.txt",
    "requirements.lock",
    os.path.join("requirements", "runtime.txt"),
    os.path.join("requirements", "gpu.txt"),
)


@pytest.mark.parametrize("relative", PILLOW_PIN_FILES)
def test_every_pillow_pin_agrees_on_the_fixed_version(relative):
    """Four files pin Pillow; one of them drifting re-opens the advisories.

    Pillow 12.3.0 carries a CRITICAL and six HIGH advisories whose fix version
    is 13.0.0, and `pip-audit --strict` plus the image scans both block on them.
    A file left behind would keep the failure alive in exactly one install path.
    """
    path = os.path.join(PROJECT_ROOT, relative)
    assert os.path.exists(path), f"{relative} is gone; the pin moved somewhere this gate does not see"
    with open(path, encoding="utf-8") as handle:
        body = handle.read()
    pins = re.findall(r"^\s*[Pp]illow\s*==\s*(\S+)\s*$", body, re.MULTILINE)
    assert pins, f"{relative} no longer pins Pillow; the audit would float"
    for pin in pins:
        assert pin == PILLOW_PIN, (
            f"{relative} pins Pillow {pin}, expected {PILLOW_PIN}: the fixed version "
            "is what clears the advisories"
        )


def test_the_readme_version_tables_match_the_pinned_pillow():
    """The user-facing table is documentation of the pin, not decoration."""
    for name in ("README.md", "README_CN.md"):
        path = os.path.join(PROJECT_ROOT, name)
        assert os.path.exists(path), name
        with open(path, encoding="utf-8") as handle:
            body = handle.read()
        mentions = re.findall(r"Pillow\s+([0-9]+(?:\.[0-9]+)+)", body)
        assert mentions, f"{name} no longer states a Pillow version"
        for mention in mentions:
            assert mention == PILLOW_PIN.rsplit(".", 1)[0], (
                f"{name} advertises Pillow {mention}, expected "
                f"{PILLOW_PIN.rsplit('.', 1)[0]} to match the pin {PILLOW_PIN}"
            )


# ------------------------------------------------------ release workflow (P39)
#: The workflow's write jobs. P41 split the old `publish-images` matrix into
#: `publish-version-tags` and `promote-latest`, because a matrix shard is an
#: independent job and a later step in the same matrix is not a global barrier.
PUBLISH_JOBS = ("stage-images", "publish-version-tags", "promote-latest", "release")


def _release():
    import yaml

    with open(os.path.join(WORKFLOWS, "release.yml"), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _release_steps(job_id: str) -> list[dict]:
    return (_release().get("jobs") or {}).get(job_id, {}).get("steps") or []


def _release_script(job_id: str) -> str:
    return "\n".join(str(step.get("run") or "") for step in _release_steps(job_id))


def test_release_declares_read_only_permissions_at_the_top():
    """P39: the workflow default must not be able to write anything."""
    permissions = _release().get("permissions") or {}
    assert permissions == {"contents": "read"}, (
        f"release.yml top-level permissions are {permissions!r}; they must be exactly "
        "contents: read so only the publishing jobs can write"
    )


def test_only_publishing_jobs_hold_write_permissions():
    """Write scopes belong to the two image jobs and the Release job, nowhere else."""
    jobs = _release().get("jobs") or {}
    for job_id, job in jobs.items():
        granted = {k: v for k, v in (job.get("permissions") or {}).items() if v == "write"}
        if job_id in PUBLISH_JOBS:
            assert granted, f"{job_id} publishes but declares no write permission"
            assert set(granted) <= {"contents", "packages"}, (
                f"{job_id} requests an unexpected write scope: {granted}"
            )
        else:
            assert not granted, f"non-publishing job {job_id} holds write access: {granted}"


def test_release_manual_dispatch_is_always_a_dry_run():
    """No input may turn a manual run into a publish."""
    document = _release()
    trigger = document.get(True) or document.get("on") or {}
    assert "workflow_dispatch" in trigger, "workflow_dispatch was removed"
    # A bare `workflow_dispatch:` parses as None; any mapping would mean inputs.
    dispatch = trigger.get("workflow_dispatch") or {}
    assert not dispatch.get("inputs"), (
        "workflow_dispatch declares inputs; P39 removed the publish switch so a manual "
        f"run can never publish, found: {sorted(dispatch.get('inputs') or {})}"
    )

    decide = next(
        (step for step in _release_steps("preflight") if step.get("id") == "decide"), None
    )
    assert decide is not None, "the publish decision step is gone"
    script = str(decide.get("run") or "")
    assert 'GITHUB_EVENT_NAME" = "push"' in script, (
        "the publish decision no longer keys on the event being a tag push"
    )
    assert "publish=true" in script and "publish=false" in script


def test_release_preflight_checks_tag_version_and_release_material():
    """The checks that need no tag object stay inline and stay early."""
    script = _release_script("preflight")
    required = {
        "tag matches pyproject": "does not match pyproject version",
        "release notes present": "release notes",
    }
    for what, needle in required.items():
        assert needle in script, f"the preflight no longer checks {what}: {needle!r}"
    assert "git fetch --no-tags origin main" in script, (
        "the preflight must compare against origin/main fetched at run time"
    )


def test_release_preflight_proves_the_tag_through_the_guard():
    """The annotated-tag proof must not be read from the checkout.

    actions/checkout@v5 rewrites the checkout-owned refs/tags/<tag> to the peeled
    commit before any step runs, so a workflow that asks that ref for its object
    type rejects every valid annotated tag. That is exactly how the v3.1.1
    release failed, and the old assertion here -- "the error message exists" --
    was green while the behaviour was impossible. The guard owns the proof.
    """
    script = _release_script("preflight")
    assert "source scripts/ci/release_tag_guard.sh" in script, (
        "the annotated-tag proof must come from the guard, which fetches the tag "
        "into refs/release-tags/<tag> instead of trusting the checkout"
    )
    assert 'release_tag_guard "$GITHUB_REF_NAME" "$main_sha"' in script, (
        "the guard must receive the tag from the event and the run-time origin/main"
    )
    assert "release_tag_write_outputs" in script, (
        "the validated values must reach GITHUB_OUTPUT for the release job"
    )
    for forbidden in ("cat-file", "git tag -l", "GITHUB_REF_NAME}^{commit}"):
        assert forbidden not in script, (
            f"preflight still reads the checkout-owned tag ref via {forbidden!r}; "
            "actions/checkout has already rewritten it to the peeled commit"
        )


def test_release_preflight_exports_the_validated_title_and_sha():
    outputs = (_release().get("jobs") or {})["preflight"].get("outputs") or {}
    assert outputs.get("tag_sha") == "${{ steps.verify.outputs.tag_sha }}"
    assert outputs.get("release_title") == "${{ steps.verify.outputs.release_title }}"


def test_release_title_comes_from_the_guard_through_the_environment():
    """The title is data: it must arrive through env:, never as script text."""
    draft = {step.get("id"): step for step in _release_steps("release")}["draft"]
    env = draft.get("env") or {}
    assert env.get("RELEASE_TITLE") == "${{ needs.preflight.outputs.release_title }}", (
        "the release title must be passed through env: so it cannot be evaluated"
    )
    script = str(draft.get("run") or "")
    assert "title=$RELEASE_TITLE" in script, (
        "the release job no longer consumes the validated title"
    )
    assert "refusing to guess one" in script, (
        "a blank title must fail closed instead of silently titling the release "
        "after the tag name"
    )
    for forbidden in ("cat-file", "git tag -l", "title=$GITHUB_REF_NAME"):
        assert forbidden not in script, (
            f"the release job still derives its title locally via {forbidden!r}"
        )


def test_release_writes_are_all_gated_by_the_publish_decision():
    """Every job that touches a registry or a Release must be behind publish == true."""
    jobs = _release().get("jobs") or {}
    for job_id in PUBLISH_JOBS:
        condition = str((jobs.get(job_id) or {}).get("if") or "")
        assert "publish == 'true'" in condition, (
            f"{job_id} is not guarded by the publish decision: {condition!r}"
        )


def test_release_test_and_artifact_gates_run_before_any_registry_write():
    """The package must build and the suite pass before an image is even staged."""
    jobs = _release().get("jobs") or {}
    stage_needs = set((jobs.get("stage-images") or {}).get("needs") or [])
    assert {"tests", "artifacts", "image-preflight"} <= stage_needs, (
        f"stage-images needs {sorted(stage_needs)}; it must depend on the test matrix, "
        "the package build and both image preflights"
    )


def test_release_requires_both_image_preflights_before_staging():
    """One broken image must not be staged because the other succeeded."""
    jobs = _release().get("jobs") or {}
    preflight = jobs.get("image-preflight") or {}
    matrix = (preflight.get("strategy") or {}).get("matrix") or {}
    assert set(matrix.get("image") or []) == {"api", "worker"}, (
        f"image-preflight covers {matrix.get('image')}; both images are required"
    )
    for step in preflight.get("steps") or []:
        if step.get("id") == "build":
            with_block = step.get("with") or {}
            assert with_block.get("push") is False, (
                "the preflight image build must not push"
            )
            # build-push-action keeps the result in the buildx cache unless it is
            # loaded; without this the following `docker run` verifies nothing.
            assert with_block.get("load") is True, (
                "the preflight build must set load: true, otherwise the image it "
                "is supposed to run and verify never reaches the local docker store"
            )


def test_release_stages_by_digest_and_promotes_version_tags_before_latest():
    """Two-stage images: digest-only staging, then version tags, then latest."""
    stage_script = _release_script("stage-images")
    stage_build = next(
        (step for step in _release_steps("stage-images") if step.get("id") == "build"), None
    )
    assert stage_build is not None, "the staging job has no build step"
    outputs = str((stage_build.get("with") or {}).get("outputs") or "")
    assert "push-by-digest=true" in outputs, (
        "staging must push by digest; a mutable tag here would leave a version tag "
        f"pointing at a half-validated build if a later job fails (outputs={outputs!r})"
    )
    # The by-digest `outputs` is the single publishing declaration; a top-level
    # `push: true` beside it is redundant and can drift on a later edit.
    assert "push" not in (stage_build.get("with") or {}), (
        "stage-images declares both push and a by-digest outputs; outputs alone is "
        "the publishing declaration"
    )
    assert "imagetools create --tag" not in stage_script, (
        "staging must not create tags; promotion belongs to the publishing jobs"
    )

    # --- P41: the version-tag/latest split must be a *job* barrier, not a step
    # order. A matrix shard is an independent job, so `version -> latest` as two
    # steps of one matrix would let api:latest finish before worker:3.1.1 exists.
    version_scripts = [str(step.get("run") or "") for step in _release_steps("publish-version-tags")]
    version_job = (_release().get("jobs") or {}).get("publish-version-tags") or {}
    assert (version_job.get("strategy") or {}).get("matrix"), (
        "publish-version-tags is no longer a matrix; both images must get a version tag"
    )
    assert any("imagetools create" in s and ":latest" not in s for s in version_scripts), (
        "no step promotes the version tag"
    )
    assert not any(":latest" in s for s in version_scripts), (
        "latest is promoted inside the version-tag matrix; that is per-shard and "
        "cannot wait for the other image's version tag"
    )

    latest_job = (_release().get("jobs") or {}).get("promote-latest") or {}
    latest_needs = set(latest_job.get("needs") or [])
    assert "publish-version-tags" in latest_needs, (
        f"promote-latest needs {sorted(latest_needs)}; it must depend on the whole "
        "version-tag matrix so latest can only start after both tags exist"
    )
    latest_scripts = [str(step.get("run") or "") for step in _release_steps("promote-latest")]
    assert any(":latest" in s and "imagetools create" in s for s in latest_scripts), (
        "promote-latest does not promote latest"
    )

    # `imagetools` is a buildx subcommand, so the promotion jobs need a builder
    # even though they build nothing.
    for job_id in ("publish-version-tags", "promote-latest"):
        uses = [str(step.get("uses") or "") for step in _release_steps(job_id)]
        assert any("docker/setup-buildx-action" in u for u in uses), (
            f"{job_id} never initialises buildx, so imagetools create cannot run"
        )


def test_release_creates_a_draft_then_publishes_it():
    """The Release must exist as a draft before anything becomes public."""
    scripts = [str(step.get("run") or "") for step in _release_steps("release")]
    draft_at = next((i for i, s in enumerate(scripts) if "gh release create" in s), None)
    publish_at = next(
        (i for i, s in enumerate(scripts) if "--draft=false" in s), None
    )
    assert draft_at is not None, "no step creates the GitHub Release"
    assert "--draft" in scripts[draft_at], (
        "the Release is created without --draft, so it becomes public before the "
        "image promotion is known to have succeeded"
    )
    assert publish_at is not None and publish_at > draft_at, (
        "the Release is never published after the draft (or published first)"
    )


def test_release_notes_are_versioned_and_exist_for_the_packaged_version():
    """The Release body comes from a reviewed file, not from commit subjects."""
    import tomllib

    with open(os.path.join(PROJECT_ROOT, "pyproject.toml"), "rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    notes = os.path.join(PROJECT_ROOT, "docs", "releases", f"v{version}.md")
    assert os.path.exists(notes), (
        f"docs/releases/v{version}.md is missing; the release job consumes it"
    )
    with open(notes, encoding="utf-8") as handle:
        body = handle.read()
    assert body.strip(), f"docs/releases/v{version}.md is empty"
    assert version in body, f"docs/releases/v{version}.md does not name its version"

    scripts = _release_script("release")
    assert "docs/releases/v$version.md" in scripts, (
        "the release body is no longer composed from the versioned notes file"
    )
    assert "git log" not in scripts, (
        "the release body still derives its content from commit subjects"
    )


def test_release_dry_run_jobs_never_write():
    """The dry-run summary must not log in, push, stage, promote or release."""
    script = _release_script("dry-run-summary")
    for forbidden in ("docker login", "imagetools create", "gh release", "push: true"):
        assert forbidden not in script, f"the dry-run path contains {forbidden!r}"


def test_release_has_no_swallowed_failures_in_publish_steps():
    """No publish/build step may end in a way that cannot fail the job."""
    for job_id in PUBLISH_JOBS + ("preflight", "tests", "artifacts", "image-preflight"):
        script = _release_script(job_id)
        lines = [ln.strip() for ln in script.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        assert lines, f"{job_id} has no runnable steps"
        assert not SWALLOW.search(lines[-1]), (
            f"{job_id} ends with a swallowed failure: {lines[-1]}"
        )
        for step in _release_steps(job_id):
            assert step.get("continue-on-error") is not True, (
                f"{job_id}/{step.get('name')} has continue-on-error"
            )


def test_release_preflight_failure_happens_before_any_write():
    """Structurally: no write step precedes the preflight job's checks."""
    jobs = _release().get("jobs") or {}
    for job_id in PUBLISH_JOBS:
        needs = set((jobs.get(job_id) or {}).get("needs") or [])
        assert "preflight" in needs, (
            f"{job_id} does not depend on preflight, so its tag/version checks are "
            "not ordered before this write"
        )
    # The write jobs also transitively require the test matrix and artifacts.
    for job_id in ("stage-images", "publish-version-tags", "promote-latest", "release"):
        script = _release_script(job_id)
        assert "::error::" not in script or "digest" in script, (
            f"{job_id} contains an error path unrelated to its own write"
        )


# ------------------------------------------------ release workflow (P41 fixes)
def _jobs_map() -> dict:
    return _release().get("jobs") or {}


def test_release_latest_is_a_separate_job_behind_the_whole_version_matrix():
    """P40/P41 G1: a matrix shard is an independent job.

    `version -> latest` as two steps of one matrix job looks ordered but is not:
    the api shard can promote `api:latest` while the worker shard has not yet
    created `worker:3.1.1`. Only a separate job whose `needs` names the matrix
    waits for every shard.
    """
    jobs = _jobs_map()
    assert "promote-latest" in jobs, "latest promotion is not its own job"
    assert "publish-version-tags" in jobs, "version promotion is not its own job"

    version_scripts = _release_script("publish-version-tags")
    assert ":latest" not in version_scripts, (
        "latest is still promoted inside the version-tag matrix job"
    )

    needs = set(jobs["promote-latest"].get("needs") or [])
    assert "publish-version-tags" in needs, (
        f"promote-latest needs {sorted(needs)}; it must wait for the entire "
        "version-tag matrix, not just its own shard"
    )
    release_needs = set(jobs["release"].get("needs") or [])
    assert "promote-latest" in release_needs, (
        f"release needs {sorted(release_needs)}; it must wait for latest promotion"
    )
    assert "publish-version-tags" not in release_needs or "promote-latest" in release_needs


def test_release_manual_dry_run_still_validates_release_material():
    """P40/P41 G3: a dry run must fail for missing/mismatched notes too.

    The tag-specific checks legitimately move behind the manual early-exit; the
    notes check must not, or a dry run rehearses a release whose reviewed notes
    do not exist.
    """
    script = _release_script("preflight")
    early_exit = script.find('GITHUB_EVENT_NAME" != "push"')
    assert early_exit != -1, "the manual early-exit disappeared"
    notes_check = script.find("release notes $notes are missing")
    version_check = script.find('grep -qF "$declared"')
    assert notes_check != -1, "the notes existence check is gone"
    assert version_check != -1, "the notes version check is gone"
    assert notes_check < early_exit, (
        "the notes existence check runs after the manual early-exit, so a dry run "
        "passes without its reviewed release notes"
    )
    assert version_check < early_exit, (
        "the notes version check runs after the manual early-exit"
    )
    # The tag-only proof stays behind it: a dry run must not demand a tag.
    assert script.find("release_tag_guard ") > early_exit, (
        "the guard runs before the manual early-exit, so a dry run would fail for "
        "lacking the tag it is not supposed to have"
    )


def test_release_body_carries_both_image_digests():
    """P40/P41 G2/G4: a tag is mutable; the body must name the digest."""
    jobs = _jobs_map()
    downloads = [
        str((step.get("with") or {}).get("name") or "")
        for step in jobs["release"].get("steps") or []
        if "download-artifact" in str(step.get("uses") or "")
    ]
    assert "digest-api" in downloads, f"release does not download digest-api: {downloads}"
    assert "digest-worker" in downloads, f"release does not download digest-worker: {downloads}"

    script = _release_script("release")
    assert "digests/digest-api.txt" in script and "digests/digest-worker.txt" in script
    assert "@$api_digest" in script and "@$worker_digest" in script, (
        "the release body does not pin the image digests"
    )
    assert "assert_sha256_digest" in script, (
        "the body step does not validate the digest format with the strict helper"
    )


def _digest_helper() -> str:
    path = os.path.join(PROJECT_ROOT, "scripts", "ci", "image_digest.sh")
    assert os.path.exists(path), "the digest helper script is missing"
    with open(path, encoding="utf-8") as handle:
        return handle.read()


#: Repository-relative paths a `run` step may need on disk. A job that references
#: one of these must check the repository out first; a job without a checkout has
#: an empty workspace, so `set -eu` aborts at the first missing file.
REPO_PATH_RE = re.compile(r"(?:^|[\s\"'=<>|;&(])((?:scripts|docs)/[A-Za-z0-9_./-]+)")


def _jobs_in_order() -> dict:
    return _jobs_map()


def _checkout_indices(steps: list) -> list:
    return [i for i, s in enumerate(steps) if "actions/checkout" in str(s.get("uses") or "")]


def test_release_jobs_that_read_repo_files_check_the_repository_out_first():
    """P44: a job that sources a repository file must check the repo out."""
    problems = []
    for job_id, job in _jobs_in_order().items():
        steps = job.get("steps") or []
        checkouts = _checkout_indices(steps)
        for index, step in enumerate(steps):
            script = str(step.get("run") or "")
            if not script:
                continue
            referenced = sorted({m.group(1) for m in REPO_PATH_RE.finditer(script)})
            if not referenced:
                continue
            if not any(c < index for c in checkouts):
                problems.append(
                    f"{job_id}/{step.get('name')} reads {referenced} but no "
                    "actions/checkout precedes it"
                )
    assert not problems, "jobs read repository files without a checkout:\n" + "\n".join(problems)


def test_release_promotion_jobs_check_out_before_sourcing_the_helper():
    """P44 P1: the exact regression, pinned for both promotion jobs."""
    for job_id in ("publish-version-tags", "promote-latest"):
        steps = _release_steps(job_id)
        checkouts = _checkout_indices(steps)
        assert checkouts, f"{job_id} never checks the repository out"
        source_at = next(
            (i for i, s in enumerate(steps) if "scripts/ci/image_digest.sh" in str(s.get("run") or "")),
            None,
        )
        assert source_at is not None, f"{job_id} no longer sources the helper"
        assert min(checkouts) < source_at, (
            f"{job_id} sources the helper at step {source_at} but checks out at "
            f"{min(checkouts)}; the helper would not exist"
        )
        checkout = steps[min(checkouts)]
        assert (checkout.get("with") or {}).get("persist-credentials") is False, (
            f"{job_id} keeps Git credentials in a job that only needs files"
        )


def test_release_helper_hashes_raw_bytes_from_a_file_not_a_variable():
    """P44 P2: command substitution strips trailing newlines."""
    helper = _digest_helper()
    assert "mktemp" in helper, "the helper does not stage the manifest in a temp file"
    assert not re.search(r"raw=\$\(\s*docker buildx", helper), (
        "the helper captures the raw manifest into a shell variable, which strips "
        "trailing newlines and can change the digest"
    )
    # The manifest is hashed from the file, never from a captured variable.
    assert re.search(r'sha256sum\s+"\$tmp"', helper), (
        "the helper does not hash the manifest file itself"
    )
    # P47: cleanup must be trap-driven so signals are covered too, and the
    # scratch file must be private to the call (a subshell), not a global.
    assert "trap 'cleanup' EXIT" in helper, (
        "the helper does not clean up its scratch file on exit"
    )
    assert "trap 'cleanup; exit 130' INT" in helper, (
        "a SIGINT leaves the scratch file behind"
    )
    assert "trap 'cleanup; exit 143' TERM" in helper, (
        "a SIGTERM leaves the scratch file behind"
    )
    assert "_DIGEST_READ_TMP" not in helper, (
        "the helper still keeps the scratch path in a global variable"
    )


def test_release_helper_is_signal_safe_and_subsheltered():
    """P47: cleanup must cover signals, and the scratch path must be call-local.

    P46 showed that a plain `_DIGEST_READ_TMP` global with only normal-path
    cleanup leaves the file behind when a job is cancelled. The function body now
    runs in a subshell with its own EXIT/INT/TERM traps, so a signal still
    removes the file and the subshell still exits non-zero.
    """
    helper = _digest_helper()
    body = helper[helper.index("read_top_level_digest()"):]
    # First meaningful line after the function signature must open a subshell.
    after_signature = body.splitlines()[1:]
    first_code = next(
        (line.strip() for line in after_signature if line.strip() and not line.strip().startswith("#")),
        "",
    )
    assert first_code == "(", (
        f"read_top_level_digest no longer runs in a subshell (first line is {first_code!r}), "
        "so its traps would leak into the caller"
    )
    for signal_name in ("EXIT", "INT", "TERM"):
        assert "trap 'cleanup" in body and signal_name in body, (
            f"no {signal_name} trap in read_top_level_digest"
        )
    # Signal handling must terminate, not fall through to a normal return.
    assert "exit 130" in body and "exit 143" in body, (
        "INT/TERM are swallowed instead of terminating the read"
    )
    # The library must not install traps at file scope (comments aside).
    preamble = helper[:helper.index("read_top_level_digest()")]
    code_lines = [
        line for line in preamble.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    offending = [line for line in code_lines if re.search(r"(^|[;&|]\s*)trap\s+", line)]
    assert not offending, (
        f"sourcing the library installs a global trap on the caller: {offending}"
    )


def test_release_helper_does_not_rely_on_pipefail():
    """P47: a pipeline's status must not decide whether hashing succeeded."""
    helper = _digest_helper()
    body = helper[helper.index("read_top_level_digest()"):]
    assert not re.search(r"sha256sum[^\n]*\|", body), (
        "the helper still pipes sha256sum into another tool; its failure would "
        "depend on the caller enabling pipefail"
    )
    assert re.search(r'if ! sum=\$\(sha256sum\s+"\$tmp"\)', body), (
        "the helper does not capture sha256sum's exit status explicitly"
    )
    assert "_digest_extract_hash" in helper, (
        "the hash field is not extracted by an explicit helper"
    )


def test_release_write_jobs_do_not_persist_git_credentials():
    """P47: a job that can write must not leave Git credentials in the workspace.

    The publishing jobs write through the registry or the GitHub API, never by
    pushing a commit, so they have no use for persisted credentials. Read-only
    jobs that merely run the test suite are out of scope. `preflight` is
    deliberately exempt: it runs `git fetch` against the remote to compare
    `origin/main`, which needs the checkout's credentials.
    """
    problems = []
    for job_id, job in _jobs_in_order().items():
        if job_id == "preflight":
            continue
        permissions = job.get("permissions") or {}
        can_write = any(level == "write" for level in permissions.values())
        if not can_write:
            continue
        for step in job.get("steps") or []:
            if "actions/checkout" not in str(step.get("uses") or ""):
                continue
            persist = (step.get("with") or {}).get("persist-credentials")
            if persist is not False:
                problems.append(f"{job_id} can write but persists Git credentials")
    assert not problems, "write jobs keep Git credentials:\n" + "\n".join(problems)


def test_release_helper_keeps_registry_diagnostics():
    """P44 P3: `2>/dev/null` hid why a manifest could not be read."""
    helper = _digest_helper()
    assert re.search(r'inspect --raw\s+"\$ref"\s+>\s+"\$tmp"', helper), (
        "the raw read no longer redirects stdout to the scratch file"
    )
    # Any redirection of stderr on the raw read hides why it failed.
    raw_read = [
        line for line in helper.splitlines()
        if "imagetools inspect --raw" in line and not line.strip().startswith("#")
    ]
    assert raw_read, "the helper no longer reads a raw manifest"
    for line in raw_read:
        assert "2>" not in line, (
            f"the helper discards buildx/registry stderr again: {line.strip()}"
        )


def test_release_stage_validates_the_digest_before_writing_it():
    """P44 P3: fail next to the cause, not three jobs later."""
    steps = _release_steps("stage-images")
    record_at = next(
        (i for i, s in enumerate(steps) if str(s.get("name") or "") == "Record the digest"),
        None,
    )
    assert record_at is not None, "stage-images no longer records the digest"
    script = str(steps[record_at].get("run") or "")
    assert "scripts/ci/image_digest.sh" in script, (
        "the digest is written without loading the strict validator"
    )
    assert "assert_sha256_digest" in script, (
        "stage-images writes the digest artifact without validating its format"
    )
    assert script.index("assert_sha256_digest") < script.index("digest-$"), (
        "the digest artifact is written before the format is validated"
    )
    checkouts = _checkout_indices(steps)
    assert checkouts and min(checkouts) < record_at, (
        "stage-images validates the digest without checking the repository out first"
    )


def test_release_draft_probe_uses_http_status_not_error_text():
    """P44/P46: the status must be *parsed* from the response, not assumed.

    P46 found this test survived a mutation that replaced the whole parse with
    `http_status=404`: every assertion here was a bare `in script` check, and the
    variable name plus the error strings all survived. The assertions below
    therefore pin the *shape* of the assignment: its right-hand side must be a
    command substitution that reads the probe file and matches an HTTP status
    line, and it must take the last such line and extract the numeric code.
    """
    script = _release_script("release")
    assert "--include" in script, "the probe does not request HTTP status lines"
    assert "grep -q 'HTTP 404'" not in script, (
        "the probe still greps the human-readable error text"
    )
    assert "could not determine an HTTP status" in script, (
        "an unparseable probe does not fail the step"
    )
    assert "refusing to guess whether it exists" in script, (
        "a non-2xx/non-404 status does not stop the step"
    )

    # The assignment must be a command substitution, not a constant.
    assignment = re.search(r"^\s*http_status=(.*)$", script, re.MULTILINE)
    assert assignment is not None, "the probe no longer assigns http_status"
    rhs = assignment.group(1).strip()
    assert rhs.startswith("$(") and rhs.endswith(")"), (
        f"http_status is not derived from a command substitution: {rhs!r}"
    )
    pipeline = rhs[2:-1]

    # It must read the probe file (not a literal).
    assert "$probe_file" in pipeline, (
        f"the status parse does not read the probe output: {pipeline!r}"
    )
    # It must match an HTTP status line, not arbitrary text.
    assert "grep" in pipeline, f"the status parse is not a grep: {pipeline!r}"
    assert "HTTP/" in pipeline, (
        f"the status parse does not match an HTTP status line: {pipeline!r}"
    )
    # It must take the final response, not the first one seen.
    assert "tail" in pipeline, (
        f"the status parse does not take the last status line: {pipeline!r}"
    )
    # It must extract the three-digit code.
    assert "awk" in pipeline or "cut" in pipeline or "sed" in pipeline, (
        f"the status parse does not extract the numeric code: {pipeline!r}"
    )
    # And it must not be a literal status.
    assert not re.fullmatch(r"\d{3}", rhs.strip("'\"")), (
        f"http_status is hard-coded to {rhs!r}"
    )


def test_release_verifies_digest_identity_after_promotion():
    """P40/P41 G4/G8 + P42: printing a manifest is not verification.

    P42 found that the old implementation parsed `--format '{{json .Manifest}}'`
    with sed. That output is pretty-printed JSON, so the regex matched nothing and
    promotion could only fail. The fix reads the raw manifest bytes and hashes
    them, which is exactly how a registry derives the digest. This test pins the
    mechanism so the sed implementation cannot come back.
    """
    for job_id in ("publish-version-tags", "promote-latest"):
        script = _release_script(job_id)
        assert "scripts/ci/image_digest.sh" in script, (
            f"{job_id} does not use the shared digest helper"
        )
        assert "verify_promoted_digest" in script, (
            f"{job_id} never verifies the promoted tag against the staged digest"
        )
        # The P42 regression: no JSON template parsing, no sed extraction.
        assert "{{json .Manifest}}" not in script, (
            f"{job_id} parses the pretty-printed .Manifest JSON again"
        )
        assert "sed -n" not in script, (
            f"{job_id} extracts the digest with sed again; that parser cannot "
            "reliably yield one top-level digest"
        )
        assert "require_buildx" in script, (
            f"{job_id} does not probe buildx before reading a manifest"
        )

    helper = _digest_helper()
    assert "--raw" in helper, (
        "the helper does not read the raw manifest; a template field cannot be "
        "verified to be the top-level digest"
    )
    assert "sha256sum" in helper, "the helper does not hash the raw manifest bytes"
    assert r"^sha256:[0-9a-f]{64}$" in helper, (
        "the helper does not enforce strict lowercase-hex digest syntax"
    )
    assert "verify_promoted_digest" in helper and "read_top_level_digest" in helper
    # Exactly one value: the helper prints a single digest, never a list.
    assert "printf '%s' \"$digest\"" in helper, (
        "the helper does not emit exactly one digest value"
    )


def test_release_rejects_malformed_digests_everywhere():
    """P42: prefix+length is not a format check.

    `case sha256:*` plus a length test accepts `sha256:` + 64 uppercase letters,
    spaces or non-hex characters. Every digest check must go through the strict
    helper instead.
    """
    helper = _digest_helper()
    assert "[[:space:]]" in helper, "the helper does not reject embedded whitespace"
    for job_id in ("publish-version-tags", "promote-latest", "release"):
        script = _release_script(job_id)
        assert "scripts/ci/image_digest.sh" in script, (
            f"{job_id} does not source the strict digest helper"
        )
        assert "assert_sha256_digest" in script, (
            f"{job_id} does not validate its digest with the strict helper"
        )
        # The weak guard must be gone.
        assert "sha256:*)" not in script, (
            f"{job_id} still uses the prefix-only case guard"
        )
        assert '"${#' not in script, (
            f"{job_id} still checks the digest by length alone"
        )


def test_release_draft_probe_distinguishes_missing_from_errors():
    """P42/P44: `gh release view` failure is not proof the release is missing."""
    script = _release_script("release")
    assert "http_status" in script, (
        "the draft probe does not distinguish a missing release from an error"
    )
    assert "could not determine an HTTP status" in script, (
        "an inconclusive probe does not fail the step"
    )
    assert "targetCommitish" in script, (
        "an existing draft is reused without checking its target commit"
    )
    assert "existing draft is for tag" in script, (
        "an existing draft is reused without checking its tag"
    )


def test_release_draft_state_is_asserted_both_ways():
    """P40/P41 G2: `--jq .isDraft` printing a value proves nothing."""
    script = _release_script("release")
    assert 'if [ "$state" != "true" ]' in script, (
        "the draft step does not assert isDraft == true"
    )
    assert 'if [ "$state" != "false" ]' in script, (
        "the publish step does not assert isDraft == false"
    )
    assert "refusing to overwrite it" in script, (
        "an existing public release would be overwritten instead of failing"
    )
    assert "--clobber" in script, "a re-run cannot replace the draft assets"
    assert "gh release view" in script and "already exists" in script, (
        "the create path is not idempotent for an existing draft"
    )


def test_release_preflight_build_has_no_attestations():
    """P40/P41 G7: `load` uses the docker exporter, which cannot carry attestations."""
    build = next(
        (step for step in _release_steps("image-preflight") if step.get("id") == "build"),
        None,
    )
    assert build is not None, "the preflight build step is gone"
    with_block = build.get("with") or {}
    assert with_block.get("load") is True, "the preflight image is never loaded for docker run"
    assert "provenance" not in with_block, (
        "provenance attestation is requested on a load-only build; the docker "
        "exporter cannot carry an OCI index attestation"
    )
    assert "sbom" not in with_block, "sbom attestation is requested on a load-only build"
    # Attestation still belongs to the registry push.
    stage_build = next(
        (step for step in _release_steps("stage-images") if step.get("id") == "build"), None
    )
    assert (stage_build.get("with") or {}).get("provenance") == "mode=max"


def test_release_artifact_readers_request_actions_read():
    """P40/P41 G6: downloading this run's artifacts is a read scope."""
    for job_id in ("publish-version-tags", "promote-latest", "release"):
        permissions = (_jobs_map()[job_id].get("permissions") or {})
        assert permissions.get("actions") == "read", (
            f"{job_id} downloads artifacts but does not declare actions: read"
        )
        assert "write" not in {k: v for k, v in permissions.items() if k == "actions"}.values()


def test_release_publish_jobs_cannot_swallow_or_only_print_status():
    """P40/P41 G8: a status command must be able to fail the step."""
    for job_id in ("publish-version-tags", "promote-latest", "release"):
        for step in _release_steps(job_id):
            script = str(step.get("run") or "")
            if "isDraft" in script or "imagetools inspect" in script:
                assert "if [" in script or "test " in script, (
                    f"{job_id}/{step.get('name')} only prints a status value"
                )
