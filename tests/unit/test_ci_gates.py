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
