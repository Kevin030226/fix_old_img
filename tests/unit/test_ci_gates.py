"""The CI configuration itself is under test (plan 搂3.11).

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
    line, it decides the step's exit status 鈥?and then nothing downstream can
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
    only shows up as `SyntaxError` while pytest imports the file 鈥?which here
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
    """A tier that can only run on one laptop is not a gate (plan 搂4.3 Step 2).

    The PostgreSQL path was developed against a fake driver that answered exactly
    what the code asked; pointing it at a real server produced four defects the fake
    could not see (SQLite-only DDL, a reserved column name, positional row access, a
    cross-type COALESCE). So the job must have a real server as a service, must test
    the rollback as well as the upgrade 鈥?a one-way "upgrade works" check says
    nothing about `downgrade` 鈥?and must not swallow a failure.
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
    "always exit successfully" 鈥?a vulnerability report that can never be read as
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
