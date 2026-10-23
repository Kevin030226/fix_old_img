"""What each image promises to support, and what it actually installs.

`test_compose_topology.py` answers "can this image reach the transports its own
compose file offers". This file answers the two questions beside it:

* the root image must not be silently excluded from that check (it was: the
  scanner only understood the long form of `build:`, and the root compose file
  uses `build: .`, so the one image that was missing its drivers was the one
  image the gate never looked at);
* no published image may ship the test/dev stack, and the inference image is the
  only place the GPU stack belongs.
"""
import glob
import os
import re

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGES = {
    "Dockerfile": "all-in-one",
    os.path.join("docker", "api.Dockerfile"): "api",
    os.path.join("docker", "worker.Dockerfile"): "worker",
}


def _text(name: str) -> str:
    with open(os.path.join(PROJECT_ROOT, name), encoding="utf-8") as handle:
        return handle.read()


def _installed_extras(text: str) -> set[str]:
    found = re.findall(r'pip install[^\n&|;]*?"\.?\[([a-z0-9,_ -]+)\]"', text)
    return {piece.strip() for group in found for piece in group.split(",") if piece.strip()}


@pytest.mark.parametrize("name", sorted(IMAGES))
def test_a_runtime_image_never_ships_test_or_dev_tooling(name):
    """pytest, moto and ruff belong to the CI image, not to a published one."""
    extras = _installed_extras(_text(name))
    assert not (extras & {"test", "dev", "benchmark"}), (
        f"{name} ships {sorted(extras & {'test', 'dev', 'benchmark'})}"
    )


def test_only_the_inference_images_install_the_gpu_stack():
    """`gpu` is torch + dlib: it belongs where models run, not behind an HTTP API.

    The API image is deliberately torch-free so a CPU-only control plane stays
    small; the all-in-one image carries it because it also runs the worker.
    """
    installed = {name: _installed_extras(_text(name)) for name in IMAGES}
    assert "gpu" not in installed[os.path.join("docker", "api.Dockerfile")]
    assert "gpu" not in installed["Dockerfile"], (
        "the root image installs the GPU stack through requirements.txt/torch, "
        "not through the extra -- declaring the extra as well would try to "
        "resolve a second torch on top of the cu128 pin"
    )


def test_the_all_in_one_image_installs_the_transport_extras():
    """`docker-compose.yml` builds this image and offers Redis in its own comments.

    Without the extras, `FIXIMG_QUEUE_BACKEND=redis` (documented in that file at
    line 13) fails on import, and PostgreSQL/S3 have the same shape. The extras
    are the minimum that makes the documented choices reachable.
    """
    text = _text("Dockerfile")
    extras = _installed_extras(text)
    assert {"postgres", "redis", "s3"} <= extras, (
        f"the root image installs {sorted(extras) or 'no extras'}, but its "
        "compose file documents Redis, PostgreSQL and S3 as selectable backends"
    )


def test_the_all_in_one_image_still_installs_the_declared_requirement_file():
    """The extras add transports; they do not replace the pinned web/imaging stack."""
    text = _text("Dockerfile")
    assert "pip install -r requirements.txt" in text
    assert "requirements/runtime.txt" not in text, (
        "runtime.txt omits torch, which this image pins separately; switching "
        "to it would need that pin to move into the same install"
    )


@pytest.mark.parametrize("name", sorted(IMAGES))
def test_every_image_upgrades_msgpack(name):
    """GHSA-6v7p-g79w-8964 lands in the wheel the base image already carries.

    `docker/api.Dockerfile` pinned the floor after Trivy reported it; the other two
    images carry the same dependency transitively and were left behind.
    """
    text = _text(name)
    installs = " ".join(
        line for line in text.splitlines() if line.strip().startswith("RUN pip install")
    )
    if not installs:
        pytest.skip(f"{name} runs no pip install")
    assert "msgpack" in installs, f"{name} does not pin the msgpack floor"


def test_every_compose_build_reference_resolves_to_a_real_dockerfile():
    """A `build:` that names a missing file is a deploy-time failure, not a lint one."""
    import yaml

    files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "docker-compose.yml"))) + sorted(
        glob.glob(os.path.join(PROJECT_ROOT, "docker", "compose*.yaml"))
    )
    assert files, "no compose files found"
    for path in files:
        document = yaml.safe_load(open(path, encoding="utf-8")) or {}
        for service, spec in (document.get("services") or {}).items():
            build = (spec or {}).get("build")
            if build is None:
                continue
            if isinstance(build, str):
                context, dockerfile = build, "Dockerfile"
            else:
                context = build.get("context") or "."
                dockerfile = build.get("dockerfile") or "Dockerfile"
            resolved = os.path.normpath(
                os.path.join(os.path.dirname(path), context, dockerfile)
            )
            assert os.path.isfile(resolved), (
                f"{os.path.basename(path)}: service {service!r} builds {resolved}, "
                "which does not exist"
            )


# --------------------------------------------------- resolved runtime dependencies

def _docker_job() -> dict:
    """The `docker` job of the CI workflow, parsed."""
    import yaml

    with open(os.path.join(PROJECT_ROOT, ".github", "workflows", "ci.yml"),
              encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    return document["jobs"]["docker"]


def _step_named(job: dict, fragment: str) -> dict:
    matches = [step for step in job["steps"] if fragment in str(step.get("name", ""))]
    assert len(matches) == 1, f"expected one step named like {fragment!r}, found {len(matches)}"
    return matches[0]


def test_every_image_asserts_the_resolved_runtime_dependencies():
    """The files said 13.0.0 and the worker image shipped 12.3.0 anyway (P18).

    Every gate in this file reads *declarations*, and declarations were not the
    problem: `requirements.txt`, the lock and both other images all pinned 13.0.0.
    The worker image installs the torch stack first, `torchvision==0.22.1` accepts
    any `pillow>=5.3.0`, and the project was installed afterwards -- so the answer
    depended on the *order* of two installs, which no file can express.

    The only place that order is observable is the built image. So the workflow must
    ask the image what it contains, in every shard, before the scan that would
    otherwise be the first thing to notice.
    """
    job = _docker_job()
    step = _step_named(job, "Assert the resolved runtime dependencies inside the image")
    body = str(step.get("run", ""))

    # The assertion must be about the image's own metadata, not a file it ships.
    assert "importlib.metadata" in body, (
        "the runtime assertion does not read installed distribution metadata, so it "
        "cannot see what the image actually contains"
    )
    assert "matrix.image" in str(step.get("run", "")) or "matrix.image" in body, (
        "the runtime assertion is not parameterised by the matrix image"
    )
    # Pillow, at the pinned version, and the removed framework as a negative.
    assert "13.0.0" in body, "the runtime assertion does not state the expected Pillow"
    for forbidden in ("gradio", "gradio-client"):
        assert forbidden in body, (
            f"the runtime assertion does not reject {forbidden!r}, so the removed "
            "framework could return inside an image unnoticed"
        )


def test_the_runtime_assertion_runs_before_the_vulnerability_scan():
    """A scan that fails first hides *why* the image was wrong.

    Trivy does fail on a vulnerable Pillow, so a missing assertion would still be
    caught -- as "19 findings in an unnamed package", twenty minutes in, with the
    actual cause (an install-order regression) left for someone to reconstruct.
    Asserting first names the defect in one line.
    """
    job = _docker_job()
    names = [str(step.get("name", "")) for step in job["steps"]]
    assertion = next(i for i, name in enumerate(names)
                     if "Assert the resolved runtime dependencies" in name)
    scan = next(i for i, name in enumerate(names)
                if "Scan the" in name and "for known vulnerabilities" in name)
    assert assertion < scan, (
        "the runtime dependency assertion runs after the Trivy scan; a regression "
        "would be reported as a scan finding instead of as the dependency it is"
    )


@pytest.mark.parametrize("relative", sorted(IMAGES))
def test_no_image_can_skip_the_runtime_assertion(relative):
    """The step is unguarded, so every matrix shard runs it.

    A `if: matrix.image == '...'` guard would let the two images that already pass
    carry the check while the one that regressed skips it -- which is exactly the
    shape of the P18 failure.
    """
    job = _docker_job()
    step = _step_named(job, "Assert the resolved runtime dependencies inside the image")
    assert not step.get("if"), (
        "the runtime dependency assertion is conditional; every image must run it, "
        "because the image that regressed was the one with the unusual install order"
    )

# ---------------------------------------- the runtime assertion must actually run
#
# P20 shipped a step that looked like a gate and was a no-op. It read:
#
#     docker run --rm "fiximg-<image>:ci" python3 - <<'PY'
#     ... assertions ...
#     PY
#
# Without `-i`, docker does not attach the container's stdin. The heredoc stayed
# in the host shell, the container's interpreter read EOF, and an EMPTY program
# exited 0 in about six milliseconds. Three images "passed" a check that asserted
# nothing, and the only trace was the absence of output nobody was looking for.
#
# The fix passes the program as an argument (`python3 -c '...'`), so there is no
# stream to forget to connect. The gates below hold that shape: they reject the
# stdin form outright rather than trusting that someone will remember `-i`, and
# they require the assertions and the success marker to be present in the code
# that actually reaches the container.


def _docker_steps() -> list[dict]:
    """Every step of the CI docker job, parsed from the committed workflow."""
    import yaml

    with open(os.path.join(PROJECT_ROOT, ".github", "workflows", "ci.yml"),
              encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    return document["jobs"]["docker"]["steps"]


def _runtime_assert_step() -> dict:
    matches = [step for step in _docker_steps()
               if "runtime dependencies" in str(step.get("name", ""))]
    assert len(matches) == 1, (
        f"expected exactly one runtime-assertion step, found {len(matches)}; a "
        "second copy would drift from the first"
    )
    return matches[0]


def _docker_run_lines(body: str) -> list[str]:
    """The docker run commands in a step body, with continuations joined."""
    joined: list[str] = []
    pending = ""
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if pending:
            pending += " " + line
        else:
            pending = line
        if pending.endswith(chr(92)):
            pending = pending[:-1].rstrip()
            continue
        joined.append(pending)
        pending = ""
    if pending:
        joined.append(pending)
    return [line for line in joined if line.startswith("docker run")]


def _python_program(step: dict) -> str:
    """The Python source the container will actually execute.

    Extracted from the `-c` argument. If the step uses stdin instead, this
    returns an empty string -- which the caller treats as a failure, because that
    form cannot be shown to deliver the code without also checking a `-i` flag
    that is easy to omit.
    """
    body = str(step.get("run", ""))
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if "python3 -c" not in line:
            continue
        tail = line.split("python3 -c", 1)[1].lstrip()
        if not tail.startswith(chr(39)):
            continue
        collected = [tail[1:]]
        for follow in lines[index + 1:]:
            if follow.strip() == chr(39):
                break
            collected.append(follow)
        else:
            continue
        return chr(10).join(collected)
    return ""


def _run_container_program(program: str, pillow: str, extra_dists=()) -> tuple[int, str]:
    """Execute the container program against stubbed metadata.

    The program is the exact text that travels to the image, so running it here
    measures what the gate will actually do. The stub replaces importlib.metadata
    with one that reports the given Pillow version and distribution names, which
    is the only thing the program reads.
    """
    import subprocess
    import sys
    import tempfile

    harness = chr(10).join([
        "import sys, types, os",
        "os.environ.setdefault('IMAGE_KIND', 'stub')",
        "fake = types.ModuleType('importlib.metadata')",
        "class _D:",
        "    def __init__(self, name): self.metadata = {'Name': name}",
        f"fake.distributions = lambda: [_D(n) for n in {list(('Pillow',) + tuple(extra_dists))!r}]",
        f"fake.version = lambda n: {pillow!r} if n.lower() == 'pillow' else '0.0.0'",
        "sys.modules['importlib.metadata'] = fake",
        "import importlib; importlib.metadata = fake",
        "exec(compile(open(sys.argv[1], encoding='utf-8').read(), sys.argv[1], 'exec'), {'__name__': '__main__'})",
    ])
    with tempfile.TemporaryDirectory() as tmp:
        harness_path = os.path.join(tmp, "harness.py")
        program_path = os.path.join(tmp, "program.py")
        with open(harness_path, "w", encoding="utf-8") as handle:
            handle.write(harness)
        with open(program_path, "w", encoding="utf-8") as handle:
            handle.write(program)
        result = subprocess.run([sys.executable, harness_path, program_path],
                                capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def _program_rejects_installed_gradio(program: str) -> bool:
    """True when the program fails if gradio is among the installed packages."""
    code, _output = _run_container_program(program, "13.0.0",
                                           extra_dists=("fastapi", "gradio"))
    return code != 0


def _program_accepts_a_clean_image(program: str) -> tuple[bool, str]:
    """True when a correct image (Pillow 13.0.0, no framework) passes, plus output."""
    code, output = _run_container_program(program, "13.0.0",
                                          extra_dists=("fastapi", "torch"))
    return code == 0, output


def _program_rejects_wrong_pillow(program: str) -> bool:
    """True when the program fails for the version P18 actually shipped."""
    code, _output = _run_container_program(program, "12.3.0",
                                           extra_dists=("fastapi",))
    return code != 0


def test_the_runtime_assertion_delivers_its_code_as_an_argument_not_via_stdin():
    """The exact defect P20 found, rejected structurally.

    Two acceptable shapes exist: `python3 -c '<code>'`, or an explicit
    `docker run -i ... <<'PY'`. Only the first is used here, and the gate requires
    it -- a heredoc form is rejected unless `-i` is present on the same command, so
    the silent no-op cannot return by someone copying an older snippet.
    """
    step = _runtime_assert_step()
    body = str(step.get("run", ""))

    program = _python_program(step)
    assert program.strip(), (
        "the runtime assertion does not pass its program with python3 -c; the "
        "container would have to read it from stdin, which is exactly the form "
        "that silently executed nothing when -i was missing"
    )

    for command in _docker_run_lines(body):
        if "python3 -" in command and "-c" not in command:
            assert " -i" in command or " --interactive" in command, (
                "a docker run ... python3 - command is missing -i, so the "
                "container's stdin is closed and the program never arrives: " + command
            )


def test_the_container_program_asserts_pillow_exactly_and_gradio_absent():
    """The semantics, checked on the code that reaches the container."""
    program = _python_program(_runtime_assert_step())
    assert program.strip(), "no program to check"

    assert "importlib.metadata" in program, (
        "the program does not read installed distribution metadata, so it cannot "
        "see what the image actually contains"
    )
    assert "md.version(" in program, "the program does not read a distribution version"
    assert _program_rejects_wrong_pillow(program), (
        "the program does not fail for Pillow 12.3.0 -- the version the worker "
        "image shipped in P18 -- so a regression would pass"
    )

    # The framework check is measured by *running* the program, not by looking for
    # the word "gradio": the first version of this gate asserted the substring and
    # passed a mutation that emptied the rejection loop, because the literals were
    # still in the source. Executing the program against a stubbed metadata that
    # reports gradio installed must fail; that is the property.
    assert _program_rejects_installed_gradio(program), (
        "the program does not fail when gradio is installed; it would ship the "
        "framework this project removed and re-open the pillow<13 advisories"
    )

    # The marker is checked by running a *clean* image through the program and
    # requiring the marker in its output. A source-only check would pass a program
    # whose print statement was unreachable, which is the same class of defect as
    # P20: the code looked right and produced nothing.
    ok, output = _program_accepts_a_clean_image(program)
    assert ok, (
        "a correct image (Pillow 13.0.0, no UI framework) did not pass the "
        f"container program; output was: {output.strip()[:300]}"
    )
    assert "RUNTIME-ASSERT-OK" in output, (
        "the program produced no success marker for a correct image, so a silent "
        "no-op would be indistinguishable from a pass -- which is how P20's defect "
        f"survived review; output was: {output.strip()[:300]}"
    )
    assert "pillow=13.0.0" in output, (
        f"the marker does not report the Pillow version it verified: {output.strip()[:200]}"
    )
    assert "gradio=absent" in output, (
        f"the marker does not report the framework result: {output.strip()[:200]}"
    )
    assert "image=" in output, (
        f"the marker does not identify which image was asserted: {output.strip()[:200]}"
    )


def test_the_runtime_assertion_fails_the_step_on_a_mismatch():
    """An assertion is only a gate if it can fail the step.

    `python3 -c` exits non-zero on an AssertionError, and the workflow runs
    each run block under `bash -e`, so a failed assertion fails the job. What must
    not happen is a program that swallows the failure.
    """
    program = _python_program(_runtime_assert_step())
    assert "assert " in program, "the program contains no assertion"
    assert "except" not in program, (
        "the program catches exceptions, so a mismatch could be swallowed"
    )
    assert "sys.exit(0)" not in program, "the program forces a zero exit"


def test_the_runtime_assertion_runs_before_every_trivy_scan():
    """Assert first, scan second: a scan finding names a package, not a cause."""
    steps = _docker_steps()
    names = [str(step.get("name", "")) for step in steps]
    assertion = next(i for i, name in enumerate(names)
                     if "runtime dependencies" in name)
    scans = [i for i, step in enumerate(steps)
             if "trivy-action" in str(step.get("uses", ""))]
    assert scans, "no Trivy scan step in the docker job"
    assert assertion < min(scans), (
        "the runtime assertion runs after the Trivy scan; a dependency regression "
        "would then be reported as a scan finding instead of as the dependency it is"
    )


def test_every_matrix_image_runs_the_runtime_assertion():
    """One shared, unguarded step -- not three copies that can drift apart."""
    step = _runtime_assert_step()
    assert not step.get("if"), (
        "the runtime assertion is conditional; the image that regressed in P18 was "
        "the one with the unusual install order, so every shard must run it"
    )
    assert "matrix.image" in str(step.get("run", "")), (
        "the assertion is not parameterised by the matrix image"
    )
    import yaml

    with open(os.path.join(PROJECT_ROOT, ".github", "workflows", "ci.yml"),
              encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    matrix = document["jobs"]["docker"]["strategy"]["matrix"]["include"]
    assert {entry["image"] for entry in matrix} == {"api", "worker", "root"}, (
        "the docker matrix no longer covers exactly the three published images"
    )
