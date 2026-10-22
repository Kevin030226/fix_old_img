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
