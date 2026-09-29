"""The compose topologies must run what they expose (plan §3.13, §7.5).

`docker compose -f docker/compose.local.yaml up` was documented as the
single-container demo and published port 9502 while running the **worker** image's
`CMD` - a queue consumer that binds nothing. It carried `FIXIMG_HOST`,
`FIXIMG_PORT` and `FIXIMG_INLINE_WORKER`, all three of which are read only by
`fiximg.cli.api`, so the file looked correct in review and served nothing in
practice. The same class of mistake is what these checks forbid: a compose file is
a declaration of what runs where, and nothing else in the suite reads it.

Commands are resolved the way Docker resolves them: a `command:` in the file wins,
otherwise the `CMD` of the Dockerfile that service builds.
"""
import glob
import os
import re

import pytest

yaml = pytest.importorskip("yaml", reason="pyyaml not installed")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPOSE_FILES = sorted(
    glob.glob(os.path.join(PROJECT_ROOT, "docker-compose.yml"))
    + glob.glob(os.path.join(PROJECT_ROOT, "docker", "compose*.yaml"))
)
#: Entry points that serve HTTP (API + Gradio). `run.py` is the offline pipeline CLI.
SERVING = ("fiximg.cli.api", "main.py")
WORKER = ("fiximg.cli.worker", "worker.py")


def _documents():
    assert COMPOSE_FILES, "no compose files found - the gate would be vacuous"
    for path in COMPOSE_FILES:
        yield path, _load(path)


def _dockerfile_cmd(path: str, build: dict, dockerfile: str) -> list[str]:
    """The CMD of the Dockerfile a service builds, as argument words.

    `build.dockerfile` is relative to `build.context`, not to the compose file -
    resolving it any other way points at a path that does not exist.
    """
    context = os.path.join(os.path.dirname(path), (build.get("context") or "."))
    target = os.path.normpath(os.path.join(context, dockerfile))
    with open(target, encoding="utf-8") as handle:
        text = handle.read()
    match = re.search(r'(?m)^CMD\s+(\[[^\]]*\])', text)
    if not match:
        raise AssertionError(f"{target}: no JSON-form CMD - the gate cannot resolve it")
    return [str(word).lower() for word in yaml.safe_load(match.group(1))]


def _command(path: str, name: str, service: dict) -> list[str]:
    """The words a container actually runs."""
    command = service.get("command")
    if command is None:
        build = service.get("build") or {}
        dockerfile = build.get("dockerfile") if isinstance(build, dict) else None
        assert dockerfile, (
            f"{os.path.basename(path)}: {name} names only an image "
            f"({service.get('image')!r}) and no command, so nothing in this "
            "repository says what it executes - a reviewer cannot tell a serving "
            "process from a queue consumer, which is the mistake this gate exists "
            "to catch. Declare `command:`."
        )
        return _dockerfile_cmd(path, build, dockerfile)
    if isinstance(command, str):
        return command.lower().split()
    return [str(word).lower() for word in command]


def _runs_this_repo(service: dict) -> bool:
    """Is this a container built from (or of) this project?

    The rules below are about *our* entry points, so they apply to our services
    only. Requiring `command:` from `postgres:16-alpine` would be a rule no real
    compose file can meet, and a gate that cannot be met becomes a waiver.
    """
    build = service.get("build")
    image = str(service.get("image") or "").lower()
    return bool(build) or image.startswith(("fiximg", "fixoldimg"))


def _published(service: dict) -> list[str]:
    out: list[str] = []
    for entry in service.get("ports") or []:
        out.append(entry if isinstance(entry, str) else str(entry.get("published", "")))
    return out


def _words(command: list[str]) -> set[str]:
    return {word.rsplit("/", 1)[-1] for word in command}


def _serves(command: list[str]) -> bool:
    words = _words(command)
    return bool(words & set(SERVING))


def _consumes_queue_only(command: list[str]) -> bool:
    """A container that only drains the queue: it has no listener at all."""
    return bool(_words(command) & set(WORKER)) and not _serves(command)


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: os.path.basename(p))
def test_every_published_port_is_served_by_an_api_process(path):
    """A service that publishes a port must run a process that binds it.

    Otherwise `docker compose up` succeeds, the healthcheck is never reached, and
    the topology only fails when someone opens a browser.
    """
    document = _load(path)
    offenders = []
    for name, service in (document.get("services") or {}).items():
        service = service or {}
        if not _runs_this_repo(service):
            continue
        published = _published(service)
        if not published:
            continue
        command = _command(path, name, service or {})
        if not _serves(command):
            offenders.append(f"{name}: publishes {published} but runs {command}")
    assert not offenders, f"{os.path.basename(path)}: {offenders}"


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: os.path.basename(p))
def test_a_worker_container_publishes_nothing(path):
    """The queue worker has no listener; an inline worker belongs to the API."""
    document = _load(path)
    offenders = []
    for name, service in (document.get("services") or {}).items():
        service = service or {}
        if not _runs_this_repo(service):
            continue
        command = _command(path, name, service)
        if _published(service) and _consumes_queue_only(command):
            offenders.append(f"{name}: publishes a port while only consuming the queue")
        inline = str((service.get("environment") or {}).get("FIXIMG_INLINE_WORKER", "")).lower()
        if inline == "true" and not _serves(command):
            offenders.append(
                f"{name}: sets FIXIMG_INLINE_WORKER=true without serving anything, "
                f"so the queue is consumed by a process nobody can submit to: {command}"
            )
    assert not offenders, f"{os.path.basename(path)}: {offenders}"


def test_the_object_store_credentials_compose_offers_are_usable_by_the_app():
    """MinIO's root user must reach the app that talks to it, in both halves.

    `docker/compose.yaml` starts MinIO with `MINIO_ROOT_USER`/`PASSWORD` while the
    app had no settings for either: boto3's ambient credential chain finds nothing
    inside those containers, so `FIXIMG_STORAGE_BACKEND=s3` could not authenticate
    no matter what an operator configured. The keys are compared in both
    directions, because a pair that merely *exists* is not a pair that matches.
    """
    pairs = 0
    for path, document in _documents():
        services = document.get("services") or {}
        minio = next(
            (svc for svc in services.values()
             if "minio" in str((svc or {}).get("image", "")).lower()),
            None,
        )
        if not minio:
            continue
        root = minio.get("environment") or {}
        expected_user = root.get("MINIO_ROOT_USER")
        expected_secret = root.get("MINIO_ROOT_PASSWORD")
        assert expected_user and expected_secret, (
            f"{os.path.basename(path)}: MinIO starts without a root credential"
        )
        for name, service in services.items():
            env = (service or {}).get("environment") or {}
            if "FIXIMG_STORAGE_ACCESS_KEY" not in env:
                continue
            pairs += 1
            assert env["FIXIMG_STORAGE_ACCESS_KEY"] == expected_user, (
                f"{name}: FIXIMG_STORAGE_ACCESS_KEY does not match MinIO_ROOT_USER"
            )
            assert env.get("FIXIMG_STORAGE_SECRET_KEY") == expected_secret, (
                f"{name}: FIXIMG_STORAGE_SECRET_KEY does not match MinIO_ROOT_PASSWORD"
            )
            assert env.get("FIXIMG_STORAGE_REGION"), (
                f"{name}: SigV4 needs a region even for a path-style endpoint"
            )
    assert pairs, "no compose file passes storage credentials to the app at all"


#: Where a container's models live, as paths inside the image's workdir. The pipeline
#: reads these three, so they are the whole of what "the weights are available" means.
WEIGHT_MOUNTS = ("/app/weights", "/app/Global/checkpoints", "/app/Face_Enhancement/checkpoints")


def _volume_targets(service: dict) -> set[str]:
    targets = set()
    for entry in service.get("volumes") or []:
        text = entry if isinstance(entry, str) else str(entry.get("target", ""))
        target = text.split(":")[-1].strip()
        if target:
            targets.add(target)
    return targets


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: os.path.basename(p))
def test_every_named_volume_a_service_uses_is_declared(path):
    """`docker compose up` refuses an undeclared volume -- but only after the build.

    The images now carry no weights, so the volumes that hold what the operator
    downloads are load-bearing: a typo there is a worker that reports every model
    unavailable for a reason nothing in the suite explains.
    """
    document = _load(path)
    declared = set(document.get("volumes") or {})
    offenders = []
    for name, service in (document.get("services") or {}).items():
        for entry in (service or {}).get("volumes") or []:
            text = entry if isinstance(entry, str) else str(entry.get("source", ""))
            source = text.split(":")[0].strip()
            # Bind mounts (`./x:/app/y`) and absolute paths need no declaration.
            if not source or source.startswith((".", "/", "~")) or "${" in source:
                continue
            if source not in declared:
                offenders.append(f"{name}: uses undeclared volume {source!r}")
    assert not offenders, f"{os.path.basename(path)}: {offenders}"


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: os.path.basename(p))
def test_a_service_that_runs_inference_mounts_every_weight_location(path):
    """The compose half of the no-weights-in-the-image decision.

    Checked against the services that actually execute stages -- the queue consumer and
    the inline-worker app -- because a weights-free image is only a working deployment if
    the downloaded files land somewhere that survives `docker compose up -d`.
    """
    document = _load(path)
    offenders = []
    for name, service in (document.get("services") or {}).items():
        service = service or {}
        if not _runs_this_repo(service):
            continue
        environment = service.get("environment") or {}
        inline = str(environment.get("FIXIMG_INLINE_WORKER", "")).lower() == "true"
        if not (inline or _consumes_queue_only(_command(path, name, service))):
            continue
        missing = [want for want in WEIGHT_MOUNTS if want not in _volume_targets(service)]
        if missing:
            offenders.append(f"{name}: does not mount {missing}")
    assert not offenders, (
        f"{os.path.basename(path)}: the image ships no weights, so an inference service "
        f"without these mounts re-downloads them on every recreate: {offenders}"
    )


DOCKERFILES = [
    os.path.join(PROJECT_ROOT, name)
    for name in ("Dockerfile", os.path.join("docker", "api.Dockerfile"),
                 os.path.join("docker", "worker.Dockerfile"))
]


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: os.path.basename(p))
def test_building_an_image_does_not_depend_on_a_third_party_weight_host(path):
    """No weight download outside an opt-in `FIXIMG_BAKE_WEIGHTS` guard.

    `docker/worker.Dockerfile` fetched the restoration checkpoints unconditionally, so
    both `images (worker)` jobs for v3.0.0 died on `facevc.blob.core.windows.net` not
    resolving -- before the Dockerfile itself had been exercised at all. THIRD_PARTY_
    NOTICES.md also records this distribution as code-only, with two of the six artifacts
    restricted to research/non-commercial use, so baking them into a published image was
    never the shape the project licenses itself to ship.
    """
    instructions = _dockerfile_instructions(path)
    fetching = [
        text for text in instructions
        if "download_weights" in text and text.upper().startswith("RUN")
    ]
    if os.path.basename(path) == "api.Dockerfile":
        assert not fetching, "the API image runs no inference and must fetch no weights"
    for text in fetching:
        assert "FIXIMG_BAKE_WEIGHTS" in text, (
            f"{os.path.relpath(path, PROJECT_ROOT)} fetches weights in an unconditional "
            "RUN step, which makes every build depend on an upstream host: " + text[:160]
        )
    if fetching:
        arguments = [text for text in instructions if text.upper().startswith("ARG")]
        assert any(
            argument.strip().split() == ["ARG", "FIXIMG_BAKE_WEIGHTS=false"]
            for argument in arguments
        ), (
            f"{os.path.relpath(path, PROJECT_ROOT)} guards the download, but not with an "
            "off-by-default ARG -- so CI and the release build still reach for the network"
        )


def _dockerfile_instructions(path: str) -> list[str]:
    """The file's instructions, with backslash continuations joined.

    Needed because a `#` inside a continued RUN is *not* a shell comment: it would eat the
    rest of the line and quietly truncate the download to half its steps.
    """
    with open(path, encoding="utf-8") as handle:
        raw = handle.read()
    instructions: list[str] = []
    current: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not current and (not stripped or stripped.startswith("#")):
            continue
        if stripped.endswith("\\"):
            current.append(stripped[:-1].strip())
            continue
        current.append(stripped)
        instructions.append(" ".join(part for part in current if part))
        current = []
    return instructions


#: A transport name, and what a compose file says to set to select it. The set is
#: scraped from the deployment declarations rather than listed here: an image that
#: cannot reach a backend the document offers is the bug, so the document is the source.
TRANSPORT_SELECTORS = {
    "postgres": ("postgresql+psycopg://",),
    "redis": ("redis://",),
    "s3": ("FIXIMG_STORAGE_BACKEND=s3",),
}


def _compose_files_building(dockerfile: str) -> list[str]:
    """Which compose files build this Dockerfile -- resolved the way Docker resolves it."""
    target = os.path.normpath(dockerfile)
    builders: list[str] = []
    for path, document in _documents():
        for service in (document.get("services") or {}).values():
            build = (service or {}).get("build")
            if not isinstance(build, dict):
                continue
            relative = build.get("dockerfile") or "Dockerfile"
            context = os.path.join(os.path.dirname(path), build.get("context") or ".")
            if os.path.normpath(os.path.join(context, relative)) == target:
                builders.append(path)
                break
    return builders


def _offered_transports(paths: list[str]) -> set[str]:
    """Transports a deployment tells an operator they may select.

    Scraped from those files rather than listed per image, so an image is only held to the
    backends its own topology offers.
    """
    offered: set[str] = set()
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        for extra, markers in TRANSPORT_SELECTORS.items():
            if any(marker in text for marker in markers):
                offered.add(extra)
    return offered


def _installed_extras(path: str) -> set[str]:
    """Extras the Dockerfile asks pip to install, from every `.[a,b]` requirement."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    found = re.findall(r'pip install[^\n&|;]*?"\.?\[([a-z0-9,_ -]+)\]"', text)
    return {piece.strip() for group in found for piece in group.split(",") if piece.strip()}


def test_an_image_can_reach_every_transport_its_own_topology_offers():
    """`--profile platform` documents PostgreSQL, Redis and MinIO; the images had no driver.

    `docker/api.Dockerfile` installed the `test` extra -- pytest, moto and fakeredis in a
    published image -- while `psycopg` was in none of them, so the documented
    `FIXIMG_DATABASE_URL=postgresql+psycopg://postgres:5432/fiximg` failed on import the
    moment an operator tried it.
    """
    checked = 0
    for path in DOCKERFILES:
        builders = _compose_files_building(path)
        offered = _offered_transports(builders)
        installed = _installed_extras(path)
        if installed & {"test", "dev"}:
            raise AssertionError(
                f"{os.path.relpath(path, PROJECT_ROOT)} ships test/dev tooling "
                f"in a runtime image: {sorted(installed & {'test', 'dev'})}"
            )
        if not offered:
            continue
        checked += 1
        missing = sorted(offered - installed)
        assert not missing, (
            f"{os.path.relpath(path, PROJECT_ROOT)} is built by "
            f"{[os.path.basename(p) for p in builders]}, which offers {sorted(offered)}, "
            f"but the image installs {sorted(installed) or 'nothing'}; missing: {missing}"
        )
    assert checked, "no image is offered a transport, so this check would prove nothing"


def test_every_image_upgrades_its_own_install_tooling():
    """A published image carries the build tooling it was built with.

    Trivy first scanned the API image on 2026-09-29 and immediately found four
    advisories in the base image's own pip tooling rather than in anything this
    project imports:

        setuptools 70.3.0     CVE-2025-47273
        wheel      0.45.1     CVE-2026-24049
        jaraco.context 5.3.0  CVE-2026-23949  (vendored inside setuptools)
        msgpack    1.1.2      GHSA-6v7p-g79w-8964

    `pip install --upgrade pip` does not touch setuptools or wheel, so each
    image names them explicitly. A gate because the failure is invisible until a
    scan runs, and only the API image is scanned -- the worker's copy of this
    problem is fixed on the same reasoning and only a release build will prove it.
    """
    paths = sorted(glob.glob(os.path.join(PROJECT_ROOT, "*Dockerfile"))) + sorted(
        glob.glob(os.path.join(PROJECT_ROOT, "docker", "*Dockerfile"))
    )
    assert len(paths) >= 3, f"expected the root, api and worker images; found {paths}"
    for path in paths:
        text = open(path, encoding="utf-8", errors="replace").read()
        installs = " ".join(
            line for line in text.splitlines() if line.strip().startswith("RUN pip install")
        )
        if not installs:
            continue
        name = os.path.relpath(path, PROJECT_ROOT)
        assert "setuptools" in installs, (
            f"{name} installs without upgrading setuptools, so the base image's copy"
            " ships in it (CVE-2025-47273 at 70.3.0)"
        )
        assert "wheel" in installs, (
            f"{name} installs without upgrading wheel, so the base image's copy ships"
            " in it (CVE-2026-24049 at 0.45.1)"
        )
