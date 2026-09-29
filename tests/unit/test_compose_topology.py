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
