"""Regression test for the multi-GPU compose device-mapping defect.

Confirmed by Docker's documented semantics, not by running Docker (none is
installed on this machine): deploy.resources.reservations.devices with
device_ids: ["1"] exposes *only that card* inside the container, and CUDA
renumbers it -- the card the host calls 1 is cuda:0 in there. The shipped
docker/compose.multigpu.yaml requested device id 1 for worker-color while
also setting FIXIMG_WORKER_GPU: "1" and a routing table naming device 1, so
that worker asked torch for a second device the container cannot see and
every colourisation failed with "invalid device ordinal".

The invariant pinned here is the one that makes the topology coherent: a service
that reserves exactly one device by id must address it as device 0 inside the
container, in both knobs that name a device.
"""
from __future__ import annotations

import glob
import os

import pytest

yaml = pytest.importorskip("yaml", reason="pyyaml not installed")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPOSE_FILES = sorted(
    glob.glob(os.path.join(PROJECT_ROOT, "docker-compose.yml"))
    + glob.glob(os.path.join(PROJECT_ROOT, "docker", "compose*.yaml"))
)


def _reserved_device_ids(service: dict) -> list[str]:
    ids: list[str] = []
    deploy = service.get("deploy") or {}
    reservations = (deploy.get("resources") or {}).get("reservations") or {}
    for entry in reservations.get("devices") or []:
        ids.extend(str(value) for value in entry.get("device_ids") or [])
    return ids


def _routed_devices(raw: str, capabilities: set[str] | None = None) -> set[str]:
    """Device numbers a routing table names, optionally only for these capabilities.

    A table may describe devices this worker never uses; what matters is the
    device a capability *this* worker serves would resolve to.
    """
    found: set[str] = set()
    for chunk in (raw or "").split(";"):
        device, _sep, caps = chunk.partition(":")
        device = device.strip()
        if not device:
            continue
        served = {c.strip() for c in caps.split(",") if c.strip()}
        if capabilities is None or (served & capabilities):
            found.add(device)
    return found


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: os.path.basename(p))
def test_a_single_reserved_device_is_addressed_as_device_zero(path):
    """device_ids: ["1"] does not mean cuda:1 inside the container."""
    with open(path, encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}

    offenders: list[str] = []
    for name, service in (document.get("services") or {}).items():
        service = service or {}
        ids = _reserved_device_ids(service)
        if len(ids) != 1:
            continue                     # nothing reserved, or a whole range
        environment = service.get("environment") or {}
        pinned = str(environment.get("FIXIMG_WORKER_GPU", "")).strip()
        if pinned and pinned != "0":
            offenders.append(
                f"{name}: reserves device id {ids[0]} but pins FIXIMG_WORKER_GPU={pinned!r}; "
                "inside the container that card is device 0"
            )
        routing = str(environment.get("FIXIMG_GPU_ROUTING", "")).strip()
        if routing:
            served = {
                c.strip()
                for c in str(environment.get("FIXIMG_WORKER_CAPABILITIES", "")).split(",")
                if c.strip()
            }
            named = _routed_devices(routing, served or None)
            if named - {"0"}:
                offenders.append(
                f"{name}: reserves only device id {ids[0]} but routes its capabilities to "
                f"device(s) {sorted(named - {'0'})}, which do not exist in the container"
                )

    assert not offenders, f"{os.path.basename(path)}: {offenders}"


def test_the_multigpu_topology_ships_a_colour_worker():
    """Guard the file the check above is about: an empty topology proves nothing."""
    path = os.path.join(PROJECT_ROOT, "docker", "compose.multigpu.yaml")
    assert os.path.exists(path), path
    with open(path, encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    services = document.get("services") or {}
    assert any(
        "colorize" in str((svc.get("environment") or {}).get("FIXIMG_WORKER_CAPABILITIES", ""))
        for svc in services.values()
    ), "no colour worker in the multi-GPU topology"
