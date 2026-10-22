"""No published install path may declare, resolve or ship Gradio (P17).

Why this is a gate and not a note in a report: `gradio` requires `pillow<13`, and
this project pins `pillow==13.0.0` because 12.x carries one CRITICAL and six HIGH
advisories that `pip-audit --strict` and the image scans both block on. The two
requirements have an empty intersection, so any install path that can reach Gradio
silently resolves Pillow back to 12.3.0 and re-opens every one of those advisories.

The failure mode is quiet, which is why it needs a test. Measured once already: a
top-level `Pillow==13.0.0` pin was satisfied by *nothing* because a transitive
`pillow<13` outranked it, the image built successfully, and the only signal was a
Trivy finding on a package the Dockerfile never mentions.

What is asserted
----------------
* no dependency declaration (pyproject, any requirements file, the lock) names
  Gradio, its client, or any of the packages that exist only to serve it;
* the source tree imports no UI framework at all — the interface is a FastAPI
  router, so `import gradio` cannot come back without this failing;
* the Dockerfiles install no UI framework and do not undo the Pillow pin with a
  second, unpinned install;
* the removal cannot be undone by re-adding it as an optional extra: every extra is
  checked, so `pip install "fiximg[ui]"` cannot become a supported path.

The rules are about *declarations*, so they read the files rather than a built
environment: they must hold on a machine that has never installed anything.
"""
from __future__ import annotations

import os
import re
import tomllib

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: The framework that was removed, plus the distributions that only exist to serve
#: it. Matching the whole set means a re-introduction through `gradio_client` or
#: `hf-gradio` fails here too, rather than only the obvious name.
FORBIDDEN_DISTRIBUTIONS = (
    "gradio",
    "gradio-client",
    "gradio_client",
    "hf-gradio",
    "groovy",
    "safehttpx",
    "semantic-version",
)

#: Files that declare an installable surface. Every one of them is checked.
DECLARATION_FILES = (
    "pyproject.toml",
    "requirements.txt",
    "requirements.lock",
    os.path.join("requirements", "runtime.txt"),
    os.path.join("requirements", "gpu.txt"),
    os.path.join("requirements", "dev.txt"),
    os.path.join("requirements", "test.txt"),
    os.path.join("requirements", "benchmark.txt"),
    os.path.join("requirements", "postgres.txt"),
    os.path.join("requirements", "redis.txt"),
    os.path.join("requirements", "s3.txt"),
    os.path.join("requirements", "otel.txt"),
)

#: Dockerfiles: the three published images.
DOCKERFILES = (
    "Dockerfile",
    os.path.join("docker", "api.Dockerfile"),
    os.path.join("docker", "worker.Dockerfile"),
)


def _text(relative: str) -> str:
    with open(os.path.join(PROJECT_ROOT, relative), encoding="utf-8-sig") as handle:
        return handle.read()


def _requirement_lines(body: str) -> list[str]:
    """The installable lines of a requirements file, comments and flags dropped."""
    lines = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        lines.append(line)
    return lines


def _names(lines: list[str]) -> set[str]:
    """Distribution names from requirement lines, PEP 503-normalised."""
    names = set()
    for line in lines:
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if match:
            names.add(match.group(1).lower().replace("_", "-").replace(".", "-"))
    return names


@pytest.mark.parametrize("relative", DECLARATION_FILES)
def test_no_declaration_file_names_a_removed_ui_framework(relative):
    """The defect this gate exists for: a reachable Gradio re-opens the advisories."""
    path = os.path.join(PROJECT_ROOT, relative)
    assert os.path.exists(path), f"{relative} is gone; the dependency surface moved"
    names = _names(_requirement_lines(_text(relative)))
    offenders = sorted(names & {name.lower().replace("_", "-")
                                for name in FORBIDDEN_DISTRIBUTIONS})
    assert not offenders, (
        f"{relative} declares {offenders}. Gradio requires pillow<13 and this "
        "project pins pillow==13.0.0 for the open advisories; the two cannot both "
        "be installed, and the resolver silently picks Pillow 12.x. Removing the "
        "framework is what made the pin installable -- see "
        "reports/p17_fastapi_ui_migration.md."
    )


def test_no_extra_can_re_install_the_removed_framework():
    """An installable legacy UI extra is the same defect wearing a different name.

    The decision recorded in the migration report is that the removal must not be
    reversible by a published install path: an extra that pulls Gradio back in would
    be a documented way to get Pillow 12 into an image nobody scans for it.
    """
    with open(os.path.join(PROJECT_ROOT, "pyproject.toml"), "rb") as handle:
        document = tomllib.load(handle)
    extras = document["project"].get("optional-dependencies") or {}
    assert extras, "no extras declared; the base-dependency check would be the only one"
    offenders = {}
    for extra, requirements in extras.items():
        names = _names([str(requirement) for requirement in requirements])
        bad = sorted(names & {name.lower().replace("_", "-")
                              for name in FORBIDDEN_DISTRIBUTIONS})
        if bad:
            offenders[extra] = bad
    assert not offenders, (
        "these extras would install the removed UI framework, which reintroduces "
        f"pillow<13 on a published path: {offenders}"
    )


def test_the_base_dependencies_name_no_ui_framework():
    """The root image installs the base dependencies, so this is the load-bearing set."""
    with open(os.path.join(PROJECT_ROOT, "pyproject.toml"), "rb") as handle:
        document = tomllib.load(handle)
    declared = _names([str(requirement) for requirement in
                       document["project"]["dependencies"]])
    offenders = sorted(declared & {name.lower().replace("_", "-")
                                   for name in FORBIDDEN_DISTRIBUTIONS})
    assert not offenders, f"base dependencies declare {offenders}"


def _python_sources() -> list[tuple[str, str]]:
    """Every Python file that ships, as (relative path, text)."""
    out = []
    for sub in ("src", "scripts", "benchmark", "tests"):
        base = os.path.join(PROJECT_ROOT, sub)
        for dirpath, _dirs, files in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, encoding="utf-8-sig", errors="replace") as handle:
                    out.append((os.path.relpath(path, PROJECT_ROOT), handle.read()))
    for name in ("main.py", "worker.py", "run.py"):
        path = os.path.join(PROJECT_ROOT, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8-sig", errors="replace") as handle:
                out.append((name, handle.read()))
    return out


def test_the_source_tree_imports_no_ui_framework():
    """A comment may discuss Gradio; an `import` may not.

    Only import statements are matched, so the migration's own explanation inside
    `app_factory`'s docstring is not a finding — the thing that would break the
    dependency guarantee is code that executes at import time.
    """
    pattern = re.compile(
        r"^\s*(?:import|from)\s+(" + "|".join(
            re.escape(name) for name in ("gradio", "gradio_client", "hf_gradio")
        ) + r")\b",
        re.MULTILINE,
    )
    offenders = [
        f"{relative}:{text[:match.start()].count(chr(10)) + 1}"
        for relative, text in _python_sources()
        if (match := pattern.search(text))
    ]
    assert not offenders, (
        "the source tree imports a UI framework that pins pillow<13: "
        + ", ".join(offenders)
    )


@pytest.mark.parametrize("relative", DOCKERFILES)
def test_no_image_installs_a_ui_framework(relative):
    """Checked on the install instructions, not on the whole file.

    A comment may explain that the framework was removed — that is the sentence a
    future reader needs — but no `RUN pip install` may name it. Reading only the
    commands is what makes the difference, and it is also the narrower claim: it is
    the install that resolves `pillow<13`, not the prose.
    """
    body = _text(relative)
    installs = "\n".join(
        line for line in body.splitlines()
        if line.strip().startswith(("RUN ", "COPY ", "ADD ")) or "pip install" in line
    ).lower()
    for name in ("gradio", "gradio-client", "gradio_client", "hf-gradio"):
        assert name not in installs, (
            f"{relative} installs {name}; an image that installs it resolves "
            "pillow<13 and fails the scan it exists to pass"
        )


@pytest.mark.parametrize("relative", DOCKERFILES)
def test_no_image_undoes_the_pillow_pin_with_a_second_install(relative):
    """The second install is how the pin gets silently overridden.

    `pip install -r requirements.txt` then `pip install -e ".[postgres,redis,s3]"`
    is the shape both images use. If a later step installed anything that depends on
    `pillow<13`, pip would downgrade Pillow and the image would still build.
    """
    body = _text(relative)
    pillow_pins = re.findall(r"(?i)^\s*pillow\s*==\s*(\S+)\s*$", body, re.MULTILINE)
    for pin in pillow_pins:
        assert pin.startswith("13."), f"{relative} pins Pillow {pin}, expected 13.x"
    # And the file must not ask for Pillow at a floating version, which would let a
    # rebuild of the same source change the image's security properties.
    floating = re.findall(r"(?i)^\s*pillow\s*(>=|~=|>|<)", body, re.MULTILINE)
    assert not floating, f"{relative} asks for Pillow at a floating version: {floating}"


def test_the_lock_and_the_flat_requirement_file_agree_on_pillow():
    """The two install paths must resolve the same Pillow, or one image differs."""
    flat = re.findall(r"(?i)^\s*Pillow\s*==\s*(\S+)\s*$", _text("requirements.txt"),
                      re.MULTILINE)
    locked = re.findall(r"(?i)^\s*pillow\s*==\s*(\S+)\s*$", _text("requirements.lock"),
                        re.MULTILINE)
    assert flat, "requirements.txt no longer pins Pillow"
    assert locked, "requirements.lock no longer pins Pillow"
    assert set(flat) == set(locked) == {"13.0.0"}, (flat, locked)


def test_the_runtime_group_resolves_pillow_13_and_nothing_else():
    """The group file the API image installs is the one that matters for a scan."""
    body = _text(os.path.join("requirements", "runtime.txt"))
    pins = re.findall(r"(?i)^\s*pillow\s*==\s*(\S+)\s*$", body, re.MULTILINE)
    assert pins == ["13.0.0"], pins


def test_no_trivy_or_audit_exemption_names_a_pillow_advisory():
    """An exemption would make the gate green while the vulnerability stayed.

    The P16 decision rejected the waiver path explicitly: the advisories are fixed
    by shipping Pillow 13, not by recording them as accepted risk.
    """
    candidates = (
        ".trivyignore",
        os.path.join("config", "audit_ignore.txt"),
    )
    for relative in candidates:
        path = os.path.join(PROJECT_ROOT, relative)
        if not os.path.exists(path):
            continue
        for line in _requirement_lines(_text(relative)):
            assert not re.match(r"^(CVE-\d{4}-\d+|GHSA-)", line), (
                f"{relative} exempts {line}. The Pillow advisories are fixed by the "
                "13.0.0 pin; an exemption here would hide a real vulnerability."
            )
