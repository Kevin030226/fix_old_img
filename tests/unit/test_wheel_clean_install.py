"""Real build/install contract tests for the wheel (v3.1.2 release gate).

`test_review_regressions_release.py` pins the path-resolution rules in memory.
This file proves them against an artifact: it builds the wheel with the project
build backend, inspects its contents, installs it into throw-away virtual
environments and runs the console scripts, the resource lookup and a minimal app
start there.

The contract it pins (see `docs/deployment.md`, section "Wheel installs and the
runtime resource tree"): the wheel is a **code-only** package. It carries
`fiximg/` and metadata, never `config/`, `models/`, `Global/`,
`Face_Enhancement/` or weight files. A non-editable install therefore requires
`FIXIMG_PROJECT_ROOT` to point at a deployment tree; without it the startup gate
`fiximg.paths.require_project_root()` fails with one actionable error. The
project does not claim that a bare wheel is standalone.

Run with `FIXIMG_WHEEL_TEST=1`; it is skipped in the ordinary unit run because it
needs a build backend and creates two virtual environments.
"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.environ.get("FIXIMG_WHEEL_TEST") != "1",
    reason="set FIXIMG_WHEEL_TEST=1 to run the real build/install contract",
)


_LEAKY_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE")


def _clean_env(extra=None):
    """The test process's environment minus anything that could leak the source tree.

    Without this, `import fiximg` inside the throw-away venv resolves to the
    checkout through the inherited PYTHONPATH and the test proves nothing.
    """
    env = {k: v for k, v in os.environ.items() if k not in _LEAKY_ENV}
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if extra:
        env.update(extra)
    return env


def _run(args, *, cwd=None, env=None, timeout=600):
    return subprocess.run(
        [str(a) for a in args], cwd=str(cwd) if cwd else None,
        env=_clean_env(env), capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )


@pytest.fixture(scope="module")
def dist(tmp_path_factory):
    out = tmp_path_factory.mktemp("dist")
    result = _run([sys.executable, "-m", "build", "--wheel", "--sdist", "--outdir", out],
                  cwd=PROJECT_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = sorted(out.glob("fiximg-*.whl"))
    sdists = sorted(out.glob("fiximg-*.tar.gz"))
    assert len(wheels) == 1 and len(sdists) == 1, (wheels, sdists)
    return {"wheel": wheels[0], "sdist": sdists[0], "dir": out}


def _venv(base: Path, name: str, *, system_site: bool = False) -> Path:
    target = base / name
    args = [sys.executable, "-m", "venv"]
    if system_site:
        args.append("--system-site-packages")
    args.append(str(target))
    result = _run(args)
    assert result.returncode == 0, result.stdout + result.stderr
    return target


def _python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _scripts(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin")


def _install(wheel: Path, venv: Path):
    return _run([_python(venv), "-m", "pip", "install", "--no-deps", "--no-index",
                 "--disable-pip-version-check", str(wheel)])


@pytest.fixture(scope="module")
def bare_venv(tmp_path_factory, dist):
    venv = _venv(tmp_path_factory.mktemp("bare"), "venv")
    result = _install(dist["wheel"], venv)
    assert result.returncode == 0, result.stdout + result.stderr
    return venv


# ------------------------------------------------------------------ content
def test_the_wheel_contains_code_only(dist):
    """No runtime resources in the wheel: the contract this file documents."""
    names = zipfile.ZipFile(dist["wheel"]).namelist()
    assert names, "empty wheel"
    for name in names:
        assert name.startswith(("fiximg/", "fiximg-3.1.2.dist-info/")), name
    forbidden_prefixes = ("config/", "configs/", "models/", "Global/",
                          "Face_Enhancement/", "Face_Detection/", "basicsr/",
                          "ddcolor/", "tests/")
    offenders = [n for n in names if n.startswith(forbidden_prefixes)]
    assert not offenders, f"runtime resources were packaged: {offenders[:10]}"
    weight_files = [n for n in names if n.lower().endswith((".pth", ".pt", ".dat", ".onnx"))]
    assert not weight_files, f"weights were packaged: {weight_files[:5]}"


def test_the_sdist_carries_no_model_weights(dist):
    import tarfile

    with tarfile.open(dist["sdist"]) as archive:
        names = archive.getnames()
    weight_files = [n for n in names if n.lower().endswith((".pth", ".pt", ".dat", ".onnx"))]
    assert not weight_files, f"weights were packaged in the sdist: {weight_files[:5]}"


# -------------------------------------------------------------- bare install
def test_the_installed_package_imports_from_the_venv(bare_venv):
    result = _run([_python(bare_venv), "-c",
                   "import fiximg, sys; print(fiximg.__file__); print(sys.prefix)"])
    assert result.returncode == 0, result.stdout + result.stderr
    first = result.stdout.splitlines()[0]
    assert str(bare_venv) in first, first


def test_the_console_scripts_are_installed_and_runnable(bare_venv):
    suffix = ".exe" if os.name == "nt" else ""
    weights = _scripts(bare_venv) / ("fiximg-weights" + suffix)
    verify = _scripts(bare_venv) / ("fiximg-verify-weights" + suffix)
    assert weights.exists(), list(_scripts(bare_venv).iterdir())
    assert verify.exists(), list(_scripts(bare_venv).iterdir())

    help_result = _run([weights, "help"])
    assert help_result.returncode == 2, help_result.stdout + help_result.stderr
    combined = (help_result.stderr + help_result.stdout).lower()
    assert "usage" in combined


def test_without_project_root_the_startup_gate_fails_actionably(bare_venv):
    env = {k: v for k, v in _clean_env().items() if k != "FIXIMG_PROJECT_ROOT"}
    result = _run([_python(bare_venv), "-c",
                   "from fiximg.paths import require_project_root; require_project_root()"],
                  env=env)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "FIXIMG_PROJECT_ROOT" in combined, combined
    assert "config" in combined and "models" in combined, combined


def test_with_project_root_the_gate_and_manifest_resolution_work(bare_venv):
    env = {"FIXIMG_PROJECT_ROOT": str(PROJECT_ROOT)}
    code = (
        "from fiximg.paths import require_project_root\n"
        "from fiximg.infrastructure.models import weights_check\n"
        "from fiximg.cli import download_weights\n"
        "print(require_project_root())\n"
        "print(weights_check.MANIFEST_PATH)\n"
        "import os\n"
        "assert os.path.exists(weights_check.MANIFEST_PATH)\n"
    )
    result = _run([_python(bare_venv), "-c", code], env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(PROJECT_ROOT) in result.stdout


# ------------------------------------------------------- minimal app start
def test_a_minimal_app_starts_in_a_dependency_provided_environment(tmp_path_factory, dist):
    """FastAPI itself comes from the host env; the wheel is installed --no-deps.

    The point is not dependency resolution (the lock files own that) but that the
    built artifact plus a deployment tree can construct and serve the app.
    """
    venv = _venv(tmp_path_factory.mktemp("app"), "venv", system_site=True)
    result = _install(dist["wheel"], venv)
    assert result.returncode == 0, result.stdout + result.stderr

    work = tmp_path_factory.mktemp("apprun")
    env = {
        "FIXIMG_PROJECT_ROOT": str(PROJECT_ROOT),
        "FIXIMG_API_TOKEN": "smoke-token",
        "FIXIMG_STORAGE_ROOT": str(work / "storage"),
        "FIXIMG_TASKS_ROOT": str(work / "tasks"),
        "FIXIMG_DATABASE_URL": "sqlite:///" + str(work / "smoke.db").replace(os.sep, "/"),
    }
    code = (
        "from fastapi.testclient import TestClient\n"
        "from fiximg.app_factory import create_app\n"
        "app = create_app()\n"
        "with TestClient(app) as client:\n"
        "    print('LIVE', client.get('/api/v1/health/live').status_code)\n"
    )
    result = _run([_python(venv), "-c", code], env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LIVE 200" in result.stdout, result.stdout + result.stderr
