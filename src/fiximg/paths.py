"""Single source of truth for filesystem locations (V3).

Before V3 every module computed its own project root with a fragile chain of
``os.path.dirname(os.path.abspath(__file__))`` calls whose depth depended on
where the file happened to live. Moving a module silently changed the root.

All path-dependent modules must import from here instead:

    from fiximg.paths import PROJECT_ROOT, CONFIG_DIR, MODELS_DIR

Package vs. deployment tree (the contract):

``python -m build`` produces a **code-only wheel** — ``fiximg/`` and nothing
else. The runtime resources (``config/weights_manifest.json``,
``models/manifest.yaml``, the ``Global``/``Face_Enhancement`` trees and the
weight files themselves) are deliberately not packaged: they are large, partly
licensed for research use only, and already deployed as a separate tree. A
non-editable install therefore needs ``FIXIMG_PROJECT_ROOT`` pointing at the
directory that holds those assets; a source checkout needs nothing.

:func:`require_project_root` is the startup gate that turns "the wheel cannot
see its resources" into one actionable error instead of a missing-manifest
traceback from deep inside a loader.

The one exception to this module owning roots is the *runtime* writable roots.
`STORAGE_ROOT` used to live here, and `config.Settings` grew `storage_root` /
`tasks_root` from the same `FIXIMG_STORAGE_ROOT` / `FIXIMG_TASKS_ROOT`
variables — two declarations of one knob, of which only the `Settings` one was
read. The writable roots are therefore owned by `config`, and this module keeps
only the roots that are a property of the deployment tree rather than of a
process.
"""
from __future__ import annotations

import os

#: Files that identify a deployment tree. Deliberately files, not just directory
#: names: ``config`` and ``models`` are common enough that a shared
#: ``site-packages`` could otherwise be mistaken for a fiximg deployment. Both
#: are small committed files that every deployment tree carries.
PROJECT_ROOT_MARKERS = (
    os.path.join("config", "weights_manifest.json"),
    os.path.join("models", "manifest.yaml"),
)


class ProjectRootNotFoundError(RuntimeError):
    """The runtime resource tree could not be located or is incomplete."""


def _looks_like_project_root(candidate: str) -> bool:
    """True when ``candidate`` carries the files a deployment tree needs."""
    if not candidate or not os.path.isdir(candidate):
        return False
    return all(
        os.path.isfile(os.path.join(candidate, name)) for name in PROJECT_ROOT_MARKERS
    )


def _resolve_project_root(module_file: str | None = None) -> str:
    """Where the deployment's runtime assets live, for a checkout or an install.

    Resolution order:

    1. ``FIXIMG_PROJECT_ROOT`` — an explicit answer always wins. This is the
       supported way to run a standard (non-editable) wheel install: point it at
       the directory that holds ``config/``, ``models/``, ``Global/`` and the
       rest.
    2. the ``src``-layout checkout this module was imported from
       (``<root>/src/fiximg/paths.py`` — up three levels);
    3. the directory containing the installed ``fiximg`` package, when the assets
       sit next to it (a flattened / ``--target`` install);
    4. the same directory's parent (``site-packages``), when the tree was
       installed there as wheel data files — this project does not package them
       today, but an operator-built layout may;
    5. the three-levels-up value anyway, so ``CONFIG_DIR`` and friends are always
       strings; :func:`require_project_root` is what refuses to start when that
       guess does not contain the assets.

    ``PROJECT_ROOT`` used to be step 2 unconditionally, which is why the
    validation report found a wheel install looking for
    ``config/weights_manifest.json`` under the environment prefix.
    """
    override = os.environ.get("FIXIMG_PROJECT_ROOT", "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))

    here = os.path.abspath(module_file or __file__)
    checkout = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    if _looks_like_project_root(checkout):
        return checkout

    package_dir = os.path.dirname(here)
    if _looks_like_project_root(package_dir):
        return package_dir

    site_packages = os.path.dirname(package_dir)
    if _looks_like_project_root(site_packages):
        return site_packages

    return checkout


def require_project_root(purpose: str = "runtime resources") -> str:
    """Return the deployment tree, or raise one actionable error.

    Called early by every entry point that needs the assets. A wheel install
    that cannot see ``config/`` must fail here, with the variable name and an
    example, rather than much later with a missing-manifest traceback from a
    loader that has no context.
    """
    root = PROJECT_ROOT
    missing = [
        name for name in PROJECT_ROOT_MARKERS
        if not os.path.isfile(os.path.join(root, name))
    ]
    if missing:
        detail = ", ".join(repr(name) for name in missing)
        raise ProjectRootNotFoundError(
            f"fiximg cannot find its {purpose}: {root!r} is missing "
            f"{detail}. A wheel install ships code only - set "
            "FIXIMG_PROJECT_ROOT to a deployment tree that contains "
            "config/weights_manifest.json and models/manifest.yaml (inference "
            "deployments additionally need the Global/, Face_Enhancement/ and "
            "Face_Detection/ trees and the downloaded weights; see "
            "docs/deployment.md, section: Wheel installs and the runtime "
            "resource tree)."
        )
    return root


#: Deployment root — ``FIXIMG_PROJECT_ROOT``, else the checkout, else the install.
PROJECT_ROOT: str = _resolve_project_root()

#: Runtime data produced by the application (SQLite file, api token, archives).
DATA_DIR: str = os.path.join(PROJECT_ROOT, "admin_data")

#: Static, versioned configuration assets (weight manifest, example users).
CONFIG_DIR: str = os.path.join(PROJECT_ROOT, "config")

#: Declarative application configuration (base/local/production profiles).
CONFIGS_DIR: str = os.path.join(PROJECT_ROOT, "configs")

#: Model weight tree + manifest (plan §6's ``models/<name>/<version>/`` layout).
#: Read through `inference.manifest.models_root()` rather than imported, so a
#: test can relocate the checkout root and have the versioned-layout resolution
#: follow it.
MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")


def ensure_legacy_importable() -> None:
    """Make the vendored model packages importable from a src-layout install.

    With ``src/fiximg`` on the path the repository root is no longer implicitly
    on ``sys.path``, but ``Global``/``ddcolor``/... are still plain top-level
    packages next to it.
    """
    import sys

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
