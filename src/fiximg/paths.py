"""Single source of truth for filesystem locations (V3).

Before V3 every module computed its own project root with a fragile chain of
``os.path.dirname(os.path.abspath(__file__))`` calls whose depth depended on
where the file happened to live. Moving a module silently changed the root.

All path-dependent modules must import from here instead:

    from fiximg.paths import PROJECT_ROOT, CONFIG_DIR, MODELS_DIR

The one exception is the *runtime* roots. `STORAGE_ROOT` used to live here, and
`config.Settings` grew `storage_root` / `tasks_root` from the same
`FIXIMG_STORAGE_ROOT` / `FIXIMG_TASKS_ROOT` variables — two declarations of one
knob, of which only the `Settings` one was read. The writable roots are therefore
owned by `config` (they are deployment configuration, and they resolve relative
to `base_dir` so two processes with different working directories still see the
same bytes), and this module keeps only the roots that are a property of the
checkout rather than of a deployment.
"""
from __future__ import annotations

import os

#: Repository root — ``<root>/src/fiximg/paths.py`` → up three levels.
PROJECT_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

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
