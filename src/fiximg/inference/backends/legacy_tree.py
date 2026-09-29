"""Which vendored legacy tree this process hosts natively (plan §3.13, §4.2 Step 2).

``Global/`` and ``Face_Enhancement/`` each expose ``options`` / ``models`` /
``util`` / ``data`` as **top-level** packages, so a single interpreter can host the
imports of only one of them.

The obvious fix — a temporary ``sys.path`` / ``sys.modules`` swap per tree — does
not hold here, and this is the reason recorded once instead of rediscovered:
both trees resolve their own submodules *at runtime*, e.g.
``Face_Enhancement/models/__init__.py:11`` and
``Face_Enhancement/util/util.py:149`` call ``importlib.import_module(...)`` with a
computed name. A module imported during such a call binds to whichever tree owns
the name *at that moment*, which after a swap has been reverted is possibly the
wrong tree — a silent miswire rather than an error.

So the boundary is the process, which is also how the plan deploys it (§3.13:
"GPU0 → restore worker, GPU1 → face/color worker"). A worker declares its tree with
``FIXIMG_NATIVE_TREE``; the other chain keeps running through
:class:`~fiximg.inference.backends.legacy_cli.LegacyCliBackend`, unchanged.
"""
from __future__ import annotations

import os
import sys

from fiximg.paths import PROJECT_ROOT

#: The trees a process can host natively. ``none`` keeps everything on subprocesses.
TREES = ("global", "face", "none")

#: Top-level package names the two legacy trees fight over.
SHARED_PACKAGES = ("options", "models", "util", "data")

GLOBAL_ROOT = os.path.join(PROJECT_ROOT, "Global")
FACE_ROOT = os.path.join(PROJECT_ROOT, "Face_Enhancement")

_ROOTS = {"global": GLOBAL_ROOT, "face": FACE_ROOT}


def tree_root(tree: str) -> str:
    """Absolute directory of a legacy tree (empty for an unknown name)."""
    return _ROOTS.get(tree, "")


def declared_tree() -> str:
    """The configured tree ownership: ``FIXIMG_NATIVE_TREE`` (default ``global``)."""
    try:
        from fiximg.config import settings

        value = str(getattr(settings, "native_tree", "global") or "global")
    except Exception:  # noqa: BLE001 — an unreadable setting must not break inference
        value = ""
    value = value.strip().lower()
    return value if value in TREES else "global"


def imported_from(tree: str) -> str | None:
    """A shared package name already loaded from ``tree``'s directory, if any."""
    root = tree_root(tree)
    if not root:
        return None
    for name in SHARED_PACKAGES:
        origin = getattr(sys.modules.get(name), "__file__", None)
        if origin and os.path.abspath(str(origin)).startswith(root + os.sep):
            return name
    return None


def foreign_tree_loaded(tree: str) -> str | None:
    """The package name the *other* legacy tree currently owns, if any.

    Narrowly scoped on purpose: a same-named third-party package (nothing imports
    a top-level ``models`` here, but a dependency could) is not our conflict, and
    treating it as one would disable native inference for no reason.
    """
    return imported_from("face" if tree == "global" else "global")


def owns(tree: str) -> bool:
    """Whether ``tree`` may load its networks in this process.

    False when the deployment declared a different tree, when the tree was set to
    ``none``, or when the other tree's packages are already imported — in every
    such case the caller falls back to the subprocess adapter and says so.
    """
    if tree not in TREES:
        return False
    declared = declared_tree()
    if declared == "none" or declared != tree:
        return False
    return foreign_tree_loaded(tree) is None


def ownership_report() -> dict:
    """What this process owns, for the models API and ``/health/ready``."""
    return {
        "declared": declared_tree(),
        "global_native": owns("global"),
        "face_native": owns("face"),
        "conflict": foreign_tree_loaded("global") or foreign_tree_loaded("face"),
    }


__all__ = [
    "GLOBAL_ROOT",
    "FACE_ROOT",
    "SHARED_PACKAGES",
    "TREES",
    "declared_tree",
    "foreign_tree_loaded",
    "imported_from",
    "owns",
    "ownership_report",
    "tree_root",
]
