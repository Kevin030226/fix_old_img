"""Layering rules the plan states as acceptance criteria (附录 B 架构 1-3).

The report asks for three properties and leaves them as prose:

* the API does not call a concrete model implementation,
* the UI does not call pipeline internals,
* a Stage does not depend on HTTP / Gradio / the database.

Prose rots. The third one already had a violation, the second had one too, and
nothing said so — a module can acquire an import in a later edit and the boundary
quietly moves. So the rules are checked here by reading the import statements of
every file in the layer, with no fixtures and no app to start: cheap enough to
run on every commit.

Deliberate scope: **module paths only**, not attribute access. A layer may import
a *protocol* or a domain type from another package (that is how dependencies
point inward); what it may not import is the named implementation module.
"""
import ast
import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(PROJECT_ROOT, "src")

#: package -> modules it must never import. Checked by prefix, so
#: `fiximg.inference.backends` also forbids `...backends.legacy_cli.list_images`.
#:
#: The lists encode the three acceptance rules and nothing more:
#:
#: * "UI 不直接调用 pipeline internals" — the whole of `fiximg.inference` is
#:   pipeline internals as far as a Gradio view is concerned; it talks to
#:   `fiximg.application`.
#: * "API 不直接调用具体模型实现" — *concrete implementations* means the backend
#:   and stage modules. `health`/`stats` reading `model_manager`, `scheduler` and
#:   `gpu_memory` is a read-only diagnostic and the plan asks for it (§3.9), so
#:   the rule is scoped to `inference.backends` / `inference.stages` rather than
#:   to the package.
#: * "Stage 不依赖 HTTP / Gradio / DB" — including the queue: a stage that
#:   acknowledges its own task would race the worker that owns the lease.
#:
#: `fiximg.infrastructure.db` is deliberately **not** forbidden for the UI. The
#: auth pages, the middleware and `ui/state.py` read `user_repository` directly
#: to answer "who is this caller". Encoding a rule the code does not follow turns
#: the gate into a permanent waiver, so it is recorded here as a known shape
#: instead — the acceptance list does not ask for it.
FORBIDDEN: dict[str, tuple[str, ...]] = {
    "fiximg.ui": ("fiximg.inference",),
    "fiximg.api": ("fiximg.inference.backends", "fiximg.inference.stages"),
    "fiximg.inference.stages": (
        "fastapi",
        "gradio",
        "fiximg.api",
        "fiximg.ui",
        "fiximg.infrastructure.db",
        "fiximg.infrastructure.queue",
    ),
}


def _module_name(path: str) -> str:
    relative = os.path.relpath(path, SRC)
    stem = relative[:-3] if relative.endswith(".py") else relative
    parts = stem.split(os.sep)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _files_in(package: str):
    root = os.path.join(SRC, *package.split("."))
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def _imported_names(path: str) -> set[str]:
    """Top-level module names an import statement refers to.

    `from a.b.c import d` contributes both `a.b.c` and `a.b.c.d`, because either
    can be the implementation the rule is about.
    """
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.module is None:  # `from . import x`
                continue
            base = node.module or ""
            if node.level:  # relative import inside the same package
                prefix = _module_name(path).split(".")[: -node.level + 1]
                base = ".".join([*prefix, base]) if base else ".".join(prefix)
            found.add(base)
            for alias in node.names:
                found.add(f"{base}.{alias.name}" if base else alias.name)
    return found


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_layer_never_imports_what_it_must_not_know_about(package):
    forbidden = FORBIDDEN[package]
    offenders = []
    for path in _files_in(package):
        for imported in sorted(_imported_names(path)):
            if any(
                imported == banned or imported.startswith(banned + ".")
                for banned in forbidden
            ):
                offenders.append(f"{_module_name(path)} -> {imported}")
    assert not offenders, (
        f"{package} crosses a boundary the plan's acceptance list names "
        f"(forbidden: {forbidden}):\n  " + "\n  ".join(offenders)
    )


def test_a_stage_receives_its_context_and_returns_a_result_without_touching_persistence():
    """The rule the parametrised check above exists to protect.

    Asserted once explicitly, so a future edit that *narrows* the forbidden list
    cannot pass every other test while silently allowing a stage to write the DB.
    """
    import inspect

    from fiximg.inference.stages.base import BaseStage

    source = inspect.getsource(inspect.getmodule(BaseStage))
    assert "import sqlite3" not in source
    assert "from fiximg.infrastructure.db" not in source
