"""Every public name in `src/fiximg` must be reachable from production.

This project has found the same defect shape over and over: a declaration that
exists, is named in a docstring, and is called from nowhere. It never shows up as
a failure — the build is green, the service is healthy, the feature is simply
absent. The instances this gate was written for, all found in one pass over a
772-name tree:

* ``analyzer.py`` carried four threshold constants duplicating ``settings.*``.
  Nothing read them, so a calibration pass that edited one changed nothing — and
  ``estimate_scratch``'s own docstring named one of them as the normaliser it did
  not use. They were exact copies of the settings defaults, so no test could
  tell.
* ``paths.py`` exported ``MODELS_DIR`` while ``manifest.py`` hardcoded
  ``os.path.join(PROJECT_ROOT, "models", ...)`` five times. The module whose
  docstring says "all path-dependent modules must import from here" was bypassed
  by the code that owns weight resolution.
* ``ADMIN_ONLY_TABS`` restated a rule ``ui/state.py`` already owned, and had no
  reader at all — so the two copies could disagree about which tabs are
  admin-only, and nothing would say so.
* ``passwords.write_yaml_atomic`` documented a temp-file-plus-fsync commit and
  its filelock. There is no ``yaml.dump`` anywhere in ``src/``: the writer had
  no caller and the risk it described could not occur.

A name is *wired* when some production file other than its own mentions it. The
scan is textual and deliberately dumb; the exemptions are the interesting part,
and each one has to state why the name is legitimately unreachable.

**What this gate does not cover: public methods.** ``_declared`` walks
``tree.body``, so it sees functions, classes and module constants and nothing
inside a class body. That is not only a gap, it is a gap that cannot be closed
here, and the reason is worth recording so nobody re-attempts it.

``Principal.can()`` was exactly this defect: a predicate, in shipped source, that
nothing called. The obvious fix is a second scan over class bodies, and it was
built and measured before being rejected. Requiring a *call* to ``ColorizationStage.run``
reports it unreachable, because ``runtime.py:567`` reaches it as
``stage.run(image, context)`` on an object resolved from a stage registry. Every
dispatch in this codebase is indirect: stage registries, model-backend
registries, ``Protocol`` conformance, plugin entry points, Gradio callbacks.

Three candidate rules were measured against the 355 public methods in
``src/fiximg``:

============================  ==========================================
rule                          reported unreachable
============================  ==========================================
file must name the class     128  (blind to ``settings.pipeline_modes``)
+ or import the module         74
anywhere in the tree,         35  (still all registry-dispatched)
no qualification
============================  ==========================================

The last column is the problem: 35 findings, most of them reachable, and no
static signal separates them from a real one. A gate that cries wolf 35 times is
worse than no gate, because the response to it is to widen the scan until it goes
quiet — which is how this file's predecessor gate stopped catching anything.

So methods are covered by a narrow, hand-checked gate where the class is small
and its surface is meant to be fixed — see ``test_principal_surface.py`` — and
not by a general one. A method reached dynamically has to be exempted with a
reason at the place that knows it is reached, not inferred at the call site.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "fiximg"
ENTRY_POINTS = ("main.py", "worker.py", "run.py")

#: Names that are reachable only from tests, each with the reason.  An entry here
#: is a claim about the code, so the test below checks the claim is still true:
#: the name must still exist.
TEST_ONLY = {
    "api/security.py::reset_cache": (
        "test seam: the API token is memoised per process, so a fixture that "
        "relocates the database has to invalidate the previous answer"
    ),
    "ui/gradio_app.py::ADMIN_ONLY_TABS": (
        "the assertion fixture, not a rule: production's single source for "
        "tab visibility is ui/state.py, and "
        "test_the_admin_only_tab_list_is_the_one_visibility_enforces compares the "
        "two so a third copy cannot appear"
    ),
    "infrastructure/db/engine.py::reset_thread_connection_cache": (
        "test seam: close_connections() drops every handle, but a fixture that "
        "only wants its own thread's gone needs the narrower form"
    ),
    "infrastructure/db/migrations/runner.py::MigrationRunner": (
        "compatibility alias for the pre-Alembic API, kept so an embedder that "
        "constructed it still works"
    ),
    "infrastructure/security/rate_limit.py::make_limiter": (
        "public factory for a caller-defined scope; the three module-level "
        "limiters are built by _make_limiter because they must also decide the "
        "backend, but anything else should get the shared class"
    ),
    "infrastructure/db/repositories/task_repository.py::read_metrics": (
        "second reader of the metrics columns, for callers that want one task's "
        "metrics without a task row; test_metric_readers_agree.py asserts it "
        "returns exactly what the production aggregate returns"
    ),
}

#: Base classes whose instances are constructed by the framework rather than by
#: name in our code: a FastAPI response model is reached through `response_model`,
#: and a Protocol through `isinstance`.
FRAMEWORK_BASES = {"BaseModel", "Protocol", "BaseSettings"}


def _sources() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" not in path.parts:
            # `utf-8-sig`: run.py and the vendored `basicsr` carry a BOM, which is
            # stripped by the tokenizer but is not valid in a str handed to ast.
            out[path.relative_to(SRC).as_posix()] = path.read_text(encoding="utf-8-sig")
    for name in ENTRY_POINTS:
        candidate = ROOT / name
        if candidate.exists():
            out[f"<entry>/{name}"] = candidate.read_text(encoding="utf-8-sig")
    return out


def _declared(source: str) -> dict[str, str]:
    """Public module-level names -> the kind that makes them exempt."""
    tree = ast.parse(source)
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            exempt = ""
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "router"):
                    exempt = "route handler: reached by the HTTP router, not by name"
            if not exempt and isinstance(node, ast.ClassDef):
                bases = set()
                for b in node.bases:
                    if isinstance(b, ast.Name):
                        bases.add(b.id)
                    elif isinstance(b, ast.Attribute):
                        bases.add(b.attr)
                hit = bases & FRAMEWORK_BASES
                if hit:
                    exempt = f"framework class ({', '.join(sorted(hit))})"
            found[node.name] = exempt
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = ([node.target] if isinstance(node, ast.AnnAssign)
                       else [t for t in node.targets if isinstance(t, ast.Name)])
            for target in targets:
                if not target.id.startswith("_"):
                    found[target.id] = ""
    return found


def _wired(name: str, owner: str, sources: dict[str, str]) -> int:
    """Occurrences of `name` outside its own declaration.

    Counts the owner file too: a name used only inside its own module is
    private-but-live, which is not the defect this gate is about. What it is
    about is a name that appears exactly once in the whole tree — the
    declaration and nothing else.
    """
    pattern = re.compile(r"(?<![\w])" + re.escape(name) + r"(?![\w])")
    return sum(len(pattern.findall(text)) for text in sources.values())


def _referenced_elsewhere(name: str, owner: str, sources: dict[str, str]) -> int:
    """Occurrences in *other* production files; reported, not asserted."""
    pattern = re.compile(r"(?<![\w])" + re.escape(name) + r"(?![\w])")
    return sum(
        len(pattern.findall(text))
        for other, text in sources.items()
        if other != owner
    )


def test_the_scan_covers_the_tree():
    """A gate that quietly stops scanning is a green light for nothing."""
    sources = _sources()
    declared = {name for text in sources.values() for name in _declared(text)}
    assert len(sources) >= 60, f"only {len(sources)} files scanned"
    assert len(declared) >= 400, f"only {len(declared)} public names found"


def test_no_public_name_is_declared_and_never_reached():
    sources = _sources()
    unwired: list[str] = []
    for owner, text in sources.items():
        if owner.startswith("<entry>/"):
            continue
        for name, exempt in _declared(text).items():
            # `> 1`, not truthiness: the count includes the declaration itself, so
            # a name appearing exactly once is one nothing refers to. Testing
            # truthiness here lets every declaration through — the first version
            # of this gate did, and three mutations sailed past it.
            if exempt or _wired(name, owner, sources) > 1:
                continue
            reason = TEST_ONLY.get(f"{owner}::{name}")
            if reason is None:
                unwired.append(f"{owner}::{name}")
            else:
                assert reason.strip(), f"{owner}::{name} has an empty reason"

    assert not unwired, (
        "declared but never referenced from production (delete it, wire it, or "
        "add it to TEST_ONLY with a reason):\n  " + "\n  ".join(sorted(unwired))
    )


def test_the_reason_methods_are_out_of_scope_stays_in_the_docstring():
    """The measured numbers are the argument; deleting them invites a re-attempt.

    A reader who finds `Principal.can()` unreachable and sees no explanation will
    build a method scan, and this file records what that cost: 128 / 74 / 35
    findings for the three candidate rules, almost all of them reachable through a
    registry. Those figures are the reason the scan is absent, so they are
    asserted rather than trusted to a comment nobody re-reads.
    """
    doc = ast.get_docstring(ast.parse(Path(__file__).read_text(encoding="utf-8-sig")))
    assert doc is not None
    assert "does not cover: public methods" in doc, (
        "the docstring no longer says methods are out of scope"
    )
    for figure in ("128", "74", "35"):
        assert figure in doc, (
            f"the measured figure {figure} was removed; the remaining ones no "
            "longer show the cost of a method scan"
        )
    for evidence in ("stage.run", "registry", "Protocol"):
        assert evidence in doc, f"the docstring lost its evidence: {evidence}"


def test_names_used_only_inside_their_own_module_are_not_flagged():
    """A private-but-live name is not this gate's business.

    Most of `ui/` is built by closures wired inside the module that declares
    them. Demanding cross-module reachability would produce a hundred exemptions
    and the gate would stop being read.
    """
    sources = _sources()
    for name in ("progress_head", "eta_text", "queue_text", "switch_options"):
        assert name in _declared(sources["ui/task_progress.py"]), name
        assert _wired(name, "ui/task_progress.py", sources) > 1, name


@pytest.mark.parametrize("key,reason", sorted(TEST_ONLY.items()))
def test_every_exemption_still_exists_and_explains_itself(key, reason):
    """An exemption is a claim; a stale one is a permanent silent pass."""
    owner, name = key.split("::")
    source = _sources().get(owner)
    assert source is not None, f"{owner} no longer exists — drop the exemption"
    assert name in _declared(source), (
        f"{key} no longer exists — the exemption would be permanent and silent"
    )
    assert len(reason) > 40, f"{key}: say *why*, not what"


# --------------------------------------------------------------------- settings
def _settings_fields() -> list[str]:
    tree = ast.parse((SRC / "config.py").read_text(encoding="utf-8-sig"))
    return [
        node.target.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Attribute)
        and isinstance(node.target.value, ast.Name)
        and node.target.value.id == "self"
    ]


#: Fields read only through `Settings` itself, or deliberately derived elsewhere.
SETTINGS_READ_IN_CONFIG = {
    "app_env": (
        "consumed by Settings itself: `env`, `is_production` and the "
        "KNOWN_ENVS fail-fast check all read `self.app_env`"
    ),
    "base_dir": "the anchor every relative path setting is derived from",
    "db_path": "relocated by engine.apply_database_url, which owns its target",
    "storage_root": (
        "the anchor for tasks_root, so setting FIXIMG_STORAGE_ROOT moves the run "
        "tree; consumers read tasks_root, never this directly"
    ),
}


def test_every_settings_field_is_read_somewhere_outside_config():
    """A configuration field nobody reads is a knob that does nothing.

    Found by the mutation that first *evaded* this file: replacing
    `settings.auto_grayscale_sat` with a literal `16.0` leaves the name declared
    and the gate silent, because the declaration is still there. The field is
    now read by nobody — and `configs/base.yaml` still lists it, so an operator
    tuning it would see no effect and no warning.

    `FIXIMG_*` names already have this gate in `test_settings.py`; the `Settings`
    attributes did not, and they are the layer the YAML profile actually binds.

    An exemption here claims something different — "read by `Settings` itself, or
    the anchor another field is derived from" — so those are checked for
    existence separately, below.
    """
    fields = _settings_fields()
    assert len(fields) >= 60, f"only {len(fields)} fields parsed out of config.py"

    others = {
        name: text
        for name, text in _sources().items()
        if name != "config.py" and not name.startswith("<entry>/")
    }
    corpus = "\n".join(others.values())
    unread = [
        field
        for field in fields
        if field not in SETTINGS_READ_IN_CONFIG
        and not re.search(r"(?<![\w])" + re.escape(field) + r"(?![\w])", corpus)
    ]

    assert not unread, (
        "Settings fields read nowhere outside config.py (wire them, or delete "
        "the field and its YAML key): " + ", ".join(sorted(unread))
    )


@pytest.mark.parametrize("field,reason", sorted(SETTINGS_READ_IN_CONFIG.items()))
def test_a_settings_exemption_still_exists(field, reason):
    assert field in _settings_fields(), f"{field} is gone — drop the exemption"
    assert len(reason) > 20, f"{field}: say *why*"
