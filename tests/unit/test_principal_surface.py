"""`Principal` is the whole authorization model, and it is meant to stay small.

`api/dependencies.py` used to carry a `scopes` field, a `_ADMIN_SCOPES` set of
six `resource:action` strings, and a `can()` predicate. None of it was
reachable: `resolve_principal()` builds the only principal this deployment can
produce, with `role="admin"` and every scope granted, so `can(scope)` was an
identity function on the single value in the domain. `require_admin` compared
`role` against `"admin"` and nothing consulted the scopes.

The problem was not dead code so much as a *misdescription*. A `scopes` field on
the authenticated actor is a promise that the service authorizes by scope. An
operator reading `dependencies.py` would reasonably conclude it did, and would
not learn otherwise until they tried to add a lesser credential and found no
scope to attach it to. The fields were removed; this file holds the line.

Why a hand-written gate rather than a scan: `test_declared_but_unwired.py`
documents, at length, why a general method-reachability scan cannot be made
sound against a codebase that dispatches through registries and Protocols. This
class has four members and no registry, so it can simply be enumerated.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from fiximg.api import dependencies
from fiximg.api.dependencies import Principal, resolve_principal
from fiximg.domain.errors import ForbiddenError

MODULE = Path(dependencies.__file__)
DECLARED_PATH = "src/fiximg/api/dependencies.py"


def _module_surface() -> tuple[set[str], dict[str, set[str]], dict[str, str]]:
    """(module-level public names, class -> public members, class -> decorators)."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8-sig"))
    module_level: set[str] = set()
    classes: dict[str, set[str]] = {}
    decorators: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                module_level.add(node.name)
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            module_level.add(node.name)
            members: set[str] = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and not item.name.startswith("_"):
                    members.add(item.name)
            classes[node.name] = members
    # The old fields, kept here so the assertion below can name them.
    for name, kind in (("scopes", "field"), ("_ADMIN_SCOPES", "constant"),
                       ("can", "method"), ("to_dict", "method")):
        decorators[name] = kind
    return module_level, classes, decorators


def _tree() -> ast.Module:
    return ast.parse(MODULE.read_text(encoding="utf-8-sig"))


def test_the_authorization_model_is_exactly_one_role_check():
    """`require_admin` is the whole model; there is nothing beside it.

    `tree.body`, not `ast.walk`: a walk descends into the class and reports
    `is_admin` as a module-level function.
    """
    assert [n.name for n in _tree().body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not n.name.startswith("_")] == [
        "resolve_principal", "require_principal", "require_admin",
    ]


def test_principal_has_no_scope_surface():
    """The fields that implied scope-based authorization are gone.

    Reinstating one of these is a feature, not a refactor: it needs a scope per
    route and a second, lesser credential for the scope to mean anything.
    """
    assert not hasattr(Principal, "can"), (
        "Principal.can() is back. The deployment has one credential and it is an "
        "admin credential, so a scope check would be an identity function."
    )
    assert not hasattr(Principal, "to_dict"), (
        "Principal.to_dict() existed only to serialise `scopes`, and nothing "
        "called it; if a diagnostic needs it, it should serialise the fields "
        "Principal actually has."
    )
    fields = Principal.__dataclass_fields__
    assert "scopes" not in fields, f"Principal grew a scopes field: {sorted(fields)}"
    # Read the code, not the file: the module docstring names every one of these
    # precisely because it explains removing them, and a text search would find
    # its own explanation and report a resurrection.
    bound = {
        t.id for n in _tree().body if isinstance(n, ast.Assign) for t in n.targets
        if isinstance(t, ast.Name)
    }
    assert "_ADMIN_SCOPES" not in bound, (
        "_ADMIN_SCOPES is assigned again but nothing reads it; require_admin "
        "compares role"
    )


def test_principals_own_surface_is_still_the_four_members_it_was_reviewed_as():
    """An addition has to be deliberate, which is what this assertion is for."""
    _module, classes, _ = _module_surface()
    assert set(classes) == {"Principal"}, sorted(classes)
    assert classes["Principal"] == {"is_admin"}, (
        f"Principal grew or lost a public member: {sorted(classes['Principal'])}. "
        "Adding one is fine — but it is a change to the authorization surface, "
        "so make it on purpose: update this test, and say why."
    )


def test_no_module_level_name_is_declared_only_for_the_scope_model():
    """A helper left behind by the removal is the usual way it creeps back."""
    module_level, classes, removed = _module_surface()
    assert "Principal" in module_level
    resurrected = sorted(
        name for name in removed
        if name in module_level
        or any(name in members for members in classes.values())
    )
    assert not resurrected, f"the scope model is partly back: {resurrected}"


def test_the_deployment_principal_is_an_admin_and_says_so():
    """The one principal the deployment can build, described accurately."""
    principal = resolve_principal()
    assert principal.is_admin is True
    assert principal.role == "admin"
    assert principal.via == "api_token"
    assert principal.user_id, "a principal must name an account, or tasks have no owner"


def test_require_admin_rejects_a_non_admin_on_role_not_on_scopes():
    """What authorization actually checks, pinned to the role comparison.

    The temptation this guards against is `if not principal.can("users:write")`,
    which would be a *different* model — and one this deployment cannot enforce,
    because the token is the only credential and it is an admin credential.
    """
    import asyncio

    plain = Principal(user_id="alice", role="user")
    assert plain.is_admin is False
    with pytest.raises(ForbiddenError):
        asyncio.run(dependencies.require_admin(plain))

    admin = Principal(user_id="root", role="admin")
    assert asyncio.run(dependencies.require_admin(admin)) is admin
