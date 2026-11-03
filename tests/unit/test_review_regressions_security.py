"""Regression tests for two P1 security findings confirmed against v3.1.2.

1. `api/security.py` compared the caller's bearer token to the expected one with
   `hmac.compare_digest(provided, expected)`. That raises `TypeError` for any
   non-ASCII `str`, and an ASGI server hands the header to the app as latin-1
   decoded text — so `Authorization: Bearer <any non-ASCII bytes>` produced a
   500 instead of a 401, on every authenticated route.

2. `storage/base.py` normalised artifact keys with `lstrip("/")` only. On Windows
   a key of `C:/Windows/evil.png` survives normalisation and `os.path.join(root,
   key)` then discards `root` entirely: the key decides the absolute path. The
   keys are generated internally today, so this is defence in depth rather than a
   live hole — but a store whose public API accepts an absolute key is a hole
   waiting for one careless caller.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from fiximg.infrastructure.storage.base import BaseArtifactStore


# ------------------------------------------------------- non-ASCII bearer token
@pytest.mark.parametrize("header", [
    "Bearer caf\u00e9",          # a raw 0xE9 byte, as ASGI delivers it
    "Bearer \u4e2d\u6587",
    "Bearer \u00ff\u00ff",
])
def test_a_non_ascii_bearer_token_is_a_401_not_a_500(header, monkeypatch):
    """`compare_digest` raises TypeError on non-ASCII strings; the route must not."""
    monkeypatch.setenv("FIXIMG_API_TOKEN", "ascii-only-token")
    import fiximg.api.security as sec

    sec.reset_cache()
    try:
        with pytest.raises(Exception) as excinfo:
            asyncio.run(sec.require_api_token(header))
    finally:
        sec.reset_cache()

    assert excinfo.value.__class__.__name__ == "HTTPException", (
        f"expected an authentication refusal, got {type(excinfo.value).__name__}: "
        f"{excinfo.value}"
    )
    assert excinfo.value.status_code == 401


def test_a_valid_ascii_token_still_authenticates(monkeypatch):
    """The half that must keep working."""
    monkeypatch.setenv("FIXIMG_API_TOKEN", "ascii-only-token")
    import fiximg.api.security as sec

    sec.reset_cache()
    try:
        assert asyncio.run(sec.require_api_token("Bearer ascii-only-token")) == "ascii-only-token"
    finally:
        sec.reset_cache()


# ------------------------------------------------------- absolute artifact keys
@pytest.mark.parametrize("key", [
    "C:/Windows/evil.png",
    "C:\\Windows\\evil.png",
    "c:/temp/x.png",
    "//server/share/x.png",
    "\\\\?\\\\C:\\\\Windows\\\\evil.png",
])
def test_an_absolute_artifact_key_is_refused(key):
    """A key must never be able to select the path the store writes to."""
    with pytest.raises(ValueError):
        BaseArtifactStore._normalise(key)


@pytest.mark.parametrize("key", [
    "tasks/2026/09/abc/output/final.png",
    "/tasks/2026/09/abc/output/final.png",
    "tasks/./2026/final.png",
])
def test_relative_artifact_keys_still_normalise(key):
    normalised = BaseArtifactStore._normalise(key)
    assert not os.path.isabs(normalised), normalised
    assert ".." not in normalised.split("/"), normalised


def test_a_normalised_key_cannot_escape_its_root(tmp_path):
    """The property the normalisation exists for, asserted end to end."""
    root = str(tmp_path)
    for key in ("tasks/a.png", "/tasks/a.png"):
        joined = os.path.join(root, BaseArtifactStore._normalise(key))
        assert os.path.abspath(joined).startswith(os.path.abspath(root) + os.sep), joined
