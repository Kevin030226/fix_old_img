"""Regression tests for the confirmed `config/ratelimit.py` X-Forwarded-For finding.

The review's critical claim was that `config/ratelimit.py` trusts the first
X-Forwarded-For segment unconditionally. Verified against the tag: the module is
**not imported by any product code** (`src/` contains no reference to it, and
`fiximg.infrastructure.security.rate_limit.client_ip` — the function the auth
router actually calls — already gates the header behind `FIXIMG_TRUSTED_PROXIES`).

So the finding is *obsolete* for the running product, but the dead module is still
shipped and still teaches the unsafe pattern to anyone who imports it. These
tests pin the safe behaviour on the module that is used, and fail loudly if the
unsafe one is ever wired back in.
"""
from __future__ import annotations

import pathlib

from fiximg.infrastructure.security import rate_limit as rlb


class _Request:
    def __init__(self, peer: str | None, forwarded: str | None = None) -> None:
        self.client = type("_C", (), {"host": peer})() if peer is not None else None
        self.headers = {} if forwarded is None else {"x-forwarded-for": forwarded}


def test_x_forwarded_for_is_ignored_from_an_untrusted_peer(monkeypatch):
    """The critical claim, asserted on the live implementation.

    `FIXIMG_TRUSTED_PROXIES` defaults to empty, so nothing is trusted and the
    header must not be able to select the rate-limit key.
    """
    monkeypatch.delenv("FIXIMG_TRUSTED_PROXIES", raising=False)
    request = _Request(peer="203.0.113.7", forwarded="198.51.100.9, 10.0.0.1")

    assert rlb.client_ip(request) == "203.0.113.7", (
        "a spoofed X-Forwarded-For selected the rate-limit key"
    )


def test_x_forwarded_for_is_honoured_from_a_trusted_peer(monkeypatch):
    """The half that must keep working: a real reverse proxy is still supported."""
    monkeypatch.setenv("FIXIMG_TRUSTED_PROXIES", "127.0.0.1,10.0.0.5")
    request = _Request(peer="127.0.0.1", forwarded="198.51.100.9, 10.0.0.1")

    assert rlb.client_ip(request) == "198.51.100.9"


def test_a_trusted_peer_without_the_header_falls_back_to_its_own_address(monkeypatch):
    monkeypatch.setenv("FIXIMG_TRUSTED_PROXIES", "127.0.0.1")
    assert rlb.client_ip(_Request(peer="127.0.0.1")) == "127.0.0.1"


def test_a_request_without_a_client_address_is_not_a_key():
    assert rlb.client_ip(_Request(peer=None)) == "unknown"


# ------------------------------------------------------- the dead module
def test_config_ratelimit_is_not_imported_by_product_code():
    """`config/ratelimit.py` is dead code; this is the assertion that keeps it so.

    The module duplicates the limiter with the *unsafe* `client_ip` (unconditional
    X-Forwarded-For). Nothing imports it today. If that ever changes, the
    duplicate becomes a second source of truth for a security decision, and this
    test is where that shows up.
    """
    src = pathlib.Path(rlb.__file__).resolve().parents[3]
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "config.ratelimit" in text or "from config import ratelimit" in text:
            offenders.append(str(path.relative_to(src)))
    assert not offenders, f"product code imports the unsafe duplicate: {offenders}"


def test_the_dead_module_is_marked_as_such():
    """A file nobody imports but that looks authoritative is a trap for the next
    reader; the warning has to be in the file itself, not only in a review."""
    module = pathlib.Path(rlb.__file__).resolve().parents[3].parent / "config" / "ratelimit.py"
    if not module.exists():          # removed outright: even better
        return
    head = module.read_text(encoding="utf-8")[:1200]
    assert "SUPERSEDED" in head or "DEPRECATED" in head, (
        "config/ratelimit.py is dead code but does not say so; its client_ip() "
        "trusts X-Forwarded-For unconditionally"
    )
