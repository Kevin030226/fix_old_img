"""Unit tests for security utilities and the V2 task repository."""

import pytest

from fiximg.api import security
from fiximg.config import settings
from fiximg.infrastructure.security.passwords import hash_password, verify_password, needs_rehash


# ------------------------------------------------------- the API token (搂3.3)
def _prepare_token_lookup(monkeypatch, tmp_path, env_token, production):
    """Pin the three inputs `get_api_token` reads: env, file, profile."""
    monkeypatch.setattr(settings, "app_env", "production" if production else "local",
                        raising=False)
    if env_token is None:
        monkeypatch.delenv("FIXIMG_API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("FIXIMG_API_TOKEN", env_token)
    monkeypatch.setattr(security, "token_file", lambda: str(tmp_path / "api_token.txt"))
    # The resolved token is cached for the process; starting from "nothing cached"
    # is what makes each of these cases test the boot path rather than a leftover.
    monkeypatch.setattr(security, "_cached", None)


def test_production_refuses_to_mint_its_own_api_token(monkeypatch, tmp_path):
    """Plan 搂3.3: "API secret 缂哄け 鈫?涓嶅惎鍔? is a production rule.

    Before this, a production boot with no token generated one, wrote it into
    `admin_data/api_token.txt` and printed it 鈥?so the service came up healthy on
    a secret nobody configured, and a second replica generated a *different* one.
    """
    _prepare_token_lookup(monkeypatch, tmp_path, env_token=None, production=True)

    with pytest.raises(Exception, match="Production configuration has no API token"):
        security.get_api_token()
    assert not (tmp_path / "api_token.txt").exists(), "it still wrote a secret to disk"


def test_production_accepts_a_token_the_operator_persisted(monkeypatch, tmp_path):
    """The rule is "no secret, no boot", not "the secret must come from the env"."""
    (tmp_path / "api_token.txt").write_text("from-mounted-secret-file", encoding="utf-8")
    _prepare_token_lookup(monkeypatch, tmp_path, env_token=None, production=True)

    assert security.get_api_token() == "from-mounted-secret-file"


def test_production_prefers_the_environment_token(monkeypatch, tmp_path):
    _prepare_token_lookup(monkeypatch, tmp_path, env_token="env-token", production=True)
    assert security.get_api_token() == "env-token"


def test_local_profile_still_bootstraps_a_development_token(monkeypatch, tmp_path, capsys):
    """The same plan allows local/demo to generate development configuration."""
    _prepare_token_lookup(monkeypatch, tmp_path, env_token=None, production=False)

    token = security.get_api_token()
    assert token and len(token) >= 16
    assert (tmp_path / "api_token.txt").read_text(encoding="utf-8").strip() == token
    assert "token" in capsys.readouterr().out.lower()


def test_hash_and_verify_roundtrip():
    stored = hash_password("s3cret")
    assert stored.startswith("pbkdf2_sha256$")
    assert verify_password("s3cret", stored)
    assert not verify_password("wrong", stored)


def test_verify_rejects_malformed():
    assert not verify_password("x", "")
    assert not verify_password("x", None)
    assert not verify_password("x", "plaintext")
    assert not verify_password("x", "md5$abc$def")


def test_needs_rehash():
    assert not needs_rehash(hash_password("x"))
    assert needs_rehash("plaintext")


def test_hash_salts_are_unique():
    assert hash_password("x") != hash_password("x")
