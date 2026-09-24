"""Unit tests for security utilities and the V2 task repository."""

from app.core.security import hash_password, verify_password, needs_rehash


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
