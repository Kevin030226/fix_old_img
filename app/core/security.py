"""Security utilities re-exported from the legacy config package (plan section 20).

The implementations (PBKDF2 hashing, atomic YAML writes) are battle-tested from V1;
they are re-exported here so the rest of the V2 code base depends on app.core only.
"""
from config.security import (  # noqa: F401
    hash_password,
    is_plaintext,
    needs_rehash,
    verify_password,
    write_yaml_atomic,
)

__all__ = [
    "hash_password",
    "is_plaintext",
    "needs_rehash",
    "verify_password",
    "write_yaml_atomic",
]
