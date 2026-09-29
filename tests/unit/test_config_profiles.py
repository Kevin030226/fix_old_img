"""Configuration profile tests (plan 搂3.3).

Pins the resolution order 鈥?defaults 鈫?base.yaml 鈫?<profile>.yaml 鈫?env vars 鈥?
and the fail-fast validation for production.

The tests inject the merged profile dict (`_PROFILE_VALUES`) instead of
reloading ``fiximg.config``: a reload would rebuild the process-wide ``settings``
singleton that every other module already holds a reference to, so the test
would pass while the application kept the old object.
"""
import os
from types import SimpleNamespace

import pytest
import yaml

import fiximg.config as config_mod
from fiximg.paths import CONFIGS_DIR, PROJECT_ROOT


def _profile_values(profile: str) -> dict:
    """Merge configs/base.yaml with the named profile, exactly as the app does."""
    merged: dict = {}
    for name in ("base", profile):
        path = os.path.join(CONFIGS_DIR, f"{name}.yaml")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            merged.update(yaml.safe_load(handle) or {})
    return merged


@pytest.fixture()
def use_profile(monkeypatch):
    """Activate a profile without reloading the module.

    Mirrors a real deployment: the profile values are injected *and* the
    selector env var is set, so ``Settings.profile`` reports the same name.
    """

    def _activate(name: str) -> None:
        monkeypatch.setattr(config_mod, "_PROFILE_VALUES", _profile_values(name))
        monkeypatch.setenv("FIXIMG_PROFILE", name)

    return _activate


def test_profile_files_exist():
    for name in ("base", "local", "docker", "production"):
        assert os.path.exists(os.path.join(CONFIGS_DIR, f"{name}.yaml")), name


# ------------------------------------------------------------------- base
def test_base_profile_values_are_applied(use_profile):
    use_profile("local")
    settings = config_mod.Settings()
    # Values that only exist in configs/base.yaml.
    assert settings.max_image_side == 4096
    assert settings.task_max_attempts == 3
    assert settings.worker_lease_seconds == 3600.0
    assert settings.concurrency_default == 1
    assert settings.tile_size == 1536


def test_local_profile_values_are_applied(use_profile):
    use_profile("local")
    settings = config_mod.Settings()
    assert settings.app_env == "local"
    assert settings.inline_worker is True
    assert settings.external_worker is False
    assert settings.queue_backend == "sqlite"
    assert settings.storage_backend == "local"
    assert settings.warmup_strategy == "lazy"
    assert settings.host == "127.0.0.1"


def test_docker_profile_splits_the_topology(use_profile):
    use_profile("docker")
    settings = config_mod.Settings()
    assert settings.app_env == "staging"
    assert settings.inline_worker is False
    assert settings.external_worker is True
    assert settings.host == "0.0.0.0"


def test_production_profile_is_strict(use_profile, monkeypatch):
    use_profile("production")
    monkeypatch.setenv(
        "FIXIMG_MODEL_MANIFEST", os.path.join(PROJECT_ROOT, "models", "manifest.yaml")
    )
    settings = config_mod.Settings()
    assert settings.is_production
    assert settings.auto_bootstrap_admin is False
    assert settings.warmup_strategy == "startup"


# ------------------------------------------------------------- precedence
def test_environment_variables_override_the_profile(use_profile, monkeypatch):
    use_profile("local")
    monkeypatch.setenv("FIXIMG_MAX_IMAGE_SIDE", "2048")
    monkeypatch.setenv("FIXIMG_TASK_MAX_ATTEMPTS", "9")
    monkeypatch.setenv("FIXIMG_INLINE_WORKER", "false")

    settings = config_mod.Settings()
    assert settings.max_image_side == 2048
    assert settings.task_max_attempts == 9
    assert settings.inline_worker is False


def test_profile_env_var_selects_the_profile(use_profile, monkeypatch):
    """FIXIMG_PROFILE names the YAML file; app_env comes from inside it."""
    use_profile("local")
    monkeypatch.setenv("FIXIMG_PROFILE", "docker")
    settings = config_mod.Settings()
    assert settings.profile == "docker"
    # The injected values are still local's, so app_env is not re-derived from
    # the profile *name* 鈥?that is exactly the separation this test protects.
    assert settings.app_env == "local"


def test_fiximg_env_is_accepted_as_a_profile_alias(use_profile, monkeypatch):
    use_profile("local")
    monkeypatch.delenv("FIXIMG_PROFILE", raising=False)
    monkeypatch.setenv("FIXIMG_ENV", "docker")
    assert config_mod.Settings().profile == "docker"


def test_fiximg_profile_wins_over_fiximg_env(use_profile, monkeypatch):
    use_profile("local")
    monkeypatch.setenv("FIXIMG_PROFILE", "local")
    monkeypatch.setenv("FIXIMG_ENV", "docker")
    assert config_mod.Settings().profile == "local"


def test_profile_name_does_not_leak_into_app_env(use_profile, monkeypatch):
    """A profile named `docker` must not produce app_env == "docker"."""
    use_profile("docker")
    settings = config_mod.Settings()
    assert settings.profile == "docker"
    assert settings.app_env == "staging"


def test_app_env_can_be_overridden_explicitly(use_profile, monkeypatch):
    use_profile("docker")
    monkeypatch.setenv("FIXIMG_APP_ENV", "production")
    assert config_mod.Settings().app_env == "production"


def test_missing_profile_file_still_applies_base(monkeypatch):
    monkeypatch.setattr(config_mod, "_PROFILE_VALUES", _profile_values("base"))
    settings = config_mod.Settings()
    assert settings.max_image_side == 4096
    assert settings.app_env == "local"  # default when no profile declares it


# -------------------------------------------------------------- validation
def test_invalid_profile_name_is_rejected(monkeypatch):
    monkeypatch.setenv("FIXIMG_PROFILE", "does-not-exist")
    with pytest.raises(ValueError, match="Unknown FIXIMG_PROFILE"):
        config_mod.Settings()


def test_invalid_backend_choice_is_rejected(monkeypatch):
    monkeypatch.setenv("FIXIMG_QUEUE_BACKEND", "rabbitmq")
    with pytest.raises(ValueError, match="FIXIMG_QUEUE_BACKEND"):
        config_mod.Settings()


def test_invalid_storage_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("FIXIMG_STORAGE_BACKEND", "ftp")
    with pytest.raises(ValueError, match="FIXIMG_STORAGE_BACKEND"):
        config_mod.Settings()


def test_invalid_warmup_strategy_is_rejected(monkeypatch):
    monkeypatch.setenv("FIXIMG_WARMUP", "eventually")
    with pytest.raises(ValueError, match="FIXIMG_WARMUP"):
        config_mod.Settings()


def test_production_requires_a_manifest(use_profile, monkeypatch, tmp_path):
    use_profile("production")
    monkeypatch.setenv("FIXIMG_MODEL_MANIFEST", str(tmp_path / "missing.yaml"))
    with pytest.raises(ValueError, match="model manifest not found"):
        config_mod.Settings()


def test_production_rejects_s3_without_a_bucket(use_profile, monkeypatch):
    use_profile("production")
    monkeypatch.setenv(
        "FIXIMG_MODEL_MANIFEST", os.path.join(PROJECT_ROOT, "models", "manifest.yaml")
    )
    monkeypatch.setenv("FIXIMG_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("FIXIMG_STORAGE_BUCKET", "")
    with pytest.raises(ValueError, match="FIXIMG_STORAGE_BUCKET"):
        config_mod.Settings()


def test_production_rejects_redis_queue_without_a_url(use_profile, monkeypatch):
    use_profile("production")
    monkeypatch.setenv(
        "FIXIMG_MODEL_MANIFEST", os.path.join(PROJECT_ROOT, "models", "manifest.yaml")
    )
    monkeypatch.setenv("FIXIMG_QUEUE_BACKEND", "redis")
    monkeypatch.setenv("FIXIMG_REDIS_URL", "")
    with pytest.raises(ValueError, match="FIXIMG_REDIS_URL"):
        config_mod.Settings()


def test_production_rejects_half_a_credential_pair(use_profile, monkeypatch):
    """An access key without its secret is not repairable by the ambient chain.

    boto3 ignores an incomplete static pair and signs with whatever identity it
    found next, so artifacts would land in somebody else's bucket instead of the
    deployment failing here.
    """
    use_profile("production")
    monkeypatch.setenv(
        "FIXIMG_MODEL_MANIFEST", os.path.join(PROJECT_ROOT, "models", "manifest.yaml")
    )
    monkeypatch.setenv("FIXIMG_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("FIXIMG_STORAGE_BUCKET", "fiximg")
    monkeypatch.setenv("FIXIMG_STORAGE_ACCESS_KEY", "fiximg")
    monkeypatch.setenv("FIXIMG_STORAGE_SECRET_KEY", "")
    with pytest.raises(ValueError, match="FIXIMG_STORAGE_SECRET_KEY"):
        config_mod.Settings()

    # Neither key is a legitimate configuration too: it means "use the chain".
    monkeypatch.setenv("FIXIMG_STORAGE_ACCESS_KEY", "")
    assert config_mod.Settings().storage_backend == "s3"


def test_the_s3_builder_carries_the_declared_identity(monkeypatch):
    """A credential the operator sets must reach the client, not just Settings.

    `build_s3_store_from_settings` passed only `endpoint_url`, so a self-hosted
    MinIO - whose root user is declared right next to it in docker/compose.yaml -
    could not be authenticated against at all, and `make smoke-services` reported
    the configured service as unreachable with no way to fix it.
    """
    import sys

    from fiximg.infrastructure.storage import s3 as s3_module

    captured: dict = {}

    def _client(service, **kwargs):
        captured.update(service=service, **kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=_client))
    settings_obj = SimpleNamespace(
        storage_backend="s3",
        storage_bucket="fiximg", storage_prefix="tasks", s3_endpoint="http://minio:9000",
        storage_access_key="fiximg", storage_secret_key="fiximg-secret",
        storage_region="us-east-1",
    )
    s3_module.build_s3_store_from_settings(settings_obj)

    assert captured["service"] == "s3"
    assert captured["endpoint_url"] == "http://minio:9000"
    assert captured["aws_access_key_id"] == "fiximg"
    assert captured["aws_secret_access_key"] == "fiximg-secret"
    assert captured["region_name"] == "us-east-1"


def test_an_unconfigured_identity_leaves_the_ambient_chain_alone(monkeypatch):
    """Empty credentials must be absent from the call, not passed as None or "".

    Passing an empty access key makes boto3 fail where its own chain (instance
    role, web identity) would have worked, so an AWS deployment that sets nothing
    would break on the strength of the settings' defaults.
    """
    import sys

    from fiximg.infrastructure.storage import s3 as s3_module

    captured: dict = {}

    def _client(service, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=_client))
    settings_obj = SimpleNamespace(
        storage_backend="s3",
        storage_bucket="fiximg", storage_prefix="", s3_endpoint="",
        storage_access_key="", storage_secret_key="", storage_region="",
    )
    s3_module.build_s3_store_from_settings(settings_obj)

    assert "aws_access_key_id" not in captured, captured
    assert "aws_secret_access_key" not in captured, captured
    assert ("region_name" not in captured or captured["region_name"] is None
            or captured["region_name"] == ""), captured


def test_local_profile_does_not_fail_fast(use_profile, monkeypatch, tmp_path):
    """Local must stay permissive: no manifest, no bucket, still boots."""
    use_profile("local")
    monkeypatch.setenv("FIXIMG_MODEL_MANIFEST", str(tmp_path / "absent.yaml"))
    assert config_mod.Settings().is_production is False


def test_bad_numeric_env_falls_back_to_the_profile_value(use_profile, monkeypatch):
    use_profile("local")
    monkeypatch.setenv("FIXIMG_PORT", "not-a-port")
    assert config_mod.Settings().port == 9502


def test_boolean_env_parsing(monkeypatch):
    for raw, expected in (("1", True), ("true", True), ("YES", True), ("on", True),
                          ("0", False), ("false", False), ("off", False)):
        monkeypatch.setenv("FIXIMG_INLINE_WORKER", raw)
        assert config_mod.Settings().inline_worker is expected, raw
