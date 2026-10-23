"""Failed configuration must stop the process, not silently fall back.

`_load_profile()` used to swallow every failure: a missing profile file, a YAML
syntax error and a non-mapping document all `continue`d, so a deployment whose
`configs/production.yaml` was lost booted on `base.yaml` + code defaults. Because
`production` is also a `KNOWN_ENVS` name, the old `_validate()` accepted it -- the
process came up claiming to be production while every fail-fast check was off,
`auto_bootstrap_admin` was back to True and GPU inference ran inside the API
process.

The rules these tests pin:

* an explicitly selected profile that does not exist            -> error
* any profile file that exists but cannot be parsed             -> error
* a profile file that is not a mapping                          -> error
* base.yaml missing, default profile, no request                -> allowed
* `database_url` in a profile (it can never take effect)        -> error
"""
import os

import pytest

import fiximg.config as config_mod


def _write(directory, name: str, text: str) -> str:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# ------------------------------------------------------- explicit selection
def test_a_selected_profile_that_does_not_exist_is_an_error(tmp_path):
    _write(tmp_path, "base.yaml", "max_image_side: 4096\n")
    with pytest.raises(config_mod.ProfileConfigurationError, match="no such file"):
        config_mod.resolve_profile("ghost", configs_dir=str(tmp_path))


def test_the_error_names_the_path_the_operator_must_create(tmp_path):
    _write(tmp_path, "base.yaml", "max_image_side: 4096\n")
    with pytest.raises(config_mod.ProfileConfigurationError) as excinfo:
        config_mod.resolve_profile("production", configs_dir=str(tmp_path))
    assert str(tmp_path / "production.yaml") in str(excinfo.value)
    assert "FIXIMG_PROFILE" in str(excinfo.value)


def test_the_environment_selector_is_honoured_the_same_way(tmp_path, monkeypatch):
    """The deployment spells the selection in the environment, not an argument."""
    _write(tmp_path, "base.yaml", "max_image_side: 4096\n")
    monkeypatch.setenv("FIXIMG_PROFILE", "docker")
    monkeypatch.delenv("FIXIMG_ENV", raising=False)
    with pytest.raises(config_mod.ProfileConfigurationError, match="docker"):
        config_mod.resolve_profile(configs_dir=str(tmp_path))


# ------------------------------------------------------------- unusable file
def test_a_broken_profile_yaml_names_the_position(tmp_path):
    _write(tmp_path, "base.yaml", "max_image_side: 4096\n")
    _write(tmp_path, "local.yaml", "max_image_side: [unclosed\n")
    with pytest.raises(config_mod.ProfileConfigurationError) as excinfo:
        config_mod.resolve_profile("local", configs_dir=str(tmp_path))
    message = str(excinfo.value)
    assert "local.yaml" in message
    assert "cannot be parsed" in message
    assert "line" in message, "the operator cannot find the problem without a position"


def test_a_profile_that_is_not_a_mapping_is_rejected(tmp_path):
    _write(tmp_path, "base.yaml", "max_image_side: 4096\n")
    _write(tmp_path, "local.yaml", "- just\n- a list\n")
    with pytest.raises(config_mod.ProfileConfigurationError, match="expected a mapping"):
        config_mod.resolve_profile("local", configs_dir=str(tmp_path))


def test_a_broken_base_file_also_stops_the_boot(tmp_path):
    """base.yaml is optional, but a base.yaml that exists and is unreadable is not.

    "Optional" describes the file being absent (a wheel ships none), not the
    file being present and broken.
    """
    _write(tmp_path, "base.yaml", "max_image_side: {{{\n")
    _write(tmp_path, "local.yaml", "max_image_side: 1\n")
    with pytest.raises(config_mod.ProfileConfigurationError, match="base.yaml"):
        config_mod.resolve_profile("local", configs_dir=str(tmp_path))


# ------------------------------------------------------- allowed default path
def test_a_checkout_without_any_configs_directory_still_runs(tmp_path, monkeypatch):
    """The default profile is not requested, so its absence is the dev path.

    The shipped wheel contains `src/fiximg` only; a pip install legitimately has
    no `configs/` tree and must still start on the code defaults. Passing a name
    *argument* is a request (and is refused when missing); taking the default is
    not.
    """
    monkeypatch.delenv("FIXIMG_PROFILE", raising=False)
    monkeypatch.delenv("FIXIMG_ENV", raising=False)
    empty = tmp_path / "no-configs"
    empty.mkdir()
    assert config_mod.resolve_profile(configs_dir=str(empty)) == {}


def test_base_overlays_the_profile_in_the_documented_order(tmp_path):
    _write(tmp_path, "base.yaml", "max_image_side: 4096\ntile_size: 1536\n")
    _write(tmp_path, "local.yaml", "max_image_side: 2048\n")
    merged = config_mod.resolve_profile("local", configs_dir=str(tmp_path))
    assert merged["max_image_side"] == 2048, "the profile must win over base"
    assert merged["tile_size"] == 1536, "base keys survive when the profile is silent"


def test_an_empty_profile_file_is_not_an_error(tmp_path):
    _write(tmp_path, "base.yaml", "max_image_side: 4096\n")
    _write(tmp_path, "local.yaml", "")
    assert config_mod.resolve_profile("local", configs_dir=str(tmp_path))["max_image_side"] == 4096


# --------------------------------------------------- database_url is env-only
def test_a_profile_cannot_declare_the_database_url(tmp_path):
    """The key would be accepted and then ignored, which is worse than refused.

    `database_url` is env-only by design: a profile value outranking a `DB_PATH`
    set by a fixture is how ~176 tests once shared one developer database.
    Refusing it at load time is what keeps "accepted" equal to "effective".
    """
    _write(tmp_path, "base.yaml", "max_image_side: 4096\n")
    _write(tmp_path, "local.yaml", "database_url: sqlite:////tmp/sneaky.db\n")
    with pytest.raises(config_mod.ProfileConfigurationError) as excinfo:
        config_mod.resolve_profile("local", configs_dir=str(tmp_path))
    message = str(excinfo.value)
    assert "database_url" in message
    assert "FIXIMG_DATABASE_URL" in message, "the message must name the real knob"


def test_the_shipped_profiles_do_not_declare_it():
    """Every profile file in the checkout must be loadable under the rule above.

    Derived from the directory rather than from a hard-coded list, so a new
    profile cannot be added and left unchecked.
    """
    shipped = sorted(os.path.splitext(name)[0] for name in os.listdir(config_mod.CONFIGS_DIR)
                     if name.endswith(".yaml"))
    assert shipped, "no configs/*.yaml found, so this test would prove nothing"
    for name in shipped:
        values = config_mod.resolve_profile(name)
        assert "database_url" not in values, name
