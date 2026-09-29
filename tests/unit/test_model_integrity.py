"""Weight integrity for the model manifest (plan 搂3.5.2, 搂3.15).

Before these tests the runtime verification was inert in two ways at once:
``models/manifest.yaml`` declared no ``sha256``, so ``verify_hashes()`` returned
``True`` for every model **without checking anything**, while a second system
(``config/weights_manifest.json``, 26 files) held the real checksums and was only
consulted by the startup CLI. The manifest now verifies against that same
baseline, so one committed file is the source of truth for both.

What is pinned here:
  * a file weight and a checkpoint *directory* are both covered;
  * an explicit YAML hash wins over the baseline for the file it names;
  * a mismatch, an unreadable file and "no baseline entry" are three different
    answers 鈥?the last one is reported as unverified, never as verified;
  * a gigabyte of legacy weights is read once, not once per health poll.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

from fiximg.domain.errors import ModelUnavailableError
from fiximg.domain.models import ModelVersion
from fiximg.infrastructure.models import weights_check
from fiximg.inference import manifest as manifest_module
from fiximg.inference.manifest import ModelManifest

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DDCOLOR_BYTES = b"ddcolor-bytes"


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """A fake project root with a few weights, plus a fake integrity baseline."""
    root = tmp_path / "repo"
    (root / "weights" / "ddcolor").mkdir(parents=True)
    (root / "checkpoints" / "stage_a").mkdir(parents=True)
    (root / "checkpoints" / "detection").mkdir(parents=True)
    (root / "weights" / "ddcolor" / "pytorch_model.pt").write_bytes(DDCOLOR_BYTES)
    (root / "checkpoints" / "stage_a" / "net_vae.pth").write_bytes(b"vae")
    (root / "checkpoints" / "stage_a" / "net_map.pth").write_bytes(b"mapping")
    (root / "checkpoints" / "detection" / "ft.pt").write_bytes(b"detector")

    monkeypatch.setattr(manifest_module, "PROJECT_ROOT", str(root))

    baseline = {
        "algorithm": "sha256",
        "files": [
            {"path": "weights/ddcolor/pytorch_model.pt", "sha256": DIGEST_A, "artifact": "ddcolor"},
            {"path": "checkpoints/stage_a/net_vae.pth", "sha256": DIGEST_A, "artifact": "global_restore"},
            {"path": "checkpoints/stage_a/net_map.pth", "sha256": DIGEST_B, "artifact": "global_restore"},
            {"path": "checkpoints/detection/ft.pt", "sha256": DIGEST_A, "artifact": "scratch_repair"},
        ],
    }
    baseline_path = tmp_path / "weights_manifest.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setattr(ModelManifest, "INTEGRITY_BASELINE_PATH", str(baseline_path))

    # "Verified at boot" must not leak between tests.
    monkeypatch.setattr(weights_check, "VERIFIED", {})
    return root


def _manifest(specs: dict[str, dict]) -> ModelManifest:
    return ModelManifest(
        models={name: ModelVersion.from_dict(name, spec) for name, spec in specs.items()},
        source="test",
    )


def _abs(root, *parts: str) -> str:
    """A weight path spelled the way the manifest spells it (normalised)."""
    return os.path.normpath(os.path.join(str(root), *parts))


def _rel(root, path: str) -> str:
    """Repo-relative with forward slashes, so assertions survive Windows."""
    return os.path.relpath(path, str(root)).replace(os.sep, "/")


def _fake_hash(monkeypatch, digest_for: dict[str, str]):
    """Replace the file hasher so a test can watch what gets read.

    Unmapped paths still get a deterministic non-empty digest: an empty string
    would be indistinguishable from an unreadable file, and the point of most of
    these tests is which files were *read*, not their contents.
    """
    calls: list[str] = []

    def _sha256_of(path, *args, **kwargs):
        normalised = os.path.normpath(path)
        calls.append(normalised)
        return digest_for.get(normalised) or hashlib.sha256(normalised.encode()).hexdigest()

    monkeypatch.setattr("fiximg.domain.artifacts.sha256_of", _sha256_of)
    return calls


# =============================== coverage ==================================
def test_a_file_weight_is_covered_by_the_baseline(sandbox):
    manifest = _manifest({"ddcolor": {"weight": "weights/ddcolor/pytorch_model.pt"}})
    assert [_rel(sandbox, p) for p in manifest.expected_weights("ddcolor")] == [
        "weights/ddcolor/pytorch_model.pt"
    ]


def test_a_checkpoint_directory_covers_every_network_beneath_it(sandbox):
    """The legacy chains load several nets per stage; they verify as one unit."""
    manifest = _manifest({"global_restore": {"checkpoint": "checkpoints/stage_a"}})
    assert sorted(os.path.basename(p) for p in manifest.expected_weights("global_restore")) == [
        "net_map.pth", "net_vae.pth",
    ]


def test_a_higher_checkpoint_directory_covers_nested_stages(sandbox):
    manifest = _manifest({"scratch_repair": {"checkpoint": "checkpoints"}})
    covered = [_rel(sandbox, p) for p in manifest.expected_weights("scratch_repair")]
    assert sorted(covered) == [
        "checkpoints/detection/ft.pt",
        "checkpoints/stage_a/net_map.pth",
        "checkpoints/stage_a/net_vae.pth",
    ]
    assert all("weights/ddcolor" not in p for p in covered), "no unrelated files swept in"


def test_a_directory_prefix_does_not_match_a_sibling(sandbox):
    """`checkpoints/stage_a` must not sweep in `checkpoints/stage_a2/`."""
    (sandbox / "checkpoints" / "stage_a2").mkdir(parents=True)
    manifest = _manifest({"global_restore": {"checkpoint": "checkpoints/stage_a"}})
    manifest._baseline = {**manifest.integrity_baseline(), "checkpoints/stage_a2/other.pth": DIGEST_A}
    assert all("stage_a2" not in p for p in manifest.expected_weights("global_restore"))


def test_a_model_declaring_no_weight_is_covered_by_nothing(sandbox):
    manifest = _manifest({"face_detection": {"framework": "legacy-cli"}})
    assert manifest.expected_weights("face_detection") == {}


# =============================== verification ==============================
def test_hashes_match_the_baseline(sandbox):
    manifest = _manifest({"global_restore": {"checkpoint": "checkpoints/stage_a"}})
    manifest._hash_cache = {
        _abs(sandbox, "checkpoints", "stage_a", "net_vae.pth"): DIGEST_A,
        _abs(sandbox, "checkpoints", "stage_a", "net_map.pth"): DIGEST_B,
    }
    assert manifest.verify_hashes() == {"global_restore": True}
    assert manifest.unverified_models() == []


def test_a_tampered_weight_fails_and_says_which_file(sandbox):
    manifest = _manifest({"ddcolor": {"weight": "weights/ddcolor/pytorch_model.pt"}})
    problems = manifest.verify_files({_abs(sandbox, "weights", "ddcolor", "pytorch_model.pt"): DIGEST_A})
    assert len(problems) == 1
    assert problems[0]["path"] == "weights/ddcolor/pytorch_model.pt"
    assert problems[0]["expected"] == DIGEST_A
    assert problems[0]["actual"] == hashlib.sha256(DDCOLOR_BYTES).hexdigest()
    assert manifest.verify_hashes() == {"ddcolor": False}


def test_a_missing_weight_file_is_a_mismatch_not_a_pass(sandbox):
    manifest = _manifest({"ddcolor": {"weight": "weights/ddcolor/absent.pt"}})
    manifest._baseline = {**manifest.integrity_baseline(), "weights/ddcolor/absent.pt": DIGEST_A}
    assert manifest.verify_hashes() == {"ddcolor": False}


def test_an_explicit_yaml_hash_wins_over_the_baseline(sandbox):
    """A deployment can pin one model without regenerating the whole baseline."""
    declared = hashlib.sha256(DDCOLOR_BYTES).hexdigest()
    manifest = _manifest({
        "ddcolor": {"weight": "weights/ddcolor/pytorch_model.pt", "sha256": declared}
    })
    expected = manifest.expected_weights("ddcolor")
    weight = _abs(sandbox, "weights", "ddcolor", "pytorch_model.pt")
    assert expected[weight] == DIGEST_A, "the baseline still describes the file"
    assert manifest.verify_hashes() == {"ddcolor": True}, "but the declared hash decides"


def test_verification_reads_each_weight_once(sandbox, monkeypatch):
    calls = _fake_hash(monkeypatch, {
        _abs(sandbox, "weights", "ddcolor", "pytorch_model.pt"): hashlib.sha256(DDCOLOR_BYTES).hexdigest()
    })
    manifest = _manifest({"ddcolor": {"weight": "weights/ddcolor/pytorch_model.pt"}})
    manifest.verify_hashes()
    manifest.verify_hashes()
    assert len(calls) == 1, "memoised per process, not recomputed per poll"


# ======================= trusting the boot integrity check ==================
def test_a_weight_proven_at_boot_is_not_hashed_again(sandbox, monkeypatch):
    """1.3 GB of legacy weights must not be re-read to answer a health poll."""
    calls = _fake_hash(monkeypatch, {})
    weights_check.VERIFIED["weights/ddcolor/pytorch_model.pt"] = DIGEST_A

    manifest = _manifest({"ddcolor": {"weight": "weights/ddcolor/pytorch_model.pt"}})
    assert manifest.verify_hashes() == {"ddcolor": True}
    assert calls == []


def test_an_unproven_weight_is_actually_hashed(sandbox, monkeypatch):
    calls = _fake_hash(monkeypatch, {
        _abs(sandbox, "weights", "ddcolor", "pytorch_model.pt"): DIGEST_A
    })
    manifest = _manifest({"ddcolor": {"weight": "weights/ddcolor/pytorch_model.pt"}})
    assert manifest.verify_hashes() == {"ddcolor": True}
    assert calls == [_abs(sandbox, "weights", "ddcolor", "pytorch_model.pt")]


def test_the_boot_registry_is_per_path_not_global(sandbox, monkeypatch):
    """Trusting boot must not make an unrelated model report verified."""
    _fake_hash(monkeypatch, {})
    weights_check.VERIFIED["checkpoints/stage_a/net_vae.pth"] = DIGEST_A
    manifest = _manifest({"global_restore": {"checkpoint": "checkpoints/stage_a"}})
    # net_map.pth was never proven and hashes to something else than DIGEST_B.
    assert manifest.verify_hashes() == {"global_restore": False}


# ============================== honesty checks =============================
def test_no_baseline_means_unverified_not_verified(sandbox, monkeypatch):
    """The old behaviour silently reported OK; that is what this pins shut."""
    monkeypatch.setattr(ModelManifest, "INTEGRITY_BASELINE_PATH", str(sandbox / "gone.json"))
    manifest = _manifest({"ddcolor": {"weight": "weights/ddcolor/pytorch_model.pt"}})
    assert manifest.integrity_baseline() == {}
    assert manifest.unverified_models() == ["ddcolor"]
    assert manifest.verify_hashes() == {"ddcolor": True}, "no evidence of tampering"


def test_health_lists_unverified_and_missing_weights(sandbox, tmp_path, monkeypatch):
    from fiximg.application.model_service import ModelService

    monkeypatch.setattr(ModelManifest, "INTEGRITY_BASELINE_PATH", str(tmp_path / "missing.json"))
    manifest = _manifest({"ddcolor": {"weight": "weights/ddcolor/absent.pt"}})

    class _Registry:
        def health_all(self):
            return {}

    class _Manager:
        def health(self):
            return {}

        def gpu_stats(self):
            return {}

    service = ModelService(
        registry=_Registry(), manager=_Manager(), manifest_provider=lambda: manifest
    )
    payload = service.health()["manifest"]
    assert payload["unverified"] == ["ddcolor"]
    assert payload["weights_present"]["ddcolor"] is False


# =========================== hot-swap validation ===========================
class _Resident:
    def __init__(self, name: str, version: str, weight_uri: str | None):
        self.name = name
        self.version = version
        self.weight_uri = weight_uri


def test_a_hot_swap_refuses_a_weight_that_does_not_match(sandbox, monkeypatch):
    from fiximg.inference.model_manager import ModelManager

    manifest = _manifest({"ddcolor": {"weight": "weights/ddcolor/pytorch_model.pt"}})
    monkeypatch.setattr(manifest_module, "get_manifest", lambda: manifest)
    weight = _abs(sandbox, "weights", "ddcolor", "pytorch_model.pt")
    with pytest.raises(ModelUnavailableError) as excinfo:
        ModelManager._validate(_Resident("ddcolor", "2.0.0", weight))
    assert excinfo.value.details["checked"] == 1
    assert excinfo.value.details["mismatches"][0]["path"] == "weights/ddcolor/pytorch_model.pt"


def test_a_hot_swap_passes_when_the_declared_hash_matches(sandbox, monkeypatch):
    from fiximg.inference.model_manager import ModelManager

    declared = hashlib.sha256(DDCOLOR_BYTES).hexdigest()
    manifest = _manifest({
        "ddcolor": {"weight": "weights/ddcolor/pytorch_model.pt", "sha256": declared}
    })
    monkeypatch.setattr(manifest_module, "get_manifest", lambda: manifest)
    ModelManager._validate(_Resident(
        "ddcolor", "2.0.0", _abs(sandbox, "weights", "ddcolor", "pytorch_model.pt")))


def test_a_hot_swap_to_an_unlisted_version_is_not_blocked(sandbox, monkeypatch):
    """A v2 directory the baseline knows nothing about is normal on a fresh install."""
    from fiximg.inference.model_manager import ModelManager

    manifest = _manifest({"ddcolor": {"framework": "pytorch"}})
    monkeypatch.setattr(manifest_module, "get_manifest", lambda: manifest)
    ModelManager._validate(_Resident("ddcolor", "2.0.0", _abs(sandbox, "unlisted.pt")))


# --------------------------------------------------- discovery exclusions
def test_the_exclude_list_names_no_directory_the_dot_rule_already_skips():
    """`.workbuddy` and `.arts` were named here, for directories this project has
    never contained, and both were dead weight: `_discover` already prunes every
    entry matching `not d.startswith(".")`, so listing a dot-directory is a no-op.

    They are leftovers from the AI tools that were used while this code was being
    written, and they had survived into shipped source. Verified equivalent at the
    time of the change: discovery returns the same 24 weight files either way.
    """
    dotted = sorted(d for d in weights_check.EXCLUDE_DIRS if d.startswith("."))
    assert not dotted, (
        f"{dotted} begin with a dot and are already pruned by the "
        'not d.startswith(".") clause in _discover, so listing them changes '
        "nothing and only records which tools happened to touch this repository"
    )


def test_discovery_ignores_dot_directories_and_caches(sandbox, monkeypatch):
    """The behaviour that makes the list above unnecessary, asserted directly."""
    root = Path(weights_check.BASE_DIR)
    decoys = (".workbuddy", ".arts", ".gates", "__pycache__", "node_modules")
    for name in decoys:
        decoy = root / name / "checkpoints"
        decoy.mkdir(parents=True, exist_ok=True)
        (decoy / "decoy_net.pth").write_bytes(b"not a real weight")
    try:
        found = sorted(rel for rel, _abs in weights_check._discover())
    finally:
        for name in decoys:
            import shutil
            shutil.rmtree(root / name, ignore_errors=True)

    leaked = sorted(f for f in found if any(d in f for d in decoys))
    assert not leaked, f"weights discovered inside an excluded directory: {leaked}"
    # And the real ones are still found, so the pruning is not over-eager.
    assert any("ddcolor" in f for f in found), found[:5]


def test_the_exclude_list_still_omits_caches_when_pruned_alone(sandbox, monkeypatch):
    """The set is not entirely redundant: `__pycache__` is load-bearing.

    `_discover`'s `not d.startswith(".")` clause handles every dot-directory, but
    `__pycache__` and `node_modules` do not begin with a dot, so removing them from
    the set makes the walker descend into build caches on every integrity check.
    """
    root = Path(weights_check.BASE_DIR)
    cache = root / "__pycache__"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "decoy.pth").write_bytes(b"x")
    try:
        as_is = sorted(rel for rel, _a in weights_check._discover())
        monkeypatch.setattr(weights_check, "EXCLUDE_DIRS", set())
        without = sorted(rel for rel, _a in weights_check._discover())
    finally:
        import shutil
        shutil.rmtree(cache, ignore_errors=True)

    extra = sorted(set(without) - set(as_is))
    assert any("__pycache__" in f for f in extra), (
        f"emptying EXCLUDE_DIRS changed discovery by {extra or 'nothing'}; if it "
        "changes by nothing then the whole set is redundant, and the comment "
        "claiming __pycache__ is load-bearing is wrong"
    )
