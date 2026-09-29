"""Inference optimisation tests (plan 搂3.5.4, 搂3.5.5).

Two capabilities under test:

* **precision policy** 鈥?the plan's ``inference_mode`` + ``autocast`` wins, gated
  by what each model declares. Nothing may be switched on globally.
* **image size policy** 鈥?the four-tier decision table, and the guarantee that
  adaptive resize does not change what the user downloads.
"""
import numpy as np
import pytest
from PIL import Image

from fiximg.inference import size_policy
from fiximg.inference.precision import (
    PrecisionPolicy,
    apply_declared_policy,
    apply_model_optimisations,
    cached_policy,
    inference_context,
    reset_policy_cache,
    resolve_policy,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_policy_cache()
    yield
    reset_policy_cache()


# --------------------------------------------------------------- policy model
def test_default_policy_is_plain_fp32():
    policy = PrecisionPolicy()
    assert policy.inference_mode is True   # always safe, always on
    assert policy.autocast_dtype is None
    assert policy.compile is False
    assert policy.channels_last is False
    assert policy.is_accelerated is False


def test_policy_reads_fp16_and_bf16():
    assert PrecisionPolicy.from_declared({"precision": "fp16"}).autocast_dtype == "float16"
    assert PrecisionPolicy.from_declared({"precision": "bf16"}).autocast_dtype == "bfloat16"
    assert PrecisionPolicy.from_declared({"precision": "half"}).autocast_dtype == "float16"


def test_policy_treats_fp32_aliases_as_no_autocast():
    for raw in ("fp32", "float32", "FP32", ""):
        assert PrecisionPolicy.from_declared({"precision": raw}).autocast_dtype is None


def test_unknown_precision_falls_back_to_fp32():
    """A typo must not enable half precision by accident."""
    assert PrecisionPolicy.from_declared({"precision": "fp8"}).autocast_dtype is None


def test_policy_reads_compile_and_channels_last():
    policy = PrecisionPolicy.from_declared(
        {"channels_last": True, "compile": True, "compile_mode": "reduce-overhead"}
    )
    assert policy.channels_last is True
    assert policy.compile is True
    assert policy.compile_mode == "reduce-overhead"
    assert policy.is_accelerated is True


def test_unknown_compile_mode_falls_back_to_default():
    assert PrecisionPolicy.from_declared({"compile_mode": "turbo"}).compile_mode == "default"


def test_policy_from_none_is_the_default():
    assert PrecisionPolicy.from_declared(None) == PrecisionPolicy()


def test_policy_describe_is_serialisable():
    described = PrecisionPolicy.from_declared({"precision": "fp16", "compile": True}).describe()
    assert described["precision"] == "float16"
    assert described["compile"] is True
    assert described["accelerated"] is True


# ------------------------------------------------------------ manifest lookup
def test_manifest_declares_ddcolor_precision():
    """The shipped manifest must declare a policy for the in-process model."""
    policy = resolve_policy("ddcolor")
    assert policy.autocast_dtype == "float16"
    assert policy.channels_last is True


def test_subprocess_models_declare_fp32():
    for name in ("global_restore", "face_detection"):
        assert resolve_policy(name).autocast_dtype is None


def test_unknown_model_gets_the_default_policy():
    assert resolve_policy("not-a-model") == PrecisionPolicy()


def test_env_override_wins_over_the_manifest(monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "precision_override", "fp32")
    assert resolve_policy("ddcolor").autocast_dtype is None


def test_env_override_of_auto_honours_the_manifest(monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "precision_override", "auto")
    assert resolve_policy("ddcolor").autocast_dtype == "float16"


def test_env_override_with_garbage_is_ignored(monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "precision_override", "fp8")
    assert resolve_policy("ddcolor").autocast_dtype == "float16"


def test_cached_policy_memoises():
    first = cached_policy("ddcolor")
    assert cached_policy("ddcolor") is first


# -------------------------------------------------------- inference context
def test_inference_context_enters_inference_mode():
    torch = pytest.importorskip("torch")

    with inference_context(PrecisionPolicy()):
        assert torch.is_inference_mode_enabled()


def test_inference_context_skips_autocast_on_cpu():
    """Half precision on CPU is unsupported; the policy must not try."""
    pytest.importorskip("torch")

    policy = PrecisionPolicy.from_declared({"precision": "fp16"})
    # Should simply not raise: autocast is only entered for CUDA devices.
    with inference_context(policy, device="cpu"):
        pass
    with inference_context(policy, device=-1):
        pass


def test_inference_context_without_torch_is_a_noop(monkeypatch):
    """CPU-only deployments may not have torch importable at all."""
    import fiximg.inference.precision as precision_mod

    monkeypatch.setattr(precision_mod, "_torch", lambda: None)
    with inference_context(PrecisionPolicy.from_declared({"precision": "fp16"})):
        pass


def test_apply_optimisations_is_a_noop_for_the_default_policy():
    class _Model:
        pass

    model = _Model()
    assert apply_model_optimisations(model, PrecisionPolicy()) is model


def test_apply_optimisations_survives_a_bad_model():
    """A model that cannot take the optimisation must not be taken offline."""
    class _Model:
        def to(self, **_kwargs):
            raise RuntimeError("unsupported memory format")

    policy = PrecisionPolicy(channels_last=True)
    model = _Model()
    assert apply_model_optimisations(model, policy) is model


class _FakeModel:
    """A torch.Module stand-in that records what it was asked to do."""

    def __init__(self, refuse_layout: bool = False) -> None:
        self.calls: list[str] = []
        self.compiled_with: str | None = None
        self.refuse_layout = refuse_layout

    def to(self, *args, **kwargs):
        if "memory_format" in kwargs:
            if self.refuse_layout:
                raise RuntimeError("unsupported memory format")
            self.calls.append("channels_last")
        return self

    def __repr__(self) -> str:
        return f"<_FakeModel {self.calls}>"


def test_declared_policy_reports_each_key_that_landed():
    """`channels_last` lands, `compile` is declined on a non-CUDA device.

    Both have to be *said*: reporting nothing about the declined one is how a
    declaration became decorative for every backend except the first.
    """
    model = _FakeModel()
    policy = PrecisionPolicy(channels_last=True, compile=True, compile_mode="default")
    returned, applied = apply_declared_policy(model, policy, device="cpu")

    assert returned is model
    assert applied == {"channels_last": True, "compile": False}, applied
    assert model.calls == ["channels_last"]


def test_a_model_that_refuses_the_layout_is_reported_as_declined():
    """A failed optimisation must neither offline the model nor claim success."""
    model = _FakeModel(refuse_layout=True)
    returned, applied = apply_declared_policy(
        model, PrecisionPolicy(channels_last=True), device="cpu"
    )
    assert returned is model, "the original model must still serve"
    assert applied == {"channels_last": False}, applied


def test_nothing_declared_nothings_reported():
    model = _FakeModel()
    assert apply_declared_policy(model, PrecisionPolicy(), device="cpu") == (model, {})
    assert model.calls == []


def test_is_cuda_recognises_device_forms():
    from fiximg.inference.precision import _is_cuda

    assert _is_cuda("cpu") is False
    assert _is_cuda(-1) is False
    # No CUDA on the CI host: 'auto' resolves to False, and that is fine.
    assert isinstance(_is_cuda("auto"), bool)


# =============================== image size policy ===============================
def test_direct_band():
    decision = size_policy.decide((800, 600), direct_max=1024, resize_max=2048, tile_max=4096)
    assert decision.strategy == size_policy.DIRECT
    assert decision.scale == 1.0
    assert decision.resizes is False


def test_direct_band_includes_the_boundary():
    decision = size_policy.decide((1024, 512), direct_max=1024, resize_max=2048, tile_max=4096)
    assert decision.strategy == size_policy.DIRECT


def test_adaptive_resize_band():
    decision = size_policy.decide((1600, 900), direct_max=1024, resize_max=2048, tile_max=4096)
    assert decision.strategy == size_policy.ADAPTIVE_RESIZE
    assert decision.target_long_side == 1024
    assert decision.scale == pytest.approx(1024 / 1600)
    assert decision.resizes is True


def test_adaptive_resize_can_be_disabled():
    decision = size_policy.decide(
        (1600, 900), direct_max=1024, resize_max=2048, tile_max=4096, adaptive_resize=False
    )
    assert decision.strategy == size_policy.DIRECT
    assert decision.resizes is False
    assert "disabled" in decision.reason


def test_tile_band():
    decision = size_policy.decide((3000, 2000), direct_max=1024, resize_max=2048, tile_max=4096)
    assert decision.strategy == size_policy.TILE
    assert decision.resizes is False


def test_tile_band_includes_the_boundary():
    decision = size_policy.decide((4096, 100), direct_max=1024, resize_max=2048, tile_max=4096)
    assert decision.strategy == size_policy.TILE


def test_reject_band():
    decision = size_policy.decide((5000, 100), direct_max=1024, resize_max=2048, tile_max=4096)
    assert decision.strategy == size_policy.REJECT
    assert decision.rejects is True
    assert "5000" in decision.reason


def test_non_positive_dimension_is_rejected():
    assert size_policy.decide((0, 100), direct_max=1024, resize_max=2048,
                              tile_max=4096).strategy == size_policy.REJECT


def test_decision_uses_the_long_side():
    """A tall image is classified by its height, not its width."""
    decision = size_policy.decide((100, 3000), direct_max=1024, resize_max=2048, tile_max=4096)
    assert decision.strategy == size_policy.TILE


def test_decision_is_serialisable():
    decision = size_policy.decide((1600, 900), direct_max=1024, resize_max=2048, tile_max=4096)
    payload = decision.to_dict()
    assert payload["strategy"] == size_policy.ADAPTIVE_RESIZE
    assert set(payload) == {"strategy", "target_long_side", "scale", "reason"}


def test_decide_from_settings_uses_the_configured_thresholds(monkeypatch):
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "size_direct_max", 100)
    monkeypatch.setattr(config_mod.settings, "size_resize_max", 200)
    monkeypatch.setattr(config_mod.settings, "size_tile_max", 300)
    monkeypatch.setattr(config_mod.settings, "size_adaptive_resize", True)

    assert size_policy.decide_from_settings((50, 50)).strategy == size_policy.DIRECT
    assert size_policy.decide_from_settings((150, 50)).strategy == size_policy.ADAPTIVE_RESIZE
    assert size_policy.decide_from_settings((250, 50)).strategy == size_policy.TILE
    assert size_policy.decide_from_settings((350, 50)).strategy == size_policy.REJECT


# ------------------------------------------------------------- resize helpers
def test_resize_for_inference_preserves_aspect_ratio():
    image = Image.new("RGB", (1600, 900), "red")
    decision = size_policy.decide((1600, 900), direct_max=1024, resize_max=2048, tile_max=4096)
    resized, original = size_policy.resize_for_inference(image, decision)

    assert original == (1600, 900)
    assert resized.size == (1024, 576)          # 1024/1600 * 900 = 576
    assert resized.mode == "RGB"


def test_resize_for_inference_is_a_noop_in_the_direct_band():
    image = Image.new("RGB", (100, 80), "blue")
    decision = size_policy.decide((100, 80), direct_max=1024, resize_max=2048, tile_max=4096)
    resized, original = size_policy.resize_for_inference(image, decision)
    assert resized is image
    assert original == (100, 80)


def test_restore_original_size_scales_the_result_back():
    image = Image.new("RGB", (1024, 576), "green")
    restored = size_policy.restore_original_size(image, (1600, 900))
    assert restored.size == (1600, 900)


def test_restore_original_size_is_a_noop_when_sizes_match():
    image = Image.new("RGB", (100, 80), "green")
    assert size_policy.restore_original_size(image, (100, 80)) is image


def test_restore_original_size_tolerates_none():
    assert size_policy.restore_original_size(None, (10, 10)) is None


def test_adaptive_resize_roundtrip_keeps_the_output_dimensions():
    """The 搂3.5.5 guarantee: the user's image comes back at its own size."""
    source = Image.new("RGB", (1600, 900), "orange")
    decision = size_policy.decide_from_settings((1600, 900))
    assert decision.resizes is True

    prepared, original = size_policy.resize_for_inference(source, decision)
    assert prepared.size != source.size

    final = size_policy.restore_original_size(prepared, original)
    assert final.size == source.size


# --------------------------------------------------- wired into the runtime
def test_runtime_normalize_input_applies_the_policy():
    from fiximg.inference.runtime import PipelineOrchestrator

    source = Image.new("RGB", (1600, 900), "orange")
    prepared, decision, original = PipelineOrchestrator._normalize_input(source)
    assert decision.strategy == size_policy.ADAPTIVE_RESIZE
    assert prepared.size == (1024, 576)
    assert original == (1600, 900)


def test_runtime_normalize_input_rejects_oversized(monkeypatch):
    import fiximg.config as config_mod
    from fiximg.domain.errors import InvalidRequestError
    from fiximg.inference.runtime import PipelineOrchestrator

    monkeypatch.setattr(config_mod.settings, "size_tile_max", 100)
    with pytest.raises(InvalidRequestError) as excinfo:
        PipelineOrchestrator._normalize_input(Image.new("RGB", (200, 50)))
    # The policy's own reason is user-facing and must survive verbatim.
    assert "200px" in str(excinfo.value)


def test_runtime_normalize_input_converts_arrays_and_modes():
    from fiximg.inference.runtime import PipelineOrchestrator

    array = np.zeros((32, 24, 3), dtype=np.uint8)
    prepared, decision, original = PipelineOrchestrator._normalize_input(array)
    assert prepared.mode == "RGB"
    assert prepared.size == (24, 32)
    assert decision.strategy == size_policy.DIRECT
    assert original == (24, 32)


# ------------------------------------------------ 搂3.5.5 output geometry
def test_the_download_keeps_the_callers_dimensions(tmp_path, monkeypatch, isolated_db):
    """The model chain rounds geometry; the user's download must not shrink.

    Measured on a live run of the app: a 298脳450 upload came back as 296脳448,
    because the vendored restoration resizes each side to a multiple of 4 (the
    detector rounds to 16), `size_decision.resizes` was False on the direct band,
    and nothing undid the rounding. The same 2 px made the quality report compare
    two different geometries, which is how an ordinary result scored SSIM -0.20.
    """
    from types import SimpleNamespace

    import fiximg.config as config_mod
    from fiximg.inference.context import StageResult
    from fiximg.inference.runtime import PipelineOrchestrator

    monkeypatch.setattr(config_mod.settings, "tasks_root", str(tmp_path / "tasks"))

    class _RoundingStage:
        """What the real chain does: work in multiples of 4."""

        name = "global_restore"
        version = "test"
        capabilities = frozenset({"restore"})

        def run(self, image, context):
            shrunk = image.resize((image.width - 2, image.height - 2))
            return StageResult(image=shrunk, metadata={"stage": "restore"})

    stage = _RoundingStage()
    orchestrator = PipelineOrchestrator(
        planner=SimpleNamespace(
            plan=lambda task_type, options=None: SimpleNamespace(
                task_type=task_type, stages=[("global_restore", {})], decisions={}
            ),
            build_stage=lambda name, kwargs: stage,
        )
    )

    from PIL import Image

    result_path, _text = orchestrator.run(
        Image.new("RGB", (298, 450), "grey"), {"username": "alice"}, "restore"
    )

    with Image.open(result_path) as result:
        assert result.size == (298, 450), (
            "the caller's geometry must survive a chain that works in /4 multiples"
        )


# ------------------- the declaration must reach every native backend (搂3.5.4)
def _load_path_source(cls) -> str:
    """Source of ``_do_load`` plus the private helpers it calls, one level deep.

    DDColor keeps its module inside a pipeline object and handles it in
    ``_apply_optimisations``, so looking at ``_do_load`` alone would report a
    backend that does apply its policy as one that ignores it.
    """
    import inspect
    import re

    text = inspect.getsource(cls._do_load)
    for name in dir(cls):
        if not name.startswith("_") or name == "_do_load":
            continue
        member = getattr(cls, name, None)
        if not callable(member):
            continue
        if re.search(rf"self\.{name}\(", text):
            try:
                text += "\n" + inspect.getsource(member)
            except (TypeError, OSError):  # pragma: no cover - builtins
                continue
    return text


def _policy_backends():
    from fiximg.inference.backends.ddcolor import DDColorBackend
    from fiximg.inference.backends.face_enhance_native import NativeFaceEnhancementBackend
    from fiximg.inference.backends.global_restore import (
        GlobalRestoreBackend,
        NativeScratchRepairBackend,
    )

    return {
        "ddcolor": DDColorBackend,
        "face_enhancement": NativeFaceEnhancementBackend,
        "global_restore": GlobalRestoreBackend,
        "scratch_repair": NativeScratchRepairBackend,
    }


@pytest.mark.parametrize("name", sorted(_policy_backends()))
def test_every_native_torch_backend_applies_its_declared_policy(name):
    """A manifest key that one backend reads and another ignores is a lie.

    ``precision``/``channels_last``/``compile`` were honoured only by DDColor, so
    declaring ``channels_last: true`` for the face or restoration chain changed
    nothing while ``GET /api/v1/models`` still presented the model as configured.
    Every backend that has a policy must route its model through
    :meth:`BaseModelBackend.apply_policy`, which also records what landed.
    """
    cls = _policy_backends()[name]
    assert getattr(cls, "policy", None) is not None, name
    assert "apply_policy" in _load_path_source(cls), (
        f"{cls.__name__} has a precision policy but its load path never applies it"
    )


def test_a_declaration_reaches_the_backend_that_serves_it():
    """Read the manifest, not the test list: any model may declare an optimisation.

    Adding ``channels_last: true`` to an entry whose backend has no policy - or
    whose serving implementation is the subprocess adapter - must fail here rather
    than silently do nothing in production.
    """
    from fiximg.inference.backends.registry import backend_registry
    from fiximg.inference.manifest import get_manifest

    with_policy = set(_policy_backends())
    offenders: list[str] = []

    for model_name in get_manifest().names():
        declared = get_manifest().get(model_name)
        metadata = getattr(declared, "metadata", {}) or {}
        asks_for_optimisation = metadata.get("channels_last") or metadata.get("compile")
        if not asks_for_optimisation:
            continue
        backend = backend_registry.get(model_name)
        if backend is None:
            offenders.append(f"{model_name}: declared but no backend serves it")
        elif model_name not in with_policy:
            offenders.append(f"{model_name}: declares an optimisation, "
                             f"{type(backend).__name__} has no policy")
        elif not getattr(backend, "policy", None):
            offenders.append(f"{model_name}: backend exposes no policy object")

    assert not offenders, offenders


def test_a_declined_optimisation_is_recorded_on_the_backend(monkeypatch):
    """The backend, not only the log, knows that its declaration did not land."""
    from fiximg.inference.backends.global_restore import GlobalRestoreBackend

    policy = PrecisionPolicy(channels_last=True, compile=True)
    monkeypatch.setattr(GlobalRestoreBackend, "policy",
                        property(lambda self: policy), raising=False)
    backend = GlobalRestoreBackend()
    backend.device = "cpu"

    model = _FakeModel(refuse_layout=True)
    returned = backend.apply_policy(model)

    assert returned is model, "a declined optimisation must not drop the model"
    reported = backend.describe()["optimisations"]
    assert reported["applied"] == {"channels_last": False, "compile": False}, reported
    assert reported["channels_last"] is True, "still declared - the record says both"


def test_unloading_clears_the_record_of_what_landed(monkeypatch):
    """`applied` describes the model that is loaded now, not the last one."""
    from fiximg.inference.backends.global_restore import GlobalRestoreBackend

    policy = PrecisionPolicy(channels_last=True)
    monkeypatch.setattr(GlobalRestoreBackend, "policy",
                        property(lambda self: policy), raising=False)
    backend = GlobalRestoreBackend()
    backend.device = "cpu"
    backend.apply_policy(_FakeModel())
    assert backend.optimisations_applied["channels_last"] is True

    monkeypatch.setattr(GlobalRestoreBackend, "_do_unload", lambda self: None)
    backend._loaded = True
    backend.unload()
    assert backend.optimisations_applied == {}


def test_is_cuda_never_claims_a_device_the_host_does_not_have():
    """A device *descriptor* is not a device.

    Autocast is entered on the strength of `_is_cuda`, and on a CPU-only host asking
    for card 1 used to satisfy that check: torch then prints "CUDA is not available,
    disabling autocast" on every call and runs fp32 鈥?so the report said the declared
    half-precision path was taken while it silently was not.
    """
    import torch

    from fiximg.inference.precision import _is_cuda

    available = bool(torch.cuda.is_available())
    assert _is_cuda("1") is available
    assert _is_cuda("cuda:0") is available
    assert _is_cuda(0) is available
    assert _is_cuda("cpu") is False
    assert _is_cuda(-1) is False
