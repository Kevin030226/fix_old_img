"""Warmup strategy resolution and boot preloading (plan 搂3.5.3).

The manifest documents `warmup:` as a per-model override of `FIXIMG_WARMUP` and
sets it on every model, and `FIXIMG_WARMUP` names three strategies 鈥?but before
this file existed nothing read the per-model key, the API process preloaded
everything whenever the global said `startup`, and the standalone GPU worker
preloaded nothing. Each of those three is a separate claim below.
"""
import pytest

from fiximg.inference.model_manager import ModelManager


class _Declared:
    """Stand-in for a manifest entry (only `metadata.warmup` is read)."""

    def __init__(self, metadata):
        self.metadata = metadata


class _Manifest:
    def __init__(self, entries):
        self._entries = entries

    def get(self, name):
        return self._entries.get(name)


@pytest.fixture()
def manager(monkeypatch):
    """A manager over a settings stub and an in-memory manifest."""

    class _Settings:
        warmup_strategy = "lazy"
        device = "cpu"
        model_keep_previous = True
        base_dir = "."

    m = ModelManager(settings_obj=_Settings())
    return m, m, monkeypatch  # (manager, settings reachable via manager.settings, monkeypatch)


def _install_manifest(monkeypatch, entries):
    import fiximg.inference.manifest as manifest_module

    monkeypatch.setattr(manifest_module, "get_manifest", lambda: _Manifest(entries))


def test_the_manifest_key_overrides_the_global_strategy(manager, monkeypatch):
    m = manager[0]
    m.settings.warmup_strategy = "lazy"
    _install_manifest(monkeypatch, {"ddcolor": _Declared({"warmup": "first-use"})})
    assert m.warmup_strategy("ddcolor") == "first-use"


def test_a_model_without_the_key_inherits_the_global_strategy(manager, monkeypatch):
    m = manager[0]
    m.settings.warmup_strategy = "startup"
    _install_manifest(monkeypatch, {"ddcolor": _Declared({})})
    assert m.warmup_strategy("ddcolor") == "startup"


def test_an_unknown_strategy_falls_back_to_the_global_one(manager, monkeypatch):
    """A typo in the manifest must not silently select a different policy."""
    m = manager[0]
    m.settings.warmup_strategy = "first-use"
    _install_manifest(monkeypatch, {"ddcolor": _Declared({"warmup": "eagerly"})})
    assert m.warmup_strategy("ddcolor") == "first-use"


def test_only_lazy_skips_the_dummy_call(manager, monkeypatch):
    m = manager[0]
    _install_manifest(monkeypatch, {
        "a": _Declared({"warmup": "lazy"}),
        "b": _Declared({"warmup": "first-use"}),
        "c": _Declared({"warmup": "startup"}),
    })
    assert m.should_warm_on_load("a") is False
    assert m.should_warm_on_load("b") is True
    assert m.should_warm_on_load("c") is True


def test_activation_passes_the_warmer_the_strategy_asked_for(manager, monkeypatch):
    """The resolved decision has to reach the registry, not just exist on paper."""
    m = manager[0]
    _install_manifest(monkeypatch, {"ddcolor": _Declared({"warmup": "lazy"})})
    seen = {}

    class _Registry:
        def current_handle(self, name):
            return None

        def current_version(self, name):
            return None

        def known_models(self):
            return ["ddcolor"]

        def activate(self, name, version, **kwargs):
            seen.update(kwargs)
            return {"active_version": version, "residents": [{"version": version, "load_ms": 1}]}

    m.registry = _Registry()
    monkeypatch.setattr(m, "declared_version", lambda name: "1.0.0")
    monkeypatch.setattr(m, "_declared_weight", lambda name: "weights/x.pt")
    monkeypatch.setattr(m, "_validate", lambda resident: None)
    monkeypatch.setattr(m, "_probe", lambda resident: True)

    m.load("ddcolor")
    assert seen["warmer"] is None, "a lazy model still paid the dummy inference"

    _install_manifest(monkeypatch, {"ddcolor": _Declared({"warmup": "first-use"})})
    m.load("ddcolor")
    assert seen["warmer"] == m._warm


# ------------------------------------------------------------- boot preloading
class _Backend:
    def __init__(self, name, calls, fail=False):
        self.name = name
        self.calls = calls
        self.fail = fail

    def load(self, device):
        self.calls.append(("load", self.name))
        if self.fail:
            raise RuntimeError("no weights on this node")

    def warmup(self):
        self.calls.append(("warm", self.name))


def _registry_of(backends):
    from fiximg.inference.backends.registry import ModelBackendRegistry

    registry = ModelBackendRegistry()
    for backend in backends:
        registry.register(backend.name, lambda b=backend: b)
    return registry


def test_startup_preloads_only_the_models_whose_strategy_says_so(manager, monkeypatch):
    m = manager[0]
    calls = []
    backends = [
        _Backend("global_restore", calls),
        _Backend("ddcolor", calls),
        _Backend("face_enhancement", calls),
    ]
    import fiximg.inference.backends.registry as registry_module

    monkeypatch.setattr(registry_module, "backend_registry", _registry_of(backends))
    _install_manifest(monkeypatch, {
        "global_restore": _Declared({"warmup": "startup"}),
        "ddcolor": _Declared({"warmup": "first-use"}),
        "face_enhancement": _Declared({"warmup": "lazy"}),
    })

    warmed = m.warm_at_process_start()

    assert warmed == ["global_restore"], warmed
    assert calls == [("load", "global_restore"), ("warm", "global_restore")]


def test_a_failing_model_does_not_stop_the_others(manager, monkeypatch):
    """Warmup is best effort: one model without weights must not cold-start the rest."""
    m = manager[0]
    calls = []
    backends = [
        _Backend("a", calls, fail=True),
        _Backend("b", calls),
    ]
    import fiximg.inference.backends.registry as registry_module

    monkeypatch.setattr(registry_module, "backend_registry", _registry_of(backends))
    _install_manifest(monkeypatch, {
        "a": _Declared({"warmup": "startup"}),
        "b": _Declared({"warmup": "startup"}),
    })

    assert m.warm_at_process_start() == ["b"]


def test_the_default_configuration_preloads_nothing(manager, monkeypatch):
    """`lazy` is the shipped default, so starting a worker must not load models."""
    m = manager[0]
    m.settings.warmup_strategy = "lazy"
    _install_manifest(monkeypatch, {"ddcolor": _Declared({"warmup": "lazy"})})
    import fiximg.inference.backends.registry as registry_module

    calls = []
    monkeypatch.setattr(registry_module, "backend_registry",
                        _registry_of([_Backend("ddcolor", calls)]))

    assert m.warm_at_process_start() == []
    assert calls == []


# ---------------------------------------------------------------- the call sites
def test_a_worker_warms_at_start(monkeypatch):
    """The process that executes inference is the one that has to be warm."""
    import fiximg.inference.model_manager as mm

    calls = []
    monkeypatch.setattr(mm.model_manager, "warm_at_process_start",
                        lambda: calls.append("warmed") or ["ddcolor"])

    from fiximg.inference.worker import PipelineWorker

    class _IdleQueue:
        kind = "test"

        def claim(self, worker_id, lease_seconds=3600.0):
            return None

        def depth(self):
            return 0

    worker = PipelineWorker(object(), poll_seconds=0.05, queue=_IdleQueue())
    worker.start()
    try:
        assert calls == ["warmed"]
    finally:
        worker.stop()


def test_a_warming_failure_still_starts_the_worker(monkeypatch):
    """A cold model is a performance problem, not a reason to refuse work."""
    import fiximg.inference.model_manager as mm

    def boom():
        raise RuntimeError("weights missing")

    monkeypatch.setattr(mm.model_manager, "warm_at_process_start", boom)

    from fiximg.inference.worker import PipelineWorker

    class _IdleQueue:
        kind = "test"

        def claim(self, worker_id, lease_seconds=3600.0):
            return None

        def depth(self):
            return 0

    worker = PipelineWorker(object(), poll_seconds=0.05, queue=_IdleQueue())
    worker.start()  # must not raise
    assert worker.is_running()
    worker.stop()


def test_the_api_process_uses_the_same_resolver(monkeypatch):
    """Two processes, one definition of 'when is a model warmed'."""
    import fiximg.app_factory as factory
    import fiximg.inference.model_manager as mm

    seen = []
    monkeypatch.setattr(mm.model_manager, "warm_at_process_start",
                        lambda: seen.append("called") or [])

    factory._warmup_models()
    assert seen == ["called"]
