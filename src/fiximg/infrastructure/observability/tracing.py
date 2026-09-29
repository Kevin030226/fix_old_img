"""Tracing seams (plan §2.10).

V3 defines one seam, :func:`get_tracer`, and every instrumented call site goes
through it. Three implementations, chosen by ``FIXIMG_TRACING``:

======================  =========================================================
``off`` (default)        :class:`NoOpTracer` — call sites unchanged, zero overhead
``sdk``                  :class:`OtelTracer` — delegates to the process-wide
                         OpenTelemetry tracer, so the *deployment* (an
                         auto-instrumentation agent or its own ``OTEL_*`` config)
                         decides exporters and propagation
``internal``             :class:`_RecordingTracer` — last N spans in memory, for
                         tests and the admin panel
======================  =========================================================

``sdk`` requires the optional extra (``pip install 'fiximg[otel]'`). Asking for
it without the package is a startup warning, not a silent fall back to no-op:
a deployment that says "trace" must be able to tell that it is not being traced.

Usage::

    from fiximg.infrastructure.observability.tracing import tracer

    with tracer.span("inference.stage", task_id=tid, stage=name):
        ...

The module-level ``tracer`` is a proxy that resolves :func:`get_tracer` on every
call, so :func:`install_tracer` takes effect everywhere — an alias bound at
import time would silently pin every call site to whichever tracer existed
first.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections.abc import Iterator


@dataclass
class Span:
    """A timed unit of work; attribute-only in the default implementation."""

    name: str
    attributes: dict = field(default_factory=dict)
    started_at: float = 0.0
    duration_s: float | None = None


class NoOpTracer:
    """Records nothing but keeps call sites identical to a real tracer."""

    enabled = False

    @contextmanager
    def span(self, name: str, **attributes) -> Iterator[Span]:
        span = Span(name=name, attributes=dict(attributes), started_at=time.perf_counter())
        try:
            yield span
        finally:
            span.duration_s = time.perf_counter() - span.started_at

    def current_context(self) -> dict:
        """Correlation ids from the logging context (request/task/user)."""
        try:
            from fiximg.infrastructure.observability.logging import current_context

            return current_context()
        except Exception:  # noqa: BLE001 — tracing must never break a request
            return {}


class _RecordingTracer(NoOpTracer):
    """Test/debug tracer that keeps the last N spans in memory.

    A span is recorded whether or not its body succeeded: the failure case is the
    one an operator opens this tracer to look at, and dropping it would make the
    recording view blind to exactly the runs worth debugging.
    """

    enabled = True

    def __init__(self, max_spans: int = 200) -> None:
        self._spans: list[Span] = []
        self._max = max_spans

    @contextmanager
    def span(self, name: str, **attributes) -> Iterator[Span]:
        span = Span(name=name, attributes=dict(attributes), started_at=time.perf_counter())
        try:
            yield span
        finally:
            span.duration_s = time.perf_counter() - span.started_at
            self._spans.append(span)
            if len(self._spans) > self._max:
                del self._spans[: len(self._spans) - self._max]

    def recorded(self) -> list[Span]:
        return list(self._spans)

    def reset(self) -> None:
        self._spans.clear()


class OtelTracer(NoOpTracer):
    """Adapter onto the process-wide OpenTelemetry tracer.

    Deliberately does *not* build a provider or exporter: in OTel's embedding
    model that is the deployment's job (auto-instrumentation, or an entry point
    that configures ``OTEL_*``). This class only mirrors fiximg's span names,
    attributes and failures into whatever the host process installed.
    """

    enabled = True

    def __init__(self, instrumentation_name: str = "fiximg") -> None:
        from opentelemetry import trace

        self._tracer = trace.get_tracer(instrumentation_name)

    @staticmethod
    def _clean(attributes: dict) -> dict:
        """OTel rejects None values and non-scalar attributes."""
        cleaned = {}
        for key, value in attributes.items():
            if value is None:
                continue
            cleaned[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
        return cleaned

    @contextmanager
    def span(self, name: str, **attributes) -> Iterator[Span]:
        from opentelemetry.trace import StatusCode

        span = Span(name=name, attributes=dict(attributes), started_at=time.perf_counter())
        with self._tracer.start_as_current_span(name, attributes=self._clean(attributes)) as otel_span:
            try:
                yield span
            except BaseException as exc:
                otel_span.record_exception(exc)
                otel_span.set_status(StatusCode.ERROR, str(exc) or type(exc).__name__)
                raise
            finally:
                span.duration_s = time.perf_counter() - span.started_at


def _configured_mode() -> str:
    """The tracing mode: settings first, raw environment as the fallback."""
    try:
        from fiximg.config import settings

        return str(getattr(settings, "tracing", "") or "").strip().lower()
    except Exception:  # noqa: BLE001 — tracing must never be the reason nothing runs
        return (os.environ.get("FIXIMG_TRACING") or "").strip().lower()


def default_tracer() -> NoOpTracer:
    """Build the tracer ``FIXIMG_TRACING`` asks for (``off`` when unset)."""
    mode = _configured_mode() or "off"
    if mode in ("off", "none", "", "0", "no"):
        return NoOpTracer()
    if mode in ("internal", "record", "records"):
        return _RecordingTracer()
    try:
        return OtelTracer()
    except ImportError:
        # Enabled-but-unavailable must be loud: the alternative is a deployment
        # believing it is exporting traces while nothing leaves the process.
        import logging

        logging.getLogger("fiximg.tracing").warning(
            "FIXIMG_TRACING=%s but opentelemetry is not installed; running untraced. "
            "Install with: pip install 'fiximg[otel]'",
            mode,
        )
        return NoOpTracer()


def get_tracer():
    """The active tracer: whatever was installed, else the configured default."""
    global _tracer
    if _tracer is None:
        _tracer = default_tracer()
    return _tracer


def install_tracer(tracer) -> None:
    """Swap in a tracer (OpenTelemetry adapter, test recorder, ...) process-wide."""
    global _tracer
    _tracer = tracer


class _TracerProxy:
    """Delegates to :func:`get_tracer` on every call.

    Module-level call sites import ``tracer``; binding the alias once would freeze
    them to the tracer that existed at import time and make
    :func:`install_tracer` a no-op for them.
    """

    @property
    def enabled(self) -> bool:
        return get_tracer().enabled

    def span(self, name: str, **attributes):
        return get_tracer().span(name, **attributes)

    def current_context(self) -> dict:
        return get_tracer().current_context()


#: Resolved at module import; None means "not installed yet, ask default_tracer()".
_tracer: NoOpTracer | None = None

#: The handle call sites use. See :class:`_TracerProxy`.
tracer = _TracerProxy()


__all__ = [
    "NoOpTracer",
    "OtelTracer",
    "Span",
    "default_tracer",
    "get_tracer",
    "install_tracer",
    "tracer",
]
