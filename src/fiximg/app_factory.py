"""Application composition root (plan §2.1 / §26).

``main.py`` shrinks to a shim; everything that wires the process together lives
here. V3 makes the assembly explicit instead of one long function:

    create_app()
      ├── register_routes()      JSON API routers (health/tasks/models/users/stats)
      ├── register_middleware()  correlation ids, acting-user attribution
      ├── register_ui()          the native web UI at "/"
      └── _app_lifespan()        DB init, security bootstrap, worker start/stop

The factory only *assembles*; it never implements business policy.

The UI is a FastAPI router (`fiximg.ui.web.routes`), not a mounted third-party
application. That is the whole of the P17 change: the shell it replaced
(`gradio.Blocks` mounted at `/`) was the only reason `gradio` was in the published
dependency graph, and `gradio` pins `pillow<13` — so the UI framework was the
single thing standing between this project and the Pillow release that clears the
open advisories. See `reports/p17_fastapi_ui_migration.md`.

GPU worker topology (plan §12/§28):
  - FIXIMG_INLINE_WORKER=true (default): a PipelineWorker thread consumes the
    queue inside this process — single-container deployments.
  - FIXIMG_INLINE_WORKER=false: run ``python worker.py`` as a separate process
    (docker-compose topology); the API then only enqueues.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from fiximg.api.errors import register_exception_handlers
from fiximg.api.routes import auth as auth_routes
from fiximg.api.routes import health, models as models_routes, stats, tasks, users
from fiximg.config import settings
from fiximg.infrastructure.db.engine import apply_database_url, init_db
from fiximg.infrastructure.db.repositories.task_repository import ensure_schema
from fiximg.ui.web.routes import register_ui_routes

#: The page title, kept in one place so the document and the API agree.
APP_TITLE = (
    "Old Photo Restoration & Scratch Repair via Deep Learning "
    "(GANs and Variational Autoencoders)"
)

# Reference to the inline PipelineWorker while the app is running (None otherwise).
_inline_worker = None


def get_inline_worker():
    """Return the in-process PipelineWorker, or None when running standalone-mode."""
    return _inline_worker
# --------------------------------------------------------------------- lifecycle
@asynccontextmanager
async def _app_lifespan(_app):
    """Initialize the database and start the inline worker at app startup."""
    global _inline_worker
    # §4.3 Step 2: an explicitly configured database URL moves DB_PATH; an
    # unsupported scheme raises here rather than being silently ignored.
    apply_database_url()
    init_db()
    ensure_schema()

    _bootstrap_security()

    _worker = None
    if settings.inline_worker:
        from fiximg.inference.runtime import PipelineOrchestrator
        from fiximg.inference.worker import PipelineWorker

        _worker = PipelineWorker(
            PipelineOrchestrator(),
            poll_seconds=settings.worker_poll_seconds,
            worker_id=f"inline-{id(_app):x}",
        )
        _worker.start()
        _inline_worker = _worker

    _warmup_models()
    _check_worker_visibility()
    try:
        yield
    finally:
        if _worker is not None:
            _worker.stop()
        _inline_worker = None


def _check_worker_visibility() -> None:
    """Say once, at boot, that the UI will run inference in this process.

    Found by doing it. A deployment running `fiximg.cli.api` with
    ``FIXIMG_INLINE_WORKER=0`` and a separate `fiximg.cli.worker` looks correct —
    the worker is there, tasks complete, `/health/ready` reports the worker
    topology. But the UI cannot see it: ``has_worker()`` reads
    ``settings.external_worker``, which is ``FIXIMG_HAS_EXTERNAL_WORKER`` and
    defaults to false, and a separate process is invisible to it. So every UI
    submission took the synchronous fallback and executed inference *inside the
    API process* while the worker held its own copy of the models.

    On one 8 GB card that is two model sets, and the failure arrives much later as

        torch.OutOfMemoryError: CUDA out of memory.
        Tried to allocate 7.22 GiB. GPU 0 has a total capacity of 7.93 GiB
        of which 0 bytes is free.

    raised from inside a vendored script, with the API log showing nothing but a
    per-submission ``ui submit sync fallback`` line that does not name the remedy.
    The flag is documented (README.md, and every compose file sets it), so this is
    a configuration mistake rather than a missing feature — but a mistake that is
    silent until it costs a CUDA OOM should not stay silent at boot.

    Only warns when a GPU is present: on CPU the synchronous path costs nothing,
    and a warning there would be noise.
    """
    from fiximg.infrastructure.observability.logging import get_logger, log_event

    if settings.inline_worker or settings.external_worker:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
    except Exception:  # noqa: BLE001 - a probe must never block startup
        return

    log_event(
        get_logger("fiximg.bootstrap"),
        "WARNING",
        "the UI cannot see a worker, so submissions run inference in this process",
        hint="set FIXIMG_HAS_EXTERNAL_WORKER=true when a separate "
             "`fiximg.cli.worker` consumes the queue",
        consequence="the API process loads its own copy of the models alongside "
                    "the worker's; on a single GPU that ends in CUDA OOM",
    )


def _bootstrap_security() -> None:
    """First-boot admin (§21) and JSON API token (§22) provisioning."""
    from fiximg.infrastructure.observability.logging import get_logger, log_event

    # First-boot admin bootstrap: env password or a random one-time password
    # printed once. Never overwrites an existing account.
    if settings.auto_bootstrap_admin:
        from fiximg.application.bootstrap import ensure_admin_user

        ensure_admin_user()

    # §22: make sure the JSON API token exists *before* the first request, so
    # operators can copy it from the startup log / read admin_data/api_token.txt
    # instead of discovering the 401 later.
    try:
        from fiximg.api.security import get_api_token

        get_api_token()
        log_event(
            get_logger("fiximg.security"),
            "INFO",
            "json api token ready",
            source="FIXIMG_API_TOKEN" if os.environ.get("FIXIMG_API_TOKEN", "").strip()
            else "generated/persisted file",
        )
    except Exception as exc:  # noqa: BLE001 — token bootstrap must never block startup
        log_event(
            get_logger("fiximg.security"),
            "ERROR",
            "api token bootstrap failed",
            error=str(exc),
        )


def _warmup_models() -> None:
    """Warm the models this process should hold at boot (plan §3.5.3).

    The selection — per-model `warmup:` key over the global `FIXIMG_WARMUP`, and
    what each strategy means — lives in
    :func:`fiximg.inference.model_manager.model_manager.warm_at_process_start`,
    which `PipelineWorker.start()` also calls. Two processes used to each carry a
    version of "when do we warm": the API preloaded everything regardless of the
    per-model declaration, and the standalone worker preloaded nothing at all.
    """
    from fiximg.infrastructure.observability.logging import get_logger, log_event
    from fiximg.inference.model_manager import model_manager

    logger = get_logger("fiximg.models")
    try:
        warmed = model_manager.warm_at_process_start()
    except Exception as exc:  # noqa: BLE001 — warmup must never block startup
        log_event(logger, "WARNING", "startup warmup skipped", error=str(exc))
        return
    if warmed:
        log_event(logger, "INFO", "warmup strategy: startup", models=warmed)


# ---------------------------------------------------------------------- routing
def register_routes(app: FastAPI) -> None:
    """Mount every JSON API router (health first so probes stay unauthenticated)."""
    app.include_router(health.router)
    # V2 defined the registration page but never mounted it, so /register 404'd.
    app.include_router(auth_routes.router)
    app.include_router(tasks.router)
    app.include_router(models_routes.router)
    app.include_router(users.router)
    app.include_router(stats.router)
    register_exception_handlers(app)


def register_middleware(app: FastAPI) -> None:
    """Correlation-id + acting-user middleware (plan §25 / §3.4.2)."""

    @app.middleware("http")
    async def _request_id_middleware(request, call_next):
        from fiximg.infrastructure.observability.logging import new_request_id

        new_request_id()
        return await call_next(request)

    @app.middleware("http")
    async def _user_id_middleware(request, call_next):
        from fiximg.infrastructure.db.repositories.user_repository import get_user
        from fiximg.infrastructure.observability.logging import set_user_id

        # Two credentials reach this process: the API bearer token, and the UI's
        # session cookie. Both are attributed here so a log line says who acted,
        # whichever door the request came through.
        auth = request.headers.get("authorization", "")
        username = ""
        if auth.lower().startswith("bearer "):
            username = auth[7:].strip()
        if not username:
            from fiximg.ui.web.session import COOKIE_NAME, resolve_session

            session = resolve_session(request.cookies.get(COOKIE_NAME))
            username = session.username if session else ""
        if username and get_user(username):
            set_user_id(username)
        return await call_next(request)


def register_ui(app: FastAPI) -> FastAPI:
    """Mount the native web UI.

    Registered last, after the JSON routers, because it owns `/` and the catch-all
    document route — the same position the Gradio mount held, and the reason route
    order still matters here.

    There is no conditional "is the UI available" branch any more. The UI is a
    router in this package: if this module imported, the UI is there. The Gradio
    version needed that branch because the framework could be absent from an
    install, which is exactly the coupling this change removed.
    """
    register_ui_routes(app)
    return app


# -------------------------------------------------------------------- assembly
def create_app() -> FastAPI:
    """Build the FastAPI application: routes → middleware → UI → lifespan."""
    app = FastAPI(title=APP_TITLE, lifespan=_app_lifespan)
    register_routes(app)
    register_middleware(app)
    return register_ui(app)


__all__ = [
    "APP_TITLE",
    "create_app",
    "get_inline_worker",
    "register_middleware",
    "register_routes",
    "register_ui",
]
