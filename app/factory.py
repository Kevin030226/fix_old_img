"""Application factory (plan section 26): main.py shrinks to `from app.factory import create_app`.

Wires FastAPI routers, the HTML middleware and the mounted Gradio demo. Database
initialization (users/history + V2 task tables) happens in the lifespan hook.

GPU worker topology (plan sections 12/28):
  - FIXIMG_INLINE_WORKER=true (default): a PipelineWorker thread consumes the
    DB-backed task queue inside this process — single-container deployments.
  - FIXIMG_INLINE_WORKER=false: run `python worker.py` as a separate process
    (docker-compose topology); the API then only enqueues.
"""
import os
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI

from app.api import health, stats, tasks, users
from app.core.config import settings
from app.db import init_db
from app.repositories.task_repository import ensure_schema
from app.ui.gradio_app import APP_TITLE, build_demo, custom_css

# Reference to the inline PipelineWorker while the app is running (None otherwise).
_inline_worker = None


def get_inline_worker():
    """Return the in-process PipelineWorker, or None when running standalone-mode."""
    return _inline_worker


@asynccontextmanager
async def _app_lifespan(_app):
    """Initialize the database at app startup (supports uvicorn app direct import)."""
    global _inline_worker
    init_db()
    ensure_schema()

    # First-boot admin bootstrap (plan §21): env password or a random one-time
    # password printed once. Never overwrites an existing account.
    if settings.auto_bootstrap_admin:
        from app.services.bootstrap import ensure_admin_user

        ensure_admin_user()

    # §22: make sure the JSON API token exists *before* the first request, so
    # operators can copy it from the startup log / read admin_data/api_token.txt
    # instead of discovering the 401 later.
    try:
        from app.api.security import get_api_token
        from app.core.logging import get_logger, log_event

        get_api_token()
        log_event(
            get_logger("fiximg.security"),
            "INFO",
            "json api token ready",
            source="FIXIMG_API_TOKEN" if os.environ.get("FIXIMG_API_TOKEN", "").strip()
            else "generated/persisted file",
        )
    except Exception as exc:  # noqa: BLE001 — token bootstrap must never block startup
        from app.core.logging import get_logger, log_event

        log_event(
            get_logger("fiximg.security"),
            "ERROR",
            "api token bootstrap failed",
            error=str(exc),
        )
    _worker = None
    if settings.inline_worker:
        from app.inference.orchestrator import PipelineOrchestrator
        from app.inference.worker import PipelineWorker

        _worker = PipelineWorker(
            PipelineOrchestrator(),
            poll_seconds=settings.worker_poll_seconds,
            worker_id=f"inline-{id(_app):x}",
        )
        _worker.start()
        _inline_worker = _worker
    try:
        yield
    finally:
        if _worker is not None:
            _worker.stop()
        _inline_worker = None


def create_app() -> FastAPI:
    """Build the FastAPI application with all routers and the Gradio UI mounted."""
    app = FastAPI(title=APP_TITLE, lifespan=_app_lifespan)

    # §25 correlation: one request_id per HTTP request (JSON logs + events).
    @app.middleware("http")
    async def _request_id_middleware(request, call_next):
        from app.core.logging import new_request_id

        new_request_id()
        return await call_next(request)

    # §25: attach the acting user_id to logs of authenticated JSON API calls.
    @app.middleware("http")
    async def _user_id_middleware(request, call_next):
        from app.core.logging import set_user_id
        from app.repositories.user_repository import get_user

        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            username = auth[7:].strip()
            if username and get_user(username):
                set_user_id(username)
        return await call_next(request)

    # JSON API routers.
    app.include_router(health.router)
    app.include_router(tasks.router)
    app.include_router(users.router)
    app.include_router(stats.router)

    # Gradio UI mounted last (route order matters for "/" and session reads).
    demo = build_demo()
    app = gr.mount_gradio_app(
        app,
        demo,
        path="/",
        auth=_auth_fn,
        auth_message="Please enter your username and password",
        max_file_size="10mb",
        show_error=False,
        css=custom_css,
    )

    # HTML middleware needs the mounted Gradio app for session reads.
    from app.ui.middleware import make_html_middleware

    _GRADIO_MOUNT = app.routes[-1] if app.routes else None
    _gradio_holder = {"app": getattr(_GRADIO_MOUNT, "app", None) if _GRADIO_MOUNT else None}
    app.middleware("http")(make_html_middleware(_gradio_holder))
    return app


def _auth_fn(username: str, password: str) -> bool:
    """Login verification used by Gradio (constant-time hash comparison)."""
    from app.services.user_service import authenticate

    return authenticate(username, password)


def _auth_message_fn(username: str) -> str:
    """Gradio login banner; warns §21 force-change accounts after login."""
    try:
        from app.repositories.user_repository import get_user

        user = get_user(username or "")
        if user and user.get("must_change_password"):
            return "⚠ Please open the Admin Panel → Account and replace the initial password."
    except Exception:  # noqa: BLE001 — the banner must never block login
        pass
    return ""
