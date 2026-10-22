"""Native web UI routes (plan §3.9) — the FastAPI replacement for the Gradio shell.

Everything here is a thin adapter: it resolves *who is calling* (a session cookie
or a bearer token), then calls the same application services the JSON API calls.
No business rule lives in this module, and nothing here touches a stage, the queue
or a model directly. That is deliberate and it is the reason the migration is
reviewable: the routes below are the shape of the old callbacks with the Gradio
component plumbing removed.

Route groups:

    GET  /                            the app shell, or the sign-in page
    GET  /ui/static/{path}            the stylesheet-free static asset (ui.js)
    POST /ui/login                    sign in, set the session cookie
    POST /ui/logout                   sign out (also POST /ui/api/session/logout)
    GET  /ui/api/session              who am I, and the CSRF token
    POST /ui/api/analysis             the Auto tab's analysis row
    POST /ui/api/tasks                submit an upload (multipart)
    GET  /ui/api/tasks/{id}           one poll frame
    GET  /ui/api/tasks/{id}/result    the finished image (inline or download)
    GET  /ui/api/history              one page of the caller's history
    GET  /ui/api/history/{id}         the selected row's previews and detail
    POST /ui/api/history/{id}/rerun   re-submit the selected task as a new one
    POST /ui/api/history/{id}/retry   re-queue the selected task under its own id
    GET  /ui/api/admin/history        every task (admin)
    DELETE /ui/api/admin/history      clear the legacy V1 rows (admin)
    GET  /ui/api/admin/archive        the archive record list (admin)
    GET  /ui/api/admin/archive/{id}   one archive record (admin)
    GET/POST/PATCH/DELETE /ui/api/admin/users[...]   user management (admin)
    GET  /ui/api/admin/models         the model table (admin)
    POST /ui/api/admin/models/{name}/{action}        warmup / unload (admin)
    POST /ui/api/session/password     change your own password

Authorisation is explicit at every admin route rather than inherited from a mount:
a route that forgets its dependency must fail loudly, so `_require_admin` is called
in the body as well as declared as a dependency where it matters.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from fiximg.api.dependencies import Principal
from fiximg.api.schemas import ui as ui_schemas
from fiximg.application import auth_service, history_service
from fiximg.application.submission import submit_upload
from fiximg.application.task_service import task_service
from fiximg.domain.errors import (
    AppError,
    ForbiddenError,
    InvalidRequestError,
    TaskNotFoundError,
)
from fiximg.infrastructure.db.repositories.user_repository import get_user
from fiximg.infrastructure.observability.logging import get_logger, log_event
from fiximg.ui import progress as progress_view
from fiximg.ui.web import pages
from fiximg.ui.web.session import (
    COOKIE_NAME,
    CSRF_HEADER,
    Session,
    csrf_matches,
    describe_cookie,
    describe_logout_cookie,
    resolve_session,
    session_payload,
    session_store,
)

logger = get_logger("fiximg.ui.web")

router = APIRouter(tags=["ui"])

#: Static assets, served from the package directory. One file, and it is named
#: here rather than taken from the URL, so the route cannot be talked into reading
#: another file: a path parameter would be a traversal surface for no benefit.
STATIC_ASSETS = {
    "ui.js": ("ui.js", "application/javascript; charset=utf-8"),
}

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

#: The bootstrap block the page reads. Polling cadence and the upload ceiling are
#: deployment facts, so they are handed to the client instead of being duplicated
#: as literals in the script.
def bootstrap_payload(session: Session) -> dict:
    """Everything the page needs before it can draw itself."""
    from fiximg.config import settings

    payload = session_payload(session)
    payload.update({
        "max_upload_mb": settings.max_upload_mb,
        "poll_ms": 1000,
        # The UI gives up where the server's own poll loop does (plan §23).
        "poll_timeout_ms": 30 * 60 * 1000,
    })
    return payload


def _ui_script() -> str:
    """The static UI script's text, read once per process."""
    global _script_cache
    if _script_cache is None:
        with open(os.path.join(STATIC_DIR, "ui.js"), encoding="utf-8") as handle:
            _script_cache = handle.read()
    return _script_cache


_script_cache: str | None = None


# --------------------------------------------------------------- dependencies
def current_session(request: Request) -> Session | None:
    """The caller's session, or None. Never raises: an anonymous request is normal."""
    return resolve_session(request.cookies.get(COOKIE_NAME))


def require_session(request: Request) -> Session:
    """A signed-in session, or a 401 in the API's own envelope."""
    session = current_session(request)
    if session is None:
        raise _unauthenticated()
    return session


def _unauthenticated() -> AppError:
    """401 with the envelope the JSON error handler already understands."""
    from fiximg.domain.errors import ErrorCode

    return InvalidRequestError(
        "Sign in to use the web interface",
        code=ErrorCode.UNAUTHENTICATED,
        status_code=401,
    )


def require_admin_session(request: Request) -> Session:
    """A signed-in administrator, or the matching refusal.

    The role is read from the session record, which `resolve_session` refreshes
    from the users table on every request — so a demoted account loses the admin
    routes on its next request rather than at its next login.
    """
    session = require_session(request)
    if not session.is_admin:
        raise ForbiddenError("Administrator privileges are required for this operation")
    return session


def require_csrf(request: Request) -> Session:
    """A session whose CSRF token matches the request header.

    Applied to every mutating route. The cookie is `SameSite=Lax`, which stops a
    cross-site fetch but not a top-level form POST, so the token is the half that
    covers the whole class.
    """
    session = require_session(request)
    if not csrf_matches(session, request.headers.get(CSRF_HEADER)):
        raise ForbiddenError(
            "Missing or invalid CSRF token; reload the page and try again"
        )
    return session


def ui_principal(session: Session) -> Principal:
    """The acting principal for a browser session.

    A `Principal` with `role` taken from the session, so the submission path reads
    its caller exactly the way it reads an API token holder. `via` names the
    credential, which is what the audit events record.
    """
    return Principal(user_id=session.username, role=session.role, via="session")


# ------------------------------------------------------------------- the shell
@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    """The application, or the sign-in page.

    The Gradio mount answered `/` with its own login form; this answers with a
    server-rendered one, and with the application shell once a session exists. Both
    are ordinary documents, so a browser with scripting disabled still signs in.
    """
    session = current_session(request)
    if session is None:
        return HTMLResponse(pages.render_sign_in())
    return HTMLResponse(
        pages.render_app(
            username=session.username,
            role=session.role,
            bootstrap=bootstrap_payload(session),
            script=_ui_script(),
        )
    )


@router.get("/ui/static/{name}", include_in_schema=False)
async def static_asset(name: str):
    """The one static asset this UI has.

    The name is matched against :data:`STATIC_ASSETS` rather than joined onto a
    directory: a path parameter used as a filesystem path is how a static route
    becomes an arbitrary-file-read, and there is exactly one file to serve.
    """
    entry = STATIC_ASSETS.get(name)
    if entry is None:
        return JSONResponse(
            {"error": {"code": "NOT_FOUND", "message": f"No asset named {name!r}"}},
            status_code=404,
        )
    filename, media_type = entry
    path = os.path.join(STATIC_DIR, filename)
    return FileResponse(
        path,
        media_type=media_type,
        # The asset changes with the package, not with the request, so a short
        # cache is safe and keeps the page from re-downloading it on every poll.
        headers={"Cache-Control": "public, max-age=300"},
    )


# --------------------------------------------------------------------- sign in
@router.post("/ui/login", include_in_schema=False)
async def login(request: Request):
    """Verify the credentials and start a session.

    A form POST (so the sign-in page works without JavaScript), and it answers with
    a redirect either way: back to `/` on success, back to the sign-in page with a
    message on failure. The failure message is deliberately identical for an
    unknown account and a wrong password.
    """
    form = await request.form()
    username = str(form.get("username") or "").strip()
    password = str(form.get("password") or "")
    if not username or not password:
        return HTMLResponse(
            pages.render_sign_in(error="Enter a username and a password", username=username),
            status_code=400,
        )
    try:
        authenticated = auth_service.authenticate(username, password)
    except Exception as exc:  # noqa: BLE001 - a login must not 500 on a DB hiccup
        log_event(logger, "ERROR", "login failed to verify", error=str(exc))
        return HTMLResponse(
            pages.render_sign_in(error="Sign-in is temporarily unavailable", username=username),
            status_code=503,
        )
    if not authenticated:
        log_event(logger, "WARNING", "login refused", username=username)
        return HTMLResponse(
            pages.render_sign_in(error="Incorrect username or password", username=username),
            status_code=401,
        )
    session = session_store.create(username)
    log_event(logger, "INFO", "login succeeded", username=username)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(**describe_cookie(session))
    return response


@router.post("/ui/logout", include_in_schema=False)
async def logout(request: Request):
    """End the session and clear the cookie (form POST, so a link can reach it)."""
    session = current_session(request)
    if session is not None:
        session_store.delete(session.session_id)
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(**describe_logout_cookie())
    return response


# -------------------------------------------------------------------- session
@router.get("/ui/api/session", response_model=ui_schemas.SessionView,
            summary="Who is signed in")
async def session_info(session: Session = Depends(require_session)):
    return session_payload(session)


@router.post("/ui/api/session/logout", response_model=ui_schemas.SignedOutView,
             summary="Sign out")
async def session_logout(session: Session = Depends(require_csrf)):
    session_store.delete(session.session_id)
    response = JSONResponse({"signed_out": True})
    response.delete_cookie(**describe_logout_cookie())
    return response


@router.post("/ui/api/session/password", response_model=ui_schemas.MessageView,
             summary="Change your own password")
async def change_own_password(
    payload: dict,
    session: Session = Depends(require_csrf),
):
    """§21 force-change entry point: replace the caller's own password.

    The same rules as the Gradio callback: at least eight characters, confirmed,
    and only ever the caller's own account — the route takes no username, so it
    cannot be pointed at someone else.
    """
    new_password = str(payload.get("new_password") or "")
    confirm = str(payload.get("confirm_password") or "")
    if len(new_password) < 8:
        raise InvalidRequestError("New password must be at least 8 characters")
    if new_password != confirm:
        raise InvalidRequestError("The two passwords do not match")
    auth_service.update_user(session.username, password=new_password)
    # The flag the submission gate reads has to be cleared with the password, or
    # the account stays locked out of the feature it just unlocked.
    auth_service.set_must_change_password(session.username, False)
    log_event(logger, "INFO", "password changed", username=session.username)
    return {"message": f"Password changed for '{session.username}'. All features are unlocked."}


# ------------------------------------------------------------------- analysis
@router.post("/ui/api/analysis", response_model=ui_schemas.AnalysisView,
             summary="Auto analysis for the current upload")
async def analysis(
    mode: str = Form("auto"),
    image: UploadFile = File(...),
    options: str = Form("{}"),
    session: Session = Depends(require_session),
):
    """§3.9.1's analysis row: what Auto Restore would do with *this* picture.

    Reads through `TaskService.auto_preview`, which is the same call the worker's
    planner makes, so the row cannot describe a different pipeline than the run.
    CPU-only CV on a 512 px copy, so it is safe in the web process.
    """
    from fiximg.application.submission import decode_upload, parse_options
    from fiximg.ui.progress import describe_preview, switch_options
    from fiximg.application.pipeline_modes import SWITCHES

    payload = await image.read()
    pil_image = decode_upload(payload)
    parsed = parse_options(options)
    names = list(SWITCHES.get(mode, {}))
    explicit = parsed.model_dump(exclude_defaults=True)
    # The client sends the switch states it holds; the analysis must see the same
    # options a submission would, or the row would describe the defaults instead of
    # what the user is about to run.
    drawn = {name: explicit.get(name, default) for name, default in SWITCHES.get(mode, {}).items()}
    chosen = switch_options(mode, names, tuple(drawn.values())) if names else None
    preview = task_service.auto_preview(pil_image, chosen)
    return {
        "available": bool(preview.get("available")),
        "text": describe_preview(preview),
        "preview": preview,
    }


# ---------------------------------------------------------------------- tasks
@router.post("/ui/api/tasks", status_code=202,
             response_model=ui_schemas.TaskCreatedView,
             summary="Queue a processing task")
async def create_task(
    mode: str = Form(..., description="auto | restore | restore_scratch | detect | colorize"),
    image: UploadFile = File(...),
    options: str = Form(None),
    priority: int = Form(0),
    session: Session = Depends(require_csrf),
):
    """Submit an upload from the browser.

    The same validation and the same queue entry as `POST /api/v1/tasks`: the
    checks live in :func:`fiximg.application.submission.submit_upload` and this route
    only supplies the caller. A UI submission therefore cannot take a shortcut past
    the rate limit, the priority ceiling or the upload guard — which is the property
    that makes "the UI must not bypass the queue" true rather than aspirational.
    """
    from fiximg.application.pipeline_modes import PIPELINE_MODES

    task_type = PIPELINE_MODES.get(mode, {}).get("task_type", mode)
    submission = submit_upload(
        principal=ui_principal(session),
        task_type=task_type,
        image_bytes=await image.read(),
        options_raw=options,
        priority=priority,
        request=None,
    )
    log_event(logger, "INFO", "ui submission queued",
              task_id=submission.task_id, task_type=task_type,
              username=session.username)
    return {
        "task_id": submission.task_id,
        "status": "queued",
        "created_at": submission.created_at,
        "priority": submission.priority,
        "poll_url": f"/ui/api/tasks/{submission.task_id}",
    }


@router.get("/ui/api/tasks/{task_id}", response_model=ui_schemas.TaskFrameView,
            summary="One poll frame for a task")
async def task_frame(task_id: str, session: Session = Depends(require_session)):
    """One poll frame, rendered by the same rules the panel's text obeys.

    Returns both the structured parts (progress, stages, flags) and the
    human-readable `text` the Gradio panel produced, so the client can choose which
    to show and the tests can assert on either.

    Ownership is enforced here: a task belonging to another user is reported as
    missing, not as forbidden, because "there is no such task" and "there is such a
    task but not for you" must not be distinguishable to a caller who guessed an id.
    """
    row = task_service.get_task(task_id)
    if not row or not _may_see(session, row):
        raise TaskNotFoundError(f"Task not found: {task_id}", details={"task_id": task_id})
    return progress_view.status_text(
        row,
        queue_depth=task_service.queue_depth(),
        decisions=task_service.planner_decisions(row.get("result_path")),
        options=_row_options(row),
    )


@router.get("/ui/api/tasks/{task_id}/result", summary="The finished image",
            # The body is a PNG, so it has no JSON model to name. Declaring the media
            # type is what keeps the OpenAPI document honest -- without it FastAPI
            # advertises application/json for a route that answers with a file.
            response_class=FileResponse,
            responses={200: {"description": "The finished image",
                             "content": {"image/png": {"schema": {"type": "string",
                                                                  "format": "binary"}}}},
                       409: {"description": "The task has not finished yet"},
                       410: {"description": "Result reclaimed by the retention TTL"}})
async def task_result(task_id: str, download: bool = Query(default=False),
                      session: Session = Depends(require_session)):
    """Serve a task's result, inline for a preview or as an attachment for a save.

    Reads through `TaskService.task_result_path`, which is the same call the JSON
    API's download route makes: it refuses a not-yet-ready task (409) and a reclaimed
    one (410) instead of answering 200 with an error page, and it re-materialises a
    published artifact when this node does not hold the file.

    The response is a `FileResponse` built from that path only. No part of the
    request names a file, so there is nothing here to traverse.
    """
    path = task_service.task_result_path(task_id)
    row = task_service.get_task(task_id) or {}
    if not _may_see(session, row):
        # Same refusal as a missing task; checked before the bytes are opened.
        raise TaskNotFoundError(f"Task not found: {task_id}", details={"task_id": task_id})
    filename = f"{task_id}.png"
    media_type = "image/png"
    if download:
        return FileResponse(path, media_type=media_type, filename=filename,
                            content_disposition_type="attachment")
    return FileResponse(path, media_type=media_type)


@router.get("/ui/api/tasks/{task_id}/input", summary="The uploaded original",
            response_class=FileResponse,
            responses={200: {"description": "The uploaded original",
                             "content": {"image/png": {"schema": {"type": "string",
                                                                  "format": "binary"}}}}})
async def task_input(task_id: str, session: Session = Depends(require_session)):
    """Serve the original upload, so the before/after slider has a left-hand side.

    The original is not an artifact kind the API exposes for download, so this reads
    the row's own `input_path` — through the service's ownership check first.
    """
    row = task_service.get_task(task_id)
    if not row or not _may_see(session, row):
        raise TaskNotFoundError(f"Task not found: {task_id}", details={"task_id": task_id})
    path = row.get("input_path") or ""
    if not path or not os.path.isfile(path):
        raise TaskNotFoundError(
            f"The original upload for {task_id} is no longer available",
            details={"task_id": task_id},
        )
    return FileResponse(path, media_type="image/png")


# -------------------------------------------------------------------- history
#: Rows per page by default, and the sizes the page offers.
DEFAULT_PAGE_SIZE = 20
PAGE_SIZES = (10, 20, 50, 100)
#: Long side of a gallery thumbnail, in pixels.
THUMBNAIL_SIDE = 180


def _clamp_page(page, page_size) -> tuple[int, int]:
    """Coerce the two query values into the range the panel supports.

    Both arrive as strings and are parsed here rather than declared `int` in the
    signature: the Gradio panel fell back to the defaults for a value it could not
    read, and a 422 from FastAPI's own validation would turn a mangled URL into an
    error page instead of the first page. Clamping to 200 is the panel's ceiling too.
    """
    try:
        size = max(1, min(int(page_size or DEFAULT_PAGE_SIZE), 200))
    except (TypeError, ValueError):
        size = DEFAULT_PAGE_SIZE
    try:
        number = max(1, int(page or 1))
    except (TypeError, ValueError):
        number = 1
    return number, size


def _may_see(session: Session, row: dict) -> bool:
    """Users see their own tasks; administrators see every task (§3.9.3).

    One implementation, in `fiximg.ui.state.visible_user_id`, which is the same rule
    the history listing and the admin panels use. A second copy here is how the
    listing and the detail view start disagreeing about what "your tasks" means.
    """
    from fiximg.ui.state import visible_user_id

    owner = visible_user_id(session.user_state)
    return owner is None or (row.get("user_id") or "") == owner


def _row_options(row: dict) -> dict:
    """The options a row was submitted with, decoded."""
    from fiximg.domain.tasks import decode_json_column

    decoded = decode_json_column((row or {}).get("options_json"), {})
    return decoded if isinstance(decoded, dict) else {}


def _history_row(row: dict) -> dict:
    """One table row, in the panel's own vocabulary."""
    from fiximg.domain.tasks import decode_json_column
    from fiximg.ui.history import metrics_text, stages_text, when_text, duration_text

    return {
        "task_id": row.get("id", ""),
        "task_type": row.get("task_type", ""),
        "created": when_text(row.get("created_at")),
        "duration": duration_text(row.get("duration_ms")),
        "status": row.get("status", ""),
        "progress": f"{int(row.get('progress') or 0)}%",
        "metrics": metrics_text(decode_json_column(row.get("metrics"), {})),
        "stages": stages_text(decode_json_column(row.get("stages"), [])),
    }


@router.get("/ui/api/history", response_model=ui_schemas.HistoryPageView,
            summary="One page of the caller's history")
async def history_page(
    page: str | None = Query(default=None),
    page_size: str | None = Query(default=None),
    session: Session = Depends(require_session),
):
    """A page of tasks, newest first, scoped to the caller.

    The pagination arithmetic is the panel's own (fetch one row past the page so
    "is there a next page?" needs no COUNT, and clamp a page past the end to the
    last one), moved here unchanged from `ui/history_panel.list_history` so the
    numbers a user saw before the migration are the numbers they see after it.
    """
    from fiximg.ui.state import visible_user_id

    number, size = _clamp_page(page, page_size)
    owner = visible_user_id(session.user_state)
    try:
        tasks = task_service.list_tasks(limit=size * number + 1, user_id=owner)
    except Exception as exc:  # noqa: BLE001 - the panel must never break the page
        log_event(logger, "ERROR", "history listing failed", error=str(exc))
        return {
            "rows": [],
            "thumbnails": [],
            "page": 1,
            "page_size": size,
            "page_label": "Page 1 of 1",
            "status": f"⚠ Could not load history: {type(exc).__name__}",
        }

    total_fetched = len(tasks)
    last_page = max(1, -(-total_fetched // size))
    number = min(number, last_page)
    start = (number - 1) * size
    window = tasks[start:start + size]
    has_more = total_fetched > start + size

    rows = [_history_row(task) for task in window]
    thumbnails = []
    for task in window:
        thumb = thumbnail_path(task.get("result_path") or "")
        if thumb:
            thumbnails.append({
                "task_id": task.get("id", ""),
                "thumbnail_url": f"/ui/api/tasks/{task.get('id', '')}/thumbnail",
                "caption": f"{task.get('task_type', '')} · {task.get('id', '')}",
            })

    scope = "all users" if owner is None else f"user '{owner}'"
    return {
        "rows": rows,
        "thumbnails": thumbnails,
        "page": number,
        "page_size": size,
        "page_label": f"Page {number} of {number + 1 if has_more else number}",
        "status": (
            f"{len(rows)} row(s) on this page — {scope}, newest first. "
            f"Page size {size}."
        ),
    }


@router.get("/ui/api/tasks/{task_id}/thumbnail", summary="A cached thumbnail",
            response_class=FileResponse,
            responses={200: {"description": "A cached thumbnail",
                             "content": {"image/png": {"schema": {"type": "string",
                                                                  "format": "binary"}}}}})
async def task_thumbnail(task_id: str, session: Session = Depends(require_session)):
    """A cached thumbnail of a task's result, generated on first request.

    The gallery renders a page of results at once, so serving multi-megapixel
    originals makes the panel crawl. The cache is keyed by (path, mtime, size), so a
    re-render does not re-encode anything — the same policy `history_panel` had.
    """
    row = task_service.get_task(task_id)
    if not row or not _may_see(session, row):
        raise TaskNotFoundError(f"Task not found: {task_id}", details={"task_id": task_id})
    thumb = thumbnail_path(row.get("result_path") or "")
    if not thumb:
        raise TaskNotFoundError(
            f"No thumbnail is available for {task_id}", details={"task_id": task_id}
        )
    return FileResponse(thumb, media_type="image/png")


# --------------------------------------------------------------- history detail
@router.get("/ui/api/history/{task_id}", response_model=ui_schemas.HistoryDetailView,
            summary="One history row's previews")
async def history_detail(task_id: str, session: Session = Depends(require_session)):
    """Everything the selected row shows: four previews, the slider pair, the detail.

    Plan §3.9.2 asks for more than the two-state original/result slider — the mask and
    the first aligned face crop are the two artifacts that explain what a run did. The
    paths come from `task_service`, never from a stage's output tree (plan §3.2).
    """
    row = task_service.get_task(task_id)
    if not row or not _may_see(session, row):
        raise TaskNotFoundError(f"Task not found: {task_id}", details={"task_id": task_id})

    from fiximg.ui.history import detail_text

    result_path = row.get("result_path") or ""
    input_path = row.get("input_path") or ""
    has_result = bool(result_path) and os.path.isfile(result_path)
    mask = artifact_path(task_id, "mask")
    face = first_face_crop(task_id)

    return {
        "task_id": task_id,
        "detail": detail_text(row, result_present=has_result),
        # A URL per slot, or null. The client sets an <img> src, so a missing file is
        # "no preview" rather than a broken image element.
        "input_url": f"/ui/api/tasks/{task_id}/input"
                     if input_path and os.path.isfile(input_path) else None,
        "result_url": f"/ui/api/tasks/{task_id}/result" if has_result else None,
        "mask_url": f"/ui/api/tasks/{task_id}/artifact/mask" if mask else None,
        "face_url": f"/ui/api/tasks/{task_id}/artifact/faces_dir" if face else None,
        "download_url": f"/ui/api/tasks/{task_id}/result?download=1" if has_result else None,
        "status": row.get("status"),
        "task_type": row.get("task_type"),
        # §3.8: retry re-queues the same id and only applies to a finished failure;
        # the client disables the button instead of explaining a refusal.
        "can_retry": row.get("status") in ("failed", "cancelled"),
        "can_rerun": bool(input_path) and os.path.isfile(input_path),
    }


@router.get("/ui/api/tasks/{task_id}/artifact/{kind}",
            summary="A task artifact preview",
            response_class=FileResponse,
            responses={200: {"description": "One artifact preview",
                             "content": {"image/png": {"schema": {"type": "string",
                                                                  "format": "binary"}}}}})
async def task_artifact(
    task_id: str,
    kind: str,
    session: Session = Depends(require_session),
):
    """Serve one artifact by *kind*, never by path.

    The kind is matched against the two the UI previews, so the request cannot name a
    file. A directory-valued artifact (`faces_dir`) resolves to its first image, which
    is what the Gradio panel's "Face crop" slot showed.
    """
    row = task_service.get_task(task_id)
    if not row or not _may_see(session, row):
        raise TaskNotFoundError(f"Task not found: {task_id}", details={"task_id": task_id})
    if kind == "mask":
        path = artifact_path(task_id, "mask")
        media_type = "image/png"
    elif kind == "faces_dir":
        path = first_face_crop(task_id)
        media_type = "image/png"
    else:
        raise TaskNotFoundError(
            f"Unknown artifact kind: {kind}", details={"task_id": task_id}
        )
    if not path or not os.path.isfile(path):
        raise TaskNotFoundError(
            f"No {kind} artifact is available for {task_id}", details={"task_id": task_id}
        )
    return FileResponse(path, media_type=media_type)


# ------------------------------------------------------------------- re-run
@router.post("/ui/api/history/{task_id}/rerun", response_model=ui_schemas.RerunView,
             summary="Re-submit a task as a new one")
async def history_rerun(task_id: str, session: Session = Depends(require_csrf)):
    """Submit the stored input again, as a **new** task.

    Deliberately different from retry: re-run starts a fresh job from the stored
    image, which is what a user means by "do that again with these settings". The
    queue position is carried over, because a task somebody paid the
    `FIXIMG_PRIORITY_MAX` budget for must not silently drop to FIFO because it was
    submitted a second time.
    """
    from PIL import Image


    row = task_service.get_task(task_id)
    if not row or not _may_see(session, row):
        raise TaskNotFoundError(f"Task not found: {task_id}", details={"task_id": task_id})
    input_path = row.get("input_path") or ""
    if not input_path or not os.path.isfile(input_path):
        raise InvalidRequestError(
            "Cannot re-run: the input image is no longer available",
            details={"task_id": task_id},
        )

    mode = _mode_for_task_type(row.get("task_type") or "")
    options = {
        key: value for key, value in _row_options(row).items()
        # Internal keys (the ground-truth path) point at artifacts of the *old* run.
        if not key.startswith("_")
    }
    try:
        with Image.open(input_path) as handle:
            image = handle.convert("RGB")
            new_id = task_service.enqueue(
                image, {"username": session.username}, mode,
                options=options or None,
                priority=int(row.get("priority") or 0),
            )
    except AppError:
        raise
    except Exception as exc:  # noqa: BLE001 - the caption is the user-facing answer
        log_event(logger, "ERROR", "history re-run failed",
                  task_id=task_id, error=str(exc))
        raise InvalidRequestError(f"Re-run failed: {exc}") from exc

    log_event(logger, "INFO", "history re-run queued", source_task=task_id, task_id=new_id)
    return {
        "task_id": new_id,
        "message": f"🔁 Re-queued {row.get('task_type')} as {new_id}.",
    }


@router.post("/ui/api/history/{task_id}/retry", response_model=ui_schemas.RerunView,
             summary="Re-queue a failed task as itself")
async def history_retry(task_id: str, session: Session = Depends(require_csrf)):
    """Re-queue the **same** task id (plan §3.8 `/tasks/{id}/retry`).

    The row, its artifacts and its metrics keep describing one job. Retry only
    applies to a failed or cancelled task; asking for it on a completed one is
    refused with that reason rather than silently duplicating GPU work.
    """
    row = task_service.get_task(task_id)
    if not row or not _may_see(session, row):
        raise TaskNotFoundError(f"Task not found: {task_id}", details={"task_id": task_id})
    if row.get("status") not in ("failed", "cancelled"):
        raise InvalidRequestError(
            f"{task_id} is {row.get('status')}; retry only applies to a failed or "
            "cancelled task — use Re-run to start a fresh one.",
            details={"task_id": task_id, "status": row.get("status")},
        )
    try:
        requeued = task_service.retry_task(task_id)
    except AppError:
        raise
    except Exception as exc:  # noqa: BLE001
        log_event(logger, "ERROR", "history retry failed", task_id=task_id, error=str(exc))
        raise InvalidRequestError(f"Retry failed: {exc}") from exc
    if not requeued:
        raise InvalidRequestError(
            f"{task_id} was not requeued (claimed or finished in the meantime).",
            details={"task_id": task_id},
        )
    log_event(logger, "INFO", "history retry queued", task_id=task_id)
    return {
        "task_id": task_id,
        "message": f"↻ Re-queued {task_id} under the same id; attempt count reset.",
    }


def _mode_for_task_type(task_type: str) -> str:
    """task_type → UI mode, so a history row re-runs through the same path."""
    from fiximg.application.pipeline_modes import PIPELINE_MODES

    for mode, cfg in PIPELINE_MODES.items():
        if cfg["task_type"] == task_type:
            return mode
    return task_type


# ------------------------------------------------------------------ artifacts
def thumbnail_path(path: str, side: int = THUMBNAIL_SIDE) -> str | None:
    """Return a cached thumbnail of `path`, or None when unusable.

    Cache keyed by (path, mtime, size) so a re-render re-encodes nothing. Moved from
    `ui/history_panel` unchanged; the only difference is that the cache lives under
    the storage root rather than beside it, which is where the deployment's volume is.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        stat = os.stat(path)
        cache_dir = thumbnail_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        target = os.path.join(cache_dir, f"{stem}_{stat.st_mtime_ns}_{side}.png")
        if os.path.exists(target):
            return target

        from PIL import Image

        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((side, side), Image.Resampling.LANCZOS)
            thumb.save(target, format="PNG", optimize=True)
        return target
    except Exception as exc:  # noqa: BLE001 — a thumbnail is cosmetic
        log_event(logger, "WARNING", "thumbnail failed", path=path, error=str(exc))
        return None


def thumbnail_cache_dir() -> str:
    """Where gallery thumbnails are cached (next to the artifact storage root)."""
    from fiximg.config import settings

    # The separator set is written as a literal pair rather than a string with a
    # backslash escape: this file is edited by tools that do not always preserve one.
    return os.path.join(
        os.path.dirname(settings.tasks_root.rstrip("/" + chr(92))), "thumbs"
    )


def artifact_path(task_id: str, kind: str) -> str | None:
    """First existing artifact of `kind` for a task, or None."""
    try:
        rows = task_service.get_task_artifacts(task_id)
    except Exception:  # noqa: BLE001 — previews must never break the panel
        return None
    for row in rows or []:
        if row.get("kind") != kind:
            continue
        path = row.get("uri") or row.get("path")
        if path and os.path.isfile(path):
            return path
    return None


def first_face_crop(task_id: str) -> str | None:
    """First aligned face from the detection stage (a directory of crops)."""
    try:
        return task_service.first_face_crop(task_id)
    except Exception:  # noqa: BLE001 — previews must never break the panel
        return None

# ---------------------------------------------------------------------- admin
#: The message every admin refusal carries. One string, so the panel and the API
#: describe the same refusal the same way.
PERMISSION_DENIED_MSG = "⛔ Permission denied: admin only"


@router.get("/ui/api/admin/history", response_model=ui_schemas.AdminHistoryView,
            summary="Every task, newest first")
async def admin_history(session: Session = Depends(require_admin_session)):
    """The *Task Management* tab: every task plus the deployment's totals.

    Reads through `history_service`, which reads the `tasks` table — the store the
    pipeline actually writes. The Gradio panel read the V1 `history` table, which
    nothing populated, so its Refresh returned "No records" on a deployment that had
    just finished a task; `tests/ui/test_admin_history_reads_tasks.py` pins that
    property and this route inherits it by using the same service.
    """
    records = history_service.list_history(50)
    rows = [
        {
            "timestamp": record.get("timestamp", ""),
            "user": record.get("user", ""),
            "type": record.get("type", ""),
            "status": record.get("status", ""),
            "psnr": _text(record.get("psnr")),
            "ssim": _text(record.get("ssim")),
            "mae": _text(record.get("mae")),
            "input_name": os.path.basename(record.get("input_path", "") or ""),
            "task_id": record.get("id", ""),
        }
        for record in records
    ]
    return {
        "rows": rows,
        "stats": history_service.history_stats(),
        "empty": not records,
    }


@router.delete("/ui/api/admin/history", response_model=ui_schemas.MessageView,
               summary="Clear the legacy history rows")
async def admin_clear_history(session: Session = Depends(require_csrf)):
    """Clear the legacy V1 rows and say what that does *not* delete.

    `history_service.clear_history` removes rows from the V1 `history` table only;
    task records are the queue's state, so a button labelled "Clear All Records" must
    not delete them. The message names the scope rather than reporting a bare success.
    """
    if not session.is_admin:
        raise ForbiddenError(PERMISSION_DENIED_MSG)
    history_service.clear_history()
    return {
        "message": "✅ Legacy history records cleared. Task records were kept.",
    }


@router.get("/ui/api/admin/archive", response_model=ui_schemas.ArchiveListView,
            summary="The archive record list")
async def admin_archive(session: Session = Depends(require_admin_session)):
    """The *Photo Archive* tab's selector: one entry per task, labelled the way the
    Gradio dropdown labelled it."""
    records = history_service.list_history()
    return {
        "records": [
            {
                "task_id": record.get("id", ""),
                "label": (
                    f"[{record.get('timestamp', '')}] {record.get('type', '')} — "
                    f"{os.path.basename(record.get('input_path', '') or '')}"
                ),
            }
            for record in records
        ],
    }


@router.get("/ui/api/admin/archive/{task_id}",
            response_model=ui_schemas.ArchiveEntryView,
            summary="One archive record")
async def admin_archive_entry(task_id: str,
                             session: Session = Depends(require_admin_session)):
    """One record's images and processing info.

    Prefers the archived copy of each file and falls back to wherever the run left it
    — the archive is filled by a sweeper, so a recent task's copy may not exist yet and
    the detail pane still has to show something. A file that exists nowhere is
    reported as no preview rather than as an error.
    """
    record = history_service.get_history_record(task_id)
    if not record:
        raise TaskNotFoundError(f"Record not found: {task_id}", details={"task_id": task_id})

    prefix = record.get("id", "")
    in_name = os.path.basename(record.get("input_path", "") or "")
    out_name = os.path.basename(record.get("output_path", "") or "")

    archived_in = os.path.join(history_service.archive_input_dir(), f"{prefix}_{in_name}")
    stored_in = record.get("input_path", "") or ""
    archived_out = os.path.join(history_service.archive_output_dir(), f"{prefix}_{out_name}")
    stored_out = record.get("output_path", "") or ""

    def _first_existing(*candidates: str) -> str | None:
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
        return None

    input_file = _first_existing(archived_in, stored_in)
    output_file = _first_existing(archived_out, stored_out)
    return {
        "task_id": task_id,
        "detail": (
            f"Task: {task_id}\n"
            f"Time: {record.get('timestamp', '')}\n"
            f"User: {record.get('user', '')}\n"
            f"Task Type: {record.get('type', '')}\n"
            f"PSNR: {record.get('psnr', '')}  SSIM: {record.get('ssim', '')}  "
            f"MAE: {record.get('mae', '')}"
        ),
        "input_url": f"/ui/api/admin/archive/{task_id}/file/input" if input_file else None,
        "output_url": f"/ui/api/admin/archive/{task_id}/file/output" if output_file else None,
    }


@router.get("/ui/api/admin/archive/{task_id}/file/{slot}",
         summary="An archived image",
         response_class=FileResponse,
         responses={200: {"description": "An archived original or result",
                          "content": {"image/png": {"schema": {"type": "string",
                                                               "format": "binary"}}}}})
async def admin_archive_file(task_id: str, slot: str,
                             session: Session = Depends(require_admin_session)):
    """Serve the archived original or result for one record.

    The slot is matched against two names, never used as a path: the record decides
    which file is read, so a request cannot name one.
    """
    if slot not in ("input", "output"):
        raise TaskNotFoundError(f"Unknown archive slot: {slot}")
    record = history_service.get_history_record(task_id)
    if not record:
        raise TaskNotFoundError(f"Record not found: {task_id}", details={"task_id": task_id})
    prefix = record.get("id", "")
    if slot == "input":
        name = os.path.basename(record.get("input_path", "") or "")
        archived = os.path.join(history_service.archive_input_dir(), f"{prefix}_{name}")
        stored = record.get("input_path", "") or ""
    else:
        name = os.path.basename(record.get("output_path", "") or "")
        archived = os.path.join(history_service.archive_output_dir(), f"{prefix}_{name}")
        stored = record.get("output_path", "") or ""
    for candidate in (archived, stored):
        if candidate and os.path.isfile(candidate):
            return FileResponse(candidate, media_type="image/png")
    raise TaskNotFoundError(
        f"No {slot} file is available for {task_id}", details={"task_id": task_id}
    )


# ---------------------------------------------------------------------- users
def _user_view(user: dict) -> dict:
    return {
        "username": user.get("username", ""),
        "role": user.get("role") or "user",
        "status": user.get("status"),
        "created_at": user.get("created_at"),
    }


@router.get("/ui/api/admin/users", response_model=ui_schemas.UserListView,
            summary="List users")
async def admin_users(session: Session = Depends(require_admin_session)):
    return {"users": [_user_view(user) for user in auth_service.list_users()]}


@router.post("/ui/api/admin/users", status_code=201,
             response_model=ui_schemas.UserResultView, summary="Create a user")
async def admin_create_user(payload: dict, session: Session = Depends(require_csrf)):
    """Create an account. Refuses an existing name and an empty password.

    The same three checks the Gradio callback made, in the same order, with the same
    wording — a migration that changed them would change what an operator sees for no
    reason.
    """
    if not session.is_admin:
        raise ForbiddenError(PERMISSION_DENIED_MSG)
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")
    role = str(payload.get("role") or "user")
    if not username or not password:
        raise InvalidRequestError("Username and password cannot be empty")
    if get_user(username):
        raise InvalidRequestError(
            f"User '{username}' already exists", code="USER_EXISTS", status_code=409
        )
    if role not in ("user", "admin"):
        raise InvalidRequestError(f"Unknown role: {role}")
    auth_service.create_user(username, password, role=role)
    return {
        "user": _user_view(get_user(username) or {"username": username, "role": role}),
        "message": f"✅ User '{username}' added",
    }


@router.patch("/ui/api/admin/users/{username}",
              response_model=ui_schemas.UserResultView, summary="Update a user")
async def admin_update_user(username: str, payload: dict,
                            session: Session = Depends(require_csrf)):
    """Change a user's password and/or role."""
    if not session.is_admin:
        raise ForbiddenError(PERMISSION_DENIED_MSG)
    if not get_user(username):
        raise InvalidRequestError(
            f"User '{username}' does not exist", code="USER_NOT_FOUND", status_code=404
        )
    password = payload.get("password") or None
    role = payload.get("role") or None
    if password is None and role is None:
        raise InvalidRequestError("Nothing to update: provide password and/or role")
    if role is not None and role not in ("user", "admin"):
        raise InvalidRequestError(f"Unknown role: {role}")
    auth_service.update_user(username, password=password, role=role)
    return {
        "user": _user_view(get_user(username) or {"username": username}),
        "message": f"✅ User '{username}' updated",
    }


@router.delete("/ui/api/admin/users/{username}", status_code=204,
                   summary="Delete a user")
async def admin_delete_user(username: str, session: Session = Depends(require_csrf)):
    """Delete an account, refusing the built-in admin.

    The built-in admin is the account a fresh deployment signs in with; deleting it
    would leave an instance nobody can administer.
    """
    if not session.is_admin:
        raise ForbiddenError(PERMISSION_DENIED_MSG)
    if username == "admin":
        raise InvalidRequestError("Cannot delete the built-in admin account")
    if not get_user(username):
        raise InvalidRequestError(
            f"User '{username}' does not exist", code="USER_NOT_FOUND", status_code=404
        )
    auth_service.delete_user(username)
    return None


# --------------------------------------------------------------------- models
def _model_view(model: dict) -> dict:
    return {
        "name": model.get("name", ""),
        "version": model.get("version"),
        "declared_version": model.get("declared_version"),
        "status": model.get("status"),
        "loaded": bool(model.get("loaded")),
        "device": model.get("device"),
        "weight_present": model.get("weight_present"),
        "capabilities": model.get("capabilities") or [],
    }


@router.get("/ui/api/admin/models", response_model=ui_schemas.ModelListView,
            summary="The model table")
async def admin_models(session: Session = Depends(require_admin_session)):
    """What this deployment declares and can run.

    The Gradio admin panel never reached the models API at all; the native UI adds the
    tab because the routes existed and an operator had no way to see them without a
    token and curl.
    """
    from fiximg.application.model_service import model_service

    return {"models": [_model_view(model) for model in model_service.list_models()]}


@router.post("/ui/api/admin/models/{name}/{action}",
             response_model=ui_schemas.ModelActionView,
             summary="Warm up or unload a model")
async def admin_model_action(name: str, action: str,
                            session: Session = Depends(require_csrf)):
    """`warmup` or `unload` — the two admin model actions with a UI affordance.

    Reload/rollback are deliberately *not* here: they take a version argument and
    change what serves traffic, which is an operator action with a token, not a
    button on a page.
    """
    if not session.is_admin:
        raise ForbiddenError(PERMISSION_DENIED_MSG)
    from fiximg.application.model_service import model_service

    if action == "warmup":
        model = model_service.warmup(name)
    elif action == "unload":
        model = model_service.unload(name)
    else:
        raise InvalidRequestError(f"Unknown model action: {action}")
    return {"model": _model_view(model), "message": f"✅ {name}: {action} requested"}


def _text(value) -> str:
    """A metric as the panel's table shows it: a string, or empty when absent."""
    return "" if value in (None, "") else str(value)


# ------------------------------------------------------------------- the page
def register_ui_routes(app) -> None:
    """Mount the UI router (called from the composition root).

    One function so the composition root names one thing. The router carries no
    prefix: it owns `/` itself, which is the route the Gradio mount used to own.
    """
    app.include_router(router)


__all__ = [
    "PERMISSION_DENIED_MSG",
    "bootstrap_payload",
    "current_session",
    "register_ui_routes",
    "require_admin_session",
    "require_csrf",
    "require_session",
    "router",
    "ui_principal",
]
