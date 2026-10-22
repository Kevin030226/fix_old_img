"""Response models for the native web UI's JSON routes (plan §2.5 / §3.9).

The JSON API declares a model for every response so the OpenAPI document is
complete and a renamed field breaks the build instead of a client. The UI's routes
are no different in kind — they are JSON over HTTP like any other — so they declare
theirs too. `tests/api/test_diagnostics_schemas.py` holds every route to that rule,
and it does not exempt a route because a browser happens to be the client.

Two of these shapes are deliberately permissive. The progress frame and the history
detail carry blocks whose keys come from the database (a stage's own metrics, a
task's decoded `options`), so those are typed as `dict` rather than restated field
by field: a model that enumerated them would need an edit every time a stage
reported a new measurement, and the resulting drift is silent.

Nothing here builds HTML. The document itself is rendered by
:mod:`fiximg.ui.web.pages` and returned as `text/html`, which the same gate exempts
for the reason it exempts `text/plain`: a document is not a data model.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SessionView(BaseModel):
    """`GET /ui/api/session` — who is signed in and what the page may do."""

    username: str
    role: str
    is_admin: bool
    csrf_token: str
    must_change_password: bool
    can_submit: bool


class MessageView(BaseModel):
    """A one-line result, for the routes whose answer is only a sentence."""

    message: str


class TaskCreatedView(BaseModel):
    """`POST /ui/api/tasks` — the queue entry, plus where to poll it."""

    task_id: str
    status: str
    created_at: str | None = None
    priority: int | None = None
    poll_url: str


class StageRowView(BaseModel):
    """One line of the progress checklist."""

    name: str
    status: str
    icon: str
    duration_ms: int | None = None
    duration_text: str = ""
    message: str = ""
    reason: str = ""


class TaskFrameView(BaseModel):
    """`GET /ui/api/tasks/{id}` — one poll frame.

    Carries both the structured parts and the `text` the panel used to yield, so a
    client can render either and the two cannot disagree about what a row means.
    """

    task_id: str | None = None
    status: str
    progress: int = Field(default=0, ge=0, le=100)
    current_stage: str | None = None
    task_type: str = ""
    label: str = ""
    stages: list[StageRowView] = Field(default_factory=list)
    stage_text: str = ""
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: int | None = None
    queue_depth: int | None = None
    result_ready: bool = False
    text: str = ""
    failed: bool = False
    cancelled: bool = False
    completed: bool = False


class AnalysisView(BaseModel):
    """`POST /ui/api/analysis` — the Auto tab's row.

    `preview` is the service's own dict; its shape is the analyzer's business, so it
    stays a dict here rather than being frozen into a schema that would have to
    track every new measurement.
    """

    available: bool
    text: str
    preview: dict[str, Any] = Field(default_factory=dict)


class HistoryRowView(BaseModel):
    """One row of the history table, already formatted for display."""

    task_id: str
    task_type: str
    created: str
    duration: str
    status: str
    progress: str
    metrics: str
    stages: str


class ThumbnailView(BaseModel):
    """One gallery tile."""

    task_id: str
    thumbnail_url: str
    caption: str


class HistoryPageView(BaseModel):
    """`GET /ui/api/history` — one page, with the pagination the page shows."""

    rows: list[HistoryRowView] = Field(default_factory=list)
    thumbnails: list[ThumbnailView] = Field(default_factory=list)
    page: int
    page_size: int
    page_label: str
    status: str


class HistoryDetailView(BaseModel):
    """`GET /ui/api/history/{id}` — the selected row's previews and detail.

    Each `*_url` is null when the file it would serve is not on disk, so a client
    sets an `<img>` source rather than rendering a broken image element.
    """

    task_id: str
    detail: str
    input_url: str | None = None
    result_url: str | None = None
    mask_url: str | None = None
    face_url: str | None = None
    download_url: str | None = None
    status: str | None = None
    task_type: str | None = None
    can_retry: bool = False
    can_rerun: bool = False


class RerunView(BaseModel):
    """`POST /ui/api/history/{id}/rerun|retry` — the id that was queued."""

    task_id: str
    message: str


class AdminHistoryRowView(BaseModel):
    """One row of the admin task table."""

    timestamp: str = ""
    user: str = ""
    type: str = ""
    status: str = ""
    psnr: str = ""
    ssim: str = ""
    mae: str = ""
    input_name: str = ""
    task_id: str = ""


class AdminHistoryView(BaseModel):
    """`GET /ui/api/admin/history` — every task plus the deployment's totals."""

    rows: list[AdminHistoryRowView] = Field(default_factory=list)
    stats: str
    empty: bool


class ArchiveRecordView(BaseModel):
    """One entry in the archive selector."""

    task_id: str
    label: str


class ArchiveListView(BaseModel):
    """`GET /ui/api/admin/archive`."""

    records: list[ArchiveRecordView] = Field(default_factory=list)


class ArchiveEntryView(BaseModel):
    """`GET /ui/api/admin/archive/{id}` — one record's images and detail."""

    task_id: str
    detail: str
    input_url: str | None = None
    output_url: str | None = None


class UserView(BaseModel):
    """One account."""

    username: str
    role: str = "user"
    status: str | None = None
    created_at: str | None = None


class UserListView(BaseModel):
    """`GET /ui/api/admin/users`."""

    users: list[UserView] = Field(default_factory=list)


class UserResultView(BaseModel):
    """`POST`/`PATCH /ui/api/admin/users[...]` — the account and a sentence."""

    user: UserView
    message: str


class ModelView(BaseModel):
    """One model, as the Models tab shows it."""

    name: str
    version: str | None = None
    declared_version: str | None = None
    status: str | None = None
    loaded: bool = False
    device: str | None = None
    weight_present: bool | None = None
    capabilities: list[str] = Field(default_factory=list)


class ModelListView(BaseModel):
    """`GET /ui/api/admin/models`."""

    models: list[ModelView] = Field(default_factory=list)


class ModelActionView(BaseModel):
    """`POST /ui/api/admin/models/{name}/{action}`."""

    model: ModelView
    message: str


class SignedOutView(BaseModel):
    """`POST /ui/api/session/logout`."""

    signed_out: bool


__all__ = [
    "AdminHistoryRowView",
    "AdminHistoryView",
    "AnalysisView",
    "ArchiveEntryView",
    "ArchiveListView",
    "ArchiveRecordView",
    "HistoryDetailView",
    "HistoryPageView",
    "HistoryRowView",
    "MessageView",
    "ModelActionView",
    "ModelListView",
    "ModelView",
    "RerunView",
    "SessionView",
    "SignedOutView",
    "StageRowView",
    "TaskCreatedView",
    "TaskFrameView",
    "ThumbnailView",
    "UserListView",
    "UserResultView",
    "UserView",
]
