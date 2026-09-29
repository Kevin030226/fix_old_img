"""Response models for the diagnostics endpoints (plan §2.5 point 1, §3.8).

`/api/v1/stats` and the health probes used to return hand-built dicts, which made
them the two holes in the "every response is a declared model" rule: OpenAPI
showed free-form objects, so a client generator produced `dict[str, Any]` for the
one endpoint an operator actually polls. These models describe what the producers
build today — `tests/api/test_diagnostics_schemas.py` fails if the payload and the
schema drift in either direction.

Keys that are genuinely dynamic (one stage timing per stage, one metric series per
recorded name) stay mappings; everything with a fixed shape is a model.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class TaskTotals(BaseModel):
    """Task counts over the requested window."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    pending: int = 0
    users: int = Field(default=0, description="Distinct accounts that submitted in the window")
    avg_ms: int | None = None
    success_rate: float | None = Field(
        default=None, description="completed / total, None when there are no tasks"
    )


class StageTiming(BaseModel):
    """Per-stage duration statistics, measured in SQL (plan §2.7)."""

    runs: int = 0
    avg_ms: int | None = None
    p50_ms: int | None = None
    p95_ms: int | None = None


class MetricsSnapshot(BaseModel):
    """The in-process registry (plan §2.10)."""

    uptime_seconds: float = 0.0
    counters: dict[str, float] = Field(default_factory=dict)
    histograms: dict[str, dict[str, float | int | None]] = Field(default_factory=dict)


class ModelStats(BaseModel):
    """Model-manager diagnostics.

    `gpu_memory_peak_mb` appears only when the process can read a CUDA device, so
    the block's key set is environment-dependent by design: the schema declares both
    fields (so the document describes a GPU node's response too), and the test checks
    every emitted key is declared rather than comparing sets for equality.
    """

    load_times_ms: dict[str, float] = Field(default_factory=dict)
    gpu_memory_peak_mb: float | None = Field(
        default=None, description="Peak VRAM read by this process; absent without CUDA"
    )

    model_config = {"extra": "allow"}


class SchedulerStats(BaseModel):
    """GPU concurrency policy and in-flight slots (plan §2.4)."""

    policy: dict = Field(default_factory=dict)
    active: dict[str, int] = Field(default_factory=dict)
    acquired_total: int = 0

    model_config = {"extra": "allow"}


class StatsResponse(BaseModel):
    window_days: int | None = Field(default=None, description="0/None = all time")
    tasks: TaskTotals = Field(default_factory=TaskTotals)
    by_type: dict[str, int] = Field(
        default_factory=dict, description="task rows per task_type, within the window"
    )
    metric_averages: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "AVG() of each numeric metric within the window; a metric recorded only "
            "as text (PSNR '+inf') has no number to average and is absent"
        ),
    )
    stages: dict[str, StageTiming] = Field(
        default_factory=dict, description="keyed by stage name"
    )
    models: ModelStats = Field(default_factory=ModelStats)
    metrics: MetricsSnapshot = Field(default_factory=MetricsSnapshot)
    scheduler: SchedulerStats = Field(default_factory=SchedulerStats)

    # Extra keys pass through: a response model that silently dropped a producer's
    # new field would be worse than no model at all. The drift is caught by
    # `tests/api/test_diagnostics_schemas.py` instead, which compares both directions.
    model_config = {"extra": "allow"}


class WorkerReport(BaseModel):
    """The worker/queue view inside the readiness probe."""

    topology: str = Field(default="standalone", description="inline | standalone")
    poll_seconds: float = 0.5
    max_queue: int = 0
    lease_seconds: float = 0.0
    max_attempts: int = 1
    pinned_gpu: str | None = None
    capabilities: list[str] | None = None
    queue_backend: str | None = None
    #: Which server actually holds the queue table: the backend value names the
    #: transport family (`FIXIMG_QUEUE_BACKEND`), the driver names the database.
    queue_driver: str | None = None
    queued: int | None = None
    worker_running: bool | None = Field(
        default=False,
        description="This process' worker. None when the topology is standalone and "
        "nothing is running, which is idle-and-unknown rather than dead.",
    )
    #: Standalone topology only: workers that heartbeated a running task recently,
    #: and those whose beat is overdue. Read from the rows the worker itself writes.
    workers_reporting: int | None = None
    workers_silent: int | None = None
    oldest_queued_seconds: float | None = Field(
        default=None, description="How long the backlog's oldest waiter has waited."
    )
    worker_liveness: str | None = Field(
        default=None,
        description="alive | silent | backlogged | starting | idle-or-unknown | unknown",
    )

    model_config = {"extra": "allow"}


class ModelHealth(BaseModel):
    """`ModelManager.health()`: what is resident on this node and what CUDA sees."""

    cuda_available: bool = False
    loaded_models: list[str] = Field(default_factory=list)
    lazy_models: list[str] = Field(default_factory=list)
    versions: dict[str, dict] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class GpuReport(BaseModel):
    topology: dict = Field(default_factory=dict)
    scheduler: dict = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class PluginReport(BaseModel):
    enabled: bool = False
    entry_points: list[dict] = Field(default_factory=list)
    loaded: list[str] = Field(default_factory=list)
    models: list[dict] = Field(default_factory=list)
    stages: list[dict] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class RecentEventView(BaseModel):
    """One deployment-wide event, as `GET /api/v1/stats/events` returns it.

    The same shape the SSE stream emits per task (see `TaskEventSchema`), so a
    client that already renders a task's events renders these without a second
    model. `task_id` is null for events that belong to no task — a rejected
    submission, a worker lifecycle entry — which is precisely why they need this
    endpoint rather than `/tasks/{id}/events`.
    """

    #: Optional for the same reason `TaskEventSchema.seq` is: the domain type
    #: allows an event built in memory, with no stored row behind it. A row read
    #: back from `system_events` always carries its autoincrement id, so a client
    #: ordering by this gets a total order in practice — but the schema states
    #: what the domain actually holds rather than asserting a guarantee nobody
    #: enforces.
    seq: int | None = None
    event: str = "message"
    level: str = "info"
    task_id: str | None = None
    user_id: str | None = None
    message: str | None = None
    data: dict | None = None
    created_at: str | None = None


class RecentEventsResponse(BaseModel):
    """Newest-first page of deployment events."""

    events: list[RecentEventView] = Field(default_factory=list)


class LivenessResponse(BaseModel):
    """The process is up; nothing else is asserted."""

    status: str = "ok"


class ReadinessResponse(BaseModel):
    """Dependencies are usable, with the reason when they are not (plan §3.8)."""

    status: str = Field(description="ready | not_ready")
    database: str = "ok"
    detail: str | None = Field(default=None, description="exception name when database=error")
    models: ModelHealth = Field(default_factory=ModelHealth)
    model_health: dict[str, dict] = Field(
        default_factory=dict, description="per-backend health, keyed by model name"
    )
    storage_backend: str = "local"
    worker: WorkerReport = Field(default_factory=WorkerReport)
    gpu: GpuReport = Field(default_factory=GpuReport)
    plugins: PluginReport = Field(default_factory=PluginReport)

    model_config = {"extra": "allow"}
