# Architecture (V3)

> Audience: contributors and operators who need to know where a change belongs.

## Layering

```
                 ┌──────────────────────────────┐
                 │  Gradio UI  ·  JSON API      │   src/fiximg/ui, src/fiximg/api
                 └──────────────┬───────────────┘
                                │  (never talks to a model directly)
                 ┌──────────────▼───────────────┐
                 │  Application services        │   src/fiximg/application
                 │  task · model · auth         │
                 └──────────────┬───────────────┘
                                │
                 ┌──────────────▼───────────────┐
                 │  Domain                      │   src/fiximg/domain
                 │  Task · Artifact · Event      │
                 │  errors · enums              │
                 └──────────────┬───────────────┘
                                │
      ┌─────────────────────────┼─────────────────────────┐
      ▼                         ▼                         ▼
┌───────────────┐      ┌─────────────────┐      ┌────────────────────┐
│  Inference    │      │  Infrastructure │      │  Config / paths    │
│  runtime      │      │  db · queue     │      │  fiximg.config     │
│  planner      │      │  storage        │      │  fiximg.paths      │
│  registry     │      │  observability  │      │  configs/*.yaml    │
│  scheduler    │      │  security       │      │                    │
│  backends     │      │                 │      │                    │
│  stages       │      │                 │      │                    │
└───────────────┘      └─────────────────┘      └────────────────────┘
```

Dependency direction is strictly downward. `inference` may import `domain` and
`infrastructure`, never `api` or `ui`.

The rule is checked by reading the import statements of every file in the layer
(`tests/unit/test_architecture_boundaries.py`), and it decides real designs: the Auto
tab's analysis row needs the analyzer, so it asks `TaskService.auto_preview()` instead of
importing `fiximg.inference` — which is also what keeps the preview and the submission
running the same two calls rather than two copies of the same decision.

## Package map

| Path | Responsibility |
|---|---|
| `src/fiximg/api/routes/` | HTTP endpoints: `tasks`, `models`, `users`, `stats`, `health`, `auth` |
| `src/fiximg/api/schemas/` | Pydantic request/response contract (single source of truth) |
| `src/fiximg/api/errors.py` | Exception handlers → one error envelope |
| `src/fiximg/api/dependencies.py` | `Principal` resolution and admin guard |
| `src/fiximg/application/` | Task/model/auth/history services, pipeline mode labels |
| `src/fiximg/domain/` | Framework-free models, enums, error codes |
| `src/fiximg/inference/runtime.py` | `PipelineOrchestrator`: plan → stages → persist |
| `src/fiximg/inference/planner.py` | Task type → ordered stage list |
| `src/fiximg/inference/registry.py` | Stage registry, capability-driven lookup |
| `src/fiximg/inference/scheduler.py` | Per-device, per-capability GPU concurrency policy |
| `src/fiximg/inference/gpu.py` | Capability → device routing and worker pinning |
| `src/fiximg/inference/backends/` | `ModelBackend` implementations + registry (`native` in-process, `legacy-cli` subprocess; each reports which) |
| `src/fiximg/inference/versions.py` | Two-version residency, atomic switch, rollback |
| `src/fiximg/inference/stages/` | One class per processing capability |
| `src/fiximg/inference/evaluation/` | Reference / no-reference / IQA metrics |
| `src/fiximg/inference/worker.py` | Queue consumer: lease, heartbeat, retry |
| `src/fiximg/infrastructure/db/` | Dialect + connection layer (SQLite default, PostgreSQL), repositories, Alembic runner |
| `src/fiximg/infrastructure/queue/` | `QueueBackend`: SQLite and Redis Streams |
| `src/fiximg/infrastructure/storage/` | `ArtifactStore`: local today, S3/MinIO (P2) |
| `src/fiximg/infrastructure/observability/` | JSON logging, metrics registry, tracing (no-op / OTel / in-memory) |
| `src/fiximg/infrastructure/security/` | Password hashing, rate limiting |
| `src/fiximg/ui/` | Gradio blocks, history panel, HTML pages, middleware, progress stream |
| `src/fiximg/cli/` | Entry points: `api`, `worker`, `batch`, weights, migration |

Vendored model code (`Global/`, `Face_Detection/`, `Face_Enhancement/`,
`ddcolor/`, `basicsr/`) stays at the repository root: it is third-party code
that the legacy CLI adapters invoke as subprocesses.

## Execution flow

```
POST /api/v1/tasks
   → TaskService.enqueue
       → persist input under storage/tasks/<y>/<m>/<task_id>/input/
       → INSERT tasks(status=queued, priority, idempotency_key)
   → (inline thread | standalone process) PipelineWorker
       → QueueBackend.claim(worker_id, lease)
       → PipelineOrchestrator.execute_queued
            → Planner → [Stage …] under GpuScheduler slots
            → artifacts + metrics persisted
       → ack, or retry with backoff until max_attempts
   → GET /api/v1/tasks/{id}          status
     GET /api/v1/tasks/{id}/events   SSE stream
     GET /api/v1/tasks/{id}/result   image
```

### Delivery semantics (plan §2.6)

**At-least-once execution, deduplicated at submission.** Not exactly-once, and the
difference is visible in three places:

| aspect | behaviour |
|---|---|
| duplicate submissions | `Idempotency-Key` has a partial unique index, so a repeated `POST` returns the existing task rather than queueing a second one |
| duplicate execution | none is prevented: a worker that dies mid-run has its lease expired by the stale sweep, and the task is requeued and executed again; `max_attempts` bounds how often |
| who decides a claim | the database row. A claim is an atomic `UPDATE` (status, `attempt_count`, `worker_id`, `lease_until`, capability filter). On the Redis topology the stream entry is only the *wake-up*: `claim()` reads the stream, then the row decides whether anything is actually runnable |
| the transport's side | the stream entry is acknowledged when a task reaches a terminal state (finished, or attempts exhausted). A task scheduled for another attempt is **not** acknowledged, because `queue.retry()` — and the operator retry — republish it |

Consequences an operator can rely on:

- a run directory is keyed by task id, so re-execution **overwrites** the previous
  attempt's bytes and the artifact table keeps exactly one row per kind, always
  describing the current file (`GET /tasks/{id}/artifacts` cannot hand back a
  digest of bytes that no longer exist);
- publication to a remote object store writes the same key per attempt, so the
  stored object also ends up being the last attempt's;
- a task can be executed more than once and observed as `running` more than once:
  consumers that trigger their own side effects on `task.completed` should key on
  `task_id`, which is stable across attempts.

Ordering between tasks is by `priority` (`ORDER BY priority DESC, created_at, id` in the
claim), and `priority` is a field of `POST /api/v1/tasks`, bounded by
`FIXIMG_PRIORITY_MAX` — a submission that omits it is FIFO. What is *not* provided is any
exactly-once guarantee: retries re-execute the same task id.

Stages acquire their model through `ModelManager.acquire(name)`, which pins the
active *version* for the duration of the call. That is what lets a hot update
switch routing mid-run without interrupting an inference already in flight
(plan §3.15).

With `FIXIMG_EVAL_MODE=async` the task is marked `completed` before the quality
metrics are computed, so the result is downloadable immediately and the numbers
arrive as a `task.evaluated` event (plan §3.16).

### The JSON columns have two shapes

`get_task` / `list_tasks` fold the stage checklist, the metric map, the option blob and
the capability hint into single columns. **What type they come back as depends on the
engine**: `json_group_array` yields TEXT on SQLite, while psycopg decodes `json_agg`
before the caller sees it. So every reader goes through
`fiximg.domain.tasks.decode_json_column`, and a test over the production tree fails any
module that reads one of those columns without it.

That rule is not stylistic. Two readers had their own `json.loads` handling and one had
none, and each failure pointed the other way:

* the progress panel treated a string stage list as "no stages yet", so on the default
  (SQLite) engine its stage checklist never appeared;
* `_decode_capabilities` wrapped `json.loads` in `except TypeError`, so on PostgreSQL the
  already-decoded list raised, the hint became the empty set — a subset of everything —
  and workers claimed tasks whose capabilities they do not serve.

Both were invisible to a suite whose doubles handed back neatly decoded lists.

## Extension points

Adding a model touches exactly two places:

1. Implement `ModelBackend` in `src/fiximg/inference/backends/`.
2. Register it in `build_default_backend_registry()`.

Adding a *stage* (a pipeline step) adds one more:

3. Implement `BaseStage` in `src/fiximg/inference/stages/` and register it in
   `build_default_registry()` with its capabilities.

The planner resolves stages by capability, so no orchestrator, API or UI code
changes. See `docs/api.md` for the HTTP surface.
