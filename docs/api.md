# HTTP API (V3)

Base URL: `http://<host>:9502`. Interactive docs: `/docs` (Swagger UI) and
`/redoc`; the raw schema is at `/openapi.json`.

Every JSON response is a declared model, and `tests/api/test_diagnostics_schemas.py`
fails if the OpenAPI document shows a free-form JSON body — that endpoint set used to
include `/api/v1/stats` and the health probes, which is precisely what an operator
polls, so a generated client typed them as `dict[str, Any]`. Three routes return
something other than JSON and say so in the document rather than leaving FastAPI to
default them to it: `/api/v1/stats/metrics` (`text/plain`, Prometheus exposition),
`/api/v1/tasks/{id}/events` (`text/event-stream`, one `TaskEvent` per `data:` line)
and `/api/v1/tasks/{id}/result` (`image/png`).

## Authentication

Every `/api/v1/*` route requires a bearer token; the health probes are public.

```http
Authorization: Bearer <token>
```

The token comes from `FIXIMG_API_TOKEN`, or is generated on first start and
persisted to `admin_data/api_token.txt` (mode 0600) with a one-time console
notice. API-submitted tasks are attributed to `FIXIMG_API_TOKEN_USER`
(default `api`), so audit records name a real account rather than a placeholder.

## Error envelope

Every failure returns the same body, so clients branch on `code`:

```json
{
  "error": {
    "code": "IMAGE_TOO_LARGE",
    "message": "Image long side exceeds 4096 px",
    "request_id": "1f0c8a7b93de"
  }
}
```

Common codes: `INVALID_REQUEST`, `INVALID_IMAGE`, `IMAGE_TOO_LARGE`,
`UPLOAD_TOO_LARGE`, `INVALID_OPTIONS`, `UNSUPPORTED_TASK_TYPE`, `UNAUTHENTICATED`,
`FORBIDDEN`, `TASK_NOT_FOUND`, `TASK_NOT_READY`, `TASK_NOT_CANCELLABLE`,
`TASK_CANCELLED`, `ARTIFACT_EXPIRED`, `QUEUE_FULL`, `RATE_LIMITED`, `INTERNAL_ERROR`.

`RATE_LIMITED` (HTTP 429) comes from `POST /api/v1/tasks` when one principal exceeds
`FIXIMG_SUBMIT_MAX` submissions in `FIXIMG_SUBMIT_WINDOW` seconds; the envelope's
`details` carry the effective `limit`, `window_seconds`, which `scope` keyed the
counter (`principal` or `client-address`) and the `remaining` allowance, and the
rejected call never reaches the queue. Capacity is a separate answer: `QUEUE_FULL`
comes from the enqueue itself, whatever the caller's rate. See
[deployment.md](deployment.md#rate-limits) for the knobs and the Redis-backed shared
counters.

## Tasks

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/tasks` | Queue a task (multipart) |
| `GET` | `/api/v1/tasks` | List tasks (`limit`, `offset`, `user_id`) |
| `GET` | `/api/v1/tasks/{id}` | Status, progress, current stage |
| `GET` | `/api/v1/tasks/{id}/events` | Live events (SSE) |
| `GET` | `/api/v1/tasks/{id}/artifacts` | Artifact metadata |
| `GET` | `/api/v1/tasks/{id}/report` | Evaluation + planner decisions |
| `GET` | `/api/v1/tasks/{id}/result` | The finished image (PNG) |
| `POST` | `/api/v1/tasks/{id}/cancel` | Cancel a task that has not finished |
| `POST` | `/api/v1/tasks/{id}/retry` | Requeue a failed/cancelled task |

### Create

```http
POST /api/v1/tasks
Authorization: Bearer <token>
Content-Type: multipart/form-data
Idempotency-Key: 01J8Z...
```

| Field | Required | Notes |
|---|---|---|
| `type` | yes | `restore`, `restore_scratch`, `detect_scratch`, `colorize`, `auto_restore` |
| `image` | yes | Image file; ≤ `FIXIMG_MAX_UPLOAD_MB`, long side ≤ `FIXIMG_MAX_IMAGE_SIDE` |
| `options` | no | JSON object: `hr`, `face_enhance`, `auto_colorize` (see **Options**) |
| `priority` | no | `0`–`FIXIMG_PRIORITY_MAX` (default 0 = FIFO); higher is claimed first |
| `ground_truth` | no | Reference photo; enables LPIPS in the report |

`Idempotency-Key` makes the call safely retryable: a repeated key returns the
original task instead of running the work twice — including its original `priority`,
which the reply reports back as stored rather than as requested.

```json
{ "task_id": "20260925-034000-a1b2c3d4", "status": "queued", "created_at": "2026-09-25T03:40:00.148932Z", "priority": 7 }
```

Every timestamp is a fixed-width UTC instant (report §2.7): `Z`-suffixed with
microsecond precision, so two submissions in the same second stay distinguishable
and no value depends on the server's zone.

#### `options`

| Key | Type | Default if omitted | What `false` / `true` mean |
|---|---|---|---|
| `face_enhance` | tri-state | on for `restore`/`restore_scratch`; for `auto_restore`, "when the analysis finds a face" | `false` never runs the face chain; `true` runs it even when the analysis found none |
| `auto_colorize` | tri-state | off for the restoration types; for `auto_restore`, "when the photo reads as black and white" | `false` never colorizes; `true` colorizes regardless |
| `hr` | bool | off | selects the 512-px face weights inside the face stages. Not a plan switch: it never adds or removes a stage |

The two plan switches are tri-state because *silence is a value*: the route forwards only
the keys the caller named, so an omitted switch leaves the decision where that plan keeps
it (task type, or image analysis) while an explicit one is an instruction in both
directions. This matters most on `auto_restore`, whose stages come from the picture: the
analysis runs on a copy downscaled to 512 px, so `{"face_enhance": true}` is how a client
insists on the face chain for a large scan whose faces it could not count.

Unknown keys are rejected with `INVALID_OPTIONS` rather than ignored, so a misspelled
switch cannot silently submit the default pipeline. That rejection is also what keeps the
list honest: there used to be a fourth key, `strength`, which was accepted, stored in
`tasks.options_json`, published in this document — and read by nothing, so a client that
asked for `{"strength": 0.3}` got the identical image and a row that seemed to say
otherwise. A test now fails the build for any declared option without a reader in the
inference layer, and for any option the pipeline reads that the API does not accept.

### Status

```json
{
  "task_id": "20260925-034000-a1b2c3d4",
  "status": "running",
  "progress": 60,
  "current_stage": "face_enhancement",
  "duration_ms": null,
  "attempt_count": 1,
  "max_attempts": 3,
  "priority": 7,
  "stages": [{ "stage_name": "global_restore", "status": "completed", "duration_ms": 18422 }],
  "metrics": {}
}
```

`priority` is the queue position the submission asked for and the row kept (0 = FIFO). The
claim orders `priority DESC, created_at`, so it is the answer to "why did that job get the
GPU first", and it is read back from the row rather than echoed from the request: an
idempotent replay returns the earlier task and therefore the earlier task's position.
Above `FIXIMG_PRIORITY_MAX` the create call is refused with `INVALID_REQUEST` and the
deciding number in `details` — clamping silently would hand the caller a queue position it
never asked for.

`status` ∈ `queued` · `running` · `completed` · `failed` · `cancelled`.

A stage's `status` ∈ `pending` · `running` · `completed` · `failed` · `skipped`.
`skipped` means the stage declined its own work — the face chain reports it on an
install without dlib, and again when a photo has no aligned face — while the task
as a whole still completes. Read the stage's `message` for the reason; the status
is what a client should branch on.

`metrics` values are JSON **numbers** where the measurement is one
(`{"psnr": 22.5, "ssim": 0.93}`); what a number cannot hold stays a string —
`"psnr": "inf"` after a bit-identical output, or a label (report §2.7).

PSNR/SSIM/MAE compare the output with **the degraded input** (an
input-output difference, not a quality score). They are reported for
`restore`, `restore_scratch` and `auto_restore` — the three task types whose
stages restore the picture — and not for `colorize` or `detect_scratch`, where
comparing against the grayscale original or reading a mask as a photo is not a
meaningful number. When a run also colorizes a grayscale input (`auto_restore`
because the analysis said so, or `restore` with `options.auto_colorize`), the
report says so, because that deliberate change is part of what the difference
measures.
`face_count` and `enhanced_count` are **faces** the chain detected and enhanced.
`degraded_count` is not a face count: it counts *images* the warp-back stage
returned unrestored, so a photo with no face at all reads
`face_count=0, enhanced_count=0, degraded_count=1` with reason
`no_face_detected` in the stage message. `identity_faces_input`,
`identity_faces_output` and `identity_faces_paired` are what the §17 identity
metric's own detector saw in the two pictures. They are separate keys on purpose —
on an install without dlib the first group is zero while the second is not — and
`identity_similarity` from the fallback descriptor is an appearance-stability
number, not a biometric match.

`gpu_peak_mb` is the highest allocation that stage caused on the device named by
`gpu_device`; the task-level value is the largest single stage, not a sum. A stage
that measured nothing omits the key — that is different from a stage that measured
zero, and the run report says so when it happens.
`gpu_util_pct` is the **mean driver-reported utilisation of that device while the
stage ran**, sampled every 0.5 s, with `gpu_util_max_pct` (the highest single
reading in the window) and `gpu_samples` (how many readings it averaged). At task
level the mean is re-weighted by those sample counts, the maximum is the largest
stage's maximum, and the count is the total pooled — a 30 s stage and a 200 ms
stage must not count equally. All three are absent when the host has no driver to
ask, which is reported as such rather than as 0 %; see
[deployment.md](deployment.md) for the probe's `gpu.topology.utilization` block.
`GET /api/v1/stats`'s `models.gpu_memory_peak_mb` is the largest peak the
**deployment** has recorded — it folds in figures another process wrote, because an
API node that never loaded a model would otherwise report `0.0` beside stage rows
saying `334.8`. The same field in `GET /api/v1/models` is that process only.

The colorization stage also reports where its model actually executed:
`metadata.device` is the realised device (`"cuda:0"`, `"cpu"`, …) and
`metadata.device_requested` appears only when the scheduled card could not be
honoured. A device the host does not have is a decline, not a number to echo.

### Artifacts

```
GET /api/v1/tasks/{id}/artifacts
```

```json
{"task_id": "20260928-035125-31a9cccd",
 "artifacts": [
   {"kind": "output", "uri": "2026/09/20260928-035125-31a9cccd/output/final.png",
    "mime_type": "image/png", "width": 450, "height": 630, "size_bytes": 514847,
    "sha256": "b3ff…", "model_version": "warp_back@2.0"}]}
```

One row per `kind`, and `sha256`/`size_bytes`/`width`/`height` describe the bytes
that row points at — after a retry they describe the *new* attempt, because the row
is replaced rather than appended.

`uri` is a key, not a server path: the object-store key when the deployment
publishes to a remote store, and otherwise a path relative to the task root. An API
client fetches bytes through `GET /tasks/{id}/result`; it never opens the server's
disk.

| `kind` | what it is |
|---|---|
| `input` | the uploaded picture as received (also present for a task still queued) |
| `output` | what `GET /result` serves: the last stage's image, back at the input's geometry |
| `ground_truth` | the §16 reference photo, when the submission supplied one |
| `mask` | the scratch-detection mask |
| `restored` | stage 1's result, before the face chain |
| `result` | a backend's own output image, before the stage composed it |
| `final` | the vendored warp-back composite at the model's geometry (not what `/result` serves) |
| `faces_dir` | directory of detection crops |
| `each_img_dir` | directory of enhanced crops |

`model_version` names the stage and the model version that wrote the row
(`<stage>@<version>`) — `stage.version` as reported by the loaded backend, not the
manifest's declaration. `input` and `ground_truth` have none, because nothing in the
pipeline produced those bytes.

### Cancel and retry

```
POST /api/v1/tasks/{id}/cancel
POST /api/v1/tasks/{id}/retry
```

Any task that has not finished is cancellable — `queued` **and** `running`. A
`running` cancellation is cooperative: the row flips to `cancelled` immediately (so
this endpoint, the status view and the SSE stream agree at once), and the worker that
holds the task stops **at the next stage boundary**. It is not killed mid-stage,
because no stage in the pipeline is written to survive being interrupted half way
through a model call — so on a single-stage plan the stage in flight still finishes,
and then nothing starts after it.

What the user gets:

- the `task.cancelled` event, whose `data.effective` is `"stage boundary"` for a
  cancellation that arrived while a worker held the task and `"immediate"` for one
  that had not started;
- no `task.failed` event and no `error_code` on the row: a cancellation is a decision,
  not a fault;
- no further `task.progress` frames — the bar freezes where the task was stopped;
- no result: `GET /result` keeps answering 409/404 the way it does for any task that
  never produced bytes;
- no automatic retry. The attempt it was on is the only one it spends, and
  `POST /tasks/{id}/retry` requeues it deliberately if the user changes their mind.

Cancelling a task that already completed answers 409 `TASK_NOT_CANCELLABLE`, which is
the only case that refuses.

### Events (SSE)

```
GET /api/v1/tasks/{id}/events?last_event_id=0&follow=true
```

Replays persisted events after `last_event_id`, then tails new ones and closes
on a terminal event. Event names: `task.created`, `task.enqueued`,
`task.started`, `task.progress`, `stage.started`, `stage.completed`,
`stage.skipped`, `stage.failed`, `stage.retried`, `task.completed`,
`task.evaluated`, `task.failed`, `task.cancelled`, `task.retrying`.

`stage.retried` is informational and neither closes the stream nor counts as a
stage outcome: after it the stage still emits its own `stage.completed` /
`stage.failed`, so a client that counts stage results must ignore it. It appears
only when a stage ran out of VRAM on one device and was re-run on another.

`task.rejected` is deliberately **not** on this list. It is written when
`FIXIMG_WORKER_MAX_QUEUE` is reached, at which point no task row exists yet, so
there is no task id to stream against; the submitting client gets HTTP 503
`QUEUE_FULL` for that request instead.

`task.progress` is emitted while a stage runs, throttled to whole 5 % steps, so a
client watching a single-stage plan sees movement instead of waiting for the
stage to end. Its `data` carries `{"progress": <0-99>, "stage": <name>, "detail":
<stage message>}`; the task row always holds the latest value, the events are the
history.

Every stage ends in exactly one of `stage.completed` / `stage.skipped` /
`stage.failed`, so counting stage events is a valid progress signal even when a
stage opted out of its work.

```
id: 12
event: stage.completed
data: {"event":"stage.completed","task_id":"...","data":{"stage":"global_restore","duration_ms":18422}}
```

When `FIXIMG_EVAL_MODE=async` the stream stays open for a bounded window
(60 s) after `task.completed` so the follow-up `task.evaluated` — carrying the
quality metrics — reaches the client without reconnecting. With the default
`inline` mode, `task.completed` is the last event.

## Models

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/models` | Declared + runnable models (incl. active/previous version) |
| `GET` | `/api/v1/models/health` | Per-model health probe |
| `GET` | `/api/v1/models/{name}` | One model |
| `GET` | `/api/v1/models/{name}/versions` | Resident versions (this process) + recorded versions (the deployment) |
| `POST` | `/api/v1/models/{name}/reload` | Hot swap to a version (admin) |
| `POST` | `/api/v1/models/{name}/rollback` | Return to the previous version (admin) |
| `POST` | `/api/v1/models/{name}/unload` | Free memory (admin) |
| `POST` | `/api/v1/models/{name}/warmup` | Explicit warmup (admin) |

Each entry carries `framework` (what `models/manifest.yaml` declares) and
`implementation` (what is actually serving): `native` keeps the weights in this
process, `legacy-cli` spawns the vendored pipeline per run. They differ on purpose
— the manifest cannot know that the native backend was skipped for missing weights
— so check `implementation` when reasoning about GPU residency. Every registered
model reports one of the two; the name of the row is the registry key, not whatever
the serving object calls itself, because one adapter class serves the detection,
enhancement and restoration stages. `GET
/api/v1/models/health` additionally reports `hashes_ok`, `unverified` and
`weights_present` per model: a weight the integrity baseline does not cover is
listed as unverified rather than reported as verified.

`GET /api/v1/models/{name}/versions` answers two different scopes in one body, and
they are not interchangeable:

| Field | Scope |
|---|---|
| `active_version`, `previous_version`, `residents` | **this process** — residency is per-process by construction, and a pure API node legitimately has none |
| `catalog` | **the deployment** — one row per `(name, version)` that any process loaded, from `model_versions` (plan §6's ModelVersion) |

Each `catalog` entry carries the declared half (`framework`, `weight_uri`, `sha256`,
`metadata`) and the observed half (`status`, `loaded_at`, `load_ms`, `load_count`,
`last_error`, `writer`). Rules worth knowing before you graph them:

* `load_count` counts loads, because one version is one row however often it reloads.
* `loaded_at` is set only by a *successful* load. A row that has never come up keeps
  it `null`, so "tried and failed" cannot read as "ran at 03:14".
* A failed activation records `status: unhealthy` plus `last_error`, and a later
  success clears the error — the row is the latest observation, not a history.
* `sha256` is `null` when the deployment declares no digest for that weight; a reload
  that could not read the manifest leaves the previous digest in place rather than
  erasing it.
* `writer` names the process that recorded the row (`FIXIMG_WORKER_ID`, else `pid-N`),
  which is how a cross-node read stays interpretable.

The catalog covers the models the version registry loads — today `ddcolor`, the one
resident in-process model. The `legacy-cli` chains have no resident handle here; their
provenance is per artifact instead (`artifacts.model_version`, filled from the stage
that produced the file). A deployment that has never loaded a version reports an
empty `catalog`, which is not the same as the endpoint failing.

A native entry also carries `optimisations` — the precision policy in force
(`precision`, `channels_last`, `compile`) plus `applied`, one boolean per key the
manifest *declared*. `applied` is the difference between a configuration and a wish:
a declared layout the loaded network refused reads `false` there instead of going
unmentioned, and the field is `null` for an entry whose serving implementation has
no torch policy to apply (a subprocess adapter, or dlib). See "Precision policy" in
`docs/deployment.md`.

### Hot update

```http
POST /api/v1/models/ddcolor/reload?version=2.0.0
```

The candidate is built, checksum-validated, warmed and health-checked while the
current version keeps serving; routing then switches atomically and the old
version is drained once its in-flight calls finish. A failing candidate returns
`switched: false` and changes nothing:

```json
{
  "name": "ddcolor",
  "version": "2.0.0",
  "switched": false,
  "reason": "ModelUnavailableError: Checksum mismatch",
  "active_version": "1.0.0",
  "residents": [{ "version": "1.0.0", "status": "ready", "refcount": 0 }]
}
```

Reloading the version that is already active reports `switched: false` with
`reason: "already active"` — weights are never overwritten in place while the
service runs; ship new weights under a new version.


## Users

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/users` | List users (admin) |
| `POST` | `/api/v1/users` | Create a user (admin) |
| `PATCH` | `/api/v1/users/{username}` | Update password/role (admin) |
| `DELETE` | `/api/v1/users/{username}` | Delete a user (admin) |

## Stats and health

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/stats` | Task totals (incl. distinct `users` and `by_type` counts), per-metric averages, per-stage P50/P95, metrics, scheduler state |
| `GET` | `/api/v1/stats/metrics` | Prometheus text exposition |
| `GET` | `/api/v1/health/live` | Liveness (public) |
| `GET` | `/api/v1/health/ready` | Readiness: DB, models, worker, storage (public) |
| `GET` | `/health`, `/health/ready` | V2 aliases, kept for existing deployments |

## Gradio UI

The Gradio app is mounted at `/` behind the same user accounts (form login).
It talks to the service layer in-process, so it needs no API token.

Each tab that submits a restoration pipeline shows the plan §3.9.1 layout: the input
image, the option switches, the result column, and one progress strip that carries all
four numbers the plan asks for:

```text
[██████████░░░░░░░░]  45%  ·  Restore (no scratches)  ·  global_restore  ·  ETA  ~55 s  ·  Queue  2
```

| Field | Where it comes from |
|---|---|
| percent, current stage | `tasks.progress` / `tasks.current_stage`, written by the runtime that owns the plan |
| `ETA` | extrapolated from those two: `elapsed × (100 − progress) / progress`, aged against the task's **own run start** (not submit time), so queue waiting cannot inflate it |
| `Queue` | the queue's depth — the deployment backlog, which is the same number `GET /api/v1/health/ready` reports |

`ETA --` means *no estimate yet*, and it is shown until there is one: below 20 %
progress or one second of execution, an extrapolation is noise, and a panel that
printed "~3 s" from a 200 ms sample would be wrong in the direction users notice.
`Queue ?` is the same honesty for a backlog that could not be read.

### Option checkboxes

The switches are the `options` keys the JSON API accepts — `face_enhance`,
`auto_colorize` and `hr` — and a gate compares them against `TaskOptionsSchema`
rather than a second list. An unchecked box always sends `false`, never "use the
task-type default": a control the user can see must not have a second, invisible
meaning.

What a **checked** box sends depends on where the plan comes from:

| Tab | Plan source | Checked sends | Unchecked sends |
|---|---|---|---|
| Restore / Restore with scratches | `plan(task_type, options)` | the key with `true` (the chain / colorization is scheduled) | `false` (declined) |
| Auto Restore | `plan_auto_restore(analysis, options)` | **nothing** — the image analysis stays the authority | `false` (declined whatever the analysis says) |

Auto Restore therefore captions the same two keys differently ("only when the analysis
finds a face"), and the difference is not cosmetic: `true` on that tab means *force* —
run the face chain even though the analyzer found nothing. The analyzer works on a copy
downscaled to 512 px, so that is exactly what a user needs on a large scan with small
faces, and an API client can ask for it directly:

```http
options={"face_enhance": true, "auto_colorize": false}
```

with `type=auto_restore`. Omitting a key leaves the decision with the analysis;
`hr` is never a plan switch — it selects the face weights inside the stages.

What the switches decided is recorded in the run report next to the raw analysis
(`planner_decisions.options` beside `planner_decisions.analysis`), so a shortened
pipeline can be told apart from an analyzer that simply found nothing.

Colourisation and scratch detection offer no switches: their plan has no face chain and
no grayscale branch to turn down.

### Auto analysis row

§3.9.1's input column has three things: the image, **Auto Analysis**, and the parameters.
The row sits above the switches on the Auto tab and shows, before any GPU work:

```text
Analysis: grayscale=no scratch=0.79 blur=0.00 faces=1 size=496x624
Planned pipeline: scratch_repair → face_detection → face_enhancement → warp_back
Forced by your switches, not by the image: colorization
```

* It is produced by `TaskService.auto_preview()` — the same `analyze_image()` plus
  `plan_auto_restore(analysis, options)` the worker will run, so a preview that drifted
  from the submission is not possible. The stage labels are the names the task row
  records (`global_restore` on the scratch branch reports itself as `scratch_repair`), and
  the numbers are `format_analysis_summary()`'s own text, which is also what the worker
  writes into its log line.
* It recomputes when the upload changes **and** when any switch changes, because the
  switches are part of the answer. The last two lines appear only when a switch actually
  refused or forced something, so an untouched tab claims no decisions it did not make.
* Only Auto Restore has the row. The restoration tabs take their stages from the task
  type, so a list of measurements there would describe nothing.
* Cost, measured on this machine: 7–11 ms warm, 30–50 ms cold (the first call also
  resolves and loads the face backend, 12 ms of that). It is CPU CV on a copy downscaled
  to 512 px — no GPU tensor, no restoration weights — so it is safe in the web process;
  with `FIXIMG_INLINE_WORKER` true the same process was already running this analyzer for
  every Auto submission, and the only new footprint on a standalone-worker deployment is
  the detector itself (YuNet: a 232 KB ONNX file, read once per process).

