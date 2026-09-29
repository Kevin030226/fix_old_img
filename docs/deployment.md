# Deployment (V3)

## Profiles

Configuration resolves in this order, lowest precedence first:

```
code defaults → configs/base.yaml → configs/<profile>.yaml → FIXIMG_* environment variables
```

Select the profile with `FIXIMG_PROFILE` (or `FIXIMG_ENV`), default `local`.

| Profile | File | Topology |
|---|---|---|
| `local` | `configs/local.yaml` | One process: API + Gradio + inline worker, SQLite, local artifacts, lazy models |
| `docker` | `configs/docker.yaml` | API and worker as separate processes sharing a DB and artifact volume |
| `production` | `configs/production.yaml` | Split topology, startup warmup, no auto-provisioned accounts, fail-fast validation |

### What a production boot refuses

`app_env=production` (set by `configs/production.yaml`) turns on fail-fast checks;
a boot that would otherwise come up "healthy" on accidental configuration stops
with every problem listed at once:

| refused | how |
|---|---|
| no API token | `FIXIMG_API_TOKEN` unset **and** no token file present — the service does not mint its own secret. An operator-placed file counts as configured. (Plan §3.3 allows auto-generated development configuration *locally only*; a generated token also means a second replica boots with a different secret.) |
| `storage_backend=s3` with no bucket | `FIXIMG_STORAGE_BUCKET` required |
| `queue_backend=redis` with no URL | `FIXIMG_REDIS_URL` required |
| missing model manifest | `FIXIMG_MODEL_MANIFEST` must point at an existing file |

Two of plan §3.3's four gates are **not** enforced here, on purpose:

* *"`DATABASE_URL` missing → do not boot"* — SQLite is the shipped single-node
  production topology (`configs/production.yaml` sets no database URL and
  `docker-compose.yml` mounts a volume for it), so "missing" is a valid
  configuration rather than a mistake. The production requirement is expressed as
  something the code can actually check: the schema must not lag the code, which
  `make db-check` and the boot-time integrity probe report.
* *"invalid object-storage credential → do not boot"* — deciding that needs a
  network round trip at start-up, and the boto3 credential chain legitimately
  resolves through an instance role with no environment keys, so "no env
  credentials" is not evidence of an invalid configuration. That probe lives in
  `make smoke-services`, which is run against a real endpoint, and a failed
  publish degrades to a warning instead of a failed task.

## Local development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[gpu,test,dev]"      # or ".[test]" for a CPU-only setup
python -m fiximg.cli.download_weights download
python main.py                        # http://127.0.0.1:9502
```

Equivalent Make targets: `make install-gpu`, `make weights`, `make serve`.

Useful commands:

| Command | Purpose |
|---|---|
| `make test` | CPU test suite (GPU-marked tests deselected) |
| `make test-gpu` | the tier that needs CUDA and/or the model weights |
| `make test-postgres` | the data path against a live PostgreSQL (`FIXIMG_TEST_POSTGRES_URL`) |
| `make lint` / `make typecheck` | ruff / mypy |
| `make syntax` | parse every file with *this* interpreter's grammar — CI runs it on 3.11, the deployment target |
| `make bench` | latency + throughput + memory benchmarks |
| `make worker` | standalone GPU worker process |
| `make migrate` | V1 history → task tables (dry run) |
| `make verify-weights` | verify weights against the manifest |

## Containers

| File | Topology | Use when |
|---|---|---|
| `Dockerfile` | All-in-one (API + UI + inline worker) | Personal machine, demo |
| `docker/compose.local.yaml` | Single container | Same, with GPU passthrough declared |
| `docker-compose.yml` | API + GPU worker | Single GPU host, production-ish |
| `docker/compose.yaml` | API + worker (+ Redis/PostgreSQL/MinIO behind `--profile platform`) | Multi-worker / multi-node |
| `docker/compose.multigpu.yaml` | API + one worker per GPU | 2+ GPUs, specialised workers |

```bash
# single container
docker compose -f docker/compose.local.yaml up -d --build

# split topology
docker compose up -d --build && docker compose logs -f worker

# full platform
docker compose -f docker/compose.yaml --profile platform up -d

# multi-GPU
docker compose -f docker/compose.multigpu.yaml up -d --build
```

The API image (`docker/api.Dockerfile`) is deliberately torch-free: it
validates requests, persists tasks and serves results, so it stays small and
starts fast. The worker image (`docker/worker.Dockerfile`) carries the full GPU
stack and the vendored model code.

## Release

`.github/workflows/release.yml` publishes what CI proves. Pushing a tag
`vX.Y.Z` runs four jobs:

1. **version** — the tag must equal `project.version` in `pyproject.toml`, or the
   run fails. The alternative (a release whose name and whose code disagree) is
   silently possible and loudly confusing afterwards.
2. **artifacts** — `python -m build`, then the wheel is installed into a throwaway
   venv and the backend registry is built *from the installed package*. A source
   tree always looks complete; only an installed wheel shows a missing module.
3. **images** — `fiximg-api` and `fiximg-worker` to `ghcr.io/<owner>/…`, tagged
   `:<version>` and `:latest`, with SLSA provenance and an SBOM attached.
4. **release** — a GitHub Release carrying the wheel, the sdist and their
   `sha256sums.txt`, plus the changelog since the previous tag.

A `workflow_dispatch` run does all of the above **except** pushing, and that is
the default for manual runs — rehearsing the pipeline must not publish anything.
The step that records the image digest says `(not pushed — dry run)` in that case
rather than printing an empty field next to a green check.

## Scaling out

The queue transport and the artifact store are selected by configuration, not by
code changes:

```bash
FIXIMG_QUEUE_BACKEND=redis   FIXIMG_REDIS_URL=redis://redis:6379/0
FIXIMG_STORAGE_BACKEND=s3    FIXIMG_STORAGE_BUCKET=fiximg
FIXIMG_S3_ENDPOINT=http://minio:9000
FIXIMG_DATABASE_URL=postgresql+psycopg://user:pw@postgres:5432/fiximg
```

Both backends implement the same protocols (`QueueBackend`, `ArtifactStore`), so
the worker, runtime and API are untouched. `FIXIMG_WORKER_ID` must be unique per
worker replica.

What travels through the store is **both ends of a run**, not just its result: the
uploaded input is published at submit time (`artifacts.uri` for kind `input`), and
a worker that claims a task whose file it has never seen fetches it back before
stage 1 opens it. Without that half, a remote store supported downloading a result
on another node while a worker on that same node failed on the input — a split
topology that looked supported and was not. The same key rule is used for both
(`object_key_for`, relative to `FIXIMG_TASKS_ROOT`), and the retention sweeper
reclaims published inputs alongside published outputs.

**The queue transport is load-bearing in a way the store is not, and both needed
their producer wired.**

A submission is written to the database **and** published to the Redis stream
(`task_service._notify_queue`), because `claim()` reads the stream while the
database only decides what a claim *means* — lease, attempts, capabilities. Skip
that second step and every Redis worker polls an empty stream while nothing runs;
the SQLite transport hides the omission, since its `enqueue` is a documented
no-op. The publish is best-effort: a Redis hiccup never fails a submission the
database already accepted, and the stale-lease sweep or the next retry recovers
the wake-up.

Artifacts are published through the store when the store is remote. On completion
the runtime calls `PipelineOrchestrator._publish_output()`, which uploads the
result image under the task's own key (`2026/09/<task_id>/output/final.png`) and
records it in `artifacts.uri`; `task_result_path()` reads that key back through
the store when the local file is not there, and `purge_expired_published()`
reclaims objects whose row passed `FIXIMG_RESULT_TTL` (the local backend is left
to the directory sweep — two reclaimers over the same files is how artifacts
disappear early).

| Setting | Default | Meaning |
|---|---|---|
| `FIXIMG_STORAGE_ROOT` | `<project>/storage` | Parent of the artifact tree. Set it when the system volume is small |
| `FIXIMG_TASKS_ROOT` | `<storage_root>/tasks` | The run directories the TTL sweeper walks and the publisher resolves keys against. A relative value is resolved against the project root, not the working directory, so two processes started from different places cannot disagree about where the bytes live |

Both are read when the settings object is built, so a container that mounts a
volume at `/data/storage` should set `FIXIMG_STORAGE_ROOT=/data/storage` rather
than expect `base_dir` to carry the artifacts with it.

That chain is what makes `FIXIMG_STORAGE_BACKEND=s3` mean something: before it,
the pipeline wrote through `storage.local` regardless, the protocol existed with no
production caller, and a second node serving downloads answered
`410 Gone` for tasks that had completed successfully. `uri` stays NULL for a local
store on purpose — every reader does `row["uri"] or row["path"]`, and a *relative*
key in local mode would resolve against the process's working directory.

**What is still unverified:** the S3 leg runs against `moto` (the real S3 API in
process, so real boto3 code paths, not a hand-written double) but never against a
MinIO or S3 endpoint over a socket. `make smoke-services` is the check that does
that, and it needs a running instance.

## Warmup strategies (plan §3.5.3)

`FIXIMG_WARMUP` picks when a model pays its load and its dummy inference; a model
can override it with a `warmup:` key in `models/manifest.yaml`. One resolver
(`ModelManager.warmup_strategy`) answers the question, so the boot preloader and
the activation-time warmer cannot each invent their own precedence:

| strategy | behaviour |
|---|---|
| `lazy` (default) | load on first request, **no** dummy call — nothing extra happens before someone needs the model |
| `first-use` | load on first request and run one tiny dummy inference, so the first real request is not the slow one |
| `startup` | preload and warm at process boot |

Who runs this is the process that executes inference: `PipelineWorker.start()`
calls `warm_at_process_start()`, which walks the backend registry and warms every
model whose *effective* strategy is `startup`. The API process runs the same
function (an inline-worker deployment is the same process anyway); a pure API node
in front of remote workers therefore no longer holds GPU weights it will never use,
and a standalone worker no longer boots cold. One model failing to warm is logged
and skipped — the worker still starts and takes work.

Before this existed, `warmup:` was documented in the manifest header and set on
every model but read by nothing, and the boot preload lived only in `create_app()`.

## GPU concurrency

Execution is serialised per capability by default (`FIXIMG_CONCURRENCY=1`),
which is the safe setting for models that share VRAM. Independent capabilities
can be allowed to overlap:

```bash
FIXIMG_CONCURRENCY=1                     # default for everything
FIXIMG_CONCURRENCY_SCRATCH_DETECTION=2   # detection is cheap and CPU-bound
```

Limits are tracked **per device**, so two GPUs each get their own budget.

## Multi-GPU routing

Routing is by *capability*, never by model name, so the table survives a model
swap (plan §3.14.2). Two mechanisms work together:

**1. In-process routing** — `FIXIMG_GPU_ROUTING` assigns capabilities to devices,
so one worker can run restoration on GPU0 and colorization on GPU1 without
loading every model on every card:

```bash
FIXIMG_GPU_ROUTING="0:restore,scratch_repair,face_restore;1:colorize"
```

Any capability not listed falls back to `FIXIMG_DEVICE`, which keeps an
unconfigured deployment behaving exactly as before.

**2. Worker pinning** — `FIXIMG_WORKER_GPU` pins a worker to one card and
`FIXIMG_WORKER_CAPABILITIES` declares what it may claim. A worker only claims
tasks whose `required_capabilities` are a subset of what it serves, so a
colour-only worker never picks up a restoration task it could not finish:

The hint is computed by the planner from the submitted options, and it is deliberately
empty for `auto_restore`: that plan is chosen from the image, and the scratch branch
still is even when the face and colorize switches pin the other two stages down. A
hint that overstated the requirement would strand such a task on a queue where every
worker looks healthy, so an `auto_restore` task can be claimed by any worker, and the
stages it cannot serve report `skipped` instead of being fabricated.

```bash
FIXIMG_WORKER_GPU=1
FIXIMG_WORKER_CAPABILITIES="colorize"
FIXIMG_WARMUP=startup      # preload only the models this worker serves
```

A ready-made two-GPU topology lives in `docker/compose.multigpu.yaml`. The
resolved topology and per-device scheduler state are visible at
`GET /api/v1/health/ready` (under `gpu`) and `GET /api/v1/stats` (under
`scheduler`).

> The task's required capabilities are derived from the planner — from the plan its
> options produce, not from its task type. So `{"auto_colorize": true}` adds `colorize`
> (a worker without the colour model will not take that task) and
> `{"face_enhance": false}` removes the face-chain capabilities (a worker without dlib
> can still take it). `auto_restore` picks its plan from the image, so it records no
> requirement and can be claimed by any worker.

## Model hot update

Two versions stay resident, so upgrading a model never takes it offline:

```
load v2 → validate (sha256) → warmup → health → SWITCH → drain v1 → unload v1
```

```bash
# 1. ship the new weights under a new path and bump models/manifest.yaml
# 2. trigger the switch; the running version serves throughout
curl -X POST -H "Authorization: Bearer $TOKEN" \
     "$HOST/api/v1/models/ddcolor/reload?version=2.0.0"
# 3. if the new version misbehaves, go back — the old one is still resident
curl -X POST -H "Authorization: Bearer $TOKEN" \
     "$HOST/api/v1/models/ddcolor/rollback"
```

In-flight inferences hold a lease on the version they started with, so a switch
never interrupts a running call; the old version is unloaded only once the last
lease is released. A candidate that fails validation, warmup or the health probe
is discarded and the running version is left untouched.

| Setting | Default | Meaning |
|---|---|---|
| `FIXIMG_MODEL_KEEP_PREVIOUS` | `true` | Keep the replaced version resident as the rollback target. `false` trades rollback speed for GPU memory |

> Weights are never overwritten in place while the service runs. Pick up new
> weights by publishing a new version, not by replacing the `.pth` file.

### Which version ran, and in which process

`model_versions` (revision `0005`, plan §6's ModelVersion) holds one row per
`(name, version)` that any process has loaded. Residency itself stays per-process, so
`GET /api/v1/models/{name}/versions` answers two scopes: `residents` for the process
you asked, `catalog` for the deployment.

The registry writes the row at the two moments that are facts rather than claims:

| Event | Row written |
|---|---|
| A version built and warmed successfully | `status: ready`, `loaded_at`, `load_ms`, `load_count += 1`, `last_error` cleared |
| Validation, warmup or the health probe failed | `status: unhealthy`, `last_error`, **no** `loaded_at` |

Nothing is written on unload: one process dropping a version says nothing about
whether another still serves it, and a `registered` row would be a claim about the
deployment that no single process is entitled to make. The declared fields
(`framework`, `weight_uri`, `sha256`, `metadata`) come from `models/manifest.yaml` at
write time and survive a later reload that could not read the manifest, so a
verification digest is never erased by an unrelated failure.

Measured live on this box — PostgreSQL 18.6, an API node with
`FIXIMG_INLINE_WORKER=0` beside a GPU worker (`FIXIMG_WORKER_ID=worker-gpu1`), one
real `colorize` task:

```text
before any load anywhere:      residents [] · catalog []
after the worker ran the task: residents [] · active_version null        ← still true
                               catalog: [{name: ddcolor, version: 1.0.0, status: ready,
                                         framework: pytorch,
                                         weight_uri: weights/ddcolor/pytorch_model.pt,
                                         sha256: null,             # none declared
                                         loaded_at: 2026-09-28T00:09:07.318036Z,
                                         load_ms: 3418, load_count: 1,
                                         writer: worker-gpu1,
                                         metadata: {channels_last: true, precision: fp16,
                                                    compile: false, warmup: first-use}}]
```

`sha256: null` is what "this deployment declares no digest for that weight" looks
like; it is not a verification result. Only `ddcolor` is a version-registry model, so
only it leaves rows — the `legacy-cli` chains record provenance per artifact
(`artifacts.model_version`) instead.

## Rate limits

Two different protections, and the reply shape differs because they answer at
different layers:

| What | Key | Knobs | Reply |
|---|---|---|---|
| `POST /api/v1/tasks` submissions | the authenticated principal (`user:<id>`), falling back to the client address when a request resolves no identity | `FIXIMG_SUBMIT_MAX` (default `60`) per `FIXIMG_SUBMIT_WINDOW` (`60` s); `0` disables | the canonical error envelope: `429` + `code: RATE_LIMITED` + `details.{limit, window_seconds, scope, remaining}` |
| account registration (`/register`, HTML) | client IP, a global counter, and the username | `FIXIMG_REGISTER_MAX` (`5`), `FIXIMG_REGISTER_GLOBAL_MAX` (`20`), `FIXIMG_REGISTER_USERNAME_MAX` (`3`), all inside `FIXIMG_REGISTER_WINDOW` (`600` s) | the HTML page with a "try again later" status line |

Three properties worth knowing before you tune them:

* The allowance is **per key, not per endpoint**: two accounts sharing one NAT
  egress do not exhaust each other, and one account cannot spend the queue from
  three browsers.
* When `FIXIMG_REDIS_URL` is set the counters live in Redis, so replicas share one
  allowance instead of each keeping its own. A Redis failure falls back to the
  in-process window (fail-open) — a cache outage must not take the service down.
* `RATE_LIMITED` is not capacity. When the queue itself is full the answer is
  `QUEUE_FULL` at enqueue time; the submission limit only stops one caller from
  flooding the endpoint, and it is checked **before** the uploaded image is decoded,
  which is the work a flood is paying for.

`FIXIMG_TRUSTED_PROXIES` (comma-separated peer addresses) decides whether
`X-Forwarded-For` is believed at all: a request arriving from any other peer is
keyed by its real address, so the header cannot be used to bypass a per-IP limit.

## Quality evaluation

| Setting | Default | Meaning |
|---|---|---|
| `FIXIMG_EVAL_MODE` | `inline` | `inline` computes metrics before the task is marked completed (V2 behaviour). `async` completes the task and makes the result downloadable first, then writes the metrics and emits a `task.evaluated` event (plan §3.16) |

Use `async` when result latency matters more than having the numbers in the same
response. Metrics are diagnostics: a failing evaluator never fails the task.

PSNR/SSIM/MAE compare the output with **the degraded input** — an input-output
difference, not a quality score. Check them against an independent implementation
before trusting one: this project shipped a transposed `sigma2_sq` term in its
hand-rolled SSIM for a while, which reported `-0.67` for a normal restoration and
*inflated* the score for a random image, so both a wrong-looking and a
plausible-looking number came off the same bug.

### An installation without dlib

`dlib` is the face chain's only hard dependency and it needs a C++ toolchain to
build, so a plain `pip install .` has no face enhancement. That is a supported
degradation, not a failure:

| what you see | meaning |
|---|---|
| `face_detection` stage status `skipped`, message names dlib | the whole face chain opted out before spawning a subprocess that could only fail; stages 3 and 4 pass the restored image through |
| the task still reports `completed` and the result downloads | the restoration itself is unaffected — this used to fail the task after three retries and throw the finished restoration away |
| `face_count` / `enhanced_count` / `degraded_count` all `0` | what the *pipeline* processed |
| `identity_faces_input` / `_output` / `_paired` possibly non-zero | what the §17 identity metric's own detector sees; different measurement, deliberately different keys |
| `Identity Similarity` with `backend: gradient_fallback` | an untrained gradient-histogram descriptor: a 0.98 means "this crop still looks like this crop", not a biometric match. Install dlib (and its two `.dat` files) to get `dlib_resnet` |

`GET /api/v1/models` shows `implementation: native` for face detection wherever
dlib and `shape_predictor_68_face_landmarks.dat` are present; without them it
reports `legacy-cli`, which is also what `FIXIMG_FACE_DETECT_NATIVE=0` asks for —
see below.

## Native inference and legacy-tree ownership

Whole-image quality restoration runs **in-process with resident weights** by
default (plan §3.5.1/§4.2): the three quality networks are built once per worker
instead of being re-read by a `python run.py` child for every request.

`Global/` and `Face_Enhancement/` both ship top-level `options`/`models`/`util`/`data`
packages, so **one process can host one of them natively** — that is not a
limitation to design around but a property of the vendored code: both trees resolve
their own submodules at runtime through `importlib.import_module()`
(`Face_Enhancement/models/__init__.py:11`, `util/util.py:149`), so a temporary
`sys.path`/`sys.modules` swap would let a dynamically imported module bind to
whichever tree happened to own the name at that moment — a silent miswire. The
boundary is therefore the process, which is also the plan's deployment shape
(§3.13: "GPU0 → restore worker, GPU1 → face/color worker"):

| `FIXIMG_NATIVE_TREE` | Resident in this process | Still subprocess |
|---|---|---|
| `global` (default) | quality restoration **and** scratch repair (both branches of stage 1) | face enhancement (stage 3) |
| `face` | face enhancement (stage 3) | quality restoration, scratch repair |
| `none` | nothing (DDColor only) | every vendored stage |

Both selectors fall back rather than fail, and the fallback is visible: stage
metadata records `"backend": "native" | "legacy-cli"`, and `GET /api/v1/models`
reports `implementation` per model. Availability also covers torch being absent
and the branch's weights being missing.

Face **detection** (stage 2) is independent of this choice: `Face_Detection/`
declares no colliding packages, so it can be resident in either worker. It is off by
default for a different reason — see the next subsection.

### Equivalence evidence

The native paths reuse the vendored code rather than reimplementing it — same
`parameter_set`/`data_transforms`, same `FaceTestDataset` and `Pix2PixModel`, same
`torchvision` save calls (including stage 1's `normalize=True`, which is a per-image
contrast stretch rather than a clip). Measured against the adapter:

| chain | result | timing |
|---|---|---|
| Global quality restoration | PSNR `inf`, SSIM `1.0`, max abs diff `0`, identical geometry | ~4.5 s per request → ~0.9 s warm |
| Scratch restoration (image + mask) | mask bit-identical; image within the reference's own 1-LSB / few-pixel branch band, and bit-identical to itself run after run — see below | 6.6 s per request → 2.2 s one-off load + 1.7 s per image |
| Face enhancement (3 crops) | byte-identical on CPU; tolerance of 1 level allowed on CUDA, where cuDNN picks kernels | ~4 s subprocess → resident model after that |

The scratch row's qualifier is measured, not hedged. Five runs of the **vendored**
pipeline over one sample, compared pairwise (10 pairs): the mask was bit-identical
every time, and the restored image was either identical or `1 LSB at 3 pixels`
apart — 6 pairs the former, 4 the latter. So the reference is *bimodal*: each
process lands on one of two outcomes one LSB apart. That also rules out the
explanation this project used to give (the detector's sigmoid flipping its `0.4`
threshold), since the mask never moved; the branch is in the triplet restoration's
conv path, where torch's CPU reductions are not order-stable. The native side, by
contrast, reproduced itself bit-for-bit in every run.

Two consequences, and they pull in opposite directions:

* `diff == 0` against *one* sampled reference run is not a gate — it is a coin flip
  that mostly passes, so the test allows the documented band (1 LSB, a few pixels)
  for the image and demands exactness for the mask;
* "the two reference runs agreed, therefore demand exactness from native" is also
  not a gate — it just fails on the samplings where native happened to hold the
  *other* branch. That escalation used to live in
  `tests/gpu/test_scratch_equivalence.py` and turned the suite red roughly one run in
  three; it is replaced by the property that *is* exact: native run twice returns
  identical bytes.

All three are tests, not claims:
`tests/gpu/test_global_equivalence.py`, `tests/gpu/test_scratch_equivalence.py` and
`tests/gpu/test_face_enhance_equivalence.py` run the two paths and compare the bytes —
the scratch test also compares the mask, because a drifted mask means the detector
preprocessing drifted. The face-enhancement test launches the native side in a **child process**,
because the pytest process may already own the Global tree — the test honours the
constraint instead of pretending it away.

Face detection's equivalence test
(`tests/gpu/test_face_detect_equivalence.py`) runs wherever dlib is installed and
compares the in-process crops with the subprocess ones byte for byte; it passed on
dlib 20.0.1 with `shape_predictor_68_face_landmarks.dat`, which is what made
`FIXIMG_FACE_DETECT_NATIVE` the default. Without dlib or that model it reports the
missing prerequisite and the registry keeps using the subprocess adapter.

Two guards exist because the vendored loader fails quietly. `base_model.load_network`
prints "…not exists yet" and continues with **randomly initialised** networks, which
produces a plausible image and a successful task; so the native backend refuses to
load when its branch's weights are absent, and the adapter refuses to start when
`Global/checkpoints/restoration` is *partially* present (an uninstalled model
directory is reported through availability instead, so a weight-free checkout and CI
are unaffected). This matters most for `--HR` scratch, which selects
`mapping_Patch_Attention/` — a directory not shipped upstream.

### Face detection

`Face_Detection/detect_all_dlib.py` (stage 2) can also run in-process: unlike the
Global tree, it imports no top-level `options`/`models`/`util`/`data` package, so it
coexists with the native restoration backend in one worker. It is **on by default**,
gated on `native_available()` — dlib importable and `shape_predictor_68_face_landmarks.dat`
present — and `FIXIMG_FACE_DETECT_NATIVE=0` puts stage 2 back on the subprocess
adapter. The default was set only after
`tests/gpu/test_face_detect_equivalence.py` ran on such an install (dlib 20.0.1) and
required the in-process crops to match the adapter's crop-for-crop, byte-for-byte; the
contract tests cover the rest: the geometry is delegated to the vendored
`search`/`compute_transformation_matrix` with the vendored arguments, crops are named
`<stem>_<n>.png` (stage 4 pairs them back by name), and the folder contract is honoured.

HR alignment (`detect_all_dlib_HR.py`) maps onto a 512
canvas rather than 256 — a different transform, not a flag — so `hr=True` always
uses the adapter, and the native backend raises instead of approximating.

One vendored bug had to be fixed to load the tree at all: `Global/options/base_options.py`
declared a help string containing a bare `%`, which Python 3.14's argparse rejects
eagerly at `add_argument()` ("badly formed help string"). It is now `%%`, which
older Pythons render the same way. On 3.14 before that fix, the whole legacy chain
could not start.

## Precision policy: declared, applied, reported

`models/manifest.yaml` decides per model how it is executed (plan §3.5.4). Nothing
is switched on globally, because the plan is right that legacy checkpoints break
under `torch.compile` and half precision:

```yaml
  ddcolor:
    precision: fp16          # torch.autocast dtype, CUDA only
    channels_last: true      # memory format conversion at load
    compile: false           # torch.compile, CUDA only
    compile_mode: default    # default | reduce-overhead | max-autotune
```

`fiximg.inference.precision` turns that into a `PrecisionPolicy`, and every native
torch backend routes its loaded network through
`BaseModelBackend.apply_policy()` — `inference_mode` is always on, autocast wraps
the call, `channels_last` / `compile` are applied once at load. The record of what
actually landed is reported, not implied:

```json
{"name": "ddcolor", "implementation": "native",
 "optimisations": {"precision": "fp16", "channels_last": true, "compile": false,
                   "accelerated": true, "applied": {"channels_last": true}}}
```

Three rules keep that object honest:

* **A declaration cannot be decorative.** A model may only declare an optimisation
  whose serving backend applies it: `test_a_declaration_reaches_the_backend_that_serves_it`
  reads the manifest and fails if an entry declares `channels_last` / `compile`
  while `LegacyCliBackend` or dlib serves it. Before this, only DDColor read the
  keys, so declaring them for another chain did nothing and said nothing.
* **A declined optimisation is named.** `applied` has an entry per declared key. If
  the layout conversion raises, or `compile` is asked for on a CPU device, the model
  keeps serving *and* the record says `false` — the failure is never reported as
  success, and unloading clears the record so a reload on another device cannot
  quote a stale answer.
* **The proof runs on hardware.** `tests/gpu/test_ddcolor_gpu.py` asserts a real
  Conv2d weight of the loaded pipeline is channels-last contiguous, and
  `test_every_native_torch_backend_applies_its_declared_policy` is a static check
  over every backend that owns a policy. Neither is satisfiable by a stub: flipping
  `channels_last: false` in the manifest turns the hardware test red.

`FIXIMG_PRECISION=fp32` overrides the dtype for every model, which is the valve an
operator needs when a checkpoint turns out not to tolerate half precision in
production. Autocast is never entered on CPU: it is a no-op at best there and
unsupported for several operators these chains use.

## Tracing

| Setting | Default | Meaning |
|---|---|---|
| `FIXIMG_TRACING` | `off` | `off` times spans and discards them (no third-party overhead). `sdk` / `otel` export through OpenTelemetry — needs `pip install 'fiximg[otel]'`, and the deployment chooses the exporter (`OTEL_*` or auto-instrumentation). `internal` keeps the last 200 spans in memory for tests and the admin panel |

Three spans exist today: `task.execute` (one per claimed task, carrying the queue
wait), `inference.stage` (one per stage, with task id, stage name, order and
device) and `model.load` (one per load or hot swap, with the version and weight
URI). Asking for `sdk` without the package logs a warning and runs untraced — it
does not pretend to be tracing (plan §2.10).

## Reliability

| Concern | Mechanism |
|---|---|
| Worker crash mid-task | `lease_until` expiry → task requeued (`FIXIMG_WORKER_LEASE`). A row with no lease falls back to `started_at`, aged against a bound of the same shape — see below |
| Defect *around* a task | The loop guards each task end to end: anything raising outside the pipeline (event write, queue-wait accounting) fails that task at once with `error_code=worker_error`, and the worker keeps polling. Without the guard the worker thread died and the queue stopped draining — which looks like a hung queue, not a crashed process |
| Transient failure | Attempts retried with backoff until `FIXIMG_TASK_MAX_ATTEMPTS` |
| Duplicate submission | `Idempotency-Key` header, backed by a unique index |
| Queue saturation | `FIXIMG_WORKER_MAX_QUEUE` → HTTP 503 `QUEUE_FULL` |
| Urgent work | `priority` on `POST /api/v1/tasks`, bounded by `FIXIMG_PRIORITY_MAX` (default `10`). The claim orders `priority DESC, created_at`, so a larger value is claimed first; above the ceiling the submission is refused rather than clamped |
| User cancellation | `POST /tasks/{id}/cancel` on a `queued` or `running` task. The row flips immediately; the owning worker stops **at the next stage boundary** (`interruption()`), never mid-stage — see below |
| Lease handed over | The same boundary check ends the losing worker's run when `worker_id` no longer matches its claim, so an expired lease cannot produce two concurrent executions of one task |
| Disk growth | `FIXIMG_RESULT_TTL` reclamation, swept by the worker every `FIXIMG_ARTIFACT_SWEEP_SECONDS` |
| Weight tampering | `config/weights_manifest.json` verified at startup |

A cancellation is a decision, not a fault, and the state machine says so: the row
keeps `error_code = NULL`, no `task.failed` event is written, the progress bar
freezes where the task was stopped (`update_progress` is fenced at `running`), the
attempt it was on is the only one it spends, and nothing requeues it.
`GET /result` keeps answering 409 because no bytes were produced.

Measured on the real GPU worker: a `restore` on a 640×960 photo was cancelled 0.35 s
after the claim was visible; the reply was `200 cancelled`, the live stream received
`task.cancelled` and closed, `global_restore` (already in flight) recorded
`completed`, the three later stages never started, and the artifact rows held only
`input`.

Delivery is **at-least-once**, deduplicated at submission; see *Delivery
semantics* in `docs/architecture.md` for what that guarantees about re-execution,
artifact rows and object-store keys.

Retention is the worker's job, not the request path's. `purge_stale_runs()` walks
`FIXIMG_TASKS_ROOT` and deletes run directories past their TTL, and it runs from
`PipelineWorker._scan_stale()` on a fixed cadence — a worker that is idle still
reclaims disk. (It used to be reachable only from the synchronous
`PipelineOrchestrator.run()`, so on the API + GPU-worker topology below the TTL
was a number nothing read; the sweep still fires opportunistically after a sync
run, which is what a no-worker deployment relies on.)

### Each clock is aged against a bound of its own shape

`reset_stale_running()` compares strings, which is exact for fixed-width UTC
instants (§2.7) and meaningless across conventions. A row written before the codec
stores `2026-03-14 17:59:00` — local wall clock, space-separated, no zone — and a
space sorts before the `T` of a canonical instant, so aged against the canonical
bound that row reads as *older than any cutoff* and the task is requeued away from
the worker running it; a genuinely two-day-old row, conversely, survives whenever
the local date has rolled past the UTC one. Both directions are real, and which one
you get depends on the hour of day.

The sweep therefore ages each shape against a bound written the same way:
`lease_until` (always canonical) and canonical `started_at` against
`timestamps.cutoff()`, pre-codec `started_at` against `timestamps.legacy_cutoff()`
— local naive, the same convention `timestamps.parse()` already assumes for legacy
values. A row with neither a lease nor any instant cannot be dated at all, so it is
left alone and counted by `running_rows_without_a_clock()` (logged as a warning)
rather than guessed about.

The test that covered this had itself been passing on the coincidence — it backdated
with `datetime('now','localtime','-2 hours')`, so it held only while the local and UTC
calendar dates matched, and it started failing on its own once this box crossed
midnight. It now injects one fictional instant spelled both ways, so both directions
are pinned whatever the host's offset is; deleting either bound turns it red.

## Operations

| Endpoint | Use |
|---|---|
| `/api/v1/health/live` | Container liveness probe |
| `/api/v1/health/ready` | Readiness: DB, models, worker topology, queue depth, worker liveness |
| `/api/v1/stats` | Task counts, success rate, per-stage P50/P95, scheduler state |
| `/api/v1/stats/metrics` | Prometheus scrape target |

The `worker` block distinguishes what an API process can know about a worker that
lives in another process. `queue_backend` is the transport family — the value
`FIXIMG_QUEUE_BACKEND` takes — so it stays `sqlite` even when that queue table is
PostgreSQL; `queue_driver` says which server holds it. `worker_liveness` then answers
from what the worker writes while it works:

| `worker_liveness` | meaning |
|---|---|
| `alive` | a worker holds a running task and heartbeated it within `max(3 s, 3 × poll)` |
| `silent` | a running task exists whose heartbeat is overdue — the process is wedged, not idle |
| `backlogged` | nothing is running, work is waiting, and the oldest waiter has outlived the grace bound |
| `starting` | work is waiting but has not yet waited long enough to conclude anything |
| `idle-or-unknown` | nothing running and nothing queued; an alive-and-idle worker looks exactly like this |

Only the inline topology reports a definite `worker_running: false` for "no worker",
because in that topology the probe and the worker share a process.

Logs are single-line JSON with `request_id`, `user_id`, `task_id`, `worker_id`,
`gpu_id` correlation fields, so a task can be traced from HTTP request through
worker execution to artifact write.

### What the web process computes

The Auto tab's analysis row (`docs/api.md`, "Auto analysis row") is computed in the web
process, on upload and on every switch change, by the same `analyze_image()` +
`plan_auto_restore()` the execution path uses. What that costs in a deployment:

| Property | Measured here |
|---|---|
| Wall time | 7–11 ms warm; 30–50 ms cold, of which ~12 ms is the face backend's first load |
| Device work | none — CV2 on a copy downscaled to 512 px, no GPU tensor, no CUDA context |
| Weights touched | no restoration or colorization model; the detector only (YuNet 232 KB ONNX, or dlib's frontal predictor, which the identity metric already requires) |
| What it can cost a split deployment | the API node loads the detector once, which an inline topology was loading anyway (the same process executes the analysis for every Auto submission) |

There is deliberately no `FIXIMG_*` switch for it. The row is two function calls on the
path every Auto submission already takes, and a toggle would be a second configuration
surface whose only effect is to hide a number the submission then recomputes anyway — the
same reason `strength` is no longer accepted.

Schema migrations:

```bash
python -m fiximg.infrastructure.db.migrations.runner status
python -m fiximg.infrastructure.db.migrations.runner upgrade
```

## Dependency lock groups

`requirements.lock` is a flat freeze, so every install pulls the GPU stack and the
test tools. The groups are split by purpose (report §3.12):

```bash
make lock-freeze     # requirements.lock <- the environment you are deploying to
make lock            # regenerate requirements/<group>.txt from that lock
make lock-check      # CI: fail if the group files no longer describe the lock
```

```bash
pip install -r requirements/runtime.txt   # production: no torch, no pytest
pip install -r requirements/gpu.txt       # inference stack
pip install -r requirements/test.txt      # CI
```

`requirements/<group>.txt` is derived, not hand-written — the header of each file
says so. The split is a closure over *installed* metadata rather than a real
resolver, so every group pins versions already known to work together. For a
fully resolved per-group set, use `pip-compile` against `pyproject.toml`.

### One environment owns the lock, and it says which

A closure is a property of an installed environment: a CPU wheel of `torch`
depends on different packages than a `+cu128` one, and the versions of
`numpy`/`timm`/`dlib` in the two are not the same set either. So the lock names
its own environment — `requirements.lock` carries
`# Frozen-environment: python 3.11 / win32` — and:

* `make lock-freeze` and `make lock` must run **in that environment**. Generating
  from a different interpreter would write files that claim to pin the deployment
  stack while describing a different one, so the exporter refuses.
* `make lock-check` enforces the two tiers below and prints the third's skip
  reason instead of passing it off as verified.

| Tier | Where it runs | What is red |
|---|---|---|
| Lock portability | anywhere | a pin another machine cannot install (`torch @ file:///D:/...`) |
| Pin agreement | anywhere | a line in a group file that is not `name==version`, or differs from the lock |
| Closure re-derivation | only in the frozen environment, and only for groups whose roots are installed | the file is not what that environment produces; a declared root that the environment does not provide |

The first two need nothing installed, so CI on any platform still gets a real
gate. The third is skipped elsewhere with both environment names printed, which
is the honest version of what used to be reported as "out of date" on machines
that merely had a different Python.

### Why the freeze is not `pip freeze > requirements.lock`

`pip freeze` writes `dlib @ file:///C:/bld/dlib-split_.../work` and
`torch @ file:///D:/Soft/fixoldimg_wheels/torch-2.7.1+cu128-...whl` for anything
installed from a conda build or a local wheel. Those paths exist only on the
machine that produced them, and the splitter copied them into
`requirements/gpu.txt` — so the documented GPU install could not run anywhere
else. `scripts/export_locks.py --freeze` pins those as plain `name==version` and
records the original source as a comment, which is both installable and
traceable. `+cu128` local versions still need the PyTorch wheel index (see the
install steps at the top of `requirements.txt`).

Every requirement file is ASCII on purpose — `pip-audit` and older `pip` open it
with the platform encoding, which breaks on Windows locales that are not UTF-8.

### Image scanning

The CI image scan runs trivy over CRITICAL/HIGH with `exit-code: "1"`, so a finding
fails the build. `.trivyignore` is committed and empty: an accepted finding is recorded
there with its CVE id and the reason, never by turning the flag back to `"0"`.
`tests/unit/test_ci_gates.py` enforces both halves of that rule — no action step may be
parametrised to succeed regardless of its findings, and any ignore file a workflow names
must exist in the repository (an ignore file nobody committed is an unbounded exemption).

## Dependency audit

```bash
pip-audit --requirement requirements.txt --strict      # what CI runs
```

The audit is a gate: any advisory against the pinned runtime requirements fails
the build. Exemptions are lines in `config/audit_ignore.txt` — one GHSA/CVE id
each, with the reason, the affected pin and a revisit date written next to it.

That file replaces a step that ended in `|| true` beside a placeholder
`--ignore-vuln GHSA-0000-0000-0000`, which could neither fail nor be read. An
exemption is now *visible and auditable* (and `pip-audit` rejects an id that no
longer applies, so a stale line fails the build instead of rotting quietly).
`tests/unit/test_ci_gates.py` keeps both properties: no workflow step may set
`continue-on-error`, and no step's **final** command may swallow a failure — the
two shapes this repository has actually been bitten by.

## External service verification

The Redis / PostgreSQL / MinIO adapters have contract tests against fakes; this
verifies a real deployment:

```bash
docker compose -f docker/compose.yaml --profile platform up -d
make smoke-services
```

```
database (postgresql)  PASS  connection + SELECT 1
redis                  SKIP  FIXIMG_REDIS_URL not set
object storage         SKIP  FIXIMG_STORAGE_BACKEND=local
queue backend          PASS  kind=sqlite, driver=postgresql, depth=1
queue setting          PASS  FIXIMG_QUEUE_BACKEND=sqlite

Every configured service is reachable.
```

That is a real run against PostgreSQL with Redis and MinIO left unconfigured — the
two `SKIP` lines become `PASS` once `FIXIMG_REDIS_URL` and
`FIXIMG_STORAGE_BACKEND=s3` are set, and `kind` then reads `redis`. `kind` names the
queue implementation, `driver` the database it is actually talking to: the SQL queue
is still called `sqlite` on a PostgreSQL deployment, so reading only `kind` looks
like two settings disagree.

An unconfigured service reports `SKIP` (not a failure); a configured but
unreachable one reports `FAIL` and exits non-zero. Every probe bounds its own
connect attempt (`PROBE_CONNECT_TIMEOUT`, 5 s): a diagnostic that hangs is worse
than one that fails, and psycopg without `connect_timeout` was measured never to
return against a refused port.

### PostgreSQL against a real server

`FIXIMG_DATABASE_URL` pointing at PostgreSQL is not just supported but verified:
`tests/integration/test_postgres_server.py` runs the repository, the legacy
`users`/`history` tables, the queue claim under four concurrent workers, the
SQL-side percentiles and the events stream against a live server, and the CI
`postgres` job does the migration round trip (`upgrade` → `status` →
`downgrade base` → `upgrade` → `check`) on a `postgres:16` service container.

```bash
docker compose -f docker/compose.yaml --profile postgres up -d
FIXIMG_TEST_POSTGRES_URL=postgresql://fiximg:fiximg@localhost:5432/fiximg_test \
    make test-postgres
```

Without that variable the tier skips and says so — which matters, because running
it for the first time against PostgreSQL 18 found four defects the fake-driver
tests could not see, since a fake that answers exactly what the code asks cannot
reject what the server rejects:

| Defect | Symptom on a real server |
|---|---|
| Migration 0001 executed the shipped SQLite DDL verbatim | `syntax error at near "AUTOINCREMENT"` — a fresh PostgreSQL database had **no schema at all** |
| `history.user` | `user` is reserved in PostgreSQL: the column and both INSERTs must quote it |
| `SELECT COUNT(*)` read as `row[0]` | `KeyError` under psycopg's `dict_row`; the fake supported positional access, so the tests passed |
| `COALESCE(metric_value, metric_text)` | `DatatypeMismatch` once `metric_value` is `DOUBLE PRECISION`; and SQLite's variadic `json_object(...)` was emitted where PostgreSQL's `json_object_agg` is a two-argument aggregate |

`downgrade base` needed its own fix: on PostgreSQL the rollback's `UPDATE` cannot
write text into a still-numeric column, so 0003 now widens `metric_value` to
`TEXT` first. SQLite tolerates either order, which is why only the server caught
it.

### Database URL

`FIXIMG_DATABASE_URL` is validated at startup. SQLite and PostgreSQL are served:

```
sqlite:///admin_data/fixoldimg.db        default
postgresql://user:pw@host:5432/fiximg    needs pip install 'fiximg[postgres]'
```

An empty value uses `admin_data/fixoldimg.db` under the project root. Only the
environment variable relocates the database — the shipped profiles deliberately
declare no `database_url`, so a `DB_PATH` set by a test or an embedder stays
authoritative. A scheme with no driver (e.g. `mysql://`) raises
`UnsupportedDatabaseError` rather than falling back to SQLite: a silent fallback
would look like a successful switch while writing to the wrong database.

## Plugins

Third-party packages can add models and stages without forking this repository
(report §3.14.1):

```toml
# in the plugin package's pyproject.toml
[project.entry-points."fiximg.models"]
my_model = "my_package.plugin:MyModelPlugin"

[project.entry-points."fiximg.stages"]
my_stage = "my_package.plugin:MyStagePlugin"
```

```python
class MyModelPlugin:
    id = "my_model"
    version = "1.0.0"
    capabilities = {"super_resolution"}      # the planner matches on these

    def create_backend(self, config=None):
        return MyBackend(config)
```

Discovery runs when the default registries are built. A broken plugin is logged
and skipped — it never stops the service. Set `FIXIMG_PLUGINS=0` to disable
discovery entirely, and check `GET /api/v1/health/ready` → `plugins` for what was
found.

## Memory-aware scheduling

When a capability can be served by several devices, the roomiest one wins:

```bash
FIXIMG_GPU_MEMORY_AWARE=true          # default
FIXIMG_GPU_MEMORY_HEADROOM_MB=256     # free VRAM a device must have
```

A stage that fails with CUDA OOM is retried once on another device that serves
the same capability; if there is none, the original error is reported. Each
stage records its own peak VRAM as `gpu_peak_mb` in the run report, measured as a
delta so an earlier stage's high-water mark cannot inflate it.

The device in the report is the device the model **ran** on. Every chain takes
`context.gpu` into the code that loads or moves its weights — including DDColor,
whose vendored builder otherwise picks card 0 whenever torch can see one, so a
`FIXIMG_DEVICE=cpu` worker used to place ~1 GB of colour weights on a GPU it had
been told not to use. When the scheduled card cannot be honoured, the stage says
so instead of echoing the request: `metadata.device` is where it ran and
`metadata.device_requested` is what was asked.

`reset_peak_memory_stats()` — how a per-stage delta is measured — needs a CUDA
context to exist, and in a fresh process the first call raises `Invalid device
argument`. Since the first window is the stage that loads the model, that meant
the heaviest stage of every run reported no memory figure at all, silently. The
sampler now initialises CUDA before resetting, and if the measurement still
cannot be taken it logs one `WARNING` (`GPU memory measurement unavailable…`)
rather than letting "no number" look like "a small number".

Without CUDA every probe returns nothing, selection falls back to the first
candidate and no peak is recorded — a CPU deployment behaves exactly as before.

### GPU utilisation

Memory says how much of the card a stage *held*; utilisation says how hard the
card was *working*. PyTorch has no API for it, so the reading comes from the
driver (`nvidia-smi --query-gpu=…`), collected by one background thread per
process on a 0.5 s interval:

| Where | What |
|---|---|
| `GET /api/v1/health/ready` → `gpu.topology.utilization` | `status`, `interval_seconds`, `age_seconds`, `devices[]`, `busy_devices` |
| each stage's metrics | `gpu_util_pct` (window mean), `gpu_util_max_pct`, `gpu_samples` |
| `GET /metrics`, `/api/v1/stats` | `gpu_utilization_percent{stage,device}` histogram |

Three properties matter to an operator:

* **It forks a tool, so it only does so when someone cares.** The sampler polls
  while a stage window is open, or for two intervals after a reader asked. An idle
  worker costs nothing. The first scrape of a fresh process answers
  `status: "starting"` because the reading has not been taken yet, and a scrape
  after a long quiet period answers with the last reading plus a large
  `age_seconds` — which is why the age is part of the payload.
* **`age_seconds` travels with the readings.** Since polling stops, a `0 %` could
  be minutes old; the field says how old. `status` is one of `idle`, `starting`,
  `measuring`, `unavailable` — and `unavailable` (no driver, or it answers nothing)
  means the metric is **absent**, never a fabricated 0 %. A CPU-only host and an
  idle GPU look different on purpose.
* **It is a device-wide reading.** Two workers pinned to one card each attribute
  the other's work to their own stage window; that is what the driver reports. Size
  the concurrency policy from `gpu_util_max_pct`, and read the per-stage mean across
  the stages of one task rather than trusting a single window.

Measured live on one RTX 5060 (`auto_restore` of `examples/old/f.png`, 372×524,
deployment interpreter, task total 14.5 s):

```text
task-level   gpu_util_pct 13.4 over 26 readings, max 87
scratch_repair    5358 ms   mean 32.4 %   max 87 %   10 readings
face_detection     807 ms    mean  0.0 %   max  0 %    1 reading
face_enhancement  4216 ms   mean  3.0 %   max 24 %    8 readings
warp_back         4098 ms   mean  0.0 %   max  0 %    7 readings
```

That is the number §2.4's policy exists to raise, and it localises the cost: the
card is idle for most of a face pipeline, so the wins are in the stages' CPU-side
and subprocess work, not in giving the GPU more concurrency.

## Schema migrations

Alembic owns the schema (report §4.3 Step 2). The database URL is never read from
`alembic.ini` — it comes from `FIXIMG_DATABASE_URL`, so a migration and the
service always mean the same database:

```bash
make db-status      # applied revision + anything pending
make db-upgrade     # alembic upgrade head
make db-downgrade   # roll back one revision
make db-history     # the revision chain
make db-adopt       # stamp an existing database as current (runs nothing)
make db-check       # fail if rows still use a format the code has left behind
```

```bash
alembic upgrade head --sql        # review the DDL before applying it
FIXIMG_DATABASE_URL=... alembic history --verbose
```

```
0003  <- 0002 (head)   typed instants and numeric metrics
0002  <- 0001          columns added after the V2 schema
0001  <- base          V3 schema baseline
```

The startup path still creates the schema idempotently, so a cold start works
without running Alembic. Adopt Alembic on an existing database with
`make db-adopt` — it records the current revision without touching anything.

`0003` is the one revision where stamping is the wrong answer, because it changes
*values* and not only shape (report §2.7). Instants move from
`2026-06-01 08:00:00` — naive, local, one-second granularity — to
`2026-06-01T00:00:00.000000Z`: same instant, but now sortable and comparable
across a DST boundary. A database that stamps instead of migrating keeps working
quietly, because the readers accept both spellings, and then mis-reads a lease: on
a UTC+8 host a legacy `lease_until` of `2026-09-27 23:00:00` *is* 15:00Z but
sorts before a canonical `2026-09-27T02:00:00Z`, so a task that is still being
worked on looks expired and gets requeued. `make db-check` is how an operator sees
that instead of discovering it:

```
applied: 0002
head:    0003
legacy-format rows (run `fiximg-db upgrade`):
  tasks.created_at: 412 row(s)
  tasks.lease_until: 1 row(s)
```

The same revision makes `metrics.metric_value` a REAL so a PSNR can be averaged
in SQL rather than pulled into Python as a string. What a REAL cannot hold —
`inf` after a bit-identical comparison, or a label — is moved to `metric_text`
*before* the cast, and readers `COALESCE` the two, so nothing is lost.

## Benchmarks

Three commands, all runnable without weights or a GPU (synthetic stage), and CI
runs them:

```bash
python -m benchmark.latency --runs 4 --sizes small     # exits 1 on regression
python -m benchmark.throughput --tasks 20              # exits 1 if a task is left undrained
python -m benchmark.memory                             # peak RSS (+ VRAM where present)
```

**The throughput run gates drain completeness.** Its table reports `tasks` (what was
submitted) and `executed` (what came back) as separate columns, and exits 1 when any
phase drained fewer tasks than it was given. Those used to be the same number — the
drain row printed `tasks = done` — so a run that finished 3 of 25 tasks reported a
plausible rate for 3 and passed. `--timeout` is the knob that produces a short drain.

**The measured numbers are committed data.** `benchmark/baseline.json` records p50/p95
and the platform overhead per size as measured on a named reference platform, and every
latency run prints its own p50 against them (`small: p50 0.0513s vs recorded 0.0514s ->
x1.00`). Plan §3.10.3 asks for a p50/p95 *baseline*; before the file existed the numbers
lived in prose, so neither a reviewer nor a test could compare against them. The file and
the code constants are checked against each other by
`tests/unit/test_benchmark_samples.py`, so the record cannot quietly disagree with the
gate it describes.

**The latency run is a gate, not a printout.** The synthetic stage sleeps a known
time, so anything above the sleep is platform cost — queue claim, stage dispatch,
artifact rows, logging — and if that exceeds `--max-overhead` (default 1.0 s, on
the smallest measured size) the command exits 1. Only the smallest size is gated:
image preparation scales with pixels, and a 4096 px run would exceed a ceiling
tuned for 512 px for reasons unrelated to the platform. Until this existed the
step could not fail, so "benchmark regression" in CI was a table nobody read.

**Percentiles mean one thing everywhere.** `ceil(p·n)`, clamped to the sample —
in the benchmark (`benchmark.common`), in `/api/v1/stats/metrics`
(`observability.metrics.nearest_rank`), and in SQL (`task_repository._ceil_rank`).
Interpolation was rejected because SQLite has no counterpart, and two engines
reporting different numbers for identical rows is worse than a coarser statistic
reported consistently. The rule is pinned by a test that compares the SQL result
with the benchmark's for the same durations.

Two bugs this consistency exposed, both invisible at large n and both live in CI's
regime (`--runs 2`): `_percentile` computed `ordered[int(n·p) − 1]`, which returns
the **minimum** as the p95 of two samples — so the latency table printed a p95
below its own p50.

## Testing against real protocols

The queue backend is tested with **fakeredis**, which implements the real Redis
Streams commands (consumer groups, pending entries, `XAUTOCLAIM`'s three-element
reply) in-process. To run the same tests against a live server:

```bash
FIXIMG_TEST_REDIS_URL=redis://localhost:6379/15 pytest tests/unit/test_redis_queue.py
```

## Database dialect

The SQL that differs between engines lives in one module
(`fiximg.infrastructure.db.dialect`):

| | SQLite | PostgreSQL |
|---|---|---|
| placeholder | `?` | `%s` |
| JSON array | `json_group_array` | `json_agg` |
| JSON object | `json_group_object` | `json_object_agg` |
| column introspection | `PRAGMA table_info(x)` | `information_schema.columns` |
| write lock | `BEGIN IMMEDIATE` | none needed |
| claim lock | none | `FOR UPDATE SKIP LOCKED` |
| conflicting insert | `INSERT OR IGNORE INTO` | `ON CONFLICT (…) DO NOTHING` |
| connections | one shared handle (`check_same_thread=False`) | one per thread |
| instant | `TEXT` — fixed-width UTC ISO-8601 | same `TEXT`, same representation |

That last row is a decision, not an omission. SQLite has no datetime type, so
`TEXT` in that layout *is* its canonical instant representation, and a
lexicographic comparison on it is a chronological one that both engines can serve
from an index. PostgreSQL could use `timestamptz`, but every lease and backoff
comparison in the repositories passes a Python string, and psycopg binds a `str`
as `text` — so the ~10 comparisons would each need a dialect-specific `::timestamptz`
cast that no check running here could execute. The precision and the missing zone
were the bug; the declared type was not. The UI renders these instants on the
viewer's wall clock, so the convention never reaches a person as an eight-hour
error.

The SQLite implementation returns exactly the strings the repository used before
the module existed, so introducing it could not change behaviour. The PostgreSQL
implementation is verified three ways: the generated statements are compiled
against SQLAlchemy's real `postgresql` dialect;
`tests/unit/test_postgres_data_path.py` boots `init_db()`, `ensure_schema()` and
the queue claim over a fake DB-API driver, asserting that no `PRAGMA`,
`INSERT OR IGNORE` or `BEGIN IMMEDIATE` reaches the wire; and
`tests/integration/test_postgres_server.py` plus the CI `postgres` job run the
same paths against a live server, which is where the four defects listed under
"PostgreSQL against a real server" were found. The fake is deliberately no more
forgiving than the driver it stands in for — it refuses positional row access,
because answering `row[0]` is what hid one of those defects.

**Still not claimed:** a run against a live Redis or MinIO. Both have protocol-level
tests (`fakeredis`, `moto`) that open no socket; `make smoke-services` probes a
configured instance for real, and the queue/transport tiers skip with that reason
named until one is pointed at them.
