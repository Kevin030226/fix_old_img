"""Centralized application settings (V3).

Configuration resolution order (plan §3.3), lowest precedence first:

    code defaults  →  configs/base.yaml  →  configs/<profile>.yaml  →  FIXIMG_* env vars

Every setting is exposed through the ``settings`` singleton. Modules must
import settings from here instead of reading ``os.environ`` directly, so
configuration stays in one place and can be validated once at startup.

The profile is chosen by ``FIXIMG_PROFILE`` (falling back to ``FIXIMG_ENV``,
default ``local``) and selects ``configs/<profile>.yaml``. YAML keys are the
lower-case attribute names: ``max_image_side`` is fed by either the YAML key or
``FIXIMG_MAX_IMAGE_SIDE``, with the environment winning.
"""
from __future__ import annotations

import os

from fiximg.paths import CONFIGS_DIR, PROJECT_ROOT

#: Backwards-compatible alias for the repository root.
BASE_DIR = PROJECT_ROOT


def _load_profile() -> dict:
    """Merge ``configs/base.yaml`` with the selected profile's overrides."""
    profile = (
        os.environ.get("FIXIMG_PROFILE")
        or os.environ.get("FIXIMG_ENV")
        or "local"
    ).strip() or "local"

    merged: dict = {}
    try:
        import yaml
    except ImportError:  # pragma: no cover - PyYAML is a declared dependency
        return merged

    for name in ("base", profile):
        path = os.path.join(CONFIGS_DIR, f"{name}.yaml")
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except Exception:  # noqa: BLE001 — a broken profile must not block boot
            continue
        if isinstance(data, dict):
            merged.update(data)
    return merged


#: Profile values resolved once at import time (env vars still take precedence).
_PROFILE_VALUES: dict = _load_profile()


def _profile_value(env_name: str):
    """Look up the profile value for a ``FIXIMG_*`` env var name."""
    if not env_name.startswith("FIXIMG_"):
        return None
    return _PROFILE_VALUES.get(env_name[len("FIXIMG_"):].lower())


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is not None:
        return value
    from_profile = _profile_value(name)
    return default if from_profile is None else str(from_profile)


def _path_env(env_name: str, default: str) -> str:
    """A filesystem location from the environment, resolved against the project root.

    Relative on purpose-debatable: an operator who writes `storage` in a compose file
    means "next to the app", not "wherever the process was started", and two processes
    with different working directories must find the same task directory.
    """
    value = _env(env_name, default)
    return value if os.path.isabs(value) else os.path.join(BASE_DIR, value)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").lower() in ("1", "true", "yes", "on")


#: Semantic environments the application recognises (distinct from profile names).
KNOWN_ENVS: frozenset[str] = frozenset({"local", "development", "staging", "production"})


class Settings:
    """Aggregate application configuration (paths, limits, model options)."""

    def __init__(self) -> None:
        # --- Runtime / environment ---
        #: Which configs/<name>.yaml was loaded (local | docker | production | ...).
        #: A pure selector: it is never derived from the file's own contents.
        self.profile: str = (
            os.environ.get("FIXIMG_PROFILE")
            or os.environ.get("FIXIMG_ENV")
            or "local"
        )
        #: Semantic environment (local | staging | production). Declared by the
        #: profile file, so a profile name like "docker" does not leak into the
        #: environment name; FIXIMG_APP_ENV overrides explicitly.
        self.app_env: str = (
            os.environ.get("FIXIMG_APP_ENV")
            or str(_PROFILE_VALUES.get("app_env") or "")
            or (self.profile if self.profile in KNOWN_ENVS else "local")
        )
        #: Backwards-compatible alias for the environment name.
        self.env: str = self.app_env
        self.base_dir: str = BASE_DIR

        # --- Web server ---
        self.host: str = _env("FIXIMG_HOST", "127.0.0.1")
        self.port: int = _env_int("FIXIMG_PORT", 9502)

        # --- Storage layout (Artifact Storage, plan section 15) ---
        # storage/tasks/<year>/<month>/<req_id>/ under the project root. Both roots are
        # relocatable, because a deployment on a small system volume otherwise fills it
        # with task images: `FIXIMG_STORAGE_ROOT` moves the whole tree,
        # `FIXIMG_TASKS_ROOT` moves just the task runs (the documented retention knob,
        # which `scripts/smoke_services.py` and the TTL sweeper both resolve against).
        # A relative value is resolved against the project root rather than the working
        # directory, so two processes started from different places cannot disagree
        # about where the bytes live.
        self.storage_root: str = _path_env("FIXIMG_STORAGE_ROOT",
                                           os.path.join(self.base_dir, "storage"))
        self.tasks_root: str = _path_env("FIXIMG_TASKS_ROOT",
                                         os.path.join(self.storage_root, "tasks"))
        # Legacy archive roots kept for backward compatibility with V1 admin UI.
        self.archive_input_dir: str = os.path.join(self.base_dir, "admin_data", "archive_inputs")
        self.archive_output_dir: str = os.path.join(self.base_dir, "admin_data", "archive_outputs")

        # --- Inference ---
        self.device: str = _env("FIXIMG_DEVICE", "auto")
        self.max_image_side: int = _env_int("FIXIMG_MAX_IMAGE_SIDE", 4096)
        # API upload guard (plan section 22); Gradio uses the same via max_file_size.
        self.max_upload_mb: int = _env_int("FIXIMG_MAX_UPLOAD_MB", 10)
        # §18 ImageTiler: tile_size <= 0 disables tiling entirely.
        self.tile_size: int = _env_int("FIXIMG_TILE_SIZE", 1536)
        self.tile_overlap: int = _env_int("FIXIMG_TILE_OVERLAP", 128)

        # --- Auto Restore thresholds (plan §10/§11) — env-tunable for calibration ---
        self.auto_grayscale_sat: float = _env_float("FIXIMG_AUTO_GRAYSCALE_SAT", 16.0)
        self.auto_sharp_laplacian: float = _env_float("FIXIMG_AUTO_SHARP_LAPLACIAN", 120.0)
        self.auto_scratch_structure_threshold: int = _env_int("FIXIMG_AUTO_SCRATCH_STRUCTURE", 30)
        self.auto_scratch_fraction_full: float = _env_float("FIXIMG_AUTO_SCRATCH_FRACTION_FULL", 0.06)
        self.auto_scratch_threshold: float = _env_float("FIXIMG_AUTO_SCRATCH_THRESHOLD", 0.5)
        self.auto_blur_threshold: float = _env_float("FIXIMG_AUTO_BLUR_THRESHOLD", 0.5)

        # --- Identity Preservation (plan §17) ---
        # "auto": dlib ResNet embedding when weights are present, else a
        # lightweight gradient-histogram fallback; "off" disables the metric.
        self.identity_backend: str = _env("FIXIMG_IDENTITY_BACKEND", "auto")
        self.result_ttl: int = _env_int("FIXIMG_RESULT_TTL", 2 * 3600)

        # --- GPU worker (plan sections 12 and 28) ---
        # True: run the DB-queue worker as a background thread inside the API
        # process (single-container deployments). False: a standalone worker
        # process (`python worker.py`) consumes the queue (compose topology).
        self.inline_worker: bool = _env_bool("FIXIMG_INLINE_WORKER", True)
        self.worker_poll_seconds: float = _env_float("FIXIMG_WORKER_POLL", 0.5)
        self.worker_max_queue: int = _env_int("FIXIMG_WORKER_MAX_QUEUE", 100)
        #: Ceiling for the `priority` a submission may ask for (plan §2.6). The queue
        #: orders `priority DESC, created_at`, so this is how far a caller may jump the
        #: line; 0 is FIFO and is what the UI always sends. Out-of-range requests are
        #: refused rather than clamped, because a silently clamped value is a contract
        #: that lies about what was accepted.
        self.priority_max: int = _env_int("FIXIMG_PRIORITY_MAX", 10)
        # Set to true when a standalone `python worker.py` process consumes the
        # queue, so the Gradio UI can use the async path (plan §23).
        self.external_worker: bool = _env_bool("FIXIMG_HAS_EXTERNAL_WORKER", False)

        # --- DDColor colorization model ---
        self.ddcolor_weights: str = _env(
            "FIXIMG_DDCOLOR_MODEL",
            os.path.join(self.base_dir, "weights", "ddcolor", "pytorch_model.pt"),
        )
        self.ddcolor_input_size: int = _env_int("FIXIMG_DDCOLOR_INPUT_SIZE", 512)
        self.ddcolor_model_size: str = _env("FIXIMG_DDCOLOR_MODEL_SIZE", "large")
        # There is no separate colourisation TTL: `result_ttl` already governs
        # every run directory, and a colourisation-specific value would have to
        # be consulted at a second place in the sweeper to be honoured. The
        # knob that existed (`FIXIMG_COLORIZE_TTL`) was read by nobody.

        # --- Security bootstrap (plan §21) ---
        # Set FIXIMG_ADMIN_PASSWORD to inject the initial admin password via the
        # deployment (Docker secret / compose env). Leave unset to get a random
        # one-time password printed once on first boot. "false" disables the
        # bootstrap entirely (e.g. accounts provisioned by an external IdP).
        self.admin_username: str = _env("FIXIMG_ADMIN_USERNAME", "admin")
        self.auto_bootstrap_admin: bool = _env_bool("FIXIMG_AUTO_BOOTSTRAP_ADMIN", True)
        #: V3 (plan §2.5 point 3): the account the deployment API token acts as,
        #: so tasks submitted over the JSON API are attributed to a real user
        #: instead of the hard-coded "api" placeholder.
        self.api_token_user: str = _env("FIXIMG_API_TOKEN_USER", "api")

        # --- Redis (plan §20/§22, P3) ---
        # Reserved for the optional Redis-backed queue and shared rate limiter.
        # Empty = everything stays on the SQLite/in-memory defaults; when set,
        # rate limiting shares counters across processes (needs `redis` pkg).
        self.redis_url: str = _env("FIXIMG_REDIS_URL", "")

        # --- Database / history ---
        self.db_path: str = os.path.join(self.base_dir, "admin_data", "fixoldimg.db")
        self.history_max: int = _env_int("FIXIMG_HISTORY_MAX", 2000)
        self.archive_ttl: int = _env_int("FIXIMG_ARCHIVE_TTL", 7 * 24 * 3600)

        # --- V3: pluggable backends (plan §2.7/§2.8/§3.3) ---
        #: Database URL. Empty means "use ``db_path``", and that default is
        #: deliberate: the engine prefers an explicit URL over ``DB_PATH``, so a
        #: URL derived from ``db_path`` here would outrank the ``DB_PATH`` an
        #: isolation fixture or an embedder sets — every test would then share
        #: one database, and the shipped profiles all declare a URL anyway.
        self.database_url: str = _env("FIXIMG_DATABASE_URL", "")
        #: queue transport: "sqlite" (default, DB-backed) or "redis" (P2).
        self.queue_backend: str = _env("FIXIMG_QUEUE_BACKEND", "sqlite")
        #: artifact storage: "local" (default) or "s3" (S3/MinIO).
        self.storage_backend: str = _env("FIXIMG_STORAGE_BACKEND", "local")
        self.storage_bucket: str = _env("FIXIMG_STORAGE_BUCKET", "")
        self.storage_prefix: str = _env("FIXIMG_STORAGE_PREFIX", "")
        self.s3_endpoint: str = _env("FIXIMG_S3_ENDPOINT", "")
        #: Credentials for the object store. Left empty, boto3 uses its ambient
        #: chain (instance profile, web identity, `~/.aws`), which is what an AWS
        #: deployment wants; set them for a self-hosted MinIO, whose root user is
        #: declared in `docker/compose.yaml` and had no way to reach the app.
        #: A region matters even there: SigV4 signs with one, and an endpoint with
        #: no resolvable region fails before the first byte moves.
        self.storage_access_key: str = _env("FIXIMG_STORAGE_ACCESS_KEY", "")
        self.storage_secret_key: str = _env("FIXIMG_STORAGE_SECRET_KEY", "")
        self.storage_region: str = _env("FIXIMG_STORAGE_REGION", "")
        #: model manifest declaring versions/capabilities (plan §3.5.2).
        self.model_manifest: str = _env(
            "FIXIMG_MODEL_MANIFEST", os.path.join(self.base_dir, "models", "manifest.yaml")
        )
        #: warmup strategy: lazy | first-use | startup (plan §3.5.3).
        self.warmup_strategy: str = _env("FIXIMG_WARMUP", "lazy")
        #: V3 (plan §3.15): keep the version replaced by the last hot swap
        #: resident as the rollback target. False trades rollback speed for GPU
        #: memory (the old version is evicted once its in-flight calls drain).
        self.model_keep_previous: bool = _env_bool("FIXIMG_MODEL_KEEP_PREVIOUS", True)
        #: V3 (plan §3.16): when to compute quality metrics.
        #: ``inline``  — before the task is marked completed (V2 behaviour);
        #: ``async``   — after, so the result is downloadable immediately and the
        #:               metrics arrive as a ``task.evaluated`` event.
        self.eval_mode: str = _env("FIXIMG_EVAL_MODE", "inline")
        #: V3 (plan §2.10): which tracer the observability seam uses.
        #: ``off`` — spans are timed and discarded (no third-party overhead);
        #: ``sdk``/``otel`` — OpenTelemetry (needs the `otel` extra; the
        #:            deployment decides exporters, typically via OTEL_*);
        #: ``internal`` — keep the last N spans in memory (tests, admin panel).
        self.tracing: str = _env("FIXIMG_TRACING", "off")
        #: V3 (plan §4.2 Step 2, §3.13): which vendored legacy tree this process may
        #: load natively. ``Global/`` and ``Face_Enhancement/`` both expose top-level
        #: ``options``/``models``/``util``/``data`` packages, so one process hosts one
        #: tree; the other chain keeps running as a subprocess. Set per worker
        #: (restore worker vs face worker) rather than trying to share a process.
        #: ``global`` | ``face`` | ``none``.
        self.native_tree: str = _env("FIXIMG_NATIVE_TREE", "global")
        #: V3 (plan §4.2 Step 2): dlib face detection in-process too. On by default:
        #: the crop-level equivalence test (`tests/gpu/test_face_detect_equivalence.py`)
        #: has now run where dlib is installed — dlib 20.0.1 plus
        #: `shape_predictor_68_face_landmarks.dat` — and required the in-process crops
        #: to be byte-identical to the subprocess ones. It stays a switch because an
        #: install without dlib must be able to say so; where dlib or the landmark
        #: model is missing the registry falls back to the subprocess adapter by
        #: itself. It is orthogonal to which tree ``native_tree`` names —
        #: Face_Detection/ declares no colliding packages, so detection can be
        #: resident in a worker owning either one, or owning none — and
        #: ``native_tree=none`` deliberately does not disable it, so that this flag
        #: stays the single switch for stage 2.
        self.face_detect_native: bool = _env_bool("FIXIMG_FACE_DETECT_NATIVE", True)

        # --- V3: multi-GPU routing (plan §4.3 Step 4, P2) ---
        #: capability → device table, e.g. "0:restore,scratch_repair;1:colorize".
        #: Capabilities not listed fall back to `device`. Routing is by capability
        #: (never by model name), so it survives a model swap.
        self.gpu_routing: str = _env("FIXIMG_GPU_ROUTING", "")
        #: Pin this worker process to one device (0, 1, ...). Empty = not pinned.
        self.worker_gpu: str = _env("FIXIMG_WORKER_GPU", "")
        #: Capabilities this worker serves; empty = every capability. Combined
        #: with `worker_gpu` it stops a specialised worker from claiming tasks it
        #: cannot complete.
        self.worker_capabilities: str = _env("FIXIMG_WORKER_CAPABILITIES", "")

        # --- V3: memory-aware scheduling (plan §7.1 / §4.3 Step 4) ---
        #: When True, a stage with several capable devices goes to the roomiest
        #: one instead of the first declared match. No effect without CUDA.
        self.gpu_memory_aware: bool = _env_bool("FIXIMG_GPU_MEMORY_AWARE", True)
        #: Free VRAM a device must have before it is preferred, in MiB.
        self.gpu_memory_headroom_mb: int = _env_int("FIXIMG_GPU_MEMORY_HEADROOM_MB", 256)

        # --- V3: Image Size Policy (plan §3.5.5) ---
        #: Long-side thresholds, in pixels, for the four size tiers:
        #:   <= size_direct_max            -> direct
        #:   <= size_resize_max            -> adaptive resize (then restored)
        #:   <= size_tile_max              -> tiled inference (plan §18)
        #:   >  size_tile_max              -> rejected with a clear error
        self.size_direct_max: int = _env_int("FIXIMG_SIZE_DIRECT_MAX", 1024)
        self.size_resize_max: int = _env_int("FIXIMG_SIZE_RESIZE_MAX", 2048)
        self.size_tile_max: int = _env_int("FIXIMG_SIZE_TILE_MAX", 4096)
        #: When False the adaptive band is passed through untouched (the models
        #: then apply their own internal resize, which is the V2 behaviour).
        self.size_adaptive_resize: bool = _env_bool("FIXIMG_SIZE_ADAPTIVE_RESIZE", True)

        #: Overrides the manifest's per-model `precision` (plan §3.5.4). Empty /
        #: "auto" honours the declaration; set to fp32 to disable half precision
        #: everywhere without editing models/manifest.yaml.
        self.precision_override: str = _env("FIXIMG_PRECISION", "")

        # --- V3: queue reliability (plan §2.6) ---
        #: how long a worker may hold a task before it is considered abandoned.
        self.worker_lease_seconds: float = _env_float("FIXIMG_WORKER_LEASE", 3600.0)
        #: heartbeat interval written to tasks.last_heartbeat.
        self.worker_heartbeat_seconds: float = _env_float("FIXIMG_WORKER_HEARTBEAT", 15.0)
        #: retry policy for failed attempts.
        self.task_max_attempts: int = _env_int("FIXIMG_TASK_MAX_ATTEMPTS", 3)
        self.task_retry_backoff_seconds: float = _env_float("FIXIMG_TASK_RETRY_BACKOFF", 30.0)
        #: how often a worker reclaims expired run artifacts (plan §2.8). It runs
        #: on the background process rather than per request because the tree walk
        #: costs more as traffic grows — exactly when skipping it would be tempting.
        self.artifact_sweep_seconds: float = _env_float("FIXIMG_ARTIFACT_SWEEP_SECONDS", 900.0)

        # --- V3: GPU concurrency policy (plan §2.4) ---
        #: default concurrent executions per capability; per-capability overrides
        #: use FIXIMG_CONCURRENCY_<CAPABILITY>.
        self.concurrency_default: int = _env_int("FIXIMG_CONCURRENCY", 1)

        self._validate()

    # ------------------------------------------------------------- validation
    def _validate(self) -> None:
        """Fail fast on contradictory configuration (plan §3.3)."""
        if self.app_env not in KNOWN_ENVS:
            raise ValueError(
                f"Unknown FIXIMG_APP_ENV: {self.app_env!r} (expected one of "
                f"{', '.join(sorted(KNOWN_ENVS))})"
            )
        # A profile name that matches neither a known environment nor a file is
        # a typo that would silently load base.yaml only — reject it instead.
        if self.profile not in KNOWN_ENVS and not os.path.exists(
            os.path.join(CONFIGS_DIR, f"{self.profile}.yaml")
        ):
            raise ValueError(
                f"Unknown FIXIMG_PROFILE: {self.profile!r} "
                f"(no configs/{self.profile}.yaml)"
            )
        if self.queue_backend not in ("sqlite", "redis"):
            raise ValueError(f"Unknown FIXIMG_QUEUE_BACKEND: {self.queue_backend!r}")
        if self.storage_backend not in ("local", "s3"):
            raise ValueError(f"Unknown FIXIMG_STORAGE_BACKEND: {self.storage_backend!r}")
        if self.warmup_strategy not in ("lazy", "first-use", "startup"):
            raise ValueError(f"Unknown FIXIMG_WARMUP: {self.warmup_strategy!r}")
        if self.eval_mode not in ("inline", "async"):
            raise ValueError(f"Unknown FIXIMG_EVAL_MODE: {self.eval_mode!r}")
        if self.tracing not in ("off", "sdk", "otel", "internal"):
            raise ValueError(
                f"Unknown FIXIMG_TRACING: {self.tracing!r} "
                "(accepted: off | sdk | otel | internal)"
            )
        if self.native_tree not in ("global", "face", "none"):
            raise ValueError(
                f"Unknown FIXIMG_NATIVE_TREE: {self.native_tree!r} "
                "(accepted: global | face | none)"
            )

        if self.is_production:
            problems = []
            if self.storage_backend == "s3" and not self.storage_bucket:
                problems.append("FIXIMG_STORAGE_BUCKET is required with storage_backend=s3")
            if self.storage_backend == "s3" and bool(self.storage_access_key) != bool(self.storage_secret_key):
                # One half of a key pair is never a mistake the ambient chain can
                # repair: boto3 ignores an incomplete static pair and signs with
                # whatever identity it found, so the artifacts land in someone
                # else's bucket instead of failing here.
                problems.append(
                    "FIXIMG_STORAGE_ACCESS_KEY and FIXIMG_STORAGE_SECRET_KEY must "
                    "be set together (or neither, to use the ambient chain)"
                )
            if self.queue_backend == "redis" and not self.redis_url:
                problems.append("FIXIMG_REDIS_URL is required with queue_backend=redis")
            if not os.path.exists(self.model_manifest):
                problems.append(f"model manifest not found: {self.model_manifest}")
            if problems:
                raise ValueError("Invalid production configuration:\n  - " + "\n  - ".join(problems))

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def pipeline_modes(self) -> dict:
        """Compatibility alias used by the legacy Gradio UI (plan section 4)."""
        from fiximg.application.pipeline_modes import PIPELINE_MODES

        return PIPELINE_MODES


settings = Settings()
