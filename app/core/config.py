"""Centralized application settings (V2).

All configuration is read from FIXIMG_* environment variables at import time and
exposed through the `settings` singleton. Modules must import settings from here
instead of calling os.environ.get() directly, so configuration stays in one place.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


class Settings:
    """Aggregate application configuration (paths, limits, model options)."""

    def __init__(self) -> None:
        # --- Runtime / environment ---
        self.env: str = _env("FIXIMG_ENV", "development")
        self.base_dir: str = BASE_DIR

        # --- Web server ---
        self.host: str = _env("FIXIMG_HOST", "127.0.0.1")
        self.port: int = _env_int("FIXIMG_PORT", 9502)

        # --- Storage layout (Artifact Storage, plan section 15) ---
        # storage/tasks/<year>/<month>/<req_id>/ under the project root.
        self.storage_root: str = os.path.join(self.base_dir, "storage")
        self.tasks_root: str = os.path.join(self.storage_root, "tasks")
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
        self.inline_worker: bool = _env("FIXIMG_INLINE_WORKER", "true").lower() in (
            "1", "true", "yes", "on"
        )
        self.worker_poll_seconds: float = _env_float("FIXIMG_WORKER_POLL", 0.5)
        self.worker_max_queue: int = _env_int("FIXIMG_WORKER_MAX_QUEUE", 100)
        # Set to true when a standalone `python worker.py` process consumes the
        # queue, so the Gradio UI can use the async path (plan §23).
        self.external_worker: bool = _env("FIXIMG_HAS_EXTERNAL_WORKER", "false").lower() in (
            "1", "true", "yes", "on"
        )

        # --- DDColor colorization model ---
        self.ddcolor_weights: str = _env(
            "FIXIMG_DDCOLOR_MODEL",
            os.path.join(self.base_dir, "weights", "ddcolor", "pytorch_model.pt"),
        )
        self.ddcolor_input_size: int = _env_int("FIXIMG_DDCOLOR_INPUT_SIZE", 512)
        self.ddcolor_model_size: str = _env("FIXIMG_DDCOLOR_MODEL_SIZE", "large")
        self.colorize_result_ttl: int = _env_int("FIXIMG_COLORIZE_TTL", 2 * 3600)

        # --- Security bootstrap (plan §21) ---
        # Set FIXIMG_ADMIN_PASSWORD to inject the initial admin password via the
        # deployment (Docker secret / compose env). Leave unset to get a random
        # one-time password printed once on first boot. "false" disables the
        # bootstrap entirely (e.g. accounts provisioned by an external IdP).
        self.admin_username: str = _env("FIXIMG_ADMIN_USERNAME", "admin")
        self.auto_bootstrap_admin: bool = _env("FIXIMG_AUTO_BOOTSTRAP_ADMIN", "true").lower() in (
            "1", "true", "yes", "on"
        )

        # --- Redis (plan §20/§22, P3) ---
        # Reserved for the optional Redis-backed queue and shared rate limiter.
        # Empty = everything stays on the SQLite/in-memory defaults; when set,
        # rate limiting shares counters across processes (needs `redis` pkg).
        self.redis_url: str = _env("FIXIMG_REDIS_URL", "")

        # --- Database / history ---
        self.db_path: str = os.path.join(self.base_dir, "admin_data", "fixoldimg.db")
        self.history_max: int = _env_int("FIXIMG_HISTORY_MAX", 2000)
        self.archive_ttl: int = _env_int("FIXIMG_ARCHIVE_TTL", 7 * 24 * 3600)

    @property
    def pipeline_modes(self) -> dict:
        """Compatibility alias used by the legacy Gradio UI (plan section 4)."""
        from app.services.pipeline_modes import PIPELINE_MODES

        return PIPELINE_MODES


settings = Settings()
