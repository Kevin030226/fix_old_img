"""Web service entry point (plan §3.2 `cli/api.py`).

    python main.py            # http://127.0.0.1:9502 (override: FIXIMG_HOST / FIXIMG_PORT)

All wiring lives in :mod:`fiximg.app_factory`; this module only performs the
weight self-check and starts Uvicorn, so the process bootstrap stays readable.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from fiximg.app_factory import create_app  # noqa: E402
from fiximg.config import settings  # noqa: E402

#: ASGI application object (importable as ``fiximg.cli.api:app``).
app = create_app()


def run() -> int:
    """Verify weights, then serve HTTP until interrupted."""
    from fiximg.infrastructure.models.weights_check import (
        WeightsIntegrityError,
        verify_weights,
    )

    try:
        n = verify_weights()
        print(f"[Self-check] Weight integrity OK ({n} files)")
    except WeightsIntegrityError as exc:
        print(
            f"[Self-check] Weight verification failed; the service refuses to start:\n{exc}",
            file=sys.stderr,
        )
        return 1

    import uvicorn

    print(f"Service started: http://{settings.host}:{settings.port}")
    uvicorn.run(app, host=settings.host, port=settings.port, reload=False)
    return 0


if __name__ == "__main__":
    sys.exit(run())
