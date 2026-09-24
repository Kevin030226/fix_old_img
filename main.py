"""Old photo restoration system — V2 web entry (plan section 26).

    python main.py            # http://127.0.0.1:9502 (override: FIXIMG_HOST / FIXIMG_PORT)

All wiring lives in app/factory.py; this file only performs the weight
self-check and starts Uvicorn.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from app.core.config import settings  # noqa: E402
from app.factory import create_app  # noqa: E402

app = create_app()


if __name__ == "__main__":
    from config.weights_check import WeightsIntegrityError, verify_weights

    try:
        n = verify_weights()
        print(f"[Self-check] Weight integrity OK ({n} files)")
    except WeightsIntegrityError as exc:
        print(
            f"[Self-check] Weight verification failed; the service refuses to start:\n{exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Service started: http://{settings.host}:{settings.port}")
    import uvicorn

    uvicorn.run(app="main:app", host=settings.host, port=settings.port, reload=False)
