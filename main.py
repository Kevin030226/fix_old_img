"""Old photo restoration system — V3 web entry point.

    python main.py            # http://127.0.0.1:9502 (override: FIXIMG_HOST / FIXIMG_PORT)

This file is intentionally a thin shim: it only puts ``src/`` on ``sys.path``
(for source checkouts that did not ``pip install -e .``) and delegates to
:mod:`fiximg.cli.api`, which owns the real bootstrap.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from fiximg.cli.api import app, run  # noqa: E402,F401  (app: uvicorn "main:app")

if __name__ == "__main__":
    sys.exit(run())
