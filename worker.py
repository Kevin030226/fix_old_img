"""Standalone GPU worker process (V3).

    python worker.py                     # poll forever, one GPU task at a time
    FIXIMG_DEVICE=auto python worker.py

Thin shim: puts ``src/`` on ``sys.path`` and delegates to
:mod:`fiximg.cli.worker`, which owns the worker bootstrap.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from fiximg.cli.worker import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
