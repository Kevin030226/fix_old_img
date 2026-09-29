"""Entry point for the legacy inference pipeline CLI.

The implementation lives in :mod:`fiximg.cli.batch`, which is where the four
stages are separately runnable (``--stages 1,2,3,4``, plan §3.7). This file is
only the path `LegacyCliBackend` spawns — it is pinned by
``backends/legacy_cli.py: self.cli_path = ... os.path.join(PROJECT_ROOT,
"run.py")`` and by the equivalence tests, so it cannot be renamed or inlined.

    python run.py --input_folder ./test_images/old --output_folder ./output
    python run.py --input_folder IN --output_folder OUT --stages 1

`PROGRESS_PREFIX` is re-exported because it is a contract with the *other* side:
`LegacyCliBackend` parses this process' stdout for the marker, so the constant
has to be reachable from both modules and has to be the same string in both.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "src"
))

from fiximg.cli.batch import (  # noqa: E402
    PROGRESS_PREFIX,
    StageError,
    main,
)

__all__ = ["PROGRESS_PREFIX", "StageError", "main"]

if __name__ == "__main__":
    try:
        sys.exit(main())
    except StageError as exc:
        print(f"\n[Pipeline failed] {exc}", file=sys.stderr)
        sys.exit(2)
