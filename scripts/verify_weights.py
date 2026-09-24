"""python -m scripts.verify_weights — thin alias (plan §19 unified commands)."""
import sys

from scripts.download_weights import main

if __name__ == "__main__":
    # Accept bare invocation: default to verify.
    argv = sys.argv if len(sys.argv) > 1 else [sys.argv[0], "verify"]
    sys.exit(main(argv))
