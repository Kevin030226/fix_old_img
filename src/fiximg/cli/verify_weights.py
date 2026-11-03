"""`python -m fiximg.cli.verify_weights` — thin alias (plan §19 unified commands).

The `fiximg-verify-weights` console script resolves to `main` in *this* module,
so the module needs its own `main` rather than only the `__main__` block: pip
generates `sys.exit(main())` with no arguments, and re-exporting the
`download_weights.main` that required an `argv` positional made every invocation
fail with a TypeError. The wrapper defaults the sub-command to `verify`, which is
what the module name promises, and forwards everything else untouched.
"""
import sys

from fiximg.cli.download_weights import main as _download_main


def main(argv: list | None = None) -> int:
    """Run the weight verifier; defaults to the `verify` sub-command."""
    resolved = list(sys.argv if argv is None else argv)
    if len(resolved) <= 1:
        resolved = [resolved[0] if resolved else "fiximg-verify-weights", "verify"]
    return _download_main(resolved)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
