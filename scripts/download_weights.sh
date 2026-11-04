#!/usr/bin/env bash
# Legacy entry point: download the model weights through the audited Python CLI.
#
# Historically this script ran its own curl/wget copy per artifact, skipped any
# file that merely existed, never checked config/weights_manifest.json, and used
# an Azure endpoint that no longer resolves. It now delegates to
# `fiximg.cli.download_weights download`, which verifies every downloaded file
# against the committed manifest (path + size + sha256), exits non-zero on any
# mismatch, and leaves the manifest untouched. Changing the baseline is a
# maintainer action on the Python CLI (generate), not something this entry point
# can do implicitly.
#
# Usage:
#   bash scripts/download_weights.sh
#   FIXIMG_PYTHON=/path/to/python bash scripts/download_weights.sh
#
# Sub-commands are intentionally not accepted here: verify and generate stay
# explicit on `python -m fiximg.cli.download_weights`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ "$#" -ne 0 ]; then
  echo "download_weights.sh: this compatibility entry point only downloads and verifies." >&2
  echo "Use: python -m fiximg.cli.download_weights [download|verify|generate]" >&2
  exit 2
fi

# Interpreter selection: an explicit override wins (PATH lookups differ between
# Git Bash, distro python3 and virtualenvs), then python3, then python.
PYTHON="${FIXIMG_PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi
if [ -z "$PYTHON" ]; then
  echo "download_weights.sh: no Python interpreter found; set FIXIMG_PYTHON" >&2
  exit 127
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "download_weights.sh: interpreter not executable: $PYTHON" >&2
  exit 127
fi

# A source checkout is not installed, so `python -m fiximg.*` needs src/ on the
# import path; keep any caller-supplied entries after it.
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON" -m fiximg.cli.download_weights download
