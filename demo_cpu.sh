#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv/bin/activate"
if [[ ! -f "$VENV_PATH" ]]; then
  echo "Virtual environment not found at $VENV_PATH" >&2
  exit 1
fi

source "$VENV_PATH"
python scripts/run_inference.py --force-cpu "$@"
