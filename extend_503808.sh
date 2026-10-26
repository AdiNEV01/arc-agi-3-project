#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
runner=.venv-arc-cu126/bin/python
if [ ! -x "$runner" ]; then
  echo "Extract this patch into the existing ~/Downloads/arc_dqn_cu126 directory, containing the working .venv-arc-cu126."
  exit 1
fi
exec "$runner" -u extend_503808.py "$@"
