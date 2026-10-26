#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
runner=.venv-arc-cu126/bin/python
if [ ! -x "$runner" ]; then
  echo "The existing .venv-arc-cu126 recovery environment was not found. Extract this patch into ~/Downloads/arc_dqn_cu126, where the CUDA check already passed."
  exit 1
fi
exec "$runner" -u continue_numerical.py "$@"
