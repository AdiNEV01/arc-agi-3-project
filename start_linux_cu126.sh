#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
nvidia-smi
bootstrap_python="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/}python"
if ! command -v "$bootstrap_python" >/dev/null 2>&1; then bootstrap_python=python3; fi
runner=.venv-arc-cu126/bin/python
compatible() {
  "$1" -c 'import sys,torch,numpy,yaml,filelock,importlib.metadata as m; assert sys.version_info[:3]==(3,13,16); assert str(torch.__version__)=="2.11.0+cu126"; assert torch.version.cuda=="12.6"; assert numpy.__version__=="2.5.4"; assert all(m.version(k)==v for k,v in {"arc-agi":"0.9.9","arcengine":"0.9.3","gymnasium":"1.4.0"}.items())' 2>/dev/null
}
if [ -x "$runner" ] && compatible "$runner"; then
  echo "Using the separate CUDA 12.6 recovery environment."
  exec "$runner" -u cu126_recovery.py "$@"
fi
if [ -d .venv-arc-cu126 ] && [ ! -f .venv-arc-cu126/ARC_RECOVERY_ENV ]; then
  echo "STOPPED: .venv-arc-cu126 already exists without this launcher's ownership marker. Use a new delivery directory."
  exit 1
fi
echo "Preparing CUDA 12.6 in a NEW environment. No school driver or existing venv will be changed."
# Install uv in our own small bootstrap venv, never into the active environment.
if [ ! -x .uv-bootstrap-cu126/bin/python ]; then
  "$bootstrap_python" -m venv .uv-bootstrap-cu126
fi
uv_python=.uv-bootstrap-cu126/bin/python
"$uv_python" -m pip install uv
"$uv_python" -m uv python install 3.13.16
if [ ! -x "$runner" ]; then
  "$uv_python" -m uv venv --seed --python 3.13.16 .venv-arc-cu126
  printf '%s\n' 'ARC explicit cu130-to-cu126 recovery environment' > .venv-arc-cu126/ARC_RECOVERY_ENV
fi
cat > .venv-arc-cu126/recovery-constraints.txt <<'PINS'
torch==2.11.0+cu126
numpy==2.5.4
arc-agi==0.9.9
arcengine==0.9.3
gymnasium==1.4.0
PyYAML==6.0.3
filelock==4.2.4
PINS
"$runner" -m pip install 'torch==2.11.0+cu126' --index-url https://download.pytorch.org/whl/cu126
"$runner" -m pip install --constraint .venv-arc-cu126/recovery-constraints.txt numpy==2.5.4 arc-agi==0.9.9 arcengine==0.9.3 gymnasium==1.4.0 PyYAML==6.0.3 filelock==4.2.4
compatible "$runner"
exec "$runner" -u cu126_recovery.py "$@"
