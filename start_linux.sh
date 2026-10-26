#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ "${1:-}" = "--cuda-build" ]; then
  build="${2:-}"
  shift 2
  case "$build" in
    cu126) exec bash start_linux_cu126.sh "$@" ;;
    cu130) ;;
    *) echo "Unsupported CUDA build: $build (choose cu126 or cu130)"; exit 1 ;;
  esac
fi
nvidia-smi
current_python="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/}python"
if ! command -v "$current_python" >/dev/null 2>&1; then current_python=python3; fi
compatible() {
  "$1" -c 'import sys,torch,numpy,yaml,filelock,importlib.metadata as m; assert sys.version_info[:3]==(3,13,16); assert str(torch.__version__)=="2.11.0+cu130"; assert numpy.__version__=="2.5.4"; assert all(m.version(k)==v for k,v in {"arc-agi":"0.9.9","arcengine":"0.9.3","gymnasium":"1.4.0"}.items())' 2>/dev/null
}
if compatible "$current_python"; then
  echo "Using your compatible active environment."
  exec "$current_python" linux_resume.py "$@"
fi
runner=.venv-arc-colab/bin/python
if [ -x "$runner" ] && compatible "$runner"; then
  echo "Reusing the compatible migration environment."
  exec "$runner" linux_resume.py "$@"
fi
echo "Preparing a separate pinned environment; your existing venv is preserved."
"$current_python" -m pip install uv
"$current_python" -m uv python install 3.13.16
if [ ! -x "$runner" ]; then
  "$current_python" -m uv venv --seed --python 3.13.16 .venv-arc-colab
fi
"$runner" -m pip install --upgrade pip
"$runner" -m pip install arc-agi==0.9.9 arcengine==0.9.3 gymnasium==1.4.0 numpy==2.5.4 PyYAML==6.0.3 filelock==4.2.4
"$runner" -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
exec "$runner" linux_resume.py "$@"
