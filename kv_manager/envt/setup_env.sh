#!/usr/bin/env bash
# Builds the kv_manager virtualenv at kv_manager/envt/.venv.
#
# Everything (uv, the managed Python 3.12, the venv, the uv download cache) stays under
# kv_manager/envt/, so nothing outside this folder is modified. CPU/network only; no GPU use.
#
#   bash kv_manager/envt/setup_env.sh          # create / update the venv from the lock
#   bash kv_manager/envt/setup_env.sh --relock # re-resolve requirements.txt -> requirements.lock.txt first
#   source kv_manager/envt/.venv/bin/activate
set -euo pipefail

ENVT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_VERSION="3.12"
UV_VERSION="0.13.6"
VENV="$ENVT_DIR/.venv"
REQS="$ENVT_DIR/requirements.txt"
LOCK="$ENVT_DIR/requirements.lock.txt"

export UV_PYTHON_INSTALL_DIR="$ENVT_DIR/.python"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ENVT_DIR/.uv-cache}"
export UV_LINK_MODE=copy

UV="$ENVT_DIR/.tools/bin/uv"
if [ ! -x "$UV" ]; then
  echo ">> bootstrapping uv $UV_VERSION into $ENVT_DIR/.tools"
  python3 -m pip install --quiet --target "$ENVT_DIR/.tools" "uv==$UV_VERSION"
fi

if [ "${1:-}" = "--relock" ] || [ ! -f "$LOCK" ]; then
  echo ">> resolving $REQS -> $LOCK"
  "$UV" pip compile "$REQS" -o "$LOCK" \
    --python-version "$PY_VERSION" \
    --python-platform x86_64-manylinux_2_28
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo ">> creating venv $VENV (Python $PY_VERSION)"
  "$UV" venv --python "$PY_VERSION" --seed "$VENV"
fi

echo ">> installing from $LOCK"
"$UV" pip install --python "$VENV/bin/python" -r "$LOCK"

"$VENV/bin/python" - <<'EOF'
import sys, torch, transformers, vllm
print(f"python {sys.version.split()[0]} | vllm {vllm.__version__} | torch {torch.__version__} "
      f"(CUDA {torch.version.cuda}) | transformers {transformers.__version__}")
assert vllm.__version__ == "0.27.1", vllm.__version__
EOF
echo ">> done. Activate with: source $VENV/bin/activate"
