#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install from https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

uv python install 3.11
uv venv --python 3.11 .venv

# shellcheck disable=SC1091
source .venv/bin/activate

# Installs Gymnasium + SB3 base dependencies plus CUDA-enabled PyTorch wheels.
uv sync --extra cuda

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
PY
