#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash /workspace/setup_project_venv.sh
#   bash /workspace/setup_project_venv.sh /workspace/AI4Finance_DRL_DDPG_Algo
#   bash /workspace/setup_project_venv.sh /workspace/AI4Finance_DRL_DDPG_Algo /workspace/AI4Finance_DRL_DDPG_Algo/.venv

PROJECT_ROOT="${1:-/workspace/AI4Finance_DRL_DDPG_Algo}"
VENV_DIR="${2:-${PROJECT_ROOT}/.venv}"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "Error: project folder not found: ${PROJECT_ROOT}" >&2
  exit 1
fi

python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# TensorFlow is provided by the base image; install project/runtime dependencies in venv.
python -m pip install \
  gym \
  "gym[atari]" \
  filelock \
  matplotlib \
  pandas \
  pytest \
  opencv-python \
  lockfile \
  mpi4py

if [[ -d "${PROJECT_ROOT}/baselines" ]]; then
  python -m pip install -e "${PROJECT_ROOT}/baselines"
else
  echo "Info: ${PROJECT_ROOT}/baselines not found. Clone it first if you need OpenAI baselines."
fi

echo "Done. Activate with:"
echo "source ${VENV_DIR}/bin/activate"
