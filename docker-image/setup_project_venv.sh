#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash /workspace/setup_project_venv.sh
#   bash /workspace/setup_project_venv.sh /workspace/AI4Finance
#   bash /workspace/setup_project_venv.sh /workspace/AI4Finance /workspace/AI4Finance/.venv
#   INSTALL_LEGACY_BASELINES=1 bash /workspace/setup_project_venv.sh

PROJECT_ROOT="${1:-/workspace/AI4Finance}"
VENV_DIR="${2:-${PROJECT_ROOT}/.venv}"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "Error: project folder not found: ${PROJECT_ROOT}" >&2
  exit 1
fi

if command -v python3.11 >/dev/null 2>&1; then
  PY_BIN="python3.11"
else
  PY_BIN="python3"
fi

"${PY_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Install runtime dependencies for TF2-native reproduction.
python -m pip install \
  tensorflow==2.16.1 \
  numpy==1.26.4 \
  filelock \
  matplotlib \
  pandas \
  pytest \
  opencv-python \
  lockfile

if [[ "${INSTALL_LEGACY_BASELINES:-0}" == "1" ]]; then
  python -m pip install "setuptools<81" wheel
  python -m pip install gym==0.15.7 cloudpickle==1.2.2 pyglet==1.5.0 scipy tqdm joblib click mpi4py
  if [[ -d "${PROJECT_ROOT}/DQN-DDPG_Stock_Trading/baselines" ]]; then
    python -m pip install -e "${PROJECT_ROOT}/DQN-DDPG_Stock_Trading/baselines" --no-build-isolation || true
  else
    echo "Info: legacy baselines folder not found at ${PROJECT_ROOT}/DQN-DDPG_Stock_Trading/baselines"
  fi
fi

echo "Done. Activate with:"
echo "source ${VENV_DIR}/bin/activate"
echo "Suggested next step:"
echo "python -m tf2_repro.data_pipeline"
