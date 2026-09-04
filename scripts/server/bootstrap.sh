#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${VENV_PATH:-.venv-cahnm}"
PYTHON_CMD="${PYTHON_CMD:-python3}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

"${PYTHON_CMD}" -m venv "${VENV_PATH}"
PYTHON_BIN="${VENV_PATH}/bin/python"
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel

# CUDA 12.6 PyTorch wheels retain Volta/sm_70 support. Force replacement so
# rerunning this bootstrap repairs an incompatible cu128/cu130 installation.
"${PYTHON_BIN}" -m pip install --force-reinstall --no-cache-dir \
  "torch==${TORCH_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"

"${PYTHON_BIN}" -m pip install -e ".[dense,dev]"

if [[ "${REQUIRE_CUDA}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/check_server_environment.py --require-cuda --out runs/paper_experiments/environment.json
else
  "${PYTHON_BIN}" scripts/check_server_environment.py --out runs/paper_experiments/environment.json
fi

echo "Environment ready. Activate with: source ${VENV_PATH}/bin/activate"
