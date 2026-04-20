#!/usr/bin/env bash
# ==============================================================================
# install_osworld.sh — sets up the `osworld` env with xlang-ai/OSWorld.
#
# Usage:   bash install/install_osworld.sh [OSWORLD_CLONE_DIR]
# Default OSWORLD_CLONE_DIR: /fs/gamma-projects/vlm-robot/OSWorld
# ==============================================================================
set -euo pipefail

ENV_NAME="osworld"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YML="${SCRIPT_DIR}/osworld.environment.yml"
OSW_DIR="${1:-/fs/gamma-projects/vlm-robot/OSWorld}"

command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found"; exit 1; }
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "[1/4] Creating env '$ENV_NAME' ..."
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "      Env already exists — skipping."
else
    conda env create -f "$YML"
fi

conda activate "$ENV_NAME"

echo "[2/4] Cloning OSWorld to $OSW_DIR ..."
if [[ ! -d "$OSW_DIR" ]]; then
    git clone --depth=1 https://github.com/xlang-ai/OSWorld.git "$OSW_DIR"
else
    echo "      Already cloned."
fi

echo "[3/4] Installing OSWorld requirements + editable package ..."
pip install -r "${OSW_DIR}/requirements.txt"
pip install -e "${OSW_DIR}"

echo "[4/4] Running smoke test ..."
python "$SCRIPT_DIR/osworld_smoke.py"

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
echo "Note: to run real desktop tasks you also need a VM backend"
echo "      (Docker: docker pull happysixd/osworld-docker, or VMware)."
