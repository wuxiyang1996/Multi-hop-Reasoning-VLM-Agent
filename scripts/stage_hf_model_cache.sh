#!/usr/bin/env bash
# Stage a Hugging Face model cache on node-local NVMe before FSDP starts.
# Source this file when the caller should inherit HF_HOME/HF_HUB_OFFLINE:
#   source scripts/stage_hf_model_cache.sh Qwen/Qwen3.5-9B

set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen3.5-9B}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_HF_HOME="${SOURCE_HF_HOME:-${REPO_ROOT}/.hf_cache}"
LOCAL_HF_HOME="${LOCAL_HF_HOME:-/scratch0/${USER}/mh-vlm-hf-cache}"
CACHE_NAME="models--${MODEL_ID//\//--}"
SOURCE_MODEL_DIR="${SOURCE_HF_HOME}/hub/${CACHE_NAME}"
LOCAL_HUB_DIR="${LOCAL_HF_HOME}/hub"

if [[ ! -d "${SOURCE_MODEL_DIR}" ]]; then
    echo "ERROR: shared model cache not found: ${SOURCE_MODEL_DIR}" >&2
    return 1 2>/dev/null || exit 1
fi

mkdir -p "${LOCAL_HUB_DIR}"
START_S="${SECONDS}"
if command -v rsync >/dev/null 2>&1; then
    rsync -a "${SOURCE_MODEL_DIR}/" "${LOCAL_HUB_DIR}/${CACHE_NAME}/"
else
    cp -a "${SOURCE_MODEL_DIR}" "${LOCAL_HUB_DIR}/"
fi
ELAPSED_S="$((SECONDS - START_S))"

export HF_HOME="${LOCAL_HF_HOME}"
export HF_HUB_OFFLINE=1
echo "Staged ${MODEL_ID} to ${LOCAL_HUB_DIR}/${CACHE_NAME} in ${ELAPSED_S}s"
echo "HF_HOME=${HF_HOME}; HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
