#!/usr/bin/env bash
#SBATCH --job-name=eval-harness-v1
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SOURCE_RUNTIME_REPO="${SOURCE_RUNTIME_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-source-fresh-v1}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
LOCAL_HF_HOME="/scratch0/${USER}/harness-eval-${SLURM_JOB_ID}/hf"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
DATASET="${DATASET:-${REPO_ROOT}/runs/harness_pretraining_v1/sft/source_held_out.jsonl}"
ADAPTER="${ADAPTER:-${REPO_ROOT}/runs/harness_pretraining_v1/checkpoints/game_receipt_lora_v1/adapter}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/docs/results/harness_pretraining_v1_source_held_out.json}"

[[ -f "${DATASET}" ]] || { echo "ERROR missing ${DATASET}" >&2; exit 2; }
[[ -f "${ADAPTER}/adapter_model.safetensors" ]] || {
    echo "ERROR missing trained adapter ${ADAPTER}" >&2
    exit 2
}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cleanup() {
    local code=$?
    trap - EXIT INT TERM
    local scratch_root="/scratch0/${USER}/harness-eval-${SLURM_JOB_ID}"
    if [[ -d "${scratch_root}" ]]; then
        rm -rf -- "${scratch_root}"
    fi
    exit "${code}"
}
trap cleanup EXIT INT TERM

SOURCE_HF_HOME="${SHARED_HF_HOME}" LOCAL_HF_HOME="${LOCAL_HF_HOME}" \
    source "${SOURCE_RUNTIME_REPO}/scripts/stage_hf_model_cache.sh" "${MODEL}"
export HF_HOME="${LOCAL_HF_HOME}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python scripts/evaluate_harness_lora.py \
  --dataset "${DATASET}" \
  --adapter "${ADAPTER}" \
  --output "${OUTPUT}" \
  --model "${MODEL}" \
  --per-objective 64 \
  --max-length 2048

echo "TRANSFERABLE_HARNESS_V1_SOURCE_HELD_OUT_COMPLETE ${OUTPUT}"
