#!/usr/bin/env bash
#SBATCH --job-name=train-harness-v1
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=10:00:00

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SOURCE_RUNTIME_REPO="${SOURCE_RUNTIME_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-source-fresh-v1}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
LOCAL_HF_HOME="/scratch0/${USER}/harness-train-${SLURM_JOB_ID}/hf"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/runs/harness_pretraining_v1/sft}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/harness_pretraining_v1/checkpoints/game_receipt_lora_v1}"
MAX_STEPS="${MAX_STEPS:-400}"

[[ -f "${DATASET_DIR}/manifest.json" ]] || {
    echo "ERROR missing SFT manifest" >&2
    exit 2
}
[[ ! -e "${OUTPUT_DIR}" ]] || {
    echo "ERROR refusing to overwrite ${OUTPUT_DIR}" >&2
    exit 2
}
mkdir -p "${OUTPUT_DIR}.launch"
scontrol show job "${SLURM_JOB_ID}" >"${OUTPUT_DIR}.launch/slurm_job_at_start.txt"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv \
    >"${OUTPUT_DIR}.launch/nvidia_smi_start.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

cleanup() {
    local code=$?
    trap - EXIT INT TERM
    local scratch_root="/scratch0/${USER}/harness-train-${SLURM_JOB_ID}"
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

python scripts/train_harness_lora.py \
    --dataset-dir "${DATASET_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --source-repo "${SOURCE_RUNTIME_REPO}" \
    --max-length 2048 \
    --max-steps "${MAX_STEPS}" \
    --gradient-accumulation 8 \
    --learning-rate 2e-4 \
    --lora-r 16 \
    --lora-alpha 32 \
    | tee "${OUTPUT_DIR}.launch/train.log"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
    >"${OUTPUT_DIR}.launch/nvidia_smi_end.txt"
echo "TRANSFERABLE_HARNESS_V1_TRAINING_COMPLETE ${OUTPUT_DIR}"
