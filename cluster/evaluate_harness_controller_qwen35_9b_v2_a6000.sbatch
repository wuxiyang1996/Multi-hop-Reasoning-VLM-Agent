#!/usr/bin/env bash
#SBATCH --job-name=eval-ctrl9b-v2
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
SOURCE_RUNTIME_REPO="${SOURCE_RUNTIME_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
DATASET="${DATASET:-${REPO_ROOT}/runs/harness_controller_sft_v2_final/source_held_out.jsonl}"
ADAPTER="${ADAPTER:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_v2/sft_v2/trainer/checkpoint-100}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_v2/sft_v2/checkpoint_100_source_family_heldout_report.json}"
SCRATCH_ROOT="/scratch0/${USER}/harness-controller-eval-v2-${SLURM_JOB_ID}"
LOCAL_HF_HOME="${SCRATCH_ROOT}/hf"
LOG_DIR="${OUTPUT}.launch"

[[ -f "${DATASET}" ]] || { echo "ERROR missing ${DATASET}" >&2; exit 2; }
[[ -f "${ADAPTER}/adapter_model.safetensors" ]] || {
  echo "ERROR missing adapter ${ADAPTER}" >&2
  exit 2
}
[[ ! -e "${OUTPUT}" ]] || { echo "ERROR refusing to overwrite ${OUTPUT}" >&2; exit 2; }
mkdir -p "${LOG_DIR}" "${SCRATCH_ROOT}"
scontrol show job "${SLURM_JOB_ID}" >"${LOG_DIR}/slurm_job_at_start.txt"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv \
  >"${LOG_DIR}/nvidia_smi_start.txt"
sha256sum "${DATASET}" "${ADAPTER}/adapter_model.safetensors" \
  >"${LOG_DIR}/input_sha256.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ -d "${SCRATCH_ROOT}" && "${SCRATCH_ROOT}" == "/scratch0/${USER}/harness-controller-eval-v2-${SLURM_JOB_ID}" ]]; then
    rm -rf -- "${SCRATCH_ROOT}"
  fi
  exit "${code}"
}
trap cleanup EXIT INT TERM

SOURCE_HF_HOME="${SHARED_HF_HOME}" LOCAL_HF_HOME="${LOCAL_HF_HOME}" \
  source "${SOURCE_RUNTIME_REPO}/scripts/stage_hf_model_cache.sh" "${MODEL}"
export HF_HOME="${LOCAL_HF_HOME}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python scripts/evaluate_harness_controller_lora_v2.py \
  --dataset "${DATASET}" \
  --adapter "${ADAPTER}" \
  --output "${OUTPUT}" \
  --model "${MODEL}" \
  --per-decision 24 \
  --batch-size 4 \
  --max-input-length 1792 \
  --max-new-tokens 256 \
  2>&1 | tee "${LOG_DIR}/evaluate.log"

jq -r '.status' "${OUTPUT}" >"${OUTPUT%.json}.status.txt"
echo "HARNESS_CONTROLLER_QWEN35_9B_V2_A6000_EVAL_COMPLETE ${OUTPUT}"
