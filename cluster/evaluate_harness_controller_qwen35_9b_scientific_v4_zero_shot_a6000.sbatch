#!/usr/bin/env bash
#SBATCH --job-name=ctrl9b-v4-zs
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SOURCE_RUNTIME_REPO="${SOURCE_RUNTIME_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
USE_LOCAL_MODEL_CACHE="${USE_LOCAL_MODEL_CACHE:-0}"
DATASET="${DATASET:-${REPO_ROOT}/runs/harness_controller_scientific_v4_zero_shot/zero_shot_eval.jsonl}"
EVALUATION_MANIFEST="${EVALUATION_MANIFEST:-${REPO_ROOT}/runs/harness_controller_scientific_v4_zero_shot/manifest.json}"
ADAPTER="${ADAPTER:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_v3/continued_sft_v1/adapter}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/runs/harness_controller_scientific_v4_zero_shot/qwen35_9b_v3_report_retry1.json}"
TASK_SCRATCH="/scratch0/${USER}/harness-controller-v4-zero-shot-${SLURM_JOB_ID}"
LOCAL_HF_HOME="${TASK_SCRATCH}/hf"
LOG_DIR="${OUTPUT}.launch"

[[ -f "${DATASET}" ]] || { echo "ERROR missing ${DATASET}" >&2; exit 2; }
[[ -f "${EVALUATION_MANIFEST}" ]] || {
  echo "ERROR missing ${EVALUATION_MANIFEST}" >&2
  exit 2
}
[[ -f "${ADAPTER}/adapter_model.safetensors" ]] || {
  echo "ERROR missing adapter ${ADAPTER}" >&2
  exit 2
}
[[ ! -e "${OUTPUT}" ]] || { echo "ERROR refusing to overwrite ${OUTPUT}" >&2; exit 2; }
mkdir -p "${LOG_DIR}" "${TASK_SCRATCH}"
scontrol show job "${SLURM_JOB_ID}" >"${LOG_DIR}/slurm_job_at_start.txt"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv \
  >"${LOG_DIR}/nvidia_smi_start.txt"
sha256sum \
  "${DATASET}" \
  "${EVALUATION_MANIFEST}" \
  "${ADAPTER}/adapter_model.safetensors" \
  >"${LOG_DIR}/input_sha256.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ -d "${TASK_SCRATCH}" && "${TASK_SCRATCH}" == "/scratch0/${USER}/harness-controller-v4-zero-shot-${SLURM_JOB_ID}" ]]; then
    rm -rf -- "${TASK_SCRATCH}"
  fi
  exit "${code}"
}
trap cleanup EXIT INT TERM

if [[ "${USE_LOCAL_MODEL_CACHE}" == "1" ]]; then
  SOURCE_HF_HOME="${SHARED_HF_HOME}" LOCAL_HF_HOME="${LOCAL_HF_HOME}" \
    source "${SOURCE_RUNTIME_REPO}/scripts/stage_hf_model_cache.sh" "${MODEL}"
  export HF_HOME="${LOCAL_HF_HOME}"
else
  export HF_HOME="${SHARED_HF_HOME}"
  echo "Using shared read-only model cache after node-local staging ENOSPC"
fi
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python scripts/evaluate_harness_controller_lora_v2.py \
  --dataset "${DATASET}" \
  --evaluation-manifest "${EVALUATION_MANIFEST}" \
  --adapter "${ADAPTER}" \
  --output "${OUTPUT}" \
  --model "${MODEL}" \
  --evaluation-kind target_ir_zero_shot \
  --balance-field target_eval_group_audit_only \
  --audit-field control_variant_audit_only \
  --per-decision 10000 \
  --batch-size 4 \
  --max-input-length 1792 \
  --max-new-tokens 256 \
  2>&1 | tee "${LOG_DIR}/evaluate.log"

nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv \
  >"${LOG_DIR}/nvidia_smi_end.txt"
jq -r '.status' "${OUTPUT}" >"${OUTPUT%.json}.status.txt"
echo "HARNESS_CONTROLLER_QWEN35_9B_SCIENTIFIC_V4_ZERO_SHOT_COMPLETE ${OUTPUT}"
