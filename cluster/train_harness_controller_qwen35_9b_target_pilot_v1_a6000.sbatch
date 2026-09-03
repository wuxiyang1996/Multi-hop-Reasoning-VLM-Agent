#!/usr/bin/env bash
#SBATCH --job-name=harness9b-target-sft
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=06:00:00

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SOURCE_RUNTIME_REPO="${SOURCE_RUNTIME_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/runs/target_harness_sft_pilot_v2}"
DATASET_AUDIT="${DATASET_AUDIT:-${DATASET_DIR}/independent_audit.json}"
INITIAL_ADAPTER="${INITIAL_ADAPTER:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_v3/continued_sft_v1/adapter}"
SOURCE_REGRESSION_DATASET="${SOURCE_REGRESSION_DATASET:-${REPO_ROOT}/runs/harness_controller_sft_v3/model_unseen_eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_target_pilot_v1/sft_v1}"
MAX_STEPS="${MAX_STEPS:-225}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
SCRATCH_ROOT="/scratch0/${USER}/harness-target-pilot-${SLURM_JOB_ID}"
LOCAL_HF_HOME="${SCRATCH_ROOT}/hf"
LAUNCH_DIR="${OUTPUT_DIR}.launch"

for path in \
  "${DATASET_DIR}/manifest.json" \
  "${DATASET_DIR}/train.jsonl" \
  "${DATASET_DIR}/validation.jsonl" \
  "${DATASET_AUDIT}" \
  "${SOURCE_REGRESSION_DATASET}" \
  "${INITIAL_ADAPTER}/adapter_model.safetensors"; do
  [[ -f "${path}" ]] || { echo "ERROR missing input ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT_DIR}" && ! -e "${LAUNCH_DIR}" ]] || {
  echo "ERROR refusing to overwrite ${OUTPUT_DIR} or ${LAUNCH_DIR}" >&2
  exit 2
}
jq -e '
  .status == "FROZEN_TARGET_DEVELOPMENT_HARNESS_SUPERVISION"
  and ([.gates[]] | all)
  and .target_data_used == true
  and .target_outcome_used_for_controller_labels == false
  and .formal_or_qualification_targets_used == false
  and .controller_training_contract.maximum_sequence_length == 2048
  and .controller_training_contract.exact_tokenizer_audit_required == true
' "${DATASET_DIR}/manifest.json" >/dev/null
jq -e '
  .status == "PASS"
  and ([.gates[]] | all)
  and .model == "Qwen/Qwen3.5-9B"
  and .maximum_sequence_length == 2048
  and .token_lengths.over_maximum == 0
' "${DATASET_AUDIT}" >/dev/null
[[ "$(jq -r '.manifest_sha256' "${DATASET_AUDIT}")" == \
   "$(sha256sum "${DATASET_DIR}/manifest.json" | cut -d' ' -f1)" ]] || {
  echo "ERROR dataset manifest changed after independent token audit" >&2
  exit 2
}
for value in "${MAX_STEPS}" "${MAX_LENGTH}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR invalid positive integer: ${value}" >&2
    exit 2
  }
done

mkdir -p "${LAUNCH_DIR}" "${SCRATCH_ROOT}"
scontrol show job "${SLURM_JOB_ID}" >"${LAUNCH_DIR}/slurm_job_at_start.txt"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv \
  >"${LAUNCH_DIR}/nvidia_smi_start.txt"
sha256sum \
  "${DATASET_DIR}/manifest.json" \
  "${DATASET_DIR}/train.jsonl" \
  "${DATASET_DIR}/validation.jsonl" \
  "${DATASET_AUDIT}" \
  "${SOURCE_REGRESSION_DATASET}" \
  "${INITIAL_ADAPTER}/adapter_model.safetensors" \
  >"${LAUNCH_DIR}/input_sha256.txt"
git -C "${REPO_ROOT}" rev-parse HEAD >"${LAUNCH_DIR}/git_head_at_start.txt"
git -C "${REPO_ROOT}" status --short >"${LAUNCH_DIR}/git_status_at_start.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ -d "${SCRATCH_ROOT}" && \
        "${SCRATCH_ROOT}" == "/scratch0/${USER}/harness-target-pilot-${SLURM_JOB_ID}" ]]; then
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

python scripts/train_harness_lora.py \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --source-repo "${SOURCE_RUNTIME_REPO}" \
  --initial-adapter "${INITIAL_ADAPTER}" \
  --max-length "${MAX_LENGTH}" \
  --max-steps "${MAX_STEPS}" \
  --gradient-accumulation 8 \
  --learning-rate "${LEARNING_RATE}" \
  --lora-r 16 \
  --lora-alpha 32 \
  2>&1 | tee "${LAUNCH_DIR}/train.log"

python scripts/evaluate_harness_controller_lora_v2.py \
  --dataset "${DATASET_DIR}/validation.jsonl" \
  --adapter "${OUTPUT_DIR}/adapter" \
  --output "${OUTPUT_DIR}/target_development_validation_report.json" \
  --model "${MODEL}" \
  --evaluation-kind target_development_validation \
  --per-decision 48 \
  --batch-size 4 \
  --max-input-length 1792 \
  --max-new-tokens 256 \
  2>&1 | tee "${LAUNCH_DIR}/evaluate_target_development.log"

python scripts/evaluate_harness_controller_lora_v2.py \
  --dataset "${SOURCE_REGRESSION_DATASET}" \
  --adapter "${OUTPUT_DIR}/adapter" \
  --output "${OUTPUT_DIR}/source_unseen_regression_report.json" \
  --model "${MODEL}" \
  --evaluation-kind source_family_heldout \
  --per-decision 24 \
  --batch-size 4 \
  --max-input-length 1792 \
  --max-new-tokens 256 \
  2>&1 | tee "${LAUNCH_DIR}/evaluate_source_regression.log"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
  >"${LAUNCH_DIR}/nvidia_smi_end.txt"
echo "HARNESS_CONTROLLER_QWEN35_9B_TARGET_PILOT_COMPLETE ${OUTPUT_DIR}"
