#!/usr/bin/env bash
#SBATCH --job-name=ctrl9b-v4-card
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=08:00:00

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
SOURCE_RUNTIME_REPO="${SOURCE_RUNTIME_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PROTOCOL="${PROTOCOL:-${REPO_ROOT}/configs/harness_controller_qwen35_9b_v4_cardinality_protocol.json}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/runs/harness_controller_sft_v4_cardinality}"
INITIAL_ADAPTER="${INITIAL_ADAPTER:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_v3/continued_sft_v1/adapter}"
TARGET_RESERVE_DIR="${TARGET_RESERVE_DIR:-${REPO_ROOT}/runs/harness_controller_v4_fresh_target_reserve}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_v4/cardinality_sft_v1}"
MAX_STEPS="${MAX_STEPS:-400}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LAUNCH_DIR="${OUTPUT_DIR}.launch"

SOURCE_REPORT="${OUTPUT_DIR}/source_cardinality_report.json"
SOURCE_QUALIFICATION="${OUTPUT_DIR}/source_cardinality_qualification.json"
TARGET_ACTIVATION="${OUTPUT_DIR}/fresh_target_activation.json"
TARGET_REPORT="${OUTPUT_DIR}/fresh_target_report.json"

[[ -f "${PROTOCOL}" ]] || { echo "ERROR missing ${PROTOCOL}" >&2; exit 2; }
[[ -f "${DATASET_DIR}/manifest.json" ]] || { echo "ERROR missing V4 dataset" >&2; exit 2; }
[[ -f "${INITIAL_ADAPTER}/adapter_model.safetensors" ]] || {
  echo "ERROR missing initial V3 adapter" >&2
  exit 2
}
[[ -f "${TARGET_RESERVE_DIR}/reserve.jsonl" ]] || {
  echo "ERROR missing pretraining-frozen target reserve" >&2
  exit 2
}
[[ ! -e "${OUTPUT_DIR}" && ! -e "${LAUNCH_DIR}" ]] || {
  echo "ERROR refusing to overwrite ${OUTPUT_DIR} or ${LAUNCH_DIR}" >&2
  exit 2
}
jq -e '
  .status == "FROZEN_BEFORE_V4_CARDINALITY_TRAINING"
  and .source_only_dataset.target_examples == 0
  and .fresh_target_reserve.activation_requires_source_cardinality_gate == true
  and .fresh_target_reserve.model_output_read_before_activation == false
' "${PROTOCOL}" >/dev/null
jq -e '
  .status == "FROZEN_SOURCE_ONLY_CONTROLLER_SUPERVISION"
  and .target_data_used == false
  and .summary.target_data_used == false
  and .cardinality_equivariance.enabled == true
  and ([.gates[]] | all)
' "${DATASET_DIR}/manifest.json" >/dev/null
jq -e '
  .status == "FROZEN_PROSPECTIVE_TO_V4_BEFORE_WEIGHT_UPDATES"
  and ([.gates[]] | all)
' "${TARGET_RESERVE_DIR}/preregistration.json" >/dev/null
for binding in \
  "source_only_dataset.manifest:${DATASET_DIR}/manifest.json" \
  "source_only_dataset.train:${DATASET_DIR}/train.jsonl" \
  "source_only_dataset.validation:${DATASET_DIR}/validation.jsonl" \
  "source_only_dataset.source_held_out:${DATASET_DIR}/source_held_out.jsonl" \
  "fresh_target_reserve.preregistration:${TARGET_RESERVE_DIR}/preregistration.json" \
  "fresh_target_reserve.dataset:${TARGET_RESERVE_DIR}/reserve.jsonl"; do
  key="${binding%%:*}"
  path="${binding#*:}"
  expected="$(jq -r ".${key}.sha256" "${PROTOCOL}")"
  actual="$(sha256sum "${path}" | cut -d' ' -f1)"
  [[ "${actual}" == "${expected}" ]] || {
    echo "ERROR protocol hash mismatch for ${path}" >&2
    exit 2
  }
done
[[ "$(sha256sum "${INITIAL_ADAPTER}/adapter_model.safetensors" | cut -d' ' -f1)" == \
   "$(jq -r '.initial_adapter.adapter_model_sha256' "${PROTOCOL}")" ]] || {
  echo "ERROR initial V3 adapter hash mismatch" >&2
  exit 2
}

mkdir -p "${LAUNCH_DIR}"
scontrol show job "${SLURM_JOB_ID}" >"${LAUNCH_DIR}/slurm_job_at_start.txt"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv \
  >"${LAUNCH_DIR}/nvidia_smi_start.txt"
sha256sum \
  "${PROTOCOL}" \
  "${DATASET_DIR}/manifest.json" \
  "${DATASET_DIR}/train.jsonl" \
  "${DATASET_DIR}/validation.jsonl" \
  "${DATASET_DIR}/source_held_out.jsonl" \
  "${INITIAL_ADAPTER}/adapter_model.safetensors" \
  "${TARGET_RESERVE_DIR}/preregistration.json" \
  "${TARGET_RESERVE_DIR}/reserve.jsonl" \
  >"${LAUNCH_DIR}/input_sha256.txt"
git -C "${REPO_ROOT}" rev-parse HEAD >"${LAUNCH_DIR}/git_head_at_start.txt"
git -C "${REPO_ROOT}" status --short >"${LAUNCH_DIR}/git_status_at_start.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${SHARED_HF_HOME}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
echo "Using shared model cache; node-local staging disabled after recorded ENOSPC"

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

# Gate 1: the new weights must solve every arity on a whole held-out source
# family.  The target reserve has not been passed to model inference yet.
python scripts/evaluate_harness_controller_lora_v2.py \
  --dataset "${DATASET_DIR}/source_held_out.jsonl" \
  --adapter "${OUTPUT_DIR}/adapter" \
  --output "${SOURCE_REPORT}" \
  --model "${MODEL}" \
  --evaluation-kind source_family_heldout \
  --balance-field candidate_count_audit_only \
  --per-decision 10000 \
  --batch-size 4 \
  --max-input-length 1792 \
  --max-new-tokens 256 \
  2>&1 | tee "${LAUNCH_DIR}/source_evaluate.log"

python scripts/qualify_harness_controller_v4_cardinality.py \
  --protocol "${PROTOCOL}" \
  --report "${SOURCE_REPORT}" \
  --training-receipt "${OUTPUT_DIR}/training_receipt.json" \
  --output "${SOURCE_QUALIFICATION}" \
  2>&1 | tee "${LAUNCH_DIR}/source_qualify.log"

# Gate 2: activation is impossible unless Gate 1 passed.  It binds the newly
# frozen source-only adapter hash to the reserve frozen before training.
python scripts/activate_harness_controller_v4_target_reserve.py \
  --protocol "${PROTOCOL}" \
  --source-qualification "${SOURCE_QUALIFICATION}" \
  --adapter "${OUTPUT_DIR}/adapter" \
  --training-receipt "${OUTPUT_DIR}/training_receipt.json" \
  --output "${TARGET_ACTIVATION}" \
  2>&1 | tee "${LAUNCH_DIR}/target_activate.log"

python scripts/evaluate_harness_controller_lora_v2.py \
  --dataset "${TARGET_RESERVE_DIR}/reserve.jsonl" \
  --evaluation-manifest "${TARGET_ACTIVATION}" \
  --adapter "${OUTPUT_DIR}/adapter" \
  --output "${TARGET_REPORT}" \
  --model "${MODEL}" \
  --evaluation-kind target_ir_zero_shot \
  --balance-field target_eval_group_audit_only \
  --audit-field control_variant_audit_only \
  --per-decision 10000 \
  --batch-size 4 \
  --max-input-length 1792 \
  --max-new-tokens 256 \
  2>&1 | tee "${LAUNCH_DIR}/target_evaluate.log"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
  >"${LAUNCH_DIR}/nvidia_smi_end.txt"
jq -r '.status' "${TARGET_REPORT}" >"${OUTPUT_DIR}/scientific_status.txt"
echo "HARNESS_CONTROLLER_QWEN35_9B_V4_CARDINALITY_COMPLETE ${OUTPUT_DIR}"
