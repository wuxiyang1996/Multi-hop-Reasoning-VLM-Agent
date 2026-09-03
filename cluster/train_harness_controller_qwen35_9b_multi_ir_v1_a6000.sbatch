#!/usr/bin/env bash
#SBATCH --job-name=ctrl9b-multi-ir
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
PROTOCOL="${PROTOCOL:-${REPO_ROOT}/configs/harness_controller_qwen35_9b_multi_ir_v1_protocol.json}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/runs/harness_multi_ir_selector_sft_v1c}"
INITIAL_ADAPTER="${INITIAL_ADAPTER:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_v4/cardinality_sft_v1/adapter}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_multi_ir_v1/source_only_sft_v1}"
MAX_STEPS="${MAX_STEPS:-400}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LAUNCH_DIR="${OUTPUT_DIR}.launch"
HELDOUT_REPORT="${OUTPUT_DIR}/source_multi_ir_report.json"
QUALIFICATION="${OUTPUT_DIR}/source_multi_ir_qualification.json"

[[ -f "${PROTOCOL}" ]] || { echo "ERROR missing protocol" >&2; exit 2; }
[[ -f "${DATASET_DIR}/manifest.json" ]] || { echo "ERROR missing dataset" >&2; exit 2; }
[[ -f "${INITIAL_ADAPTER}/adapter_model.safetensors" ]] || {
  echo "ERROR missing initial V4 adapter" >&2
  exit 2
}
[[ ! -e "${OUTPUT_DIR}" && ! -e "${LAUNCH_DIR}" ]] || {
  echo "ERROR refusing to overwrite ${OUTPUT_DIR} or ${LAUNCH_DIR}" >&2
  exit 2
}
jq -e '
  .status == "FROZEN_BEFORE_MULTI_IR_SELECTOR_WEIGHT_UPDATES"
  and .source_only_dataset.target_examples == 0
  and .model == "Qwen/Qwen3.5-9B"
' "${PROTOCOL}" >/dev/null
jq -e '
  .status == "FROZEN_SOURCE_ONLY_MULTI_IR_SELECTOR_SUPERVISION"
  and ([.gates[]] | all)
' "${DATASET_DIR}/manifest.json" >/dev/null
for binding in \
  "source_only_dataset.manifest:${DATASET_DIR}/manifest.json" \
  "source_only_dataset.train:${DATASET_DIR}/train.jsonl" \
  "source_only_dataset.validation:${DATASET_DIR}/validation.jsonl" \
  "source_only_dataset.source_held_out:${DATASET_DIR}/source_held_out.jsonl"; do
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
  echo "ERROR initial V4 adapter hash mismatch" >&2
  exit 2
}
jq -e '
  .status == "SOURCE_CARDINALITY_GATE_PASSED"
  and ([.gates[]] | all)
' "${REPO_ROOT}/$(jq -r '.initial_adapter.source_qualification' "${PROTOCOL}")" \
  >/dev/null

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
  --dataset "${DATASET_DIR}/source_held_out.jsonl" \
  --adapter "${OUTPUT_DIR}/adapter" \
  --output "${HELDOUT_REPORT}" \
  --model "${MODEL}" \
  --evaluation-kind source_multi_ir_heldout \
  --balance-field source_ir_kind_audit_only \
  --audit-field control_variant_audit_only \
  --per-decision 10000 \
  --batch-size 4 \
  --max-input-length 1792 \
  --max-new-tokens 128 \
  2>&1 | tee "${LAUNCH_DIR}/source_evaluate.log"

python scripts/qualify_harness_multi_ir_selector_v1.py \
  --protocol "${PROTOCOL}" \
  --report "${HELDOUT_REPORT}" \
  --training-receipt "${OUTPUT_DIR}/training_receipt.json" \
  --output "${QUALIFICATION}" \
  2>&1 | tee "${LAUNCH_DIR}/source_qualify.log"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
  >"${LAUNCH_DIR}/nvidia_smi_end.txt"
jq -r '.status' "${QUALIFICATION}" >"${OUTPUT_DIR}/scientific_status.txt"
echo "HARNESS_CONTROLLER_QWEN35_9B_MULTI_IR_V1_COMPLETE ${OUTPUT_DIR}"
