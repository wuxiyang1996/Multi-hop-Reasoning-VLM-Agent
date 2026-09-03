#!/usr/bin/env bash
#SBATCH --job-name=ctrl9b-mixed-v1
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
SOURCE_RUNTIME_REPO="${SOURCE_RUNTIME_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PROTOCOL="${PROTOCOL:-${REPO_ROOT}/configs/harness_controller_qwen35_9b_mixed_v1_protocol.json}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/runs/harness_mixed_source_sft_v1}"
SCALAR_HELDOUT="${SCALAR_HELDOUT:-${REPO_ROOT}/runs/harness_controller_sft_v5_schema_closure/source_held_out.jsonl}"
MULTI_IR_HELDOUT="${MULTI_IR_HELDOUT:-${REPO_ROOT}/runs/harness_multi_ir_selector_sft_v1c/source_held_out.jsonl}"
INITIAL_ADAPTER="${INITIAL_ADAPTER:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_v4/cardinality_sft_v1/adapter}"
TARGET_RESERVE_DIR="${TARGET_RESERVE_DIR:-${REPO_ROOT}/runs/harness_controller_v5_fresh_target_reserve}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_mixed_v1/source_only_sft_v1}"
MAX_STEPS="${MAX_STEPS:-600}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LAUNCH_DIR="${OUTPUT_DIR}.launch"

SCALAR_REPORT="${OUTPUT_DIR}/source_scalar_report.json"
SCALAR_PREDICTIONS="${OUTPUT_DIR}/source_scalar_report.predictions.jsonl"
MULTI_IR_REPORT="${OUTPUT_DIR}/source_multi_ir_report.json"
SOURCE_QUALIFICATION="${OUTPUT_DIR}/source_mixed_qualification.json"
TARGET_ACTIVATION="${OUTPUT_DIR}/fresh_target_activation.json"
TARGET_REPORT="${OUTPUT_DIR}/fresh_target_report.json"
TARGET_PREDICTIONS="${OUTPUT_DIR}/fresh_target_report.predictions.jsonl"
TARGET_SHADOW_AUDIT="${OUTPUT_DIR}/fresh_target_shadow_audit.json"

[[ -f "${PROTOCOL}" ]] || { echo "ERROR missing frozen mixed protocol" >&2; exit 2; }
[[ -f "${DATASET_DIR}/manifest.json" ]] || { echo "ERROR missing mixed dataset" >&2; exit 2; }
[[ -f "${SCALAR_HELDOUT}" && -f "${MULTI_IR_HELDOUT}" ]] || {
  echo "ERROR missing source held-out gates" >&2
  exit 2
}
[[ -f "${INITIAL_ADAPTER}/adapter_model.safetensors" ]] || {
  echo "ERROR missing initial V4 adapter" >&2
  exit 2
}
[[ -f "${TARGET_RESERVE_DIR}/reserve.jsonl" ]] || {
  echo "ERROR missing pretraining-frozen V5 target reserve" >&2
  exit 2
}
[[ ! -e "${OUTPUT_DIR}" && ! -e "${LAUNCH_DIR}" ]] || {
  echo "ERROR refusing to overwrite ${OUTPUT_DIR} or ${LAUNCH_DIR}" >&2
  exit 2
}

jq -e '
  .status == "FROZEN_BEFORE_MIXED_SOURCE_WEIGHT_UPDATES"
  and .model == "Qwen/Qwen3.5-9B"
  and .mixed_source_dataset.target_examples == 0
  and .training.max_steps == 600
  and .training.learning_rate == 0.00002
' "${PROTOCOL}" >/dev/null
jq -e '
  .status == "FROZEN_SOURCE_ONLY_MIXED_HARNESS_SUPERVISION"
  and .target_data_used == false
  and .target_outcome_used_for_controller_labels == false
  and .formal_or_qualification_targets_used == false
  and .video_target_data_used == false
  and ([.gates[]] | all)
' "${DATASET_DIR}/manifest.json" >/dev/null
jq -e '
  .status == "FROZEN_PROSPECTIVE_TO_V5_BEFORE_WEIGHT_UPDATES"
  and ([.gates[]] | all)
' "${TARGET_RESERVE_DIR}/preregistration.json" >/dev/null

for binding in \
  "mixed_source_dataset.manifest:${DATASET_DIR}/manifest.json" \
  "mixed_source_dataset.train:${DATASET_DIR}/train.jsonl" \
  "mixed_source_dataset.validation:${DATASET_DIR}/validation.jsonl" \
  "source_held_out_gates.scalar_executor:${SCALAR_HELDOUT}" \
  "source_held_out_gates.multi_ir_selector:${MULTI_IR_HELDOUT}" \
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
  echo "ERROR initial V4 adapter hash mismatch" >&2
  exit 2
}
INITIAL_QUALIFICATION="${REPO_ROOT}/$(jq -r '.initial_adapter.source_qualification' "${PROTOCOL}")"
jq -e '
  .status == "SOURCE_CARDINALITY_GATE_PASSED"
  and ([.gates[]] | all)
' "${INITIAL_QUALIFICATION}" >/dev/null

mkdir -p "${LAUNCH_DIR}"
scontrol show job "${SLURM_JOB_ID}" >"${LAUNCH_DIR}/slurm_job_at_start.txt"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv \
  >"${LAUNCH_DIR}/nvidia_smi_start.txt"
sha256sum \
  "${PROTOCOL}" \
  "${DATASET_DIR}/manifest.json" \
  "${DATASET_DIR}/train.jsonl" \
  "${DATASET_DIR}/validation.jsonl" \
  "${SCALAR_HELDOUT}" \
  "${MULTI_IR_HELDOUT}" \
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

# One continued LoRA update preserves the scalar executor while adding the
# four-schema program selector. Every supervised row is source-derived.
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

# Gate 1a: retain exact typed-effect execution, including missing-schema
# abstention independently at every source-authorized cardinality n=2..12.
python scripts/evaluate_harness_controller_lora_v2.py \
  --dataset "${SCALAR_HELDOUT}" \
  --adapter "${OUTPUT_DIR}/adapter" \
  --output "${SCALAR_REPORT}" \
  --model "${MODEL}" \
  --evaluation-kind source_family_heldout \
  --balance-field candidate_count_audit_only \
  --per-decision 10000 \
  --batch-size 4 \
  --max-input-length 1792 \
  --max-new-tokens 256 \
  2>&1 | tee "${LAUNCH_DIR}/source_scalar_evaluate.log"

# Gate 1b: select or abstain over four anonymous source-induced IR schemas.
python scripts/evaluate_harness_controller_lora_v2.py \
  --dataset "${MULTI_IR_HELDOUT}" \
  --adapter "${OUTPUT_DIR}/adapter" \
  --output "${MULTI_IR_REPORT}" \
  --model "${MODEL}" \
  --evaluation-kind source_multi_ir_heldout \
  --balance-field source_ir_kind_audit_only \
  --audit-field control_variant_audit_only \
  --per-decision 10000 \
  --batch-size 4 \
  --max-input-length 1792 \
  --max-new-tokens 128 \
  2>&1 | tee "${LAUNCH_DIR}/source_multi_ir_evaluate.log"

python scripts/qualify_harness_mixed_source_v1.py \
  --protocol "${PROTOCOL}" \
  --training-receipt "${OUTPUT_DIR}/training_receipt.json" \
  --scalar-report "${SCALAR_REPORT}" \
  --scalar-predictions "${SCALAR_PREDICTIONS}" \
  --multi-ir-report "${MULTI_IR_REPORT}" \
  --output "${SOURCE_QUALIFICATION}" \
  2>&1 | tee "${LAUNCH_DIR}/source_mixed_qualify.log"

# The target file is opened only after both source-only gates pass.
python scripts/activate_harness_controller_v5_target_reserve.py \
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

# Gate 3: fail closed under the structural verifier. The model output is not
# repaired by a Python argmax and may not authorize an unsafe source operator.
python scripts/audit_harness_controller_v4_shadow.py \
  --reserve "${TARGET_RESERVE_DIR}/reserve.jsonl" \
  --activation "${TARGET_ACTIVATION}" \
  --report "${TARGET_REPORT}" \
  --predictions "${TARGET_PREDICTIONS}" \
  --output "${TARGET_SHADOW_AUDIT}" \
  2>&1 | tee "${LAUNCH_DIR}/target_shadow_audit.log"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
  >"${LAUNCH_DIR}/nvidia_smi_end.txt"
jq -r '.status' "${TARGET_SHADOW_AUDIT}" >"${OUTPUT_DIR}/scientific_status.txt"
echo "HARNESS_CONTROLLER_QWEN35_9B_MIXED_V1_COMPLETE ${OUTPUT_DIR}"
