#!/usr/bin/env bash
#SBATCH --job-name=ctrl9b-six-v3
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
SOURCE_RUNTIME_REPO="${SOURCE_RUNTIME_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PROTOCOL="${PROTOCOL:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_mixed_v3_protocol/protocol.json}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/runs/harness_mixed_source_sft_v3}"
SCALAR_HELDOUT="${SCALAR_HELDOUT:-${REPO_ROOT}/runs/harness_controller_sft_v5_schema_closure/source_held_out.jsonl}"
MULTI_IR_HELDOUT="${MULTI_IR_HELDOUT:-${REPO_ROOT}/runs/harness_multi_ir_permutation_closure_sft_v3/source_held_out.jsonl}"
INITIAL_ADAPTER="${INITIAL_ADAPTER:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_mixed_v2/source_only_sft_seed20260831/adapter}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_mixed_v3/source_only_sft_seed20260901}"
MAX_STEPS="${MAX_STEPS:-800}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
SEED="${SEED:-20260901}"
LAUNCH_DIR="${OUTPUT_DIR}.launch"

SCALAR_REPORT="${OUTPUT_DIR}/source_scalar_report.json"
SCALAR_PREDICTIONS="${OUTPUT_DIR}/source_scalar_report.predictions.jsonl"
MULTI_IR_REPORT="${OUTPUT_DIR}/source_multi_ir_report.json"
SOURCE_QUALIFICATION="${OUTPUT_DIR}/source_mixed_qualification.json"
TARGET_ACTIVATION="${OUTPUT_DIR}/six_benchmark_substitution_activation.json"
TARGET_DATASET="${REPO_ROOT}/runs/harness_9b_six_benchmark_substitution_fresh_v2/route_selector_replay.jsonl"
TARGET_REPORT="${OUTPUT_DIR}/six_benchmark_route_report.json"
TARGET_PREDICTIONS="${OUTPUT_DIR}/six_benchmark_route_report.predictions.jsonl"
ACTION_AUDIT="${OUTPUT_DIR}/six_benchmark_action_equivalence.json"

[[ -f "${PROTOCOL}" ]] || { echo "ERROR missing frozen V3 protocol" >&2; exit 2; }
[[ -f "${DATASET_DIR}/manifest.json" ]] || { echo "ERROR missing V3 source dataset" >&2; exit 2; }
[[ -f "${SCALAR_HELDOUT}" && -f "${MULTI_IR_HELDOUT}" ]] || {
  echo "ERROR missing V3 source held-out gates" >&2
  exit 2
}
[[ -f "${INITIAL_ADAPTER}/adapter_model.safetensors" ]] || {
  echo "ERROR V2 source-only adapter is incomplete" >&2
  exit 2
}
[[ ! -e "${OUTPUT_DIR}" && ! -e "${LAUNCH_DIR}" ]] || {
  echo "ERROR refusing to overwrite V3 output/launch artifacts" >&2
  exit 2
}

mkdir -p "${LAUNCH_DIR}"
scontrol show job "${SLURM_JOB_ID}" >"${LAUNCH_DIR}/slurm_job_at_start.txt"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv \
  >"${LAUNCH_DIR}/nvidia_smi_start.txt"
git -C "${REPO_ROOT}" rev-parse HEAD >"${LAUNCH_DIR}/git_head_at_start.txt"
git -C "${REPO_ROOT}" status --short >"${LAUNCH_DIR}/git_status_at_start.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${SHARED_HF_HOME}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

jq -e '
  .status == "FROZEN_BEFORE_MIXED_SOURCE_WEIGHT_UPDATES"
  and .schema_version == "harness-controller-qwen35-9b-mixed-protocol-v3-permutation-closure"
  and .model == "Qwen/Qwen3.5-9B"
  and .mixed_source_dataset.target_examples == 0
  and .training.max_steps == 800
  and .training.learning_rate == 0.000005
  and .training.seed == 20260901
  and .six_benchmark_substitution.preregistration_status
      == "FROZEN_FRESH_PRESENTATION_BEFORE_SOURCE_ONLY_PERMUTATION_UPDATE"
  and .six_benchmark_substitution.native_replay_index.tasks == 1346
  and .six_benchmark_substitution.dataset.examples == 2246
' "${PROTOCOL}" >/dev/null

sha256sum \
  "${PROTOCOL}" \
  "${DATASET_DIR}/manifest.json" \
  "${DATASET_DIR}/train.jsonl" \
  "${DATASET_DIR}/validation.jsonl" \
  "${SCALAR_HELDOUT}" \
  "${MULTI_IR_HELDOUT}" \
  "${INITIAL_ADAPTER}/adapter_model.safetensors" \
  "${TARGET_DATASET}" \
  >"${LAUNCH_DIR}/input_sha256.txt"

# The only V3 update is balanced, source-only anonymous catalog permutation
# closure.  No V1 diagnostic prompt, error ID, benchmark identity, action, or
# outcome occurs in this train split.
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
  --seed "${SEED}" \
  2>&1 | tee "${LAUNCH_DIR}/train.log"

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

python scripts/activate_harness_9b_six_benchmark_substitution_v1.py \
  --protocol "${PROTOCOL}" \
  --source-qualification "${SOURCE_QUALIFICATION}" \
  --adapter "${OUTPUT_DIR}/adapter" \
  --training-receipt "${OUTPUT_DIR}/training_receipt.json" \
  --output "${TARGET_ACTIVATION}" \
  2>&1 | tee "${LAUNCH_DIR}/six_benchmark_activate.log"

python scripts/evaluate_harness_controller_lora_v2.py \
  --dataset "${TARGET_DATASET}" \
  --evaluation-manifest "${TARGET_ACTIVATION}" \
  --adapter "${OUTPUT_DIR}/adapter" \
  --output "${TARGET_REPORT}" \
  --model "${MODEL}" \
  --evaluation-kind target_model_substitution \
  --balance-field target_eval_group_audit_only \
  --audit-field route_component_audit_only \
  --per-decision 10000 \
  --batch-size 4 \
  --max-input-length 1792 \
  --max-new-tokens 128 \
  2>&1 | tee "${LAUNCH_DIR}/six_benchmark_route_evaluate.log"

python scripts/audit_harness_9b_six_benchmark_action_equivalence_v1.py \
  --activation "${TARGET_ACTIVATION}" \
  --route-report "${TARGET_REPORT}" \
  --route-predictions "${TARGET_PREDICTIONS}" \
  --output "${ACTION_AUDIT}" \
  2>&1 | tee "${LAUNCH_DIR}/six_benchmark_action_equivalence.log"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
  >"${LAUNCH_DIR}/nvidia_smi_end.txt"
jq -r '.status' "${ACTION_AUDIT}" >"${OUTPUT_DIR}/scientific_status.txt"
echo "HARNESS_CONTROLLER_QWEN35_9B_SIX_BENCHMARK_V3_COMPLETE ${OUTPUT_DIR}"
