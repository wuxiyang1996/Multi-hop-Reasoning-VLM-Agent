#!/usr/bin/env bash
#SBATCH --job-name=ctrl9b-six-v2
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
REQUEST="${REQUEST:-${REPO_ROOT}/configs/harness_controller_qwen35_9b_mixed_v2_request.json}"
PROTOCOL_DIR="${PROTOCOL_DIR:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_mixed_v2_protocol}"
PROTOCOL="${PROTOCOL:-${PROTOCOL_DIR}/protocol.json}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/runs/harness_mixed_source_sft_v2}"
SCALAR_HELDOUT="${SCALAR_HELDOUT:-${REPO_ROOT}/runs/harness_controller_sft_v5_schema_closure/source_held_out.jsonl}"
MULTI_IR_HELDOUT="${MULTI_IR_HELDOUT:-${REPO_ROOT}/runs/harness_multi_ir_selector_sft_v2/source_held_out.jsonl}"
INITIAL_ADAPTER="${INITIAL_ADAPTER:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_mixed_v1/source_only_sft_v1/adapter}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_mixed_v2/source_only_sft_seed20260831}"
MAX_STEPS="${MAX_STEPS:-300}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
SEED="${SEED:-20260831}"
LAUNCH_DIR="${OUTPUT_DIR}.launch"

SCALAR_REPORT="${OUTPUT_DIR}/source_scalar_report.json"
SCALAR_PREDICTIONS="${OUTPUT_DIR}/source_scalar_report.predictions.jsonl"
MULTI_IR_REPORT="${OUTPUT_DIR}/source_multi_ir_report.json"
SOURCE_QUALIFICATION="${OUTPUT_DIR}/source_mixed_qualification.json"
TARGET_ACTIVATION="${OUTPUT_DIR}/six_benchmark_substitution_activation.json"
TARGET_DATASET="${REPO_ROOT}/runs/harness_9b_six_benchmark_substitution_v1/route_selector_replay.jsonl"
TARGET_REPORT="${OUTPUT_DIR}/six_benchmark_route_report.json"
TARGET_PREDICTIONS="${OUTPUT_DIR}/six_benchmark_route_report.predictions.jsonl"
ACTION_AUDIT="${OUTPUT_DIR}/six_benchmark_action_equivalence.json"

[[ -f "${REQUEST}" ]] || { echo "ERROR missing frozen V2 request" >&2; exit 2; }
[[ -f "${DATASET_DIR}/manifest.json" ]] || { echo "ERROR missing V2 source dataset" >&2; exit 2; }
[[ -f "${SCALAR_HELDOUT}" && -f "${MULTI_IR_HELDOUT}" ]] || {
  echo "ERROR missing source held-out gates" >&2
  exit 2
}
[[ -f "${INITIAL_ADAPTER}/adapter_model.safetensors" ]] || {
  echo "ERROR current V1 adapter is incomplete" >&2
  exit 2
}
[[ ! -e "${PROTOCOL_DIR}" && ! -e "${OUTPUT_DIR}" && ! -e "${LAUNCH_DIR}" ]] || {
  echo "ERROR refusing to overwrite V2 protocol/output/launch artifacts" >&2
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

# This resolves and hashes the completed V1 adapter only after its source-only
# qualification passes.  The six-benchmark task set was frozen earlier and is
# still not opened here.
python scripts/freeze_harness_controller_qwen35_9b_mixed_v2_protocol.py \
  --request "${REQUEST}" \
  --output "${PROTOCOL}" \
  2>&1 | tee "${LAUNCH_DIR}/freeze_protocol.log"

jq -e '
  .status == "FROZEN_BEFORE_MIXED_SOURCE_WEIGHT_UPDATES"
  and .model == "Qwen/Qwen3.5-9B"
  and .mixed_source_dataset.target_examples == 0
  and .training.max_steps == 300
  and .training.learning_rate == 0.00001
  and .training.seed == 20260831
  and .six_benchmark_substitution.native_replay_index.tasks == 1346
  and .six_benchmark_substitution.dataset.examples == 2246
' "${PROTOCOL}" >/dev/null

sha256sum \
  "${REQUEST}" \
  "${PROTOCOL}" \
  "${DATASET_DIR}/manifest.json" \
  "${DATASET_DIR}/train.jsonl" \
  "${DATASET_DIR}/validation.jsonl" \
  "${SCALAR_HELDOUT}" \
  "${MULTI_IR_HELDOUT}" \
  "${INITIAL_ADAPTER}/adapter_model.safetensors" \
  "${TARGET_DATASET}" \
  >"${LAUNCH_DIR}/input_sha256.txt"

# Continued source-only LoRA update: the only new competence is the
# predeclared Candy temporal source contract. No target prompt, label, action,
# or outcome participates in a weight update.
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

# The target prompts can only be opened after both source-only gates pass.
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

# Paper artifacts are deliberately downstream of every fail-closed gate.  The
# builder refuses to write if any source, route, receipt, or action-equivalence
# condition is incomplete.
python scripts/build_harness_9b_six_benchmark_paper_report_v1.py \
  --protocol "${PROTOCOL}" \
  --training-receipt "${OUTPUT_DIR}/training_receipt.json" \
  --source-qualification "${SOURCE_QUALIFICATION}" \
  --route-report "${TARGET_REPORT}" \
  --action-audit "${ACTION_AUDIT}" \
  --markdown "${REPO_ROOT}/docs/HARNESS_CONTROLLER_QWEN35_9B_SIX_BENCHMARK_V2_RESULTS.md" \
  --json "${REPO_ROOT}/docs/results/harness_controller_qwen35_9b_six_benchmark_v2.json" \
  2>&1 | tee "${LAUNCH_DIR}/paper_report.log"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
  >"${LAUNCH_DIR}/nvidia_smi_end.txt"
jq -r '.status' "${ACTION_AUDIT}" >"${OUTPUT_DIR}/scientific_status.txt"
echo "HARNESS_CONTROLLER_QWEN35_9B_SIX_BENCHMARK_V2_COMPLETE ${OUTPUT_DIR}"
