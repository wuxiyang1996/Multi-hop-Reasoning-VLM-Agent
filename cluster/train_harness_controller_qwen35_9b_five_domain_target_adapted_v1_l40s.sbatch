#!/usr/bin/env bash
#SBATCH --job-name=ctrl9b-target5-v1
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=10:00:00

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SOURCE_RUNTIME_REPO="${SOURCE_RUNTIME_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PROTOCOL="${PROTOCOL:-${REPO_ROOT}/configs/harness_controller_qwen35_9b_five_domain_target_adapted_v1.json}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/runs/target_harness_sft_five_domain_v1}"
INITIAL_ADAPTER="${INITIAL_ADAPTER:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_mixed_v3/source_only_sft_seed20260901/adapter}"
SCALAR_HELDOUT="${SCALAR_HELDOUT:-${REPO_ROOT}/runs/harness_controller_sft_v5_schema_closure/source_held_out.jsonl}"
MULTI_IR_HELDOUT="${MULTI_IR_HELDOUT:-${REPO_ROOT}/runs/harness_multi_ir_permutation_closure_sft_v3/source_held_out.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_five_domain_target_adapted_v1/sft_seed20260902}"
LAUNCH_DIR="${OUTPUT_DIR}.launch"
MAX_STEPS="${MAX_STEPS:-600}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
SEED="${SEED:-20260902}"

for path in \
  "${PROTOCOL}" \
  "${DATASET_DIR}/manifest.json" \
  "${DATASET_DIR}/independent_audit.json" \
  "${DATASET_DIR}/train.jsonl" \
  "${DATASET_DIR}/validation.jsonl" \
  "${INITIAL_ADAPTER}/adapter_model.safetensors" \
  "${SCALAR_HELDOUT}" \
  "${MULTI_IR_HELDOUT}"; do
  [[ -f "${path}" ]] || { echo "ERROR missing input ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT_DIR}" && ! -e "${LAUNCH_DIR}" ]] || {
  echo "ERROR refusing to overwrite target-adapted output" >&2
  exit 2
}

jq -e '
  .status == "FROZEN_BEFORE_TARGET_ADAPTED_WEIGHT_UPDATE"
  and .model == "Qwen/Qwen3.5-9B"
  and .target_development_dataset.train_examples == 4500
  and .target_development_dataset.validation_examples == 1500
  and (.target_development_dataset.target_domains | sort)
      == (["alfworld", "discoveryworld", "tirbench", "video", "webshop"] | sort)
  and .training.max_steps == 600
  and .training.learning_rate == 0.000005
  and .training.seed == 20260902
' "${PROTOCOL}" >/dev/null
jq -e '
  .status == "FROZEN_FIVE_TARGET_DEVELOPMENT_HARNESS_SUPERVISION"
  and ([.gates[]] | all)
  and .summary.examples == 6000
  and .target_data_used == true
  and .target_outcome_used_for_controller_labels == false
  and .formal_or_qualification_targets_used == false
  and .video_target_data_used == true
' "${DATASET_DIR}/manifest.json" >/dev/null
jq -e '.status == "PASS" and ([.gates[]] | all) and .token_lengths.over_maximum == 0' \
  "${DATASET_DIR}/independent_audit.json" >/dev/null

[[ "$(sha256sum "${INITIAL_ADAPTER}/adapter_model.safetensors" | cut -d' ' -f1)" == \
   "$(jq -r '.initial_adapter.adapter_model_sha256' "${PROTOCOL}")" ]] || {
  echo "ERROR source-only initial adapter drifted" >&2; exit 2;
}
[[ "$(sha256sum "${DATASET_DIR}/manifest.json" | cut -d' ' -f1)" == \
   "$(jq -r '.target_development_dataset.manifest_sha256' "${PROTOCOL}")" ]]
[[ "$(sha256sum "${DATASET_DIR}/independent_audit.json" | cut -d' ' -f1)" == \
   "$(jq -r '.target_development_dataset.independent_audit_sha256' "${PROTOCOL}")" ]]
[[ "$(sha256sum "${DATASET_DIR}/train.jsonl" | cut -d' ' -f1)" == \
   "$(jq -r '.target_development_dataset.train_sha256' "${PROTOCOL}")" ]]
[[ "$(sha256sum "${DATASET_DIR}/validation.jsonl" | cut -d' ' -f1)" == \
   "$(jq -r '.target_development_dataset.validation_sha256' "${PROTOCOL}")" ]]

mkdir -p "${LAUNCH_DIR}"
scontrol show job "${SLURM_JOB_ID}" >"${LAUNCH_DIR}/slurm_job_at_start.txt"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv \
  >"${LAUNCH_DIR}/nvidia_smi_start.txt"
git -C "${REPO_ROOT}" rev-parse HEAD >"${LAUNCH_DIR}/git_head_at_start.txt"
git -C "${REPO_ROOT}" status --short >"${LAUNCH_DIR}/git_status_at_start.txt"
sha256sum "${PROTOCOL}" "${DATASET_DIR}/manifest.json" \
  "${DATASET_DIR}/independent_audit.json" "${DATASET_DIR}/train.jsonl" \
  "${DATASET_DIR}/validation.jsonl" "${INITIAL_ADAPTER}/adapter_model.safetensors" \
  "${SCALAR_HELDOUT}" "${MULTI_IR_HELDOUT}" >"${LAUNCH_DIR}/input_sha256.txt"

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
  --max-length 2048 \
  --max-steps "${MAX_STEPS}" \
  --gradient-accumulation 8 \
  --learning-rate "${LEARNING_RATE}" \
  --lora-r 16 \
  --lora-alpha 32 \
  --seed "${SEED}" \
  2>&1 | tee "${LAUNCH_DIR}/train.log"

python scripts/evaluate_harness_controller_lora_v2.py \
  --dataset "${DATASET_DIR}/validation.jsonl" \
  --adapter "${OUTPUT_DIR}/adapter" \
  --output "${OUTPUT_DIR}/five_domain_development_report.json" \
  --model "${MODEL}" \
  --evaluation-kind target_development_validation \
  --balance-field target_domain_audit_only \
  --audit-field target_eval_group_audit_only \
  --per-decision 10000 --batch-size 4 --max-input-length 1792 --max-new-tokens 256 \
  2>&1 | tee "${LAUNCH_DIR}/five_domain_evaluate.log"

python scripts/evaluate_harness_controller_lora_v2.py \
  --dataset "${SCALAR_HELDOUT}" \
  --adapter "${OUTPUT_DIR}/adapter" \
  --output "${OUTPUT_DIR}/source_scalar_retention_report.json" \
  --model "${MODEL}" \
  --evaluation-kind source_family_heldout \
  --balance-field candidate_count_audit_only \
  --per-decision 10000 --batch-size 4 --max-input-length 1792 --max-new-tokens 256 \
  2>&1 | tee "${LAUNCH_DIR}/source_scalar_evaluate.log"

python scripts/evaluate_harness_controller_lora_v2.py \
  --dataset "${MULTI_IR_HELDOUT}" \
  --adapter "${OUTPUT_DIR}/adapter" \
  --output "${OUTPUT_DIR}/source_multi_ir_retention_report.json" \
  --model "${MODEL}" \
  --evaluation-kind source_multi_ir_heldout \
  --balance-field source_ir_kind_audit_only \
  --audit-field control_variant_audit_only \
  --per-decision 10000 --batch-size 4 --max-input-length 1792 --max-new-tokens 128 \
  2>&1 | tee "${LAUNCH_DIR}/source_multi_ir_evaluate.log"

jq -e '.status == "TARGET_DEVELOPMENT_CONTROLLER_GATE_PASSED" and ([.gates[]] | all)' \
  "${OUTPUT_DIR}/five_domain_development_report.json" >/dev/null
jq -e '.status == "SOURCE_FAMILY_HELD_OUT_CONTROLLER_GATE_PASSED" and ([.gates[]] | all)' \
  "${OUTPUT_DIR}/source_scalar_retention_report.json" >/dev/null
jq -e '.status == "SOURCE_MULTI_IR_HELD_OUT_CONTROLLER_GATE_PASSED" and ([.gates[]] | all)' \
  "${OUTPUT_DIR}/source_multi_ir_retention_report.json" >/dev/null

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
  >"${LAUNCH_DIR}/nvidia_smi_end.txt"
echo "FIVE_DOMAIN_TARGET_ADAPTED_HARNESS_AND_SOURCE_RETENTION_GATES_PASSED" \
  >"${OUTPUT_DIR}/scientific_status.txt"
echo "HARNESS_CONTROLLER_QWEN35_9B_FIVE_DOMAIN_TARGET_ADAPTED_COMPLETE ${OUTPUT_DIR}"
