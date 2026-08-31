#!/usr/bin/env bash
#SBATCH --job-name=harness-ctrl9b-v3
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
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/runs/harness_controller_sft_v3}"
EVAL_DATASET="${EVAL_DATASET:-${DATASET_DIR}/model_unseen_eval.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-${DATASET_DIR}/model_unseen_eval.manifest.json}"
INITIAL_ADAPTER="${INITIAL_ADAPTER:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_v2/sft_v2/adapter}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_v3/continued_sft_v1}"
MAX_STEPS="${MAX_STEPS:-120}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
SCRATCH_ROOT="/scratch0/${USER}/harness-controller-v3-${SLURM_JOB_ID}"
LOCAL_HF_HOME="${SCRATCH_ROOT}/hf"
LAUNCH_DIR="${OUTPUT_DIR}.launch"

[[ -f "${DATASET_DIR}/manifest.json" ]] || {
  echo "ERROR missing controller dataset manifest" >&2
  exit 2
}
[[ -f "${EVAL_DATASET}" && -f "${EVAL_MANIFEST}" ]] || {
  echo "ERROR missing frozen model-unseen evaluation inputs" >&2
  exit 2
}
[[ -f "${INITIAL_ADAPTER}/adapter_model.safetensors" ]] || {
  echo "ERROR missing initial adapter ${INITIAL_ADAPTER}" >&2
  exit 2
}
[[ ! -e "${OUTPUT_DIR}" && ! -e "${LAUNCH_DIR}" ]] || {
  echo "ERROR refusing to overwrite ${OUTPUT_DIR} or ${LAUNCH_DIR}" >&2
  exit 2
}
jq -e '
  .status == "FROZEN_SOURCE_ONLY_CONTROLLER_SUPERVISION"
  and ([.gates[]] | all)
  and .prompt_policy.game_identity_exposed == false
  and .prompt_policy.native_actions_exposed == false
  and .prompt_policy.target_action_authority == false
  and .summary.target_data_used == false
' "${DATASET_DIR}/manifest.json" >/dev/null
jq -e '
  .status == "FROZEN_BEFORE_V3_ADAPTER_TRAINING"
  and .schema_version == "harness-controller-model-unseen-eval-v3"
  and .prompt_overlap_with_exclusions == 0
  and .target_data_used == false
  and .examples >= 20
' "${EVAL_MANIFEST}" >/dev/null
[[ "$(sha256sum "${EVAL_DATASET}" | cut -d' ' -f1)" == \
   "$(jq -r '.output_file_sha256' "${EVAL_MANIFEST}")" ]] || {
  echo "ERROR frozen evaluation file no longer matches its manifest" >&2
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
  "${EVAL_DATASET}" \
  "${EVAL_MANIFEST}" \
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
        "${SCRATCH_ROOT}" == "/scratch0/${USER}/harness-controller-v3-${SLURM_JOB_ID}" ]]; then
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
  --dataset "${EVAL_DATASET}" \
  --adapter "${OUTPUT_DIR}/adapter" \
  --output "${OUTPUT_DIR}/model_unseen_source_family_report.json" \
  --model "${MODEL}" \
  --per-decision 24 \
  --batch-size 4 \
  --max-input-length 1792 \
  --max-new-tokens 256 \
  2>&1 | tee "${LAUNCH_DIR}/evaluate.log"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
  >"${LAUNCH_DIR}/nvidia_smi_end.txt"
jq -r '.status' "${OUTPUT_DIR}/model_unseen_source_family_report.json" \
  >"${OUTPUT_DIR}/scientific_status.txt"
echo "HARNESS_CONTROLLER_QWEN35_9B_V3_COMPLETE ${OUTPUT_DIR}"
