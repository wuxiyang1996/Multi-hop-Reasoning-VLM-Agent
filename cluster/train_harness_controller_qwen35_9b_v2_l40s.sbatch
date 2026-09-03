#!/usr/bin/env bash
#SBATCH --job-name=harness-ctrl9b-v2
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
SOURCE_RUNTIME_REPO="${SOURCE_RUNTIME_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/runs/harness_controller_sft_v2_final}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_v2/sft_v2}"
MAX_STEPS="${MAX_STEPS:-240}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
SCRATCH_ROOT="/scratch0/${USER}/harness-controller-v2-${SLURM_JOB_ID}"
LOCAL_HF_HOME="${SCRATCH_ROOT}/hf"

[[ -f "${DATASET_DIR}/manifest.json" ]] || {
  echo "ERROR missing controller dataset manifest" >&2
  exit 2
}
[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "ERROR refusing to overwrite ${OUTPUT_DIR}" >&2
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
for value in "${MAX_STEPS}" "${MAX_LENGTH}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR invalid positive integer: ${value}" >&2
    exit 2
  }
done

mkdir -p "${OUTPUT_DIR}.launch" "${SCRATCH_ROOT}"
scontrol show job "${SLURM_JOB_ID}" >"${OUTPUT_DIR}.launch/slurm_job_at_start.txt"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv \
  >"${OUTPUT_DIR}.launch/nvidia_smi_start.txt"
sha256sum "${DATASET_DIR}/manifest.json" \
  >"${OUTPUT_DIR}.launch/dataset_manifest_sha256.txt"
git -C "${REPO_ROOT}" rev-parse HEAD >"${OUTPUT_DIR}.launch/git_head_at_start.txt"
git -C "${REPO_ROOT}" status --short >"${OUTPUT_DIR}.launch/git_status_at_start.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ -d "${SCRATCH_ROOT}" && "${SCRATCH_ROOT}" == "/scratch0/${USER}/harness-controller-v2-${SLURM_JOB_ID}" ]]; then
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
  --max-length "${MAX_LENGTH}" \
  --max-steps "${MAX_STEPS}" \
  --gradient-accumulation 8 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  2>&1 | tee "${OUTPUT_DIR}.launch/train.log"

python scripts/evaluate_harness_controller_lora_v2.py \
  --dataset "${DATASET_DIR}/source_held_out.jsonl" \
  --adapter "${OUTPUT_DIR}/adapter" \
  --output "${OUTPUT_DIR}/source_family_heldout_report.json" \
  --model "${MODEL}" \
  --per-decision 24 \
  --batch-size 4 \
  --max-input-length 1792 \
  --max-new-tokens 256 \
  2>&1 | tee "${OUTPUT_DIR}.launch/evaluate.log"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
  >"${OUTPUT_DIR}.launch/nvidia_smi_end.txt"
jq -r '.status' "${OUTPUT_DIR}/source_family_heldout_report.json" \
  >"${OUTPUT_DIR}/scientific_status.txt"
echo "HARNESS_CONTROLLER_QWEN35_9B_V2_COMPLETE ${OUTPUT_DIR}"
