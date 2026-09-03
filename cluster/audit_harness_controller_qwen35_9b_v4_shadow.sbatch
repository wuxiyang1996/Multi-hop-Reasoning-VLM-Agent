#!/usr/bin/env bash
#SBATCH --job-name=ctrl9b-v4-shadow
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_v4/cardinality_sft_v1}"
LAUNCH_DIR="${OUTPUT_ROOT}.shadow-audit"

[[ -f "${OUTPUT_ROOT}/fresh_target_report.json" ]] || {
  echo "ERROR target report was not produced by the parent job" >&2
  exit 3
}
[[ ! -e "${OUTPUT_ROOT}/fresh_target_shadow_audit.json" ]] || {
  echo "ERROR refusing to overwrite existing shadow audit" >&2
  exit 2
}
mkdir -p "${LAUNCH_DIR}"
scontrol show job "${SLURM_JOB_ID}" >"${LAUNCH_DIR}/slurm_job_at_start.txt"
sha256sum \
  "${REPO_ROOT}/src/motif_transfer/neural_harness_controller.py" \
  "${REPO_ROOT}/scripts/audit_harness_controller_v4_shadow.py" \
  "${OUTPUT_ROOT}/fresh_target_activation.json" \
  "${OUTPUT_ROOT}/fresh_target_report.json" \
  "${OUTPUT_ROOT}/fresh_target_report.predictions.jsonl" \
  >"${LAUNCH_DIR}/input_sha256.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/audit_harness_controller_v4_shadow.py \
  2>&1 | tee "${LAUNCH_DIR}/audit.log"

echo "HARNESS_CONTROLLER_QWEN35_9B_V4_SHADOW_AUDIT_COMPLETE ${OUTPUT_ROOT}"
