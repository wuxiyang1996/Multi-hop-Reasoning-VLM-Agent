#!/usr/bin/env bash
#SBATCH --job-name=audit-alf-adapt
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
CONDA_ENV="${CONDA_ENV:-alfworld}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/target_rollout_collection_v1/alfworld_valid_unseen/adaptation}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/configs/target_rollout_collection_v1.json}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/docs/results/alfworld_target_adaptation_v1.json}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/audit_target_adaptation_rollouts.py \
  --manifest "${MANIFEST}" \
  --input-dir "${RUN_ROOT}" \
  --cell alfworld_valid_unseen \
  --output "${OUTPUT}"

echo "ALFWORLD_TARGET_ADAPTATION_V1_AUDIT_COMPLETE ${OUTPUT}"
