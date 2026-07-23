#!/usr/bin/env bash
#SBATCH --job-name=audit-source-motif-v1
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/fresh_source_execution_motif_v1}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/docs/results/fresh_source_execution_motif_v1_integrity.json}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

python scripts/audit_fresh_source_collection.py \
  "${RUN_ROOT}/gymv_strider/evidence" \
  "${RUN_ROOT}/gymv_thunder_force_iii/evidence" \
  --expected-episodes 12 \
  --output "${OUTPUT}"

echo "FRESH_SOURCE_EXECUTION_MOTIF_V1_AUDIT_COMPLETE ${OUTPUT}"
