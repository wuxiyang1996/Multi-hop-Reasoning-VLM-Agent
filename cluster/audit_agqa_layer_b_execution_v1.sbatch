#!/usr/bin/env bash
#SBATCH --job-name=agqa-layerb-audit
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
COHORT_NAME="${COHORT_NAME:-development_v1}"
GROUNDING_FILE="${GROUNDING_FILE:-raw_video_grounding.json}"
AUDIT_FILE="${AUDIT_FILE:-intrinsic_execution_audit.json}"
python scripts/audit_agqa_layer_b_execution.py \
  --grounding "runs/agqa2_layer_b_raw_video_v1/${COHORT_NAME}/${GROUNDING_FILE}" \
  --semantic-runtime "runs/agqa2_layer_b_raw_video_v1/${COHORT_NAME}/semantic_runtime.json" \
  --source-capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json \
  --output "runs/agqa2_layer_b_raw_video_v1/${COHORT_NAME}/${AUDIT_FILE}"
