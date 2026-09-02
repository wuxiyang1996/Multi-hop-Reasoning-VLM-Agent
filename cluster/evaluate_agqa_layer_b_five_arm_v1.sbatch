#!/usr/bin/env bash
#SBATCH --job-name=agqa-layerb-5arm
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
COHORT_NAME="${COHORT_NAME:-development_v1}"
STAGE="${STAGE:-development}"
ENTRY="${ENTRY:-AGQA_balanced/train_balanced.txt}"
GROUNDING_FILE="${GROUNDING_FILE:-raw_video_grounding.json}"
FALLBACK_FILE="${FALLBACK_FILE:-shared_fallback.json}"
EVALUATION_FILE="${EVALUATION_FILE:-five_arm_evaluation.json}"
python scripts/evaluate_agqa_layer_b_five_arm.py \
  --cohort "runs/agqa2_layer_b_raw_video_v1/${COHORT_NAME}/public_cohort.json" \
  --grounding "runs/agqa2_layer_b_raw_video_v1/${COHORT_NAME}/${GROUNDING_FILE}" \
  --fallback "runs/agqa2_layer_b_raw_video_v1/${COHORT_NAME}/${FALLBACK_FILE}" \
  --source-capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry "${ENTRY}" \
  --stage "${STAGE}" \
  --output "runs/agqa2_layer_b_raw_video_v1/${COHORT_NAME}/${EVALUATION_FILE}"
