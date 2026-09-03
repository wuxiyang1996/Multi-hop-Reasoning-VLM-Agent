#!/usr/bin/env bash
#SBATCH --job-name=agqa-gdino-d48-eval
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/agqa_gdino_d48_eval_%j.out
#SBATCH --error=logs/agqa_gdino_d48_eval_%j.err
set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
python scripts/evaluate_agqa_query_open_vocabulary_detector.py \
  --grounding runs/agqa2_query_grounder_v2_development/gdino_dense48_pilot_v1.json \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/train_balanced.txt \
  --output runs/agqa2_query_grounder_v2_development/gdino_dense48_pilot_v1_evaluation.json
