#!/usr/bin/env bash
#SBATCH --job-name=agqa-sf-rel-eval
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/agqa_sf_rel_eval_%j.out
#SBATCH --error=logs/agqa_sf_rel_eval_%j.err
set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
python scripts/evaluate_agqa_relation_phrase_grounder.py \
  --grounding runs/agqa2_query_grounder_v2_development/slowfast_relation_phrase_pilot_v1.json \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --output runs/agqa2_query_grounder_v2_development/slowfast_relation_phrase_pilot_v1_evaluation.json
