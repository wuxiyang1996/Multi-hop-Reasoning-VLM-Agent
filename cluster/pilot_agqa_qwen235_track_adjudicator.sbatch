#!/usr/bin/env bash
#SBATCH --job-name=agqa-q235-track
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_q235_track_%j.out
#SBATCH --error=logs/agqa_q235_track_%j.err
set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
python scripts/pilot_agqa_qwen235_track_adjudicator.py \
  --cohort runs/agqa2_query_grounder_v2_development/frozen_cohort/public_cohort.json \
  --semantic-runtime runs/agqa2_query_grounder_v2_development/semantic_runtime.json \
  --proposal-grounding runs/agqa2_query_grounder_v2_development/slowfast_relation_phrase_pilot_v1.json \
  --cache-dir runs/agqa2_query_grounder_v2_development/cache_qwen235_track_v1 \
  --output runs/agqa2_query_grounder_v2_development/qwen235_track_pilot_v1.json
