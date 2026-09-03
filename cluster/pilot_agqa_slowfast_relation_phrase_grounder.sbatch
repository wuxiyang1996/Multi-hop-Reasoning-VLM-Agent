#!/usr/bin/env bash
#SBATCH --job-name=agqa-sf-relphrase
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_sf_relphrase_%j.out
#SBATCH --error=logs/agqa_sf_relphrase_%j.err
set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export HF_HUB_OFFLINE=1
python scripts/pilot_agqa_slowfast_relation_phrase_grounder.py \
  --cohort runs/agqa2_query_grounder_v2_development/frozen_cohort/public_cohort.json \
  --semantic-runtime runs/agqa2_query_grounder_v2_development/semantic_runtime.json \
  --action-grounding runs/agqa2_query_grounder_v2_development/slowfast_uniform48_full80_v1.json \
  --max-tasks 5 --output runs/agqa2_query_grounder_v2_development/slowfast_relation_phrase_pilot_v1.json
