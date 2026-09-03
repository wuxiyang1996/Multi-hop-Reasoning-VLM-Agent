#!/usr/bin/env bash
#SBATCH --job-name=agqa-tempura-cap
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/agqa_tempura_cap_%j.out
#SBATCH --error=logs/agqa_tempura_cap_%j.err
set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
export HF_HOME="/fs/gamma-projects/vlm-robot/huggingface"
python scripts/pilot_agqa_tempura_dense_caption.py \
  --cohort runs/agqa2_query_grounder_v2_development/frozen_cohort/public_cohort.json \
  --proposal-grounding runs/agqa2_query_grounder_v2_development/slowfast_relation_phrase_pilot_v1.json \
  --third-party /fs/gamma-projects/vlm-robot/third_party/TEMPURA-caption/src \
  --output runs/agqa2_query_grounder_v2_development/tempura_dense_caption_pilot_v1.json
