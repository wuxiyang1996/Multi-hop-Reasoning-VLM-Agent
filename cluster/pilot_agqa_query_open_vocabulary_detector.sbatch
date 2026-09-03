#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2-gdino
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=runs/agqa2_query_grounder_v2_development/gdino-%j.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export HF_HUB_OFFLINE=1

python scripts/pilot_agqa_query_open_vocabulary_detector.py \
  --cohort runs/agqa2_query_grounder_v2_development/frozen_cohort/public_cohort.json \
  --localized-grounding runs/agqa2_query_grounder_v2_development/pilot_v4.json \
  --inspection-frames 16 --box-threshold 0.18 \
  --output runs/agqa2_query_grounder_v2_development/gdino_pilot_v1.json
