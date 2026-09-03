#!/usr/bin/env bash
#SBATCH --job-name=agqa-gdino-d48
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/agqa_gdino_d48_%j.out
#SBATCH --error=logs/agqa_gdino_d48_%j.err
set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
export HF_HOME="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache"
python scripts/pilot_agqa_query_open_vocabulary_detector.py \
  --cohort runs/agqa2_query_grounder_v2_development/frozen_cohort/public_cohort.json \
  --task-ids QNKDI-1778,QNKDI-1952,SXFG6-4268,SXFG6-4278,ZL55N-7962 \
  --inspection-frames 48 \
  --box-threshold 0.08 \
  --ontology-chunk-size 6 \
  --maximum-tracks 36 \
  --output runs/agqa2_query_grounder_v2_development/gdino_dense48_pilot_v1.json
