#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2-dev-freeze
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=runs/agqa2_query_grounder_v2_development/freeze-%j.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"

python scripts/freeze_agqa_full_train_broad_v1.py \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/train_balanced.txt \
  --video-root /fs/gamma-projects/vlm-robot/datasets/AGQA2-charades-selected \
  --parser-supervision runs/agqa2_layer_b_raw_video_v1/semantic_parser_data/train.jsonl \
  --output-dir runs/agqa2_query_grounder_v2_development/frozen_cohort \
  --videos 40 --tasks-per-stratum-video 2 \
  --salt agqa-query-grounder-v2-development-20260902
