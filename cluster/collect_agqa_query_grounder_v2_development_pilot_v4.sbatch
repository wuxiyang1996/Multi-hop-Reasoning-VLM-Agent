#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2-pilot4
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=runs/agqa2_query_grounder_v2_development/pilot-v4-%j.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"

python scripts/collect_agqa_query_grounder_v2.py \
  --cohort runs/agqa2_query_grounder_v2_development/frozen_cohort/public_cohort.json \
  --semantic-runtime runs/agqa2_query_grounder_v2_development/semantic_runtime.json \
  --keys /fs/gamma-projects/vlm-robot/keys.py \
  --model qwen/qwen3-vl-32b-instruct \
  --frame-count 96 --anchor-frame-count 32 --positions 0,1,2,3,4 \
  --cache-dir runs/agqa2_query_grounder_v2_development/cache_v4 \
  --output runs/agqa2_query_grounder_v2_development/pilot_v4.json
