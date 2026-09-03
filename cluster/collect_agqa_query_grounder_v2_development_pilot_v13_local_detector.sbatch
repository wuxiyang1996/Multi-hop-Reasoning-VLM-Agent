#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2-p13-local
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=runs/agqa2_query_grounder_v2_development/pilot-v13-local-%j.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export HF_HUB_OFFLINE=1

python scripts/collect_agqa_query_grounder_v2.py \
  --cohort runs/agqa2_query_grounder_v2_development/frozen_cohort/public_cohort.json \
  --semantic-runtime runs/agqa2_query_grounder_v2_development/semantic_runtime.json \
  --keys /fs/gamma-projects/vlm-robot/keys.py --model qwen/qwen3-vl-32b-instruct \
  --frame-count 96 --anchor-frame-count 32 --open-vocab-inventory \
  --detector-inspection-frames 8 --detector-box-threshold 0.12 --detector-ontology-chunk-size 6 \
  --localized-frames-per-panel 8 --localized-panel-frame-width 160 --positions 0,1,2,3,4 \
  --cache-dir runs/agqa2_query_grounder_v2_development/cache_v13_local_detector \
  --output runs/agqa2_query_grounder_v2_development/pilot_v13_local_detector.json
