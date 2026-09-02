#!/usr/bin/env bash
#SBATCH --job-name=agqa-layerb-q235
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
MODEL="${MODEL:-qwen/qwen3-vl-235b-a22b-instruct}"
POSITIONS="${POSITIONS:-0,4,8,12,16,20,24,28}"
OUTPUT_NAME="${OUTPUT_NAME:-qwen235_grounding_pilot_v1.json}"
CACHE_NAME="${CACHE_NAME:-qwen235_pilot_call_cache}"
FRAME_COUNT="${FRAME_COUNT:-24}"
python scripts/collect_agqa_layer_b_qwen235_grounding.py \
  --cohort runs/agqa2_layer_b_raw_video_v1/acquisition_v2/public_cohort.json \
  --semantic-runtime runs/agqa2_layer_b_raw_video_v1/acquisition_v2/semantic_runtime.json \
  --keys /fs/gamma-projects/vlm-robot/keys.py \
  --model "${MODEL}" \
  --positions "${POSITIONS}" \
  --frame-count "${FRAME_COUNT}" \
  --cache-dir "runs/agqa2_layer_b_raw_video_v1/acquisition_v2/${CACHE_NAME}" \
  --output "runs/agqa2_layer_b_raw_video_v1/acquisition_v2/${OUTPUT_NAME}"
