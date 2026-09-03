#!/usr/bin/env bash
#SBATCH --job-name=agqa-layerb-target
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
python scripts/repair_agqa_layer_b_targeted_actions.py \
  --cohort runs/agqa2_layer_b_raw_video_v1/acquisition_v2/public_cohort.json \
  --input runs/agqa2_layer_b_raw_video_v1/acquisition_v2/qwen32_grounding_48f_pilot_v1.json \
  --keys /fs/gamma-projects/vlm-robot/keys.py \
  --model qwen/qwen3-vl-32b-instruct \
  --cache-dir runs/agqa2_layer_b_raw_video_v1/acquisition_v2/qwen32_targeted_action_cache_v1 \
  --output runs/agqa2_layer_b_raw_video_v1/acquisition_v2/qwen32_grounding_48f_targeted_pilot_v1.json
