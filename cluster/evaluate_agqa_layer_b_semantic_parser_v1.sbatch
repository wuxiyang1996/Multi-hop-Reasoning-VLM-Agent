#!/usr/bin/env bash
#SBATCH --job-name=agqa-layerb-sem-eval
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python scripts/evaluate_agqa_layer_b_semantic_parser.py \
  --data-dir runs/agqa2_layer_b_raw_video_v1/semantic_parser_data \
  --model-dir runs/agqa2_layer_b_raw_video_v1/semantic_parser_flan_t5_small_v1/final \
  --max-rows "${MAX_ROWS:-10000}" \
  --output "${OUTPUT:-runs/agqa2_layer_b_raw_video_v1/semantic_parser_flan_t5_small_v1/diagnostic_10000.json}"
