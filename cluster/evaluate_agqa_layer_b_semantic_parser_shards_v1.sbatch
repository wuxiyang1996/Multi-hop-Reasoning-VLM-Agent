#!/usr/bin/env bash
#SBATCH --job-name=agqa-layerb-sem-shard
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
START=$((SLURM_ARRAY_TASK_ID * 7735))
COUNT=7735
if [[ "${SLURM_ARRAY_TASK_ID}" == "3" ]]; then COUNT=7734; fi
python scripts/evaluate_agqa_layer_b_semantic_parser.py \
  --data-dir runs/agqa2_layer_b_raw_video_v1/semantic_parser_data \
  --model-dir runs/agqa2_layer_b_raw_video_v1/semantic_parser_flan_t5_small_v1/final \
  --start-row "${START}" --max-rows "${COUNT}" \
  --output "runs/agqa2_layer_b_raw_video_v1/semantic_parser_flan_t5_small_v1/heldout_shard_${SLURM_ARRAY_TASK_ID}.json"
