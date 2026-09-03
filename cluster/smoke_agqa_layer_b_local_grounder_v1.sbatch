#!/usr/bin/env bash
#SBATCH --job-name=agqa-layerb-vlm
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
python scripts/smoke_agqa_layer_b_local_grounder.py \
  --data runs/agqa2_layer_b_raw_video_v1/semantic_parser_data/validation.jsonl \
  --video-root /fs/gamma-projects/vlm-robot/datasets/STAR-official/videos/charades \
  --model Qwen/Qwen3.5-9B \
  --output runs/agqa2_layer_b_raw_video_v1/local_grounder_smoke_v1.json
