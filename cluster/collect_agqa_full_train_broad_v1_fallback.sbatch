#!/usr/bin/env bash
#SBATCH --job-name=agqa-ftb-fallback
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/agqa2_full_train_broad_v1/fallback-%j.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

python scripts/collect_agqa_layer_b_shared_fallback.py \
  --cohort runs/agqa2_full_train_broad_v1/parser_qualified_reserve_v2/public_cohort.json \
  --grounding runs/agqa2_full_train_broad_v1/qwen235_grounding_full390.json \
  --model Qwen/Qwen3.5-9B \
  --output runs/agqa2_full_train_broad_v1/shared_fallback_full390.json
