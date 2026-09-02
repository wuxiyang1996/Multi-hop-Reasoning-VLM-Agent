#!/usr/bin/env bash
#SBATCH --job-name=agqa-ftb4-fallback
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=runs/agqa2_full_train_broad_powered_v4/fallback-%j.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"; export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
python scripts/collect_agqa_layer_b_shared_fallback.py \
  --cohort runs/agqa2_full_train_broad_powered_v4/parser_qualified_reserve/public_cohort.json \
  --grounding runs/agqa2_full_train_broad_powered_v4/qwen32_grounding_full1790.json \
  --model Qwen/Qwen3.5-9B --output runs/agqa2_full_train_broad_powered_v4/shared_fallback_full1790.json
