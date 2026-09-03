#!/usr/bin/env bash
#SBATCH --job-name=agqa-otc-v15-9b
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_otc_v15_fallback_%j.out
#SBATCH --error=logs/agqa_otc_v15_fallback_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="${AGQA_RUN:-runs/agqa2_offtheshelf_compositional_development_v15}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

python scripts/collect_agqa_layer_b_shared_fallback.py \
  --cohort "${RUN}/public_cohort.json" \
  --grounding "${RUN}/shared_layer_b_grounding_v15c.json" \
  --model Qwen/Qwen3.5-9B \
  --output "${RUN}/shared_fallback_qwen9b_v15c.json"
