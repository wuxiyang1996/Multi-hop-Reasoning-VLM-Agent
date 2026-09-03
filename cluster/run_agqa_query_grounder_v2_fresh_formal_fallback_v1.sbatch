#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2-formal-9b
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/agqa_qgv2_formal_fallback_%j.out
#SBATCH --error=logs/agqa_qgv2_formal_fallback_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="${AGQA_RUN:-runs/agqa2_query_grounder_v2_fresh_formal_v1}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

python scripts/collect_agqa_layer_b_shared_fallback.py \
  --cohort "${RUN}/frozen_cohort/public_cohort.json" \
  --grounding "${RUN}/shared_layer_b_grounding.json" \
  --model Qwen/Qwen3.5-9B \
  --output "${RUN}/shared_fallback_qwen9b.json"
