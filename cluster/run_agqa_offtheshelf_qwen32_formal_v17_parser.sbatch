#!/usr/bin/env bash
#SBATCH --job-name=agqa-q32-formal-parser
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_q32_formal_v17_parser_%j.out
#SBATCH --error=logs/agqa_q32_formal_v17_parser_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_offtheshelf_qwen32_formal_v17b"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python scripts/run_agqa_layer_b_semantic_parser.py \
  --cohort "${RUN}/public_cohort.json" \
  --model-dir runs/agqa2_layer_b_raw_video_v1/semantic_parser_flan_t5_small_v1/final \
  --qualification runs/agqa2_layer_b_raw_video_v1/semantic_parser_flan_t5_small_v1/heldout_evaluation.json \
  --output "${RUN}/semantic_runtime.json" --batch-size 32
