#!/usr/bin/env bash
#SBATCH --job-name=agqa-q32-formal-dl
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/agqa_q32_formal_v17b_download_%j.out
#SBATCH --error=logs/agqa_q32_formal_v17b_download_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_offtheshelf_qwen32_formal_v17b"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

python scripts/download_agqa2_active_grounding_v4_reserve.py \
  --selection "${RUN}/download_selection.json" \
  --receipt "${RUN}/download_receipt.json"
