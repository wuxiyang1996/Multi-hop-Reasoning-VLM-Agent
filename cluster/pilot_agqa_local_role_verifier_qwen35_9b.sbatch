#!/bin/bash
#SBATCH --job-name=agqa-role-q35-9b
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_role_q35_9b_%j.out
#SBATCH --error=logs/agqa_role_q35_9b_%j.err
set -euo pipefail
cd /fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
export PYTHONPATH=src:.
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export HF_HUB_OFFLINE=1
python scripts/pilot_agqa_local_role_verifier.py \
  --model Qwen/Qwen3.5-9B \
  --cohort runs/agqa2_query_grounder_v2_development/frozen_cohort/public_cohort.json \
  --localized-grounding runs/agqa2_query_grounder_v2_development/pilot_v4.json \
  --detector-grounding runs/agqa2_query_grounder_v2_development/gdino_pilot_v4_public_aliases.json \
  --output runs/agqa2_query_grounder_v2_development/local_role_pilot_qwen35_9b_v1.json
