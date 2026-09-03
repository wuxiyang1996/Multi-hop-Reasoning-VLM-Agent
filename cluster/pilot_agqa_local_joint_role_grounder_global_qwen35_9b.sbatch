#!/bin/bash
#SBATCH --job-name=agqa-role-global9b
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_role_global9b_%j.out
#SBATCH --error=logs/agqa_role_global9b_%j.err
set -euo pipefail
cd /fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
export PYTHONPATH=src:.
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export HF_HUB_OFFLINE=1
python scripts/pilot_agqa_local_joint_role_grounder.py \
  --cohort runs/agqa2_query_grounder_v2_development/frozen_cohort/public_cohort.json \
  --detector-grounding runs/agqa2_query_grounder_v2_development/gdino_full_dev_v1.json \
  --max-tasks 5 \
  --output runs/agqa2_query_grounder_v2_development/local_joint_role_global_pilot_qwen35_9b_v1.json
