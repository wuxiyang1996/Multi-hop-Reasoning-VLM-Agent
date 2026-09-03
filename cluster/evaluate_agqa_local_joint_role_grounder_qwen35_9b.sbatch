#!/bin/bash
#SBATCH --job-name=agqa-role-joint-eval
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/agqa_role_joint_eval_%j.out
#SBATCH --error=logs/agqa_role_joint_eval_%j.err
set -euo pipefail
cd /fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
export PYTHONPATH=src:.
python scripts/evaluate_agqa_local_role_verifier.py \
  --grounding runs/agqa2_query_grounder_v2_development/local_joint_role_pilot_qwen35_9b_v1.json \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --output runs/agqa2_query_grounder_v2_development/local_joint_role_pilot_qwen35_9b_v1_evaluation.json
