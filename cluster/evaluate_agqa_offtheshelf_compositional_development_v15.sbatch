#!/usr/bin/env bash
#SBATCH --job-name=agqa-otc-v15-eval
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/agqa_otc_v15_eval_%j.out
#SBATCH --error=logs/agqa_otc_v15_eval_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="${AGQA_RUN:-runs/agqa2_offtheshelf_compositional_development_v15}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

python scripts/evaluate_agqa_offtheshelf_compositional_development_v15.py \
  --cohort "${RUN}/public_cohort.json" \
  --preoutcome "${RUN}/five_arm_preoutcome_v15c.json" \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/train_balanced.txt \
  --negative-transfer-fraction 0.05 \
  --output "${RUN}/development_evaluation_v15c.json"
