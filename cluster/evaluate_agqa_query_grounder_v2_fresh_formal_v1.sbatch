#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2-formal-eval
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_qgv2_formal_eval_%j.out
#SBATCH --error=logs/agqa_qgv2_formal_eval_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_query_grounder_v2_fresh_formal_v1"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

python scripts/evaluate_agqa_query_grounder_v2_fresh_formal.py \
  --protocol configs/agqa2_query_grounder_v2_fresh_formal_v1.json \
  --manifest "${RUN}/frozen_cohort/manifest.json" \
  --cohort "${RUN}/frozen_cohort/public_cohort.json" \
  --preoutcome "${RUN}/preoutcome_receipt.json" \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/train_balanced.txt \
  --output "${RUN}/formal_evaluation.json"
