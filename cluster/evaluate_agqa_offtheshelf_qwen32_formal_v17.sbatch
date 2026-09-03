#!/usr/bin/env bash
#SBATCH --job-name=agqa-q32-formal-eval
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/agqa_q32_formal_v17_eval_%j.out
#SBATCH --error=logs/agqa_q32_formal_v17_eval_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_offtheshelf_qwen32_formal_v17b"
PROTOCOL="runs/agqa2_offtheshelf_qwen32_formal_protocol_v17b/protocol.json"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

python scripts/evaluate_agqa_offtheshelf_compositional_development_v15.py \
  --cohort "${RUN}/public_cohort.json" \
  --preoutcome "${RUN}/qwen32_five_arm_preoutcome_v17b.json" \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/train_balanced.txt \
  --negative-transfer-fraction 0.05 \
  --formal-protocol "${PROTOCOL}" \
  --formal-manifest "${RUN}/manifest.json" \
  --output "${RUN}/qwen32_formal_evaluation_v17b.json"
