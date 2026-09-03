#!/usr/bin/env bash
#SBATCH --job-name=agqa-q32-pre-v16
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/agqa_q32_pre_v16_%j.out
#SBATCH --error=logs/agqa_q32_pre_v16_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_offtheshelf_compositional_development_v15"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

python scripts/freeze_agqa_offtheshelf_qwen32_multiview_development_preoutcome_v16.py \
  --cohort "${RUN}/public_cohort.json" \
  --semantic-runtime "${RUN}/semantic_runtime.json" \
  --routed-grounding "${RUN}/qwen32_48f_96f_routed_v16.json" \
  --grounding-view "${RUN}/qwen32_48f_full_v16.json" \
  --grounding-view "${RUN}/qwen32_96f_full_v16.json" \
  --fallback "${RUN}/qwen32_shared_fallback_qwen9b_v16.json" \
  --source-capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json \
  --anonymous-controller runs/anonymous_video_harness_v1/controller.json \
  --minimum-source-commit-fraction 0.20 \
  --maximum-permuted-commit-fraction 0.05 \
  --minimum-disagreement-fraction 0.05 \
  --output "${RUN}/qwen32_five_arm_preoutcome_v16.json"
