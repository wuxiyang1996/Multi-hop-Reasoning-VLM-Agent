#!/usr/bin/env bash
#SBATCH --job-name=agqa-q32-formal-pre
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/agqa_q32_formal_v17_pre_%j.out
#SBATCH --error=logs/agqa_q32_formal_v17_pre_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_offtheshelf_qwen32_formal_v17b"
PROTOCOL="runs/agqa2_offtheshelf_qwen32_formal_protocol_v17b/protocol.json"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

python scripts/freeze_agqa_offtheshelf_qwen32_multiview_development_preoutcome_v16.py \
  --cohort "${RUN}/public_cohort.json" \
  --semantic-runtime "${RUN}/semantic_runtime.json" \
  --routed-grounding "${RUN}/qwen32_48f_96f_routed_v17b.json" \
  --grounding-view "${RUN}/qwen32_48f_full_v17b.json" \
  --grounding-view "${RUN}/qwen32_96f_full_v17b.json" \
  --fallback "${RUN}/qwen32_shared_fallback_qwen9b_v17b.json" \
  --source-capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json \
  --anonymous-controller runs/anonymous_video_harness_v1/controller.json \
  --minimum-source-commit-fraction 0.20 \
  --maximum-permuted-commit-fraction 0.05 \
  --minimum-disagreement-fraction 0.05 \
  --executor-version v3 \
  --formal-protocol "${PROTOCOL}" \
  --formal-manifest "${RUN}/manifest.json" \
  --download-receipt "${RUN}/download_receipt.json" \
  --output "${RUN}/qwen32_five_arm_preoutcome_v17b.json"
