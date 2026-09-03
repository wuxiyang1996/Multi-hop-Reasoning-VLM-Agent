#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2sb-formal-freeze
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_qgv2sb_formal_freeze_%j.out
#SBATCH --error=logs/agqa_qgv2sb_formal_freeze_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="${AGQA_RUN:?AGQA_RUN is required}"
PROTOCOL="${AGQA_PROTOCOL:?AGQA_PROTOCOL is required}"
QUALIFICATION="${AGQA_QUALIFICATION:?AGQA_QUALIFICATION is required}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

python scripts/freeze_agqa_query_grounder_v2_strict_boundary_preoutcome.py \
  --protocol "${PROTOCOL}" \
  --manifest "${RUN}/frozen_cohort/manifest.json" \
  --qualification "${QUALIFICATION}" \
  --cohort "${RUN}/frozen_cohort/public_cohort.json" \
  --semantic-runtime "${RUN}/semantic_runtime.json" \
  --query-grounding "${RUN}/query_grounder_candidates_frozen.json" \
  --grounding "${RUN}/shared_layer_b_grounding.json" \
  --fallback "${RUN}/shared_fallback_qwen9b.json" \
  --action-grounding "${RUN}/slowfast_all_scores.json" \
  --slowfast-bindings "${RUN}/slowfast_action_bindings.json" \
  --source-capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json \
  --anonymous-controller runs/anonymous_video_harness_v1/controller.json \
  --output "${RUN}/preoutcome_receipt.json"
