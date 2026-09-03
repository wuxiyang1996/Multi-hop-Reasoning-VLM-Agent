#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2-formal-freeze
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_qgv2_formal_freeze_%j.out
#SBATCH --error=logs/agqa_qgv2_formal_freeze_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_query_grounder_v2_fresh_formal_v1"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

python scripts/freeze_agqa_query_grounder_v2_formal_preoutcome.py \
  --protocol configs/agqa2_query_grounder_v2_fresh_formal_v1.json \
  --manifest "${RUN}/frozen_cohort/manifest.json" \
  --qualification runs/agqa2_query_grounder_v2_powered_qualification_v1/qualification_evaluation.json \
  --cohort "${RUN}/frozen_cohort/public_cohort.json" \
  --semantic-runtime "${RUN}/semantic_runtime.json" \
  --query-grounding "${RUN}/query_grounder_candidates_frozen.json" \
  --grounding "${RUN}/shared_layer_b_grounding.json" \
  --fallback "${RUN}/shared_fallback_qwen9b.json" \
  --source-capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json \
  --anonymous-controller runs/anonymous_video_harness_v1/controller.json \
  --output "${RUN}/preoutcome_receipt.json"
