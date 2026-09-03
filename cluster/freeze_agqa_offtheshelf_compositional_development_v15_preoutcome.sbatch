#!/usr/bin/env bash
#SBATCH --job-name=agqa-otc-v15-pre
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/agqa_otc_v15_pre_%j.out
#SBATCH --error=logs/agqa_otc_v15_pre_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="${AGQA_RUN:-runs/agqa2_offtheshelf_compositional_development_v15}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

python scripts/freeze_agqa_offtheshelf_compositional_development_preoutcome_v15.py \
  --cohort "${RUN}/public_cohort.json" \
  --semantic-runtime "${RUN}/semantic_runtime.json" \
  --query-grounding "${RUN}/query_grounder_compositional_v15c.json" \
  --grounding "${RUN}/shared_layer_b_grounding_v15c.json" \
  --fallback "${RUN}/shared_fallback_qwen9b_v15c.json" \
  --source-capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json \
  --anonymous-controller runs/anonymous_video_harness_v1/controller.json \
  --action-grounding "${RUN}/slowfast_all_scores.json" \
  --slowfast-bindings "${RUN}/slowfast_action_bindings.json" \
  --minimum-source-commit-fraction 0.20 \
  --maximum-permuted-commit-fraction 0.05 \
  --minimum-disagreement-fraction 0.05 \
  --minimum-two-event-fraction 0.50 \
  --output "${RUN}/five_arm_preoutcome_v15c.json"
