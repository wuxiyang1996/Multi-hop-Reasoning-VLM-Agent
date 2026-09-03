#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2teq-eval
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_qgv2teq_eval_%j.out
#SBATCH --error=logs/agqa_qgv2teq_eval_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="${AGQA_RUN:-runs/agqa2_query_grounder_v2_typed_evidence_qualification_v13}"
PROTOCOL="${AGQA_PROTOCOL:-configs/agqa2_query_grounder_v2_typed_evidence_qualification_v13.json}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

python scripts/evaluate_agqa_query_grounder_v2_strict_boundary_qualification.py \
  --grounding "${RUN}/query_grounder_candidates_frozen.json" \
  --parent-grounding "${RUN}/query_grounder_candidates_strict_parent.json" \
  --preoutcome-audit "${RUN}/preoutcome_coverage.json" \
  --action-grounding "${RUN}/slowfast_all_scores.json" \
  --slowfast-bindings "${RUN}/slowfast_action_bindings.json" \
  --protocol "${PROTOCOL}" \
  --manifest "${RUN}/frozen_cohort/manifest.json" \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/train_balanced.txt \
  --output "${RUN}/qualification_evaluation.json"
