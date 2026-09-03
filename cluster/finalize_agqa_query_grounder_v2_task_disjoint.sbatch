#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2-td-final
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_qgv2_td_final_%j.out
#SBATCH --error=logs/agqa_qgv2_td_final_%j.err
set -Eeuo pipefail

REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_query_grounder_v2_task_disjoint_qualification_v1"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

python scripts/merge_agqa_layer_b_action_probe_shards.py \
  --cohort "${RUN}/frozen_cohort/public_cohort.json" \
  --expected-grounding "${RUN}/frozen_cohort/public_cohort.json" \
  --shard "${RUN}/slowfast_all_scores_shard0.json" \
  --shard "${RUN}/slowfast_all_scores_shard1.json" \
  --shard "${RUN}/slowfast_all_scores_shard2.json" \
  --shard "${RUN}/slowfast_all_scores_shard3.json" \
  --output "${RUN}/slowfast_all_scores.json"

python scripts/build_agqa_action_genome_sgdet_query_plans.py \
  --cohort "${RUN}/frozen_cohort/public_cohort.json" \
  --semantic-runtime "${RUN}/semantic_runtime.json" \
  --action-grounding "${RUN}/slowfast_all_scores.json" \
  --output "${RUN}/query_plans.json"

python scripts/compile_agqa_slowfast_action_bindings.py \
  --action-grounding "${RUN}/slowfast_all_scores.json" \
  --query-plans "${RUN}/query_plans.json" \
  --output "${RUN}/slowfast_action_bindings.json"

python scripts/compile_agqa_action_genome_query_grounder_v2.py \
  --cohort "${RUN}/frozen_cohort/public_cohort.json" \
  --semantic-runtime "${RUN}/semantic_runtime.json" \
  --sgdet "${RUN}/action_genome_sgdet_v2.json" \
  --query-plans "${RUN}/query_plans.json" \
  --slowfast-bindings "${RUN}/slowfast_action_bindings.json" \
  --minimum-object-score 0.05 \
  --minimum-candidate-confidence 0.675 \
  --sgdet-rank-weight 0.60 \
  --output "${RUN}/query_grounder_v2_frozen.json"

python scripts/evaluate_agqa_query_grounder_v2_qualification.py \
  --grounding "${RUN}/query_grounder_v2_frozen.json" \
  --protocol configs/agqa2_query_grounder_v2_task_disjoint_qualification_v1.json \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/test_balanced.txt \
  --split-audit "${RUN}/action_genome_split_audit.json" \
  --output "${RUN}/qualification_evaluation.json"
