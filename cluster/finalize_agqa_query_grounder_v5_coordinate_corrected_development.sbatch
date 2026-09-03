#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv5-final
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_qgv5_final_%j.out
#SBATCH --error=logs/agqa_qgv5_final_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_query_grounder_v5_coordinate_corrected_development"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

merge_args=()
for shard in 0 1 2 3 4 5 6 7; do
  merge_args+=(--shard "${RUN}/slowfast_all_scores_shard${shard}.json")
done
python scripts/merge_agqa_layer_b_action_probe_shards.py \
  --cohort "${RUN}/frozen_cohort/public_cohort.json" \
  --expected-grounding "${RUN}/frozen_cohort/public_cohort.json" \
  "${merge_args[@]}" --output "${RUN}/slowfast_all_scores.json"

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
  --minimum-object-score 0.05 --minimum-candidate-confidence 0.0 \
  --sgdet-rank-weight 0.60 \
  --output "${RUN}/query_grounder_candidates_frozen.json"

# This is the first command allowed to open development answers.  A failed
# gate intentionally returns non-zero after preserving the immutable report.
python scripts/evaluate_agqa_query_grounder_v5_development.py \
  --grounding "${RUN}/query_grounder_candidates_frozen.json" \
  --protocol configs/agqa2_query_grounder_v5_coordinate_corrected_calibration.json \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/train_balanced.txt \
  --output "${RUN}/stage_a_development_evaluation.json"
