#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv5-qbef
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_qgv5_qbef_%j.out
#SBATCH --error=logs/agqa_qgv5_qbef_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_query_grounder_v5_question_blind_event_development_v1d"
PARENT="runs/agqa2_query_grounder_v5_coordinate_corrected_development"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

merge_args=()
for shard in 0 1 2 3 4 5 6 7; do
  merge_args+=(--shard "${RUN}/event_inventory_shard${shard}.json")
done
python scripts/merge_agqa_question_blind_typed_event_inventory_v1.py \
  "${merge_args[@]}" \
  --protocol configs/agqa2_query_grounder_v5_question_blind_event_development_v1d.json \
  --output "${RUN}/event_inventory_frozen.json"

python scripts/freeze_agqa_question_blind_event_query_adapter_v1.py \
  --acquisition-protocol configs/agqa2_query_grounder_v5_question_blind_event_development_v1d.json \
  --cohort "${PARENT}/frozen_cohort/public_cohort.json" \
  --semantic-runtime "${PARENT}/semantic_runtime.json" \
  --sgdet "${PARENT}/action_genome_sgdet_v2.json" \
  --query-plans "${PARENT}/query_plans.json" \
  --event-inventory "${RUN}/event_inventory_frozen.json" \
  --compiler scripts/compile_agqa_question_blind_event_query_grounder_v1.py \
  --event-grounder-module src/motif_transfer/agqa_question_blind_event_grounder.py \
  --output "${RUN}/query_adapter_protocol.json"

python scripts/compile_agqa_question_blind_event_query_grounder_v1.py \
  --cohort "${PARENT}/frozen_cohort/public_cohort.json" \
  --semantic-runtime "${PARENT}/semantic_runtime.json" \
  --sgdet "${PARENT}/action_genome_sgdet_v2.json" \
  --query-plans "${PARENT}/query_plans.json" \
  --event-inventory "${RUN}/event_inventory_frozen.json" \
  --protocol "${RUN}/query_adapter_protocol.json" \
  --minimum-object-score 0.05 \
  --output "${RUN}/query_grounder_candidates_frozen.json"

# This is the first Stage-C command allowed to open development answers.
python scripts/evaluate_agqa_query_grounder_v5_development.py \
  --grounding "${RUN}/query_grounder_candidates_frozen.json" \
  --protocol "${RUN}/query_adapter_protocol.json" \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/train_balanced.txt \
  --output "${RUN}/stage_c_development_evaluation.json"
