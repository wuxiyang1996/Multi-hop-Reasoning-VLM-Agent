#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2-final
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_qgv2_finalize_%j.out
#SBATCH --error=logs/agqa_qgv2_finalize_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_query_grounder_v2_development_v2"
PARENT="runs/agqa2_query_grounder_v5_coordinate_corrected_development"
PROTOCOL="configs/agqa2_query_grounder_v5_question_blind_event_development_v2.json"
AMENDMENT="configs/agqa2_query_grounder_v5_question_blind_event_development_v2_compiler_amendment.json"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

event_shards=()
anchor_shards=()
for shard_index in {0..7}; do
  event_shards+=(--shard "${RUN}/event_inventory_shard${shard_index}.json")
  anchor_shards+=(--shard "${RUN}/anchor_localizations_canonical_shard${shard_index}.json")
done

python scripts/merge_agqa_question_blind_typed_event_inventory_v1.py \
  "${event_shards[@]}" \
  --protocol "${PROTOCOL}" \
  --output "${RUN}/event_inventory_consumed_development.json"

python scripts/merge_agqa_anchor_localizations_v2.py \
  "${anchor_shards[@]}" \
  --protocol "${PROTOCOL}" \
  --output "${RUN}/anchor_localizations_consumed_development.json"

python scripts/compile_agqa_question_blind_event_query_grounder_v2.py \
  --cohort "${PARENT}/frozen_cohort/public_cohort.json" \
  --semantic-runtime "${PARENT}/semantic_runtime.json" \
  --sgdet "${PARENT}/action_genome_sgdet_v2.json" \
  --query-plans "${PARENT}/query_plans.json" \
  --event-inventory "${RUN}/event_inventory_consumed_development.json" \
  --anchor-localizations "${RUN}/anchor_localizations_consumed_development.json" \
  --protocol "${PROTOCOL}" \
  --runtime-amendment "${AMENDMENT}" \
  --minimum-object-score 0.05 \
  --consumed-development-pilot \
  --output "${RUN}/query_grounding_v2_consumed_development.json"

python scripts/freeze_agqa_answer_blind_candidate_verifier_v2.py \
  --parent-protocol "${PROTOCOL}" \
  --runtime-amendment "${AMENDMENT}" \
  --cohort "${PARENT}/frozen_cohort/public_cohort.json" \
  --sgdet "${PARENT}/action_genome_sgdet_v2.json" \
  --candidate-grounding "${RUN}/query_grounding_v2_consumed_development.json" \
  --consumed-development-pilot \
  --output "${RUN}/candidate_verifier_protocol.json"
