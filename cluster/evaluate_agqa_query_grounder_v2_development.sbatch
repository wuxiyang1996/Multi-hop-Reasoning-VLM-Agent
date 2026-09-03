#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2-eval
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_qgv2_evaluate_%j.out
#SBATCH --error=logs/agqa_qgv2_evaluate_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_query_grounder_v2_development_v2"
PARENT="runs/agqa2_query_grounder_v5_coordinate_corrected_development"
ACQUISITION_PROTOCOL="configs/agqa2_query_grounder_v5_question_blind_event_development_v2.json"
VERIFIER_PROTOCOL="${RUN}/candidate_verifier_protocol.json"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

verification_shards=()
for shard_index in {0..7}; do
  verification_shards+=(--shard "${RUN}/candidate_verification_shard${shard_index}.json")
done

python scripts/merge_agqa_answer_blind_binary_event_verifier_v1.py \
  "${verification_shards[@]}" \
  --protocol "${VERIFIER_PROTOCOL}" \
  --candidate-grounding "${RUN}/query_grounding_v2_consumed_development.json" \
  --output "${RUN}/candidate_verification_consumed_development.json"

python scripts/compile_agqa_binary_verified_query_grounder_v1.py \
  --candidate-grounding "${RUN}/query_grounding_v2_consumed_development.json" \
  --verification "${RUN}/candidate_verification_consumed_development.json" \
  --protocol "${VERIFIER_PROTOCOL}" \
  --consumed-development-pilot \
  --output "${RUN}/query_grounding_v2_verified_consumed_development.json"

# This is the first command in V2 allowed to open development answers.
python scripts/evaluate_agqa_query_grounder_v5_development.py \
  --grounding "${RUN}/query_grounding_v2_verified_consumed_development.json" \
  --protocol "${ACQUISITION_PROTOCOL}" \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/train_balanced.txt \
  --consumed-development-grounding \
  --output "${RUN}/development_qualification_report.json"
