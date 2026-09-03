#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv5-binf
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_qgv5_binf_%j.out
#SBATCH --error=logs/agqa_qgv5_binf_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_query_grounder_v5_question_blind_event_development_v1d"
BINARY="${RUN}/binary_verifier_v1"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

merge_args=()
for shard in 0 1 2 3 4 5 6 7; do
  merge_args+=(--shard "${BINARY}/verification_shard${shard}.json")
done
python scripts/merge_agqa_answer_blind_binary_event_verifier_v1.py \
  "${merge_args[@]}" \
  --protocol "${BINARY}/protocol.json" \
  --candidate-grounding "${RUN}/query_grounder_candidates_frozen.json" \
  --output "${BINARY}/verification_frozen.json"

python scripts/compile_agqa_binary_verified_query_grounder_v1.py \
  --candidate-grounding "${RUN}/query_grounder_candidates_frozen.json" \
  --verification "${BINARY}/verification_frozen.json" \
  --protocol "${BINARY}/protocol.json" \
  --output "${BINARY}/query_grounder_candidates_frozen.json"

# First binary-verifier command allowed to open development answers.
python scripts/evaluate_agqa_query_grounder_v5_development.py \
  --grounding "${BINARY}/query_grounder_candidates_frozen.json" \
  --protocol "${BINARY}/protocol.json" \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/train_balanced.txt \
  --output "${BINARY}/development_evaluation.json"
