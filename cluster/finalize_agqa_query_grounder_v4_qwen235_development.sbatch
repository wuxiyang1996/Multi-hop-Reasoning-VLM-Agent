#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv4-qwen-final
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/agqa_qgv4_qwen_final_%j.out
#SBATCH --error=logs/agqa_qgv4_qwen_final_%j.err
set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_query_grounder_v4_qwen235_development"
PARENT="runs/agqa2_query_grounder_v3_compositional_development"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
merge_args=()
for shard in 0 1 2 3; do
  merge_args+=(--shard "${RUN}/adjudication_shard${shard}.json")
done
python scripts/merge_agqa_query_grounder_v4_adjudicator_shards.py \
  --candidate-grounding "${PARENT}/query_grounder_candidates_frozen.json" \
  "${merge_args[@]}" --output "${RUN}/adjudication_frozen.json"
python scripts/calibrate_agqa_query_grounder_v4_threshold.py \
  --adjudication "${RUN}/adjudication_frozen.json" \
  --protocol configs/agqa2_query_grounder_v4_qwen235_development.json \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/train_balanced.txt \
  --output "${RUN}/global_threshold_calibration.json"
