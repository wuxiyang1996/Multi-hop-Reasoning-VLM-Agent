#!/usr/bin/env bash
#SBATCH --job-name=agqa-q32-build-v16
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/agqa_q32_build_v16_%j.out
#SBATCH --error=logs/agqa_q32_build_v16_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_offtheshelf_compositional_development_v15"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

q48=(); q96=()
for shard in $(seq 0 3); do
  q48+=(--shard "${RUN}/qwen32_48f_shard${shard}_v16.json")
  q96+=(--shard "${RUN}/qwen32_96f_shard${shard}_v16.json")
done
python scripts/merge_agqa_layer_b_grounding_shards.py \
  --cohort "${RUN}/public_cohort.json" "${q48[@]}" \
  --output "${RUN}/qwen32_48f_full_v16.json"
python scripts/merge_agqa_layer_b_grounding_shards.py \
  --cohort "${RUN}/public_cohort.json" "${q96[@]}" \
  --output "${RUN}/qwen32_96f_full_v16.json"
python scripts/route_agqa_layer_b_shared_grounding.py \
  --cohort "${RUN}/public_cohort.json" \
  --semantic-runtime "${RUN}/semantic_runtime.json" \
  --candidate "${RUN}/qwen32_48f_full_v16.json" \
  --candidate "${RUN}/qwen32_96f_full_v16.json" \
  --shared-action-frame-budget 0 \
  --output "${RUN}/qwen32_48f_96f_routed_v16.json"
