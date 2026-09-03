#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv5-qbe
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --array=0-7
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/agqa_qgv5_qbe_%A_%a.out
#SBATCH --error=logs/agqa_qgv5_qbe_%A_%a.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_query_grounder_v5_question_blind_event_development_v1d"
PARENT="runs/agqa2_query_grounder_v5_coordinate_corrected_development"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

python scripts/collect_agqa_question_blind_typed_event_inventory_v1.py \
  --cohort "${PARENT}/frozen_cohort/public_cohort.json" \
  --sgdet "${PARENT}/action_genome_sgdet_v2.json" \
  --ontology configs/agqa2_action_genome_sgdet_ontology_v1.json \
  --protocol configs/agqa2_query_grounder_v5_question_blind_event_development_v1d.json \
  --cache-dir "${RUN}/preflight_cache" \
  --model qwen/qwen3-vl-235b-a22b-instruct \
  --clip-count 8 --frames-per-clip 8 \
  --sampling-strategy partition_complete \
  --minimum-object-score 0.05 \
  --shard-count 8 --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output "${RUN}/event_inventory_shard${SLURM_ARRAY_TASK_ID}.json"
