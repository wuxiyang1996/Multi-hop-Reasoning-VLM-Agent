#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv5-qwen
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --array=0-3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/agqa_qgv5_qwen_%A_%a.out
#SBATCH --error=logs/agqa_qgv5_qwen_%A_%a.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_query_grounder_v5_native_exact_qwen235_development"
PARENT="runs/agqa2_query_grounder_v5_coordinate_corrected_development"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

python scripts/pilot_agqa_query_grounder_v4_qwen235_adjudicator.py \
  --cohort "${PARENT}/frozen_cohort/public_cohort.json" \
  --sgdet-raw "${PARENT}/action_genome_sgdet_v2.json" \
  --candidate-grounding "${PARENT}/query_grounder_candidates_frozen.json" \
  --protocol configs/agqa2_query_grounder_v5_native_exact_qwen235_development.json \
  --cache-dir "${RUN}/cache" \
  --model qwen/qwen3-vl-235b-a22b-instruct \
  --maximum-candidates 6 --maximum-frames 12 \
  --coordinate-mode native_exact \
  --shard-count 4 --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output "${RUN}/adjudication_shard${SLURM_ARRAY_TASK_ID}.json"
