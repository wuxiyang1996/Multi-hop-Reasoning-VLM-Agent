#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2-verify
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --array=0-7
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/agqa_qgv2_verify_%A_%a.out
#SBATCH --error=logs/agqa_qgv2_verify_%A_%a.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_query_grounder_v2_development_v2"
PARENT="runs/agqa2_query_grounder_v5_coordinate_corrected_development"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

python scripts/verify_agqa_question_blind_event_candidates_v1.py \
  --cohort "${PARENT}/frozen_cohort/public_cohort.json" \
  --sgdet "${PARENT}/action_genome_sgdet_v2.json" \
  --candidate-grounding "${RUN}/query_grounding_v2_consumed_development.json" \
  --protocol "${RUN}/candidate_verifier_protocol.json" \
  --cache-dir "${RUN}/candidate_verifier_cache" \
  --model qwen/qwen3-vl-235b-a22b-instruct \
  --provider parasail \
  --maximum-frames 12 \
  --shard-count 8 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --consumed-development-pilot \
  --output "${RUN}/candidate_verification_shard${SLURM_ARRAY_TASK_ID}.json"
