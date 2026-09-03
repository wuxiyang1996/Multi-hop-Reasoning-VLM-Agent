#!/usr/bin/env bash
#SBATCH --job-name=agqa-q32-dev-v16
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/agqa_q32_dev_v16_%A_%a.out
#SBATCH --error=logs/agqa_q32_dev_v16_%A_%a.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="runs/agqa2_offtheshelf_compositional_development_v15"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

if (( SLURM_ARRAY_TASK_ID < 4 )); then
  FRAME_COUNT=48
  SHARD="${SLURM_ARRAY_TASK_ID}"
else
  FRAME_COUNT=96
  SHARD="$((SLURM_ARRAY_TASK_ID - 4))"
fi
START="$((SHARD * 64))"
END="$((START + 63))"
POSITIONS="$(seq -s, "${START}" "${END}")"

python scripts/collect_agqa_layer_b_qwen235_grounding.py \
  --cohort "${RUN}/public_cohort.json" \
  --semantic-runtime "${RUN}/semantic_runtime.json" \
  --keys /fs/gamma-projects/vlm-robot/keys.py \
  --model qwen/qwen3-vl-32b-instruct \
  --positions "${POSITIONS}" --frame-count "${FRAME_COUNT}" \
  --max-tokens 1800 --response-mode json_schema --omit-temperature \
  --cache-dir "${RUN}/qwen32_${FRAME_COUNT}f_cache_v16" \
  --output "${RUN}/qwen32_${FRAME_COUNT}f_shard${SHARD}_v16.json"
