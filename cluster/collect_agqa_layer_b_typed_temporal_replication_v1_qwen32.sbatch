#!/usr/bin/env bash
#SBATCH --job-name=agqa-lb-typed-q32
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --array=0-7%8
#SBATCH --output=runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/qwen32-%A_%a.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"

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
  --cohort runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/public_cohort.json \
  --semantic-runtime runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/semantic_runtime_repaired.json \
  --keys /fs/gamma-projects/vlm-robot/keys.py \
  --model qwen/qwen3-vl-32b-instruct \
  --positions "${POSITIONS}" \
  --frame-count "${FRAME_COUNT}" \
  --max-tokens 1800 \
  --response-mode json_schema \
  --omit-temperature \
  --cache-dir "runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/qwen32_grounding_${FRAME_COUNT}f_cache_typed_v1" \
  --output "runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/qwen32_grounding_${FRAME_COUNT}f_shard${SHARD}_typed_v1.json"
