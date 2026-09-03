#!/usr/bin/env bash
#SBATCH --job-name=agqa-ftb4-q32
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-17%8
#SBATCH --output=runs/agqa2_full_train_broad_powered_v4/qwen32-%A_%a.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
start="$((SLURM_ARRAY_TASK_ID * 100))"; end="$((start + 99))"
if (( end > 1789 )); then end=1789; fi
positions="$(seq -s, "$start" "$end")"
python scripts/collect_agqa_layer_b_qwen235_grounding.py \
  --cohort runs/agqa2_full_train_broad_powered_v4/parser_qualified_reserve/public_cohort.json \
  --semantic-runtime runs/agqa2_full_train_broad_powered_v4/parser_qualified_reserve/semantic_runtime.json \
  --keys /fs/gamma-projects/vlm-robot/keys.py --model qwen/qwen3-vl-32b-instruct \
  --positions "$positions" --frame-count 48 --max-tokens 1800 \
  --response-mode json_schema --omit-temperature \
  --cache-dir runs/agqa2_full_train_broad_powered_v4/qwen32_grounding_cache \
  --output "runs/agqa2_full_train_broad_powered_v4/qwen32_grounding_shard${SLURM_ARRAY_TASK_ID}.json"
