#!/usr/bin/env bash
#SBATCH --job-name=agqa-ftb4-claims
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-17%6
#SBATCH --output=runs/agqa2_full_train_broad_powered_v4/claims-%A_%a.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"; export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
start="$((SLURM_ARRAY_TASK_ID * 100))"; end="$((start + 99))"
if (( end > 1789 )); then end=1789; fi
positions="$(seq -s, "$start" "$end")"
python scripts/collect_agqa_layer_b_atomic_claims.py \
  --cohort runs/agqa2_full_train_broad_powered_v4/parser_qualified_reserve/public_cohort.json \
  --grounding runs/agqa2_full_train_broad_powered_v4/qwen32_grounding_full1790.json \
  --keys /fs/gamma-projects/vlm-robot/keys.py --model qwen/qwen3-vl-32b-instruct \
  --positions "$positions" --frame-count 96 --max-tokens 500 --workers 3 \
  --cache-dir runs/agqa2_full_train_broad_powered_v4/atomic_claim_cache \
  --output "runs/agqa2_full_train_broad_powered_v4/atomic_claims_shard${SLURM_ARRAY_TASK_ID}.json"
