#!/usr/bin/env bash
#SBATCH --job-name=agqa-ftb3-claims
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%4
#SBATCH --output=runs/agqa2_full_train_broad_v3/claims-%A_%a.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
start="$((SLURM_ARRAY_TASK_ID * 40))"
end="$((start + 39))"
positions="$(seq -s, "$start" "$end")"
python scripts/collect_agqa_layer_b_atomic_claims.py \
  --cohort runs/agqa2_full_train_broad_v3/frozen_reserve/public_cohort.json \
  --grounding runs/agqa2_full_train_broad_v3/qwen32_grounding_full240.json \
  --keys /fs/gamma-projects/vlm-robot/keys.py \
  --model qwen/qwen3-vl-32b-instruct \
  --positions "$positions" --frame-count 96 --max-tokens 500 --workers 3 \
  --cache-dir runs/agqa2_full_train_broad_v3/atomic_claim_cache \
  --output "runs/agqa2_full_train_broad_v3/atomic_claims_shard${SLURM_ARRAY_TASK_ID}.json"
