#!/usr/bin/env bash
#SBATCH --job-name=agqa-ftb4-cmerge
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=runs/agqa2_full_train_broad_powered_v4/claims-merge-%j.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"; export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
args=(); for shard in $(seq 0 17); do args+=(--shard "runs/agqa2_full_train_broad_powered_v4/atomic_claims_shard${shard}.json"); done
python scripts/merge_agqa_layer_b_atomic_claim_shards.py \
  --cohort runs/agqa2_full_train_broad_powered_v4/parser_qualified_reserve/public_cohort.json \
  "${args[@]}" --output runs/agqa2_full_train_broad_powered_v4/atomic_claims_full1790.json
