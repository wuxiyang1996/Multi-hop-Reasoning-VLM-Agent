#!/usr/bin/env bash
#SBATCH --job-name=agqa-lb-risk-claims
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=runs/agqa2_layer_b_raw_video_v1/risk_replication_v1/claims-%j.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"

python scripts/collect_agqa_layer_b_atomic_claims.py \
  --cohort runs/agqa2_layer_b_raw_video_v1/risk_replication_v1/public_cohort.json \
  --grounding runs/agqa2_layer_b_raw_video_v1/risk_replication_v1/routed_qwen32_48f_96f_slowfast_fill_missing_full384_risk_v1.json \
  --keys /fs/gamma-projects/vlm-robot/keys.py \
  --model qwen/qwen3-vl-32b-instruct \
  --frame-count 96 \
  --max-tokens 500 \
  --workers 8 \
  --positions all \
  --cache-dir runs/agqa2_layer_b_raw_video_v1/risk_replication_v1/atomic_claims_qwen32_scoped_cache_risk_v1 \
  --output runs/agqa2_layer_b_raw_video_v1/risk_replication_v1/atomic_claims_qwen32_scoped_full384_risk_v1.json
