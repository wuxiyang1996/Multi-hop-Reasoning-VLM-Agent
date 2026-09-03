#!/usr/bin/env bash
#SBATCH --job-name=agqa-lb-typed-download
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/download-%j.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"

python scripts/download_agqa2_active_grounding_v4_reserve.py \
  --selection runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/download_selection.json \
  --receipt runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/download_receipt.json
