#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2-dev-dl
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=runs/agqa2_query_grounder_v2_development/download-%j.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"

python scripts/download_agqa2_active_grounding_v4_reserve.py \
  --selection configs/agqa2_query_grounder_v2_development_selection.json \
  --receipt runs/agqa2_query_grounder_v2_development/download_receipt.json \
  --archive-url https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip
