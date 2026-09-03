#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2sbq-dl
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/agqa_qgv2sbq_download_%j.out
#SBATCH --error=logs/agqa_qgv2sbq_download_%j.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="${AGQA_RUN:-runs/agqa2_query_grounder_v2_strict_boundary_qualification_v2}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

python scripts/download_agqa2_active_grounding_v4_reserve.py \
  --selection "${RUN}/frozen_cohort/download_selection.json" \
  --receipt "${RUN}/frozen_cohort/download_receipt.json" \
  --archive-url https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip
python scripts/verify_agqa_query_grounder_v5_development_download.py \
  --cohort-dir "${RUN}/frozen_cohort" \
  --output "${RUN}/frozen_cohort/download_verification.json"
