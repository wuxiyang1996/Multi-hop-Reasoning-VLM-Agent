#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2-e8-gem
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=runs/agqa2_query_grounder_v2_development/eval-v8-gemini-%j.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"

python scripts/evaluate_agqa_query_grounder_v2_qualification.py \
  --grounding runs/agqa2_query_grounder_v2_development/pilot_v8_gemini.json \
  --protocol configs/agqa2_query_grounder_v2_development_protocol.json \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/train_balanced.txt \
  --output runs/agqa2_query_grounder_v2_development/pilot_v8_gemini_evaluation.json
