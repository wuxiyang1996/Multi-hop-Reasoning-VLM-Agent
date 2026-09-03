#!/usr/bin/env bash
#SBATCH --job-name=agqa-prog-eval
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -Eeuo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export HF_HOME="/fs/gamma-projects/vlm-robot/hf_cache"
export HF_HUB_OFFLINE=1
python scripts/evaluate_agqa_full_program_compiler.py \
  --data-dir runs/agqa2_full_operator_transfer_v1/compiler_data \
  --model-dir runs/agqa2_full_operator_transfer_v1/compiler_flan_t5_small_v1b/final \
  --capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities.json \
  --output runs/agqa2_full_operator_transfer_v1/compiler_flan_t5_small_v1b/heldout_evaluation.json \
  --batch-size 64
