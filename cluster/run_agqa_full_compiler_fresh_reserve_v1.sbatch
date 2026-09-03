#!/usr/bin/env bash
#SBATCH --job-name=agqa-compiler-fresh
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python scripts/run_agqa_full_compiler_fresh_reserve.py \
  --cohort runs/agqa2_full_operator_transfer_v1/fresh_question_reserve_v1/public_cohort.json \
  --model-dir runs/agqa2_full_operator_transfer_v1/compiler_flan_t5_small_v1b/final \
  --capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json \
  --compiler-qualification runs/agqa2_full_operator_transfer_v1/compiler_flan_t5_small_v1b/heldout_evaluation.json \
  --output runs/agqa2_full_operator_transfer_v1/fresh_question_reserve_v1/compiler_runtime_v1.json
