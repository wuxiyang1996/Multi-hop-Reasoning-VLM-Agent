#!/usr/bin/env bash
#SBATCH --job-name=agqa-ftb3-eval
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/agqa2_full_train_broad_v3/evaluate-%j.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
test -s runs/agqa2_full_train_broad_v3/preoutcome_receipt.json
python scripts/evaluate_agqa_layer_b_epistemic_five_arm.py \
  --cohort runs/agqa2_full_train_broad_v3/frozen_reserve/public_cohort.json \
  --semantic-runtime runs/agqa2_full_train_broad_v3/semantic_runtime.json \
  --grounding runs/agqa2_full_train_broad_v3/qwen32_grounding_full240.json \
  --claims runs/agqa2_full_train_broad_v3/atomic_claims_full240.json \
  --fallback runs/agqa2_full_train_broad_v3/shared_fallback_full240.json \
  --source-capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json \
  --anonymous-controller runs/anonymous_video_harness_v1/controller.json \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/train_balanced.txt --stage formal \
  --output runs/agqa2_full_train_broad_v3/formal_evaluation.json
