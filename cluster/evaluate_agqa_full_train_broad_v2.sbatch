#!/usr/bin/env bash
#SBATCH --job-name=agqa-ftb-eval
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/agqa2_full_train_broad_v1/evaluate-%j.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"

test -s runs/agqa2_full_train_broad_v1/preoutcome_receipt_v2.json
python scripts/evaluate_agqa_layer_b_epistemic_five_arm.py \
  --cohort runs/agqa2_full_train_broad_v1/parser_qualified_reserve_v2/public_cohort.json \
  --grounding runs/agqa2_full_train_broad_v1/qwen235_grounding_full390.json \
  --claims runs/agqa2_full_train_broad_v1/atomic_claims_full390.json \
  --fallback runs/agqa2_full_train_broad_v1/shared_fallback_full390.json \
  --source-capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json \
  --anonymous-controller runs/anonymous_video_harness_v1/controller.json \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/train_balanced.txt \
  --stage formal \
  --output runs/agqa2_full_train_broad_v1/formal_evaluation_v2.json
