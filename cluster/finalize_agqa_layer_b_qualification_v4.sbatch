#!/usr/bin/env bash
#SBATCH --job-name=agqa-lb-v4-final
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/agqa2_layer_b_raw_video_v1/qualification_v4/finalize-%j.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"

python scripts/freeze_agqa_layer_b_epistemic_preoutcome.py \
  --preregistration configs/agqa2_layer_b_raw_video_qualification_v4_preregistration.json \
  --cohort runs/agqa2_layer_b_raw_video_v1/qualification_v4/public_cohort.json \
  --semantic-runtime runs/agqa2_layer_b_raw_video_v1/qualification_v4/semantic_runtime_repaired.json \
  --grounding runs/agqa2_layer_b_raw_video_v1/qualification_v4/routed_qwen32_48f_96f_slowfast_fill_missing_full512_v4.json \
  --claims runs/agqa2_layer_b_raw_video_v1/qualification_v4/atomic_claims_qwen32_scoped_full512_v4.json \
  --fallback runs/agqa2_layer_b_raw_video_v1/qualification_v4/shared_fallback_routed_full512_v4.json \
  --source-capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json \
  --output runs/agqa2_layer_b_raw_video_v1/qualification_v4/pre_outcome_freeze_receipt.json

mkdir -p runs/agqa2_layer_b_raw_video_v1/qualification_v4/evaluator_inputs
cp runs/agqa2_layer_b_raw_video_v1/qualification_v4/public_cohort.json \
  runs/agqa2_layer_b_raw_video_v1/qualification_v4/evaluator_inputs/public_cohort.json
cp runs/agqa2_layer_b_raw_video_v1/qualification_v4/semantic_runtime_repaired.json \
  runs/agqa2_layer_b_raw_video_v1/qualification_v4/evaluator_inputs/semantic_runtime.json

python scripts/evaluate_agqa_layer_b_epistemic_five_arm.py \
  --cohort runs/agqa2_layer_b_raw_video_v1/qualification_v4/evaluator_inputs/public_cohort.json \
  --grounding runs/agqa2_layer_b_raw_video_v1/qualification_v4/routed_qwen32_48f_96f_slowfast_fill_missing_full512_v4.json \
  --claims runs/agqa2_layer_b_raw_video_v1/qualification_v4/atomic_claims_qwen32_scoped_full512_v4.json \
  --fallback runs/agqa2_layer_b_raw_video_v1/qualification_v4/shared_fallback_routed_full512_v4.json \
  --source-capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/train_balanced.txt \
  --output runs/agqa2_layer_b_raw_video_v1/qualification_v4/five_arm_epistemic_qualification_full512_v4.json \
  --stage qualification
