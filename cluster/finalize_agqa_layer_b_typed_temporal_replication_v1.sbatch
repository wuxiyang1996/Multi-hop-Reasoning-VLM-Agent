#!/usr/bin/env bash
#SBATCH --job-name=agqa-lb-typed-final
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/finalize-%j.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"

python scripts/freeze_agqa_layer_b_multiview_preoutcome.py \
  --preregistration configs/agqa2_layer_b_typed_temporal_replication_v1_preregistration.json \
  --cohort runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/public_cohort.json \
  --semantic-runtime runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/semantic_runtime_repaired.json \
  --routed-grounding runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/routed_qwen32_48f_96f_slowfast_fill_missing_full256_typed_v1.json \
  --grounding-view runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/composite_qwen32_48f_slowfast_fill_missing_full256_typed_v1.json \
  --grounding-view runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/composite_qwen32_96f_slowfast_fill_missing_full256_typed_v1.json \
  --fallback runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/shared_fallback_routed_full256_typed_v1.json \
  --source-capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json \
  --output runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/pre_outcome_freeze_receipt.json

python scripts/evaluate_agqa_layer_b_typed_multiview_five_arm.py \
  --preregistration configs/agqa2_layer_b_typed_temporal_replication_v1_preregistration.json \
  --pre-outcome-receipt runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/pre_outcome_freeze_receipt.json \
  --cohort runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/public_cohort.json \
  --semantic-runtime runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/semantic_runtime_repaired.json \
  --routed-grounding runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/routed_qwen32_48f_96f_slowfast_fill_missing_full256_typed_v1.json \
  --fallback runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/shared_fallback_routed_full256_typed_v1.json \
  --source-capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --entry AGQA_balanced/train_balanced.txt \
  --output runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/five_arm_typed_temporal_full256_v1.json
