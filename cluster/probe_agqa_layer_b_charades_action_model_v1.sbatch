#!/usr/bin/env bash
#SBATCH --job-name=agqa-lb-slowfast
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/runs/agqa2_layer_b_raw_video_v1/off_the_shelf/pytorchvideo_runtime:${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
COHORT_NAME="${COHORT_NAME:-acquisition_v2}"
GROUNDING_FILE="${GROUNDING_FILE:-qwen32_grounding_48f_full32_v1.json}"
OUTPUT_FILE="${OUTPUT_FILE:-slowfast_charades_action_probe_dense_full32_v1.json}"
python scripts/probe_agqa_layer_b_charades_action_model.py \
  --cohort "runs/agqa2_layer_b_raw_video_v1/${COHORT_NAME}/public_cohort.json" \
  --grounding "runs/agqa2_layer_b_raw_video_v1/${COHORT_NAME}/${GROUNDING_FILE}" \
  --semantic-runtime "runs/agqa2_layer_b_raw_video_v1/${COHORT_NAME}/semantic_runtime.json" \
  --checkpoint runs/agqa2_layer_b_raw_video_v1/off_the_shelf/checkpoints/SLOWFAST_8x8_R50_CHARADES.pyth \
  --classes runs/agqa2_layer_b_raw_video_v1/off_the_shelf/Charades_v1_classes.txt \
  --ontology /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/scene_graphs/AGQA_scene_graphs/ENG.txt \
  --sampling dense10x32 \
  --output "runs/agqa2_layer_b_raw_video_v1/${COHORT_NAME}/${OUTPUT_FILE}"
