#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2-sfdense
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%4
#SBATCH --output=runs/agqa2_query_grounder_v2_development/slowfast-dense-%A_%a.out
set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/runs/agqa2_layer_b_raw_video_v1/off_the_shelf/pytorchvideo_runtime:${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
python scripts/probe_agqa_layer_b_charades_action_model.py \
  --cohort runs/agqa2_query_grounder_v2_development/frozen_cohort/public_cohort.json \
  --grounding runs/agqa2_query_grounder_v2_development/gdino_full_dev_v1.json \
  --semantic-runtime runs/agqa2_query_grounder_v2_development/semantic_runtime.json \
  --checkpoint runs/agqa2_layer_b_raw_video_v1/off_the_shelf/checkpoints/SLOWFAST_8x8_R50_CHARADES.pyth \
  --classes runs/agqa2_layer_b_raw_video_v1/off_the_shelf/Charades_v1_classes.txt \
  --ontology /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/scene_graphs/AGQA_scene_graphs/ENG.txt \
  --sampling dense10x32 --store-all-scores \
  --shard-count 4 --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output "runs/agqa2_query_grounder_v2_development/slowfast_dense_all_scores_shard${SLURM_ARRAY_TASK_ID}_v3.json"
