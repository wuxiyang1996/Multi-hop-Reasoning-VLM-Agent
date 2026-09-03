#!/usr/bin/env bash
#SBATCH --job-name=agqa-otc-v15-sf
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/agqa_otc_v15_sf_%A_%a.out
#SBATCH --error=logs/agqa_otc_v15_sf_%A_%a.err

set -Eeuo pipefail
REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
RUN="${AGQA_RUN:-runs/agqa2_offtheshelf_compositional_development_v15}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/runs/agqa2_layer_b_raw_video_v1/off_the_shelf/pytorchvideo_runtime:${REPO_ROOT}/src:${REPO_ROOT}"

python scripts/probe_agqa_layer_b_charades_action_model.py \
  --cohort "${RUN}/public_cohort.json" \
  --grounding "${RUN}/public_cohort.json" \
  --semantic-runtime "${RUN}/semantic_runtime.json" \
  --checkpoint runs/agqa2_layer_b_raw_video_v1/off_the_shelf/checkpoints/SLOWFAST_8x8_R50_CHARADES.pyth \
  --classes runs/agqa2_layer_b_raw_video_v1/off_the_shelf/Charades_v1_classes.txt \
  --ontology /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/scene_graphs/AGQA_scene_graphs/ENG.txt \
  --sampling uniform48 --store-all-scores \
  --shard-count 8 --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output "${RUN}/slowfast_all_scores_shard${SLURM_ARRAY_TASK_ID}.json"
