#!/usr/bin/env bash
#SBATCH --job-name=agqa-qgv2-td-sgdet
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/agqa_qgv2_td_sgdet_%j.out
#SBATCH --error=logs/agqa_qgv2_td_sgdet_%j.err
set -Eeuo pipefail

REPO_ROOT="/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
TEMPURA_ROOT="/fs/gamma-projects/vlm-robot/third_party/unbiasedSGG"
TEMPURA_ENV="/fs/gamma-projects/vlm-robot/third_party/envs/tempura-py37"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

"${TEMPURA_ENV}/bin/python" scripts/pilot_agqa_action_genome_sgdet.py \
  --cohort runs/agqa2_query_grounder_v2_task_disjoint_qualification_v1/frozen_cohort/public_cohort.json \
  --ontology configs/agqa2_action_genome_sgdet_ontology_v1.json \
  --third-party "${TEMPURA_ROOT}" \
  --detector-checkpoint "${TEMPURA_ROOT}/fasterRCNN/models/faster_rcnn_ag.pth" \
  --relation-checkpoint "${TEMPURA_ROOT}/checkpoints/sgdet/best_Mrecall_model.tar" \
  --maximum-frames 64 \
  --output runs/agqa2_query_grounder_v2_task_disjoint_qualification_v1/action_genome_sgdet_v2.json
