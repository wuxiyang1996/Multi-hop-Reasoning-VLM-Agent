#!/usr/bin/env bash
#SBATCH --job-name=alf-target-adapt
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%4

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SOURCE_REPO="${SOURCE_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main}"
CONDA_ENV="${CONDA_ENV:-alfworld}"
KEYS="${KEYS:-/fs/gamma-projects/vlm-robot/keys.py}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/configs/target_rollout_collection_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/target_rollout_collection_v1/alfworld_valid_unseen/adaptation}"
CONFIG="${CONFIG:-${SOURCE_REPO}/configs/alfworld_base_config.yaml}"
DATA="${DATA:-${SOURCE_REPO}/.cache/alfworld_data}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

INDEX="${SLURM_ARRAY_TASK_ID}"
OUTPUT="${OUTPUT_ROOT}/task_${INDEX}.json"
CACHE="${OUTPUT_ROOT}/decision_cache/task_${INDEX}.json"
mkdir -p "${OUTPUT_ROOT}/decision_cache"

python scripts/run_alfworld_target_rollout.py \
  --keys "${KEYS}" \
  --config "${CONFIG}" \
  --data "${DATA}" \
  --manifest "${MANIFEST}" \
  --split-name adaptation \
  --task-offset "${INDEX}" \
  --output "${OUTPUT}" \
  --decision-cache "${CACHE}" \
  --alfworld-split train \
  --seed 73001 \
  --decision-max-tokens 512 \
  --max-steps 30

echo "ALFWORLD_TARGET_ADAPTATION_V1_COMPLETE ${INDEX} ${OUTPUT}"
