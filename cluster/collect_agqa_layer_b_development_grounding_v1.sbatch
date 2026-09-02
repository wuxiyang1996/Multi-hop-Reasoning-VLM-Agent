#!/usr/bin/env bash
#SBATCH --job-name=agqa-layerb-dev-vlm
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
COHORT_NAME="${COHORT_NAME:-development_v1}"
GROUNDING_FILE="${GROUNDING_FILE:-raw_video_grounding.json}"
CAPTION_FLAG=()
if [[ "${CAPTION_FIRST:-0}" == "1" ]]; then CAPTION_FLAG=(--caption-first); fi
CHECKPOINT_FLAG=()
if [[ -n "${CHECKPOINT_DIR:-}" ]]; then CHECKPOINT_FLAG=(--checkpoint-dir "${CHECKPOINT_DIR}"); fi
python scripts/collect_agqa_layer_b_local_grounding.py \
  --cohort "runs/agqa2_layer_b_raw_video_v1/${COHORT_NAME}/public_cohort.json" \
  --semantic-runtime "runs/agqa2_layer_b_raw_video_v1/${COHORT_NAME}/semantic_runtime.json" \
  --model Qwen/Qwen3.5-9B \
  --output "runs/agqa2_layer_b_raw_video_v1/${COHORT_NAME}/${GROUNDING_FILE}" \
  "${CAPTION_FLAG[@]}" "${CHECKPOINT_FLAG[@]}"
