#!/usr/bin/env bash
#SBATCH --job-name=agqa-qwen9b
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# The cluster image has prebuilt vLLM kernels but no /usr/local/cuda nvcc.
# Prevent FlashInfer from attempting a JIT build during engine warm-up.
export VLLM_USE_FLASHINFER_SAMPLER=0
python scripts/run_agqa_neural_only_qwen9b_fresh.py \
  --cohort runs/agqa2_full_operator_transfer_v1/fresh_question_reserve_v1/public_cohort.json \
  --stsg /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/scene_graphs/AGQA_scene_graphs/AGQA_test_stsgs.pkl \
  --ontology /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/scene_graphs/AGQA_scene_graphs/ENG.txt \
  --model Qwen/Qwen3.5-9B \
  --output runs/agqa2_full_operator_transfer_v1/fresh_question_reserve_v1/neural_qwen35_9b_runtime.json
