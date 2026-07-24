#!/usr/bin/env bash
#SBATCH --job-name=motif-p1-peft
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-5%2

set -Eeuo pipefail

CLEAN_REPO="${CLEAN_REPO:-${SLURM_SUBMIT_DIR}}"
SOURCE_REPO="${SOURCE_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SOURCE_REPO}/runs/b2_best_checkpoints}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-${CLEAN_REPO}/runs/phase1_fresh_a5000_v3/authentic_skill_loaded}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CLEAN_REPO}/runs/phase1_motif_extraction_peft_v1}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"

GAMES=(
  candy_crush
  tetris
  gymv_columns
  gymv_streets_of_rage_2
  gymv_strider
  gymv_thunder_force_iii
)
GAME="${GAMES[${SLURM_ARRAY_TASK_ID}]}"
EVIDENCE_DIR="${EVIDENCE_ROOT}/${GAME}/evidence"
ADAPTER="${CHECKPOINT_ROOT}/${GAME}/adapters/skillbank/segment"
RUN_DIR="${OUTPUT_ROOT}/${GAME}"

[[ -f "${EVIDENCE_DIR}/manifest.json" ]] || { echo "ERROR missing ${EVIDENCE_DIR}" >&2; exit 2; }
[[ -f "${ADAPTER}/adapter_model.safetensors" ]] || { echo "ERROR missing ${ADAPTER}" >&2; exit 2; }
[[ ! -e "${RUN_DIR}" ]] || { echo "ERROR refusing to overwrite ${RUN_DIR}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export HF_HOME="${SHARED_HF_HOME}"
export HF_HUB_OFFLINE=1
export PYTHONPATH="${CLEAN_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${CLEAN_REPO}"
python scripts/extract_matched_peft_motifs.py "${EVIDENCE_DIR}" \
  --base-model "${MODEL}" \
  --segment-adapter "${ADAPTER}" \
  --output-dir "${RUN_DIR}"

echo "PHASE1_MATCHED_PEFT_COMPLETE ${GAME} ${RUN_DIR}"
