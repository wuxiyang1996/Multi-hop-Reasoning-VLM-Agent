#!/usr/bin/env bash
#SBATCH --job-name=motif-native-rank
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-11%4
#SBATCH --exclude=gammagpu10,gammagpu11,gammagpu13,gammagpu15

set -Eeuo pipefail

CLEAN_REPO="${CLEAN_REPO:-${SLURM_SUBMIT_DIR}}"
SOURCE_REPO="${SOURCE_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SOURCE_REPO}/runs/b2_best_checkpoints}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-${CLEAN_REPO}/runs/phase1_native_policy_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CLEAN_REPO}/runs/phase1_native_rankings_v2}"
HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
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
GAME_INDEX=$((SLURM_ARRAY_TASK_ID % 6))
GAME="${GAMES[${GAME_INDEX}]}"
if (( SLURM_ARRAY_TASK_ID < 6 )); then
  CONDITION="authentic_skill_loaded"
else
  CONDITION="skill_disabled"
fi

CHECKPOINT="${CHECKPOINT_ROOT}/${GAME}"
EVIDENCE_DIR="${EVIDENCE_ROOT}/${CONDITION}/${GAME}/evidence"
OUTPUT_DIR="${OUTPUT_ROOT}/${CONDITION}/${GAME}"
SKILL_BANK="${CHECKPOINT}/banks/${GAME}/skill_bank.jsonl"
SEGMENT_ADAPTER="${CHECKPOINT}/adapters/skillbank/segment"

[[ -f "${EVIDENCE_DIR}/events.jsonl" ]] || { echo "ERROR missing evidence ${EVIDENCE_DIR}" >&2; exit 2; }
[[ -f "${SKILL_BANK}" ]] || { echo "ERROR missing bank ${SKILL_BANK}" >&2; exit 2; }
[[ -f "${SEGMENT_ADAPTER}/adapter_model.safetensors" ]] || { echo "ERROR missing adapter ${SEGMENT_ADAPTER}" >&2; exit 2; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "ERROR refusing to overwrite ${OUTPUT_DIR}" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export HF_HOME HF_HUB_OFFLINE=1
export PYTHONPATH="${CLEAN_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

python "${CLEAN_REPO}/scripts/rank_native_source_segments.py" \
  "${EVIDENCE_DIR}" \
  --skill-bank "${SKILL_BANK}" \
  --base-model "${MODEL}" \
  --segment-adapter "${SEGMENT_ADAPTER}" \
  --output-dir "${OUTPUT_DIR}"

echo "NATIVE_RANKING_COMPLETE ${CONDITION} ${GAME} ${OUTPUT_DIR}"
