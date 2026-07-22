#!/usr/bin/env bash
#SBATCH --job-name=motif-p1-extract
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
EVIDENCE_ROOT="${EVIDENCE_ROOT:-${CLEAN_REPO}/runs/phase1_fresh_a5000_v1/authentic_skill_loaded}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CLEAN_REPO}/runs/phase1_motif_extraction_v1}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
GPU_UTIL="${GPU_UTIL:-0.96}"

GAMES=(
  candy_crush
  tetris
  gymv_columns
  gymv_streets_of_rage_2
  gymv_strider
  gymv_thunder_force_iii
)
GAME="${GAMES[${SLURM_ARRAY_TASK_ID}]}"
CHECKPOINT="${CHECKPOINT_ROOT}/${GAME}"
EVIDENCE_DIR="${EVIDENCE_ROOT}/${GAME}/evidence"
RUN_DIR="${OUTPUT_ROOT}/${GAME}"
PORT=$((20000 + 20 * SLURM_ARRAY_TASK_ID))

[[ -f "${EVIDENCE_DIR}/manifest.json" ]] || { echo "ERROR missing ${EVIDENCE_DIR}" >&2; exit 2; }
for role in segment contract curator; do
  [[ -f "${CHECKPOINT}/adapters/skillbank/${role}/adapter_model.safetensors" ]] || {
    echo "ERROR missing ${role} adapter for ${GAME}" >&2
    exit 2
  }
done
[[ ! -e "${RUN_DIR}" ]] || { echo "ERROR refusing to overwrite ${RUN_DIR}" >&2; exit 2; }
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/served_adapters"
ln -s "${CHECKPOINT}/adapters/skillbank" "${RUN_DIR}/served_adapters/skillbank"
scontrol show job "${SLURM_JOB_ID}" >"${RUN_DIR}/slurm_job_at_start.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export HF_HOME="${SHARED_HF_HOME}"
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_MAX_MODEL_LEN=4096
export VLLM_MAX_NUM_SEQS=16
export VLLM_MAX_NUM_BATCHED_TOKENS=4096

cd "${SOURCE_REPO}"
python scripts/serve_rollout_pool.py \
  --model "${MODEL}" \
  --adapter-dir "${RUN_DIR}/served_adapters" \
  --gpus 0 \
  --base-port "${PORT}" \
  --gpu-util "${GPU_UTIL}" \
  --log-dir "${RUN_DIR}/logs/vllm" \
  >"${RUN_DIR}/logs/vllm_launcher.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  exit "${code}"
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + 1800))
while (( SECONDS < deadline )); do
  if curl -fsS -m 5 "http://127.0.0.1:${PORT}/v1/models" >"${RUN_DIR}/served_models.json"; then
    break
  fi
  kill -0 "${SERVER_PID}" 2>/dev/null || { echo "ERROR vLLM exited during startup" >&2; exit 3; }
  sleep 5
done
[[ -s "${RUN_DIR}/served_models.json" ]] || { echo "ERROR vLLM startup timeout" >&2; exit 3; }

cd "${CLEAN_REPO}"
python scripts/extract_frozen_motifs.py "${EVIDENCE_DIR}" \
  --endpoint "http://127.0.0.1:${PORT}/v1" \
  --segment-model "${MODEL}" \
  --condition authentic \
  --output "${RUN_DIR}/base_authentic.json"
python scripts/extract_frozen_motifs.py "${EVIDENCE_DIR}" \
  --endpoint "http://127.0.0.1:${PORT}/v1" \
  --segment-model segment \
  --condition authentic \
  --output "${RUN_DIR}/coevolved_authentic.json"
python scripts/extract_frozen_motifs.py "${EVIDENCE_DIR}" \
  --endpoint "http://127.0.0.1:${PORT}/v1" \
  --segment-model segment \
  --condition receipt_only \
  --output "${RUN_DIR}/coevolved_receipt_only.json"

echo "PHASE1_MOTIF_EXTRACTION_COMPLETE ${GAME} ${RUN_DIR}"
