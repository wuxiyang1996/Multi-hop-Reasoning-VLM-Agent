#!/usr/bin/env bash
#SBATCH --job-name=ns-source-value-v1
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --exclude=gammagpu10,gammagpu11,gammagpu13,gammagpu15

set -Eeuo pipefail

CLEAN_REPO="${CLEAN_REPO:-${SLURM_SUBMIT_DIR}}"
SOURCE_RUNTIME="${SOURCE_RUNTIME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-source-fresh-v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main/runs/b2_best_checkpoints}"
GAME="${GAME:-gymv_thunder_force_iii}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PLAN="${PLAN:-${CLEAN_REPO}/runs/source_neurosymbolic_value_smoke_v1/frozen_plan.json}"
OUTPUT="${OUTPUT:-${CLEAN_REPO}/runs/source_neurosymbolic_value_smoke_v1/execution}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
GYMV_CONDA_ENV="${GYMV_CONDA_ENV:-gymv}"
GPU_UTIL="${GPU_UTIL:-0.82}"
PORT=$((21000 + SLURM_JOB_ID % 1000))

CHECKPOINT="${CHECKPOINT_ROOT}/${GAME}"
[[ -f "${PLAN}" ]] || { echo "ERROR missing frozen plan ${PLAN}" >&2; exit 2; }
[[ -f "${CHECKPOINT}/metadata.json" ]] || { echo "ERROR missing checkpoint ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${CHECKPOINT}/adapters/decision/action_taking/adapter_model.safetensors" ]] || {
  echo "ERROR missing action_taking adapter" >&2
  exit 2
}
[[ ! -e "${OUTPUT}" ]] || { echo "ERROR refusing to overwrite ${OUTPUT}" >&2; exit 2; }

mkdir -p "$(dirname "${OUTPUT}")/logs"
scontrol show job "${SLURM_JOB_ID}" >"$(dirname "${OUTPUT}")/slurm_job_at_start.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
GYMV_PYTHON="$(conda run -n "${GYMV_CONDA_ENV}" python -c 'import sys; print(sys.executable)')"
[[ -x "${GYMV_PYTHON}" ]] || { echo "ERROR invalid Gym-V interpreter" >&2; exit 2; }
export GYMV_PYTHON
export HF_HOME="${SHARED_HF_HOME}"
export HF_HUB_OFFLINE=1
export RAG_EMBEDDER_DEVICE=cpu
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_MAX_MODEL_LEN=4096
export VLLM_MAX_NUM_SEQS=16
export VLLM_MAX_NUM_BATCHED_TOKENS=4096

SERVED_ADAPTERS="$(dirname "${OUTPUT}")/served_adapters"
mkdir -p "${SERVED_ADAPTERS}"
ln -s "${CHECKPOINT}/adapters/decision" "${SERVED_ADAPTERS}/decision"

cd "${SOURCE_RUNTIME}"
python scripts/serve_rollout_pool.py \
  --model "${MODEL}" \
  --adapter-dir "${SERVED_ADAPTERS}" \
  --gpus 0 \
  --base-port "${PORT}" \
  --gpu-util "${GPU_UTIL}" \
  --log-dir "$(dirname "${OUTPUT}")/logs/vllm" \
  >"$(dirname "${OUTPUT}")/logs/vllm_launcher.log" 2>&1 &
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
  if curl -fsS -m 5 "http://127.0.0.1:${PORT}/v1/models" \
      >"$(dirname "${OUTPUT}")/served_models.json"; then
    break
  fi
  kill -0 "${SERVER_PID}" 2>/dev/null || {
    echo "ERROR vLLM exited during startup" >&2
    exit 3
  }
  sleep 5
done
[[ -s "$(dirname "${OUTPUT}")/served_models.json" ]] || {
  echo "ERROR vLLM startup timeout" >&2
  exit 3
}

cd "${CLEAN_REPO}"
PYTHONPATH="${CLEAN_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python scripts/run_source_multihorizon_replay.py \
    --plan "${PLAN}" \
    --output "${OUTPUT}" \
    --source-runtime "${SOURCE_RUNTIME}" \
    --endpoint "http://127.0.0.1:${PORT}/v1" \
    --model "${MODEL}" \
  | tee "$(dirname "${OUTPUT}")/summary.json"

echo "SOURCE_NEUROSYMBOLIC_VALUE_SMOKE_COMPLETE ${OUTPUT}"
