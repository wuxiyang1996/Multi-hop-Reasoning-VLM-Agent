#!/usr/bin/env bash
#SBATCH --job-name=motif-p1-complete
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-11%4
#SBATCH --exclude=gammagpu10,gammagpu11,gammagpu13,gammagpu15

set -Eeuo pipefail

CLEAN_REPO="${CLEAN_REPO:-${SLURM_SUBMIT_DIR}}"
SOURCE_REPO="${SOURCE_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SOURCE_REPO}/runs/b2_best_checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CLEAN_REPO}/runs/phase1_fresh_v3}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
GYMV_CONDA_ENV="${GYMV_CONDA_ENV:-gymv}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
EPISODES="${EPISODES:-2}"
MAX_STEPS="${MAX_STEPS:-12}"
GPU_UTIL="${GPU_UTIL:-0.82}"
STAGE_MODEL_CACHE="${STAGE_MODEL_CACHE:-0}"

GAME_SET="${GAME_SET:-six}"
if [[ "${GAME_SET}" == "tetris_candy" ]]; then
  GAMES=(tetris candy_crush)
elif [[ "${GAME_SET}" == "six" ]]; then
  GAMES=(
    candy_crush
    tetris
    gymv_columns
    gymv_streets_of_rage_2
    gymv_strider
    gymv_thunder_force_iii
  )
else
  echo "ERROR unsupported GAME_SET=${GAME_SET}" >&2
  exit 2
fi

GAME_COUNT=${#GAMES[@]}
GAME_INDEX=$((SLURM_ARRAY_TASK_ID % GAME_COUNT))
GAME="${GAMES[${GAME_INDEX}]}"
if (( SLURM_ARRAY_TASK_ID < GAME_COUNT )); then
  CONDITION="authentic_skill_loaded"
else
  CONDITION="skill_disabled"
fi

CHECKPOINT="${CHECKPOINT_ROOT}/${GAME}"
SKILL_BANK="${CHECKPOINT}/banks/${GAME}/skill_bank.jsonl"
RUN_DIR="${OUTPUT_ROOT}/${CONDITION}/${GAME}"
PORT=$((18000 + 20 * SLURM_ARRAY_TASK_ID))
SEED_BASE=$((7301 + 100 * GAME_INDEX))
LOCAL_CACHE_ROOT="/scratch0/${USER}/motif-p1-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
SOURCE_OVERLAY="$(mktemp -d "/tmp/motif-source-overlay-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}-XXXXXX")"

[[ ! -e "${RUN_DIR}" ]] || { echo "ERROR refusing to overwrite ${RUN_DIR}" >&2; exit 2; }
for path in \
  "${CHECKPOINT}/metadata.json" \
  "${CHECKPOINT}/adapters/decision/skill_selection/adapter_model.safetensors" \
  "${CHECKPOINT}/adapters/decision/action_taking/adapter_model.safetensors"; do
  [[ -f "${path}" ]] || { echo "ERROR missing ${path}" >&2; exit 2; }
done
if [[ "${CONDITION}" == "authentic_skill_loaded" ]]; then
  [[ -f "${SKILL_BANK}" ]] || { echo "ERROR missing ${SKILL_BANK}" >&2; exit 2; }
fi

mkdir -p "${RUN_DIR}/logs"
scontrol show job "${SLURM_JOB_ID}" >"${RUN_DIR}/slurm_job_at_start.txt"
PYTHONPATH="${CLEAN_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python "${CLEAN_REPO}/scripts/record_source_provenance.py" \
  --source-repo "${SOURCE_REPO}" \
  --clean-repo "${CLEAN_REPO}" \
  --output "${RUN_DIR}/source_provenance_at_start.json"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if [[ "${GAME}" == gymv_* ]]; then
  GYMV_PYTHON="$(conda run -n "${GYMV_CONDA_ENV}" python -c 'import sys; print(sys.executable)')"
  [[ -x "${GYMV_PYTHON}" ]] || { echo "ERROR invalid Gym-V interpreter" >&2; exit 2; }
  export GYMV_PYTHON
fi

if [[ "${STAGE_MODEL_CACHE}" == "1" ]]; then
  mkdir -p "${LOCAL_CACHE_ROOT}"
  SOURCE_HF_HOME="${SHARED_HF_HOME}" LOCAL_HF_HOME="${LOCAL_CACHE_ROOT}/hf" \
    source "${SOURCE_REPO}/scripts/stage_hf_model_cache.sh" "${MODEL}"
  export HF_HOME="${LOCAL_CACHE_ROOT}/hf"
elif [[ "${STAGE_MODEL_CACHE}" == "0" ]]; then
  export HF_HOME="${SHARED_HF_HOME}"
else
  echo "ERROR STAGE_MODEL_CACHE must be 0 or 1" >&2
  exit 2
fi
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_MAX_MODEL_LEN=4096
export VLLM_MAX_NUM_SEQS=16
export VLLM_MAX_NUM_BATCHED_TOKENS=4096

SERVED_ADAPTERS="${RUN_DIR}/served_adapters"
mkdir -p "${SERVED_ADAPTERS}"
ln -s "${CHECKPOINT}/adapters/decision" "${SERVED_ADAPTERS}/decision"

cd "${SOURCE_REPO}"
python scripts/serve_rollout_pool.py \
  --model "${MODEL}" \
  --adapter-dir "${SERVED_ADAPTERS}" \
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
  if [[ "${SOURCE_OVERLAY}" == "/tmp/motif-source-overlay-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}-"* ]]; then
    rm -rf -- "${SOURCE_OVERLAY}"
  fi
  if [[ "${STAGE_MODEL_CACHE}" == "1" \
        && "${LOCAL_CACHE_ROOT}" == "/scratch0/${USER}/motif-p1-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}" ]]; then
    rm -rf -- "${LOCAL_CACHE_ROOT}"
  fi
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

# Keep the old source tree read-only.  The collection job shadows only the
# Python trainer package in a short-lived overlay, then applies the protocol
# reliability fix owned and reviewed by this clean repository.
cp -a "${SOURCE_REPO}/trainer" "${SOURCE_OVERLAY}/trainer"
mkdir -p "${SOURCE_OVERLAY}/scripts"
cp "${SOURCE_REPO}/scripts/run_instrumented_source_smoke.py" "${SOURCE_OVERLAY}/scripts/"
ln -s "${SOURCE_REPO}/.git" "${SOURCE_OVERLAY}/.git"
(cd "${SOURCE_OVERLAY}" && git apply "${CLEAN_REPO}/patches/source_policy_receipts.patch")
export PYTHONPATH="${SOURCE_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(
  python "${SOURCE_OVERLAY}/scripts/run_instrumented_source_smoke.py"
  --output "${RUN_DIR}/evidence"
  --game "${GAME}"
  --episodes "${EPISODES}"
  --max-steps "${MAX_STEPS}"
  --seed-base "${SEED_BASE}"
  --endpoint "http://127.0.0.1:${PORT}/v1"
  --model "${MODEL}"
  --checkpoint "${CHECKPOINT}"
  --sequential-episodes
)
if [[ "${CONDITION}" == "authentic_skill_loaded" ]]; then
  CMD+=(--skill-bank "${SKILL_BANK}")
fi
"${CMD[@]}" | tee "${RUN_DIR}/summary.json"

echo "PHASE1_COMPLETE ${CONDITION} ${GAME} ${RUN_DIR}"
