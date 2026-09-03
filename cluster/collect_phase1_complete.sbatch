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
PORT_BASE="${PORT_BASE:-18000}"
REASONING_OBSERVER="${REASONING_OBSERVER:-0}"
EXCLUDE_HUMAN_HINTS="${EXCLUDE_HUMAN_HINTS:-1}"
OFFLINE_ANNOTATION_SOURCE_ROOT="${OFFLINE_ANNOTATION_SOURCE_ROOT:-}"
MATCHED_POLICY_SKILL_ID="${MATCHED_POLICY_SKILL_ID:-}"
MATCHED_REPLAY_WORKERS="${MATCHED_REPLAY_WORKERS:-4}"

GAME_SET="${GAME_SET:-six}"
if [[ "${GAME_SET}" == "tetris_candy" ]]; then
  GAMES=(tetris candy_crush)
elif [[ "${GAME_SET}" == "streets_strider" ]]; then
  GAMES=(gymv_streets_of_rage_2 gymv_strider)
elif [[ "${GAME_SET}" == "columns_thunder" ]]; then
  GAMES=(gymv_columns gymv_thunder_force_iii)
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

# The upstream writer emits every JSONL and the manifest in one batch at the end
# of a run, so one invocation is the smallest atomic unit available without
# patching it.  CHUNK_SIZE splits a game across several array tasks to shrink
# that unit: measured cost is 1-2 min of vLLM startup against 4 min per episode,
# so 4-episode chunks buy 4x finer resume for about 8% more GPU time.
# CHUNK_SIZE=0 reproduces the original one-task-per-game layout exactly.
CHUNK_SIZE="${CHUNK_SIZE:-0}"
SEED_OFFSET=0
if (( CHUNK_SIZE > 0 )); then
  EPISODES_PER_GAME="${EPISODES_PER_GAME:-${EPISODES}}"
  CHUNKS_PER_GAME=$(( (EPISODES_PER_GAME + CHUNK_SIZE - 1) / CHUNK_SIZE ))
  GAME_INDEX=$(( SLURM_ARRAY_TASK_ID / CHUNKS_PER_GAME ))
  CHUNK_INDEX=$(( SLURM_ARRAY_TASK_ID % CHUNKS_PER_GAME ))
  (( GAME_INDEX < GAME_COUNT )) || {
    echo "ERROR array task ${SLURM_ARRAY_TASK_ID} exceeds ${GAME_COUNT} games x ${CHUNKS_PER_GAME} chunks" >&2
    exit 2
  }
  GAME="${GAMES[${GAME_INDEX}]}"
  CONDITION="${CONDITION_OVERRIDE:-authentic_skill_loaded}"
  SEED_OFFSET=$(( CHUNK_INDEX * CHUNK_SIZE ))
  EPISODES=$(( EPISODES_PER_GAME - SEED_OFFSET ))
  (( EPISODES <= CHUNK_SIZE )) || EPISODES=${CHUNK_SIZE}
else
  GAME_INDEX=$((SLURM_ARRAY_TASK_ID % GAME_COUNT))
  GAME="${GAMES[${GAME_INDEX}]}"
  if (( SLURM_ARRAY_TASK_ID < GAME_COUNT )); then
    CONDITION="authentic_skill_loaded"
  else
    CONDITION="skill_disabled"
  fi
fi

CHECKPOINT="${CHECKPOINT_ROOT}/${GAME}"
SKILL_BANK="${CHECKPOINT}/banks/${GAME}/skill_bank.jsonl"
if (( CHUNK_SIZE > 0 )); then
  RUN_DIR="${OUTPUT_ROOT}/${CONDITION}/${GAME}/chunk$(printf '%02d' "${CHUNK_INDEX}")"
else
  RUN_DIR="${OUTPUT_ROOT}/${CONDITION}/${GAME}"
fi
PORT=$((PORT_BASE + 20 * SLURM_ARRAY_TASK_ID))
# Defaults reproduce the original 7301/7401 families exactly.  A fresh collection
# passes SEED_BASE_ROOT/SEED_BASE_STRIDE so every game gets its own declared band
# instead of silently reusing one base across game sets.
SEED_BASE=$(( ${SEED_BASE:-$(( ${SEED_BASE_ROOT:-7301} + ${SEED_BASE_STRIDE:-100} * GAME_INDEX ))} + SEED_OFFSET ))
LOCAL_CACHE_ROOT="/scratch0/${USER}/motif-p1-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
SOURCE_OVERLAY="$(mktemp -d "/tmp/motif-source-overlay-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}-XXXXXX")"
PATCH_SNAPSHOT="${SOURCE_OVERLAY}/patches"

ALLOW_RESUME="${ALLOW_RESUME:-0}"
if [[ -e "${RUN_DIR}" ]]; then
  # Collection is not restartable mid-episode: the writer emits every JSONL and the
  # manifest in one batch at the end.  A finished manifest is therefore the only
  # safe resume point, and a half-written directory must still be a hard failure.
  if [[ "${ALLOW_RESUME}" == "1" && -f "${RUN_DIR}/evidence/manifest.json" ]]; then
    echo "PHASE1_ALREADY_COMPLETE ${CONDITION} ${GAME} ${RUN_DIR}"
    exit 0
  fi
  echo "ERROR refusing to overwrite ${RUN_DIR}" >&2
  exit 2
fi
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
# Snapshot executable patches before the long model startup. Queued/running
# jobs must not observe later edits in the shared working tree.
mkdir -p "${PATCH_SNAPSHOT}"
for patch_name in \
  source_policy_receipts.patch \
  source_request_seed_control.patch \
  source_matched_policy_treatments.patch \
  source_no_human_policy_hints.patch \
  source_reasoning_shadow_observer.patch; do
  cp "${CLEAN_REPO}/patches/${patch_name}" "${PATCH_SNAPSHOT}/${patch_name}"
done
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
export RAG_EMBEDDER_DEVICE=cpu
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

if [[ -n "${OFFLINE_ANNOTATION_SOURCE_ROOT}" ]]; then
  SOURCE_EVIDENCE="${OFFLINE_ANNOTATION_SOURCE_ROOT}/authentic_skill_loaded/${GAME}/evidence"
  [[ -f "${SOURCE_EVIDENCE}/manifest.json" ]] || {
    echo "ERROR missing immutable source evidence ${SOURCE_EVIDENCE}" >&2
    exit 2
  }
  PYTHONPATH="${CLEAN_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    python "${CLEAN_REPO}/scripts/annotate_source_shadow.py" \
      --evidence "${SOURCE_EVIDENCE}" \
      --output "${RUN_DIR}/annotation" \
      --endpoint "http://127.0.0.1:${PORT}/v1" \
      --model "${MODEL}"
  echo "OFFLINE_SHADOW_COMPLETE ${GAME} ${RUN_DIR}/annotation"
  exit 0
fi

# Keep the old source tree read-only.  The collection job shadows only the
# Python trainer package in a short-lived overlay, then applies the protocol
# reliability fix owned and reviewed by this clean repository.
cp -a "${SOURCE_REPO}/trainer" "${SOURCE_OVERLAY}/trainer"
mkdir -p "${SOURCE_OVERLAY}/scripts"
cp "${SOURCE_REPO}/scripts/run_instrumented_source_smoke.py" "${SOURCE_OVERLAY}/scripts/"
ln -s "${SOURCE_REPO}/.git" "${SOURCE_OVERLAY}/.git"
(cd "${SOURCE_OVERLAY}" && git apply "${PATCH_SNAPSHOT}/source_policy_receipts.patch")
(cd "${SOURCE_OVERLAY}" && git apply "${PATCH_SNAPSHOT}/source_request_seed_control.patch")
(cd "${SOURCE_OVERLAY}" && git apply "${PATCH_SNAPSHOT}/source_matched_policy_treatments.patch")
if [[ "${EXCLUDE_HUMAN_HINTS}" == "1" ]]; then
  (cd "${SOURCE_OVERLAY}" && git apply "${PATCH_SNAPSHOT}/source_no_human_policy_hints.patch")
elif [[ "${EXCLUDE_HUMAN_HINTS}" != "0" ]]; then
  echo "ERROR EXCLUDE_HUMAN_HINTS must be 0 or 1" >&2
  exit 2
fi
if [[ "${REASONING_OBSERVER}" == "1" ]]; then
  (cd "${SOURCE_OVERLAY}" && git apply "${PATCH_SNAPSHOT}/source_reasoning_shadow_observer.patch")
elif [[ "${REASONING_OBSERVER}" != "0" ]]; then
  echo "ERROR REASONING_OBSERVER must be 0 or 1" >&2
  exit 2
fi
OVERLAY_RECEIPT_CMD=(
  python "${CLEAN_REPO}/scripts/record_source_overlay_receipt.py"
  --overlay-runner "${SOURCE_OVERLAY}/trainer/coevolution/episode_runner.py"
  --overlay-vllm-client "${SOURCE_OVERLAY}/trainer/coevolution/vllm_client.py"
  --overlay-smoke-runner "${SOURCE_OVERLAY}/scripts/run_instrumented_source_smoke.py"
  --policy-receipts-patch "${PATCH_SNAPSHOT}/source_policy_receipts.patch"
  --request-seed-control-patch "${PATCH_SNAPSHOT}/source_request_seed_control.patch"
  --matched-policy-patch "${PATCH_SNAPSHOT}/source_matched_policy_treatments.patch"
  --output "${RUN_DIR}/source_overlay_receipt.json"
)
if [[ "${EXCLUDE_HUMAN_HINTS}" == "1" ]]; then
  OVERLAY_RECEIPT_CMD+=(
    --no-human-hints-patch "${PATCH_SNAPSHOT}/source_no_human_policy_hints.patch"
  )
fi
if [[ "${REASONING_OBSERVER}" == "1" ]]; then
  OVERLAY_RECEIPT_CMD+=(
    --shadow-observer-patch "${PATCH_SNAPSHOT}/source_reasoning_shadow_observer.patch"
  )
fi
"${OVERLAY_RECEIPT_CMD[@]}"
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
if [[ "${REASONING_OBSERVER}" == "1" ]]; then
  CMD+=(--reasoning-backbone-harness)
fi
if [[ -n "${MATCHED_POLICY_SKILL_ID}" ]]; then
  [[ "${CONDITION}" == "authentic_skill_loaded" ]] || {
    echo "ERROR matched policy treatments require authentic_skill_loaded" >&2
    exit 2
  }
  CMD+=(
    --matched-policy-skill-id "${MATCHED_POLICY_SKILL_ID}"
    --matched-replay-workers "${MATCHED_REPLAY_WORKERS}"
  )
fi
"${CMD[@]}" | tee "${RUN_DIR}/summary.json"

PYTHONPATH="${CLEAN_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python "${CLEAN_REPO}/scripts/record_source_provenance.py" \
  --source-repo "${SOURCE_REPO}" \
  --clean-repo "${CLEAN_REPO}" \
  --output "${RUN_DIR}/source_provenance_at_end.json"

echo "PHASE1_COMPLETE ${CONDITION} ${GAME} ${RUN_DIR}"
