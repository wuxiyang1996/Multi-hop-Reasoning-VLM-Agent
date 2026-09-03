#!/usr/bin/env bash
#SBATCH --job-name=ws-qwen9b-v1
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=18:00:00

set -Eeuo pipefail

TARGET_REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SOURCE_RUNTIME_REPO="${SOURCE_RUNTIME_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
CONFIG="${CONFIG:-${TARGET_REPO_ROOT}/configs/webshop_qwen35_9b_scale_v1_development.json}"
MODE="${MODE:-smoke}"
MODEL_PORT="${MODEL_PORT:-18180}"
WEBSHOP_PORT="${WEBSHOP_PORT:-3000}"
MODEL_ENV="${MODEL_ENV:-cosplay-candy-a100}"
TARGET_ENV="${TARGET_ENV:-vlm_benchmarks}"
SCRATCH_ROOT="/scratch0/${USER}/webshop-qwen9b-${SLURM_JOB_ID}"
LOCAL_HF_HOME="${SCRATCH_ROOT}/hf"
EMPTY_ADAPTER_DIR="${SCRATCH_ROOT}/empty-adapters"

case "${MODE}" in
  smoke|qualification) ;;
  *) echo "ERROR MODE must be smoke or qualification" >&2; exit 2 ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-${TARGET_REPO_ROOT}/runs/webshop_qwen35_9b_scale_v1/${MODE}}"
[[ -f "${CONFIG}" ]] || { echo "ERROR missing ${CONFIG}" >&2; exit 2; }
[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "ERROR refusing to overwrite ${OUTPUT_DIR}" >&2
  exit 2
}

MODEL_CONTEXT_LENGTH="${MODEL_CONTEXT_LENGTH:-$(jq -r '.runtime.model_context_length' "${CONFIG}")}"
MAXIMUM_OUTPUT_TOKENS="${MAXIMUM_OUTPUT_TOKENS:-$(jq -r '.runtime.maximum_output_tokens' "${CONFIG}")}"
MAXIMUM_STEPS="${MAXIMUM_STEPS:-$(jq -r '.runtime.maximum_steps' "${CONFIG}")}"
SCHEMA_RETRIES="${SCHEMA_RETRIES:-$(jq -r '.runtime.schema_retries' "${CONFIG}")}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-$(jq -r '.runtime.candidate_count' "${CONFIG}")}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$(jq -r '.runtime.gpu_memory_utilization' "${CONFIG}")}"
ENABLE_THINKING="${ENABLE_THINKING:-$(
  jq -r 'if .runtime.enable_thinking == false then "false" else "true" end' "${CONFIG}"
)}"
for value in "${MODEL_CONTEXT_LENGTH}" "${MAXIMUM_OUTPUT_TOKENS}" \
  "${MAXIMUM_STEPS}" "${SCHEMA_RETRIES}" "${CANDIDATE_COUNT}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR invalid positive integer runtime value: ${value}" >&2
    exit 2
  }
done
[[ "${GPU_MEMORY_UTILIZATION}" =~ ^0\.[0-9]+$ ]] || {
  echo "ERROR invalid gpu_memory_utilization: ${GPU_MEMORY_UTILIZATION}" >&2
  exit 2
}
[[ "${ENABLE_THINKING}" == "true" || "${ENABLE_THINKING}" == "false" ]] || {
  echo "ERROR invalid enable_thinking boolean: ${ENABLE_THINKING}" >&2
  exit 2
}
THINKING_ARGS=()
if [[ "${ENABLE_THINKING}" == "false" ]]; then
  THINKING_ARGS+=(--disable-thinking)
fi
mkdir -p "${OUTPUT_DIR}/logs" "${SCRATCH_ROOT}" "${EMPTY_ADAPTER_DIR}"

MODEL_PID=""
WEBSHOP_PID=""
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ -n "${WEBSHOP_PID}" ]]; then
    kill "${WEBSHOP_PID}" 2>/dev/null || true
    wait "${WEBSHOP_PID}" 2>/dev/null || true
  fi
  if [[ -n "${MODEL_PID}" ]]; then
    kill "${MODEL_PID}" 2>/dev/null || true
    wait "${MODEL_PID}" 2>/dev/null || true
  fi
  if [[ -d "${SCRATCH_ROOT}" && "${SCRATCH_ROOT}" == "/scratch0/${USER}/webshop-qwen9b-${SLURM_JOB_ID}" ]]; then
    rm -rf -- "${SCRATCH_ROOT}"
  fi
  exit "${code}"
}
trap cleanup EXIT INT TERM

scontrol show job "${SLURM_JOB_ID}" >"${OUTPUT_DIR}/slurm_job_at_start.txt"
sha256sum "${CONFIG}" >"${OUTPUT_DIR}/config_file_sha256.txt"
git -C "${TARGET_REPO_ROOT}" rev-parse HEAD >"${OUTPUT_DIR}/git_head_at_start.txt"
git -C "${TARGET_REPO_ROOT}" status --short >"${OUTPUT_DIR}/git_status_at_start.txt"

mapfile -t TASK_IDS < <(
  python - "${CONFIG}" "${MODE}" <<'PY'
import json
import sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
key = "smoke_tasks" if sys.argv[2] == "smoke" else "qualification_tasks"
for task_id in config["task_policy"][key]:
    print(task_id)
PY
)
[[ "${#TASK_IDS[@]}" -gt 0 ]] || { echo "ERROR no frozen tasks" >&2; exit 2; }
printf '%s\n' "${TASK_IDS[@]}" >"${OUTPUT_DIR}/task_ids.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MODEL_ENV}"
SOURCE_HF_HOME="${SHARED_HF_HOME}" LOCAL_HF_HOME="${LOCAL_HF_HOME}" \
  source "${SOURCE_RUNTIME_REPO}/scripts/stage_hf_model_cache.sh" "${MODEL}"
export HF_HOME="${LOCAL_HF_HOME}"
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_MAX_MODEL_LEN="${MODEL_CONTEXT_LENGTH}"
export VLLM_MAX_NUM_SEQS=8
export VLLM_MAX_NUM_BATCHED_TOKENS="${MODEL_CONTEXT_LENGTH}"

cd "${SOURCE_RUNTIME_REPO}"
python scripts/serve_rollout_pool.py \
  --model "${MODEL}" \
  --adapter-dir "${EMPTY_ADAPTER_DIR}" \
  --gpus 0 \
  --base-port "${MODEL_PORT}" \
  --gpu-util "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MODEL_CONTEXT_LENGTH}" \
  --max-num-seqs 8 \
  --max-num-batched-tokens "${MODEL_CONTEXT_LENGTH}" \
  --log-dir "${OUTPUT_DIR}/logs/vllm" \
  >"${OUTPUT_DIR}/logs/vllm_launcher.log" 2>&1 &
MODEL_PID=$!

deadline=$((SECONDS + 1800))
while (( SECONDS < deadline )); do
  if curl -fsS -m 5 "http://127.0.0.1:${MODEL_PORT}/v1/models" \
      >"${OUTPUT_DIR}/served_models.json"; then
    break
  fi
  kill -0 "${MODEL_PID}" 2>/dev/null || {
    echo "ERROR vLLM exited during startup" >&2
    exit 3
  }
  sleep 5
done
[[ -s "${OUTPUT_DIR}/served_models.json" ]] || {
  echo "ERROR vLLM startup timeout" >&2
  exit 3
}
jq -e --arg model "${MODEL}" \
  '.data | any(.id == $model)' "${OUTPUT_DIR}/served_models.json" >/dev/null

conda activate "${TARGET_ENV}"
cd "${TARGET_REPO_ROOT}"
PYTHONPATH="${TARGET_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python scripts/run_webshop_direct_server_v4.py \
    --goal-seed 233 --host 127.0.0.1 --port "${WEBSHOP_PORT}" \
    >"${OUTPUT_DIR}/logs/webshop_server.log" 2>&1 &
WEBSHOP_PID=$!

deadline=$((SECONDS + 600))
while (( SECONDS < deadline )); do
  if curl -fsS -m 5 "http://127.0.0.1:${WEBSHOP_PORT}/" >/dev/null; then
    break
  fi
  kill -0 "${WEBSHOP_PID}" 2>/dev/null || {
    echo "ERROR WebShop server exited during startup" >&2
    exit 3
  }
  sleep 2
done
curl -fsS -m 5 "http://127.0.0.1:${WEBSHOP_PORT}/" >/dev/null || {
  echo "ERROR WebShop server startup timeout" >&2
  exit 3
}

set +e
PYTHONPATH="${TARGET_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python scripts/run_webshop_search_automaton_v16.py \
    --task-ids "${TASK_IDS[@]}" \
    --role development \
    --model "${MODEL}" \
    --base-url "http://127.0.0.1:${MODEL_PORT}/v1" \
    --cache-seed "${SCRATCH_ROOT}/no-cache-seed.json" \
    --output-dir "${OUTPUT_DIR}" \
    --maximum-output-tokens "${MAXIMUM_OUTPUT_TOKENS}" \
    --schema-retries "${SCHEMA_RETRIES}" \
    --candidate-count "${CANDIDATE_COUNT}" \
    --maximum-steps "${MAXIMUM_STEPS}" \
    "${THINKING_ARGS[@]}" \
    --run-id "webshop-qwen35-9b-scale-v1-${MODE}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/runner.log"
RUNNER_CODE=${PIPESTATUS[0]}
set -e
echo "${RUNNER_CODE}" >"${OUTPUT_DIR}/runner_exit_code.txt"

[[ -f "${OUTPUT_DIR}/report.json" ]] || {
  echo "ERROR runner produced no report (exit ${RUNNER_CODE})" >&2
  exit 4
}
jq -e '.summaries | length == 5' "${OUTPUT_DIR}/report.json" >/dev/null
jq -e '[.summaries[].failures] | add == 0' "${OUTPUT_DIR}/report.json" >/dev/null

if [[ "${MODE}" == "smoke" ]]; then
  [[ "$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'webshop.0.*.json' | wc -l)" -eq 5 ]] || {
    echo "ERROR smoke did not produce five matched receipts" >&2
    exit 4
  }
fi

jq -r '.status' "${OUTPUT_DIR}/report.json" >"${OUTPUT_DIR}/scientific_status.txt"
echo "WEBSHOP_QWEN35_9B_${MODE^^}_COMPLETE ${OUTPUT_DIR}"
