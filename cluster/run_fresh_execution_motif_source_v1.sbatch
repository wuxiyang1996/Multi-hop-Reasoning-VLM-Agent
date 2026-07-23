#!/usr/bin/env bash
#SBATCH --job-name=fresh-motif-src
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
CLEAN_REPO="${CLEAN_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CLEAN_REPO}/runs/fresh_source_execution_motif_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main/runs/b2_best_checkpoints}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
GYMV_CONDA_ENV="${GYMV_CONDA_ENV:-gymv}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
EPISODES="${EPISODES:-12}"
MAX_STEPS="${MAX_STEPS:-100}"
PORT="${PORT:-8000}"
STAGE_MODEL_CACHE="${STAGE_MODEL_CACHE:-1}"

GAMES=(gymv_strider gymv_thunder_force_iii)
SKILLS=(COMMIT/POSITION COMMIT/CLEAR)
SEEDS=(91001 92001)
GAME="${GAMES[${SLURM_ARRAY_TASK_ID}]}"
MATCHED_SKILL="${SKILLS[${SLURM_ARRAY_TASK_ID}]}"
SEED_BASE="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
CHECKPOINT="${CHECKPOINT_ROOT}/${GAME}"
SKILL_BANK="${CHECKPOINT}/banks/${GAME}/skill_bank.jsonl"
RUN_DIR="${OUTPUT_ROOT}/${GAME}"
LOCAL_HF_HOME="/scratch0/${USER}/fresh-motif-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}/hf"

for path in \
    "${CHECKPOINT}/metadata.json" \
    "${CHECKPOINT}/adapters/decision/skill_selection/adapter_model.safetensors" \
    "${CHECKPOINT}/adapters/decision/action_taking/adapter_model.safetensors" \
    "${CHECKPOINT}/adapters/skillbank/segment/adapter_model.safetensors" \
    "${CHECKPOINT}/adapters/skillbank/contract/adapter_model.safetensors" \
    "${CHECKPOINT}/adapters/skillbank/curator/adapter_model.safetensors" \
    "${SKILL_BANK}"; do
    [[ -f "${path}" ]] || { echo "ERROR missing ${path}" >&2; exit 2; }
done
[[ ! -e "${RUN_DIR}" ]] || {
    echo "ERROR refusing to overwrite ${RUN_DIR}" >&2
    exit 2
}
mkdir -p "${RUN_DIR}/logs"
scontrol show job "${SLURM_JOB_ID}" >"${RUN_DIR}/slurm_job_at_start.txt"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv \
    >"${RUN_DIR}/nvidia_smi_start.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

cleanup_staged_cache() {
    local scratch_root="/scratch0/${USER}/fresh-motif-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
    if [[ "${STAGE_MODEL_CACHE}" == "1" && -d "${scratch_root}" ]]; then
        rm -rf -- "${scratch_root}"
    fi
}

GYMV_PYTHON="$(conda run -n "${GYMV_CONDA_ENV}" python -c 'import sys; print(sys.executable)')"
[[ -x "${GYMV_PYTHON}" ]] || {
    echo "ERROR invalid Gym-V interpreter: ${GYMV_PYTHON}" >&2
    exit 2
}
export GYMV_PYTHON

if [[ "${STAGE_MODEL_CACHE}" == "1" ]]; then
    SOURCE_HF_HOME="${SHARED_HF_HOME}" LOCAL_HF_HOME="${LOCAL_HF_HOME}" \
        source scripts/stage_hf_model_cache.sh "${MODEL}"
    export HF_HOME="${LOCAL_HF_HOME}"
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

SERVE_ADAPTER_DIR="${RUN_DIR}/served_adapters"
mkdir -p "${SERVE_ADAPTER_DIR}"
ln -s "${CHECKPOINT}/adapters/decision" "${SERVE_ADAPTER_DIR}/decision"

python scripts/serve_rollout_pool.py \
    --model "${MODEL}" \
    --adapter-dir "${SERVE_ADAPTER_DIR}" \
    --gpus 0 \
    --base-port "${PORT}" \
    --gpu-util 0.82 \
    --log-dir "${RUN_DIR}/logs/vllm" \
    >"${RUN_DIR}/logs/vllm_launcher.log" 2>&1 &
SERVER_PID=$!
cleanup() {
    local code=$?
    trap - EXIT INT TERM
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    cleanup_staged_cache
    exit "${code}"
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + 1800))
while (( SECONDS < deadline )); do
    if curl -fsS -m 5 "http://127.0.0.1:${PORT}/v1/models" \
        >"${RUN_DIR}/served_models.json"; then
        break
    fi
    kill -0 "${SERVER_PID}" 2>/dev/null || {
        echo "ERROR vLLM exited during startup" >&2
        exit 3
    }
    sleep 5
done
[[ -s "${RUN_DIR}/served_models.json" ]] || {
    echo "ERROR vLLM startup timeout" >&2
    exit 3
}

python - "${RUN_DIR}/served_models.json" <<'PY'
import json, sys
models = {row["id"] for row in json.load(open(sys.argv[1]))["data"]}
missing = sorted({"skill_selection", "action_taking"} - models)
if missing:
    raise SystemExit(f"checkpoint adapters not served: {missing}")
PY

python scripts/run_instrumented_source_smoke.py \
    --output "${RUN_DIR}/evidence" \
    --game "${GAME}" \
    --episodes "${EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --seed-base "${SEED_BASE}" \
    --endpoint "http://127.0.0.1:${PORT}/v1" \
    --model "${MODEL}" \
    --checkpoint "${CHECKPOINT}" \
    --skill-bank "${SKILL_BANK}" \
    --matched-policy-skill-id "${MATCHED_SKILL}" \
    --matched-replay-workers 8 \
    --sequential-episodes \
    | tee "${RUN_DIR}/summary.json"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
    >"${RUN_DIR}/nvidia_smi_end.txt"
echo "FRESH_EXECUTION_MOTIF_SOURCE_COMPLETE ${GAME} ${RUN_DIR}"
