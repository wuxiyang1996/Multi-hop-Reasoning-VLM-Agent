#!/usr/bin/env bash
#SBATCH --job-name=best-ckpt-src-smoke
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00

# Fail-closed source-rollout smoke for a full co-evolution checkpoint.
# This is a checkpoint-load/provenance test, not a performance estimate.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/runs/grpo_visual_default_2x4_a6000_20260720_v9/checkpoints/step_99800}"
SKILL_BANK="${SKILL_BANK:-${CHECKPOINT}/banks/candy_crush/skill_bank.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/best_checkpoint_source_smoke_${SLURM_JOB_ID}}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
LOCAL_HF_HOME="/scratch0/${USER}/best-ckpt-source-${SLURM_JOB_ID}/hf"
STAGE_MODEL_CACHE="${STAGE_MODEL_CACHE:-1}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
GYMV_CONDA_ENV="${GYMV_CONDA_ENV:-gymv}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
GAME="${GAME:-candy_crush}"
EPISODES="${EPISODES:-2}"
MAX_STEPS="${MAX_STEPS:-12}"
SEED_BASE="${SEED_BASE:-142}"
PORT="${PORT:-8000}"
# Historical checkpoints may contain skill-bank LoRAs that predate full
# Qwen3.5 packed-projection coverage.  Source rollouts only call the two
# decision adapters; segment/contract/curator remain preserved and hashed in
# the checkpoint receipt but are not loaded into vLLM unless explicitly asked.
SERVE_ADAPTER_SCOPE="${SERVE_ADAPTER_SCOPE:-decision}"

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
[[ ! -e "${RUN_DIR}" ]] || { echo "ERROR refusing to overwrite ${RUN_DIR}" >&2; exit 2; }
mkdir -p "${RUN_DIR}/logs"
scontrol show job "${SLURM_JOB_ID}" >"${RUN_DIR}/slurm_job_at_start.txt"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv \
    >"${RUN_DIR}/nvidia_smi_start.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

cleanup_staged_cache() {
    if [[ "${STAGE_MODEL_CACHE}" == "1" \
          && "${LOCAL_HF_HOME}" == "/scratch0/${USER}/best-ckpt-source-${SLURM_JOB_ID}/hf" ]]; then
        rm -rf -- "/scratch0/${USER}/best-ckpt-source-${SLURM_JOB_ID}"
    fi
}
trap cleanup_staged_cache EXIT

# The rollout/model process and the patched Gym-V environments intentionally
# live in separate Conda environments.  SubprocessEnv preserves this boundary
# through an explicit interpreter instead of relying on PATH inheritance.
if [[ "${GAME}" == gymv_* ]]; then
    GYMV_PYTHON="$(conda run -n "${GYMV_CONDA_ENV}" python -c 'import sys; print(sys.executable)')"
    [[ -x "${GYMV_PYTHON}" ]] || {
        echo "ERROR invalid Gym-V interpreter: ${GYMV_PYTHON}" >&2
        exit 2
    }
    "${GYMV_PYTHON}" -c \
        "import gym_v, gym_v.envs; assert any(e.startswith('Temporal/') for e in gym_v.registry)"
    export GYMV_PYTHON
fi

if [[ "${STAGE_MODEL_CACHE}" == "1" ]]; then
    SOURCE_HF_HOME="${SHARED_HF_HOME}" LOCAL_HF_HOME="${LOCAL_HF_HOME}" \
        source scripts/stage_hf_model_cache.sh "${MODEL}"
    export HF_HOME="${LOCAL_HF_HOME}"
elif [[ "${STAGE_MODEL_CACHE}" == "0" ]]; then
    export HF_HOME="${SHARED_HF_HOME}"
else
    echo "ERROR STAGE_MODEL_CACHE must be 0 or 1, got ${STAGE_MODEL_CACHE}" >&2
    exit 2
fi
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_MAX_MODEL_LEN=4096
export VLLM_MAX_NUM_SEQS=16
export VLLM_MAX_NUM_BATCHED_TOKENS=4096

if [[ "${SERVE_ADAPTER_SCOPE}" == "decision" ]]; then
    SERVE_ADAPTER_DIR="${RUN_DIR}/served_adapters"
    mkdir -p "${SERVE_ADAPTER_DIR}"
    ln -s "${CHECKPOINT}/adapters/decision" "${SERVE_ADAPTER_DIR}/decision"
    REQUIRED_ADAPTERS="skill_selection,action_taking"
elif [[ "${SERVE_ADAPTER_SCOPE}" == "all" ]]; then
    SERVE_ADAPTER_DIR="${CHECKPOINT}/adapters"
    REQUIRED_ADAPTERS="skill_selection,action_taking,segment,contract,curator"
else
    echo "ERROR invalid SERVE_ADAPTER_SCOPE=${SERVE_ADAPTER_SCOPE}" >&2
    exit 2
fi

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

python - "${RUN_DIR}/served_models.json" "${REQUIRED_ADAPTERS}" <<'PY'
import json, sys
models = {row["id"] for row in json.load(open(sys.argv[1]))["data"]}
required = set(sys.argv[2].split(","))
missing = sorted(required - models)
if missing:
    raise SystemExit(f"checkpoint adapters not served: {missing}; got={sorted(models)}")
print(f"served all checkpoint adapters: {sorted(required)}")
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
    | tee "${RUN_DIR}/summary.json"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
    >"${RUN_DIR}/nvidia_smi_end.txt"
echo "BEST_CHECKPOINT_SOURCE_SMOKE_COMPLETE ${RUN_DIR}"
