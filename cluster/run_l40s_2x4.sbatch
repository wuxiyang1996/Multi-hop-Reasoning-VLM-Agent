#!/usr/bin/env bash
#SBATCH --job-name=mh-vlm-l40s-2x4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

# Two-node layout:
#   node 0, GPUs 0-3: four Qwen3.5-9B TP=1 rollout replicas
#   node 1, GPUs 0-1: Qwen3.5-9B LoRA GRPO with FSDP=2
#   node 1, GPUs 2-3: Qwen3.5-35B-A3B judge with TP=2 + FP8

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.5-35B-A3B}"
RUN_MODE="${RUN_MODE:-smoke}"                 # smoke | train
START_MODE="${START_MODE:-from_scratch}"      # from_scratch | resume
GAMES="${GAMES:-twenty_forty_eight}"
CURRICULUM="${CURRICULUM:-none}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"
EPISODES="${EPISODES:-4}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/l40s_2x4_${SLURM_JOB_ID}}"
SHARED_HF_HOME="${SHARED_HF_HOME:-${REPO_ROOT}/.hf_cache}"
STAGE_LOCAL_CACHE="${STAGE_LOCAL_CACHE:-1}"
NO_WANDB="${NO_WANDB:-1}"
DEBUG_IO="${DEBUG_IO:-0}"

if [[ ! -f "${REPO_ROOT}/scripts/run_coevolution.py" ]]; then
    echo "ERROR: REPO_ROOT is not this repository: ${REPO_ROOT}" >&2
    exit 2
fi
if [[ "${RUN_MODE}" != "smoke" && "${RUN_MODE}" != "train" ]]; then
    echo "ERROR: RUN_MODE must be smoke or train" >&2
    exit 2
fi
if [[ "${START_MODE}" != "from_scratch" && "${START_MODE}" != "resume" ]]; then
    echo "ERROR: START_MODE must be from_scratch or resume" >&2
    exit 2
fi
if [[ "${START_MODE}" == "from_scratch" && -e "${RUN_DIR}" ]]; then
    echo "ERROR: refusing to overwrite existing from-scratch RUN_DIR: ${RUN_DIR}" >&2
    exit 2
fi

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#NODES[@]}" -ne 2 ]]; then
    echo "ERROR: expected exactly 2 nodes, got ${#NODES[@]}" >&2
    exit 2
fi
ROLLOUT_NODE="${NODES[0]}"
TRAIN_NODE="${NODES[1]}"
ROLLOUT_URLS="http://${ROLLOUT_NODE}:8000/v1,http://${ROLLOUT_NODE}:8010/v1,http://${ROLLOUT_NODE}:8020/v1,http://${ROLLOUT_NODE}:8030/v1"
JUDGE_URL="http://${TRAIN_NODE}:8004/v1"
BOOTSTRAP_ADAPTER_DIR="${RUN_DIR}/bootstrap_adapters"
TRAIN_ADAPTER_DIR="${RUN_DIR}/lora_adapters"
ROLLOUT_HF_HOME="/scratch0/${USER}/mh-vlm-hf-cache"
TRAIN_HF_HOME="/scratch0/${USER}/mh-vlm-hf-cache"
mkdir -p "${RUN_DIR}/logs"

ROLLOUT_PID=""
JUDGE_PID=""
TRAIN_PID=""
cleanup() {
    local code=$?
    trap - EXIT INT TERM
    for pid in "${TRAIN_PID}" "${JUDGE_PID}" "${ROLLOUT_PID}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    exit "${code}"
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "2x4 L40S co-evolution job ${SLURM_JOB_ID}"
echo "rollout node: ${ROLLOUT_NODE} (GPUs 0-3)"
echo "train node:   ${TRAIN_NODE} (FSDP GPUs 0-1; judge GPUs 2-3)"
echo "run dir:      ${RUN_DIR}"
echo "mode:         ${RUN_MODE}/${START_MODE}"
echo "============================================================"

MODEL_CACHE_NAME="models--${MODEL//\//--}"
USE_LOCAL_CACHE=0
if [[ "${STAGE_LOCAL_CACHE}" == "1" && -d "${SHARED_HF_HOME}/hub/${MODEL_CACHE_NAME}" ]]; then
    echo "Staging ${MODEL} to node-local NVMe on both nodes..."
    srun --overlap -N1 -n1 -c2 -w "${ROLLOUT_NODE}" \
        env REPO_ROOT="${REPO_ROOT}" SOURCE_HF_HOME="${SHARED_HF_HOME}" \
            LOCAL_HF_HOME="${ROLLOUT_HF_HOME}" MODEL="${MODEL}" \
        bash -lc 'cd "$REPO_ROOT"; source scripts/stage_hf_model_cache.sh "$MODEL"' \
        >"${RUN_DIR}/logs/stage_rollout.log" 2>&1 &
    stage_rollout_pid=$!
    srun --overlap -N1 -n1 -c2 -w "${TRAIN_NODE}" \
        env REPO_ROOT="${REPO_ROOT}" SOURCE_HF_HOME="${SHARED_HF_HOME}" \
            LOCAL_HF_HOME="${TRAIN_HF_HOME}" MODEL="${MODEL}" \
        bash -lc 'cd "$REPO_ROOT"; source scripts/stage_hf_model_cache.sh "$MODEL"' \
        >"${RUN_DIR}/logs/stage_train.log" 2>&1 &
    stage_train_pid=$!
    wait "${stage_rollout_pid}"
    wait "${stage_train_pid}"
    USE_LOCAL_CACHE=1
else
    ROLLOUT_HF_HOME="${SHARED_HF_HOME}"
    TRAIN_HF_HOME="${SHARED_HF_HOME}"
    echo "Node-local staging skipped; using ${SHARED_HF_HOME}"
fi

if [[ "${START_MODE}" == "from_scratch" ]]; then
    echo "Preparing fresh random LoRA adapters..."
    srun --overlap -N1 -n1 -c8 -w "${TRAIN_NODE}" \
        env REPO_ROOT="${REPO_ROOT}" CONDA_ENV="${CONDA_ENV}" \
            HF_HOME="${TRAIN_HF_HOME}" HF_HUB_OFFLINE="${USE_LOCAL_CACHE}" \
            MODEL="${MODEL}" ADAPTER_DIR="${BOOTSTRAP_ADAPTER_DIR}" \
            WORK_DIR="${RUN_DIR}/bootstrap_init" \
        bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; cd "$REPO_ROOT"; exec python scripts/prepare_random_adapters.py --model "$MODEL" --adapter-dir "$ADAPTER_DIR" --work-dir "$WORK_DIR"' \
        >"${RUN_DIR}/logs/prepare_adapters.log" 2>&1
    SERVE_ADAPTER_DIR="${BOOTSTRAP_ADAPTER_DIR}"
else
    SERVE_ADAPTER_DIR="${TRAIN_ADAPTER_DIR}"
    if [[ ! -f "${SERVE_ADAPTER_DIR}/decision/action_taking/adapter_config.json" ]]; then
        echo "ERROR: resume adapters not found under ${SERVE_ADAPTER_DIR}" >&2
        exit 2
    fi
fi

echo "Starting four rollout replicas on ${ROLLOUT_NODE}..."
srun --overlap -N1 -n1 -c16 -w "${ROLLOUT_NODE}" \
    env REPO_ROOT="${REPO_ROOT}" CONDA_ENV="${CONDA_ENV}" \
        HF_HOME="${ROLLOUT_HF_HOME}" HF_HUB_OFFLINE="${USE_LOCAL_CACHE}" \
        VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_MAX_MODEL_LEN=8192 \
        VLLM_MAX_NUM_SEQS=32 VLLM_MAX_NUM_BATCHED_TOKENS=8192 \
        MODEL="${MODEL}" ADAPTER_DIR="${SERVE_ADAPTER_DIR}" \
        LOG_DIR="${RUN_DIR}/logs/rollout" \
    bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; cd "$REPO_ROOT"; exec python scripts/serve_rollout_pool.py --model "$MODEL" --adapter-dir "$ADAPTER_DIR" --gpus 0 1 2 3 --base-port 8000 --gpu-util 0.85 --log-dir "$LOG_DIR"' \
    >"${RUN_DIR}/logs/rollout_launcher.log" 2>&1 &
ROLLOUT_PID=$!

echo "Starting 35B-A3B TP=2 FP8 judge on ${TRAIN_NODE} GPUs 2,3..."
srun --overlap -N1 -n1 -c8 -w "${TRAIN_NODE}" \
    env REPO_ROOT="${REPO_ROOT}" CONDA_ENV="${CONDA_ENV}" \
        HF_HOME="${SHARED_HF_HOME}" VLLM_USE_FLASHINFER_SAMPLER=0 \
        MODEL="${JUDGE_MODEL}" GPUS=2,3 TENSOR_PARALLEL=2 EXPERT_PARALLEL=1 \
        GPU_UTIL=0.90 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=4 PORT=8004 \
        HOST=0.0.0.0 QUANTIZATION=fp8 SPECULATIVE=none \
    bash -lc 'cd "$REPO_ROOT"; exec bash inference/serve_qwen35_35b_a3b.sh' \
    >"${RUN_DIR}/logs/judge.log" 2>&1 &
JUDGE_PID=$!

wait_for_endpoint() {
    local url=$1
    local owner_pid=$2
    local label=$3
    local deadline=$((SECONDS + 1800))
    while (( SECONDS < deadline )); do
        if curl -fsS -m 3 "${url}/models" >/dev/null 2>&1; then
            echo "${label} ready: ${url}"
            return 0
        fi
        if ! kill -0 "${owner_pid}" 2>/dev/null; then
            echo "ERROR: ${label} process exited during startup" >&2
            return 1
        fi
        sleep 5
    done
    echo "ERROR: timed out waiting for ${label}: ${url}" >&2
    return 1
}

for port in 8000 8010 8020 8030; do
    wait_for_endpoint "http://${ROLLOUT_NODE}:${port}/v1" "${ROLLOUT_PID}" "rollout:${port}"
done
wait_for_endpoint "${JUDGE_URL}" "${JUDGE_PID}" "judge"

read -r -a GAME_ARGS <<< "${GAMES}"
TRAIN_ARGS=(
    --games "${GAME_ARGS[@]}"
    --curriculum "${CURRICULUM}"
    --total-steps "${TOTAL_STEPS}"
    --episodes-per-game "${EPISODES}"
    --max-concurrent "${MAX_CONCURRENT}"
    --no-manage-vllm
    --vllm-url "${ROLLOUT_URLS}"
    --grpo-devices 0 1
    --run-dir "${RUN_DIR}"
    --adapter-dir "${TRAIN_ADAPTER_DIR}"
    --checkpoint-interval 1
)
if [[ "${START_MODE}" == "from_scratch" ]]; then
    TRAIN_ARGS+=(
        --from-scratch
        --load-decision-adapters "${BOOTSTRAP_ADAPTER_DIR}/decision"
        --load-skillbank-adapters "${BOOTSTRAP_ADAPTER_DIR}/skillbank"
    )
else
    TRAIN_ARGS+=(--resume)
fi
if [[ "${NO_WANDB}" == "1" ]]; then TRAIN_ARGS+=(--no-wandb); fi
if [[ "${DEBUG_IO}" == "1" ]]; then TRAIN_ARGS+=(--debug-io); fi

MAX_EPISODE_STEPS=""
if [[ "${RUN_MODE}" == "smoke" ]]; then
    MAX_EPISODE_STEPS="${COEVO_MAX_EPISODE_STEPS:-5}"
fi

echo "Starting trainer on ${TRAIN_NODE} GPUs 0,1..."
srun --overlap -N1 -n1 -c12 -w "${TRAIN_NODE}" \
    env REPO_ROOT="${REPO_ROOT}" CONDA_ENV="${CONDA_ENV}" \
        CUDA_VISIBLE_DEVICES=0,1 HF_HOME="${TRAIN_HF_HOME}" \
        HF_HUB_OFFLINE="${USE_LOCAL_CACHE}" \
        VLLM_BASE_URLS="${ROLLOUT_URLS}" \
        VLLM_BASE_URL_MAP="${JUDGE_MODEL}=${JUDGE_URL}" \
        VLM_AGENT_BACKBONE_JUDGE_MODEL="${JUDGE_MODEL}" \
        COEVO_MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
    bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; cd "$REPO_ROOT"; exec python scripts/run_coevolution.py "$@"' runner "${TRAIN_ARGS[@]}" \
    >"${RUN_DIR}/trainer.log" 2>&1 &
TRAIN_PID=$!
wait "${TRAIN_PID}"
TRAIN_PID=""
echo "Training complete: ${RUN_DIR}"
