#!/usr/bin/env bash
#SBATCH --job-name=grpo-smoke-a6000-2x4
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00

# Node 0: four persistent Qwen3.5-9B TP=1 rollout replicas.
# Node 1: Qwen3.5-9B LoRA GRPO with FSDP=4.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main/runs}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/grpo_smoke_2x4_a6000_${SLURM_JOB_ID}}"
WARM_RUN="${WARM_RUN:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main/runs/candy_crush_coevo_v4_20260519_093912}"
WARM_ADAPTER_DIR="${WARM_RUN}/lora_adapters"
WARM_BANK_DIR="${WARM_RUN}/skillbank"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
TOTAL_STEPS="${TOTAL_STEPS:-3}"
EPISODES="${EPISODES:-2}"
MAX_EPISODE_STEPS="${COEVO_MAX_EPISODE_STEPS:-12}"
GPU_GRES="${GPU_GRES:-gpu:rtxa6000}"

if [[ ! -x "${REPO_ROOT}/scripts/audit_grpo_smoke.py" ]]; then
    echo "ERROR: missing GRPO smoke audit in ${REPO_ROOT}" >&2
    exit 2
fi
for path in \
    "${WARM_ADAPTER_DIR}/decision/skill_selection/adapter_config.json" \
    "${WARM_ADAPTER_DIR}/decision/action_taking/adapter_config.json" \
    "${WARM_ADAPTER_DIR}/skillbank/segment/adapter_config.json" \
    "${WARM_ADAPTER_DIR}/skillbank/contract/adapter_config.json" \
    "${WARM_ADAPTER_DIR}/skillbank/curator/adapter_config.json" \
    "${WARM_BANK_DIR}/candy_crush/skill_bank.jsonl"; do
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: missing warm-start input: ${path}" >&2
        exit 2
    fi
done
if [[ -e "${RUN_DIR}" ]]; then
    echo "ERROR: refusing to overwrite ${RUN_DIR}" >&2
    exit 2
fi
mkdir -p "${RUN_DIR}/logs"
scontrol show job "${SLURM_JOB_ID}" >"${RUN_DIR}/slurm_job_at_start.txt"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#NODES[@]}" -ne 2 ]]; then
    echo "ERROR: expected exactly two nodes, got ${#NODES[@]}" >&2
    exit 2
fi
ROLLOUT_NODE="${NODES[0]}"
TRAIN_NODE="${NODES[1]}"
ROLLOUT_HF_HOME="/scratch0/${USER}/grpo-smoke-${SLURM_JOB_ID}/hf"
TRAIN_HF_HOME="/scratch0/${USER}/grpo-smoke-${SLURM_JOB_ID}/hf"
ROLLOUT_URLS="http://${ROLLOUT_NODE}:8000/v1,http://${ROLLOUT_NODE}:8010/v1,http://${ROLLOUT_NODE}:8020/v1,http://${ROLLOUT_NODE}:8030/v1"

echo "run_dir=${RUN_DIR}"
echo "rollout_node=${ROLLOUT_NODE} train_node=${TRAIN_NODE}"
echo "layout=4x9B-rollout + 4xFSDP-GRPO"
echo "steps=${TOTAL_STEPS} episodes=${EPISODES} max_episode_steps=${MAX_EPISODE_STEPS}"

for node in "${ROLLOUT_NODE}" "${TRAIN_NODE}"; do
    srun --exact --mem=0 --gres="${GPU_GRES}:4" -N1 -n1 -c1 -w "${node}" \
        env REPO_ROOT="${REPO_ROOT}" CONDA_ENV="${CONDA_ENV}" \
        bash -lc 'nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; exec python "$REPO_ROOT/scripts/verify_cuda_devices.py" 4' \
        >"${RUN_DIR}/nvidia_smi_start_${node}.txt" &
done
wait

echo "Staging Qwen3.5-9B to node-local NVMe..."
srun --exact --mem=0 --gres=none -N1 -n1 -c4 -w "${ROLLOUT_NODE}" \
    env REPO_ROOT="${REPO_ROOT}" SOURCE_HF_HOME="${SHARED_HF_HOME}" \
        LOCAL_HF_HOME="${ROLLOUT_HF_HOME}" \
    bash -lc 'cd "$REPO_ROOT"; source scripts/stage_hf_model_cache.sh Qwen/Qwen3.5-9B' \
    >"${RUN_DIR}/logs/stage_rollout_9b.log" 2>&1 &
stage_rollout=$!
srun --exact --mem=0 --gres=none -N1 -n1 -c4 -w "${TRAIN_NODE}" \
    env REPO_ROOT="${REPO_ROOT}" SOURCE_HF_HOME="${SHARED_HF_HOME}" \
        LOCAL_HF_HOME="${TRAIN_HF_HOME}" \
    bash -lc 'cd "$REPO_ROOT"; source scripts/stage_hf_model_cache.sh Qwen/Qwen3.5-9B' \
    >"${RUN_DIR}/logs/stage_train_9b.log" 2>&1 &
stage_train=$!
wait "${stage_rollout}" "${stage_train}"

ROLLOUT_PID=""
cleanup() {
    local code=$?
    trap - EXIT INT TERM
    if [[ -n "${ROLLOUT_PID}" ]] && kill -0 "${ROLLOUT_PID}" 2>/dev/null; then
        kill "${ROLLOUT_PID}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    exit "${code}"
}
trap cleanup EXIT INT TERM

srun --exact --mem=0 --gres="${GPU_GRES}:4" -N1 -n1 -c4 -w "${ROLLOUT_NODE}" \
    env REPO_ROOT="${REPO_ROOT}" CONDA_ENV="${CONDA_ENV}" \
        HF_HOME="${ROLLOUT_HF_HOME}" HF_HUB_OFFLINE=1 \
        VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_MAX_MODEL_LEN=4096 \
        VLLM_MAX_NUM_SEQS=16 VLLM_MAX_NUM_BATCHED_TOKENS=4096 \
        MODEL="${MODEL}" ADAPTER_DIR="${WARM_ADAPTER_DIR}" \
        LOG_DIR="${RUN_DIR}/logs/rollout" \
    bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; cd "$REPO_ROOT"; exec python scripts/serve_rollout_pool.py --model "$MODEL" --adapter-dir "$ADAPTER_DIR" --gpus 0 1 2 3 --base-port 8000 --gpu-util 0.82 --log-dir "$LOG_DIR"' \
    >"${RUN_DIR}/logs/rollout_launcher.log" 2>&1 &
ROLLOUT_PID=$!

wait_for_endpoint() {
    local url=$1
    local deadline=$((SECONDS + 1800))
    while (( SECONDS < deadline )); do
        if curl -fsS -m 3 "${url}/models" >/dev/null 2>&1; then
            echo "ready ${url}"
            return 0
        fi
        if ! kill -0 "${ROLLOUT_PID}" 2>/dev/null; then
            echo "ERROR: rollout pool exited during startup" >&2
            return 1
        fi
        sleep 5
    done
    echo "ERROR: timeout waiting for ${url}" >&2
    return 1
}
for port in 8000 8010 8020 8030; do
    wait_for_endpoint "http://${ROLLOUT_NODE}:${port}/v1"
done

echo "Starting three-step real-rollout GRPO smoke..."
srun --exact --mem=0 --gres="${GPU_GRES}:4" -N1 -n1 -c12 -w "${TRAIN_NODE}" \
    env REPO_ROOT="${REPO_ROOT}" CONDA_ENV="${CONDA_ENV}" \
        HF_HOME="${TRAIN_HF_HOME}" HF_HUB_OFFLINE=1 \
        VLLM_BASE_URLS="${ROLLOUT_URLS}" \
        COEVO_MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
        GRPO_FSDP_BATCH_SIZE=1 GRPO_REF_MICRO_BATCH=1 GRPO_MAX_SEQ_LEN=1024 \
        GRPO_NCCL_TIMEOUT_S=1200 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
        OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; cd "$REPO_ROOT"; exec python scripts/run_coevolution.py --games candy_crush --curriculum none --total-steps "$1" --episodes-per-game "$2" --episodes-per-game-overrides "{}" --max-concurrent "$2" --model Qwen/Qwen3.5-9B --temperature 0.7 --max-tokens 256 --no-manage-vllm --vllm-url "$3" --grpo-devices 0 1 2 3 --grpo-max-epochs 1 --run-dir "$4" --load-decision-adapters "$5/decision" --load-skillbank-adapters "$5/skillbank" --seed-bank-dir "$6" --checkpoint-interval 1 --no-wandb --no-vision-state-perception --no-cross-game-skill-transfer --thread-workers 12 --process-workers 2' runner \
        "${TOTAL_STEPS}" "${EPISODES}" "${ROLLOUT_URLS}" "${RUN_DIR}" \
        "${WARM_ADAPTER_DIR}" "${WARM_BANK_DIR}" \
    >"${RUN_DIR}/trainer.log" 2>&1

python "${REPO_ROOT}/scripts/audit_grpo_smoke.py" \
    --run-dir "${RUN_DIR}" --warm-adapter-dir "${WARM_ADAPTER_DIR}" \
    --expected-steps "${TOTAL_STEPS}" --output "${RUN_DIR}/smoke_audit.json" \
    >"${RUN_DIR}/logs/audit.log" 2>&1

kill "${ROLLOUT_PID}" 2>/dev/null || true
wait "${ROLLOUT_PID}" 2>/dev/null || true
ROLLOUT_PID=""
for node in "${ROLLOUT_NODE}" "${TRAIN_NODE}"; do
    srun --exact --mem=0 --gres="${GPU_GRES}:4" -N1 -n1 -c1 -w "${node}" \
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
        >"${RUN_DIR}/nvidia_smi_end_${node}.txt" &
done
wait
scontrol show job "${SLURM_JOB_ID}" >"${RUN_DIR}/slurm_job_at_end.txt"
echo "GRPO_SMOKE_COMPLETE ${RUN_DIR}"
cat "${RUN_DIR}/smoke_audit.json"
