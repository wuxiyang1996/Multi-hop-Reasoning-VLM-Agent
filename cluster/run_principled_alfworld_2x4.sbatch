#!/usr/bin/env bash
#SBATCH --job-name=skill-xfer-alf-2x4
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

# Preregistered two-node layout:
#   phase A / node 0 / GPUs 0-3: 4x 9B base rollout replicas
#   phase A / node 1 / GPUs 0-1: source-only Game SFT (one LoRA per GPU)
#   phase A / node 1 / GPUs 2-3: 35B TP=2 untrusted proposal audit
#   phase B / node 0 / GPUs 0-3: 4x 9B replicas with frozen Game SFT LoRAs
# Held-out ALFWorld clients are CPU-only and use frozen admission artifacts.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SPEC="${SPEC:-${REPO_ROOT}/configs/principled_alfworld_2x4_experiment.json}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/principled_alfworld_2x4_${SLURM_JOB_ID}}"
SHARED_HF_HOME="${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
TRAIN_ENV="${TRAIN_ENV:-cosplay-candy-a100}"
ALFWORLD_ENV="${ALFWORLD_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/alfworld}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PROPOSER_MODEL="${PROPOSER_MODEL:-Qwen/Qwen3.5-35B-A3B}"
ADMISSION_MANIFEST="${REPO_ROOT}/artifacts/admission/alfworld/manifest-c92a05274bacf88286a05a1be48c8d6bd48da6285038be7f01b1682d014e7cd1-eaa1bbf4214ff0d3.json"
BINDING_CONFIG="${REPO_ROOT}/configs/alfworld_one_shot_bindings.json"
TARGET_CONFIG="${REPO_ROOT}/configs/alfworld_pick_and_place_config.yaml"
SOURCE_SFT_DIR="${REPO_ROOT}/artifacts/source_only_game_sft"
SOURCE_SFT_MANIFEST="${SOURCE_SFT_DIR}/source_sft_manifest.json"
ADAPTER_DIR="${RUN_DIR}/source_only_lora"
EMPTY_ADAPTER_DIR="${RUN_DIR}/empty_adapters"

if [[ ! -f "${SPEC}" || ! -f "${ADMISSION_MANIFEST}" || ! -x "${REPO_ROOT}/scripts/eval_principled_alfworld.py" ]]; then
    echo "ERROR: repository or frozen experiment inputs are incomplete: ${REPO_ROOT}" >&2
    exit 2
fi
if [[ -e "${RUN_DIR}" ]]; then
    echo "ERROR: refusing to overwrite run directory: ${RUN_DIR}" >&2
    exit 2
fi
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/eval" "${EMPTY_ADAPTER_DIR}"
cp "${SPEC}" "${RUN_DIR}/experiment_spec.json"
scontrol show job "${SLURM_JOB_ID}" >"${RUN_DIR}/slurm_job_at_start.txt"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#NODES[@]}" -ne 2 ]]; then
    echo "ERROR: expected exactly two nodes, got ${#NODES[@]}" >&2
    exit 2
fi
ROLLOUT_NODE="${NODES[0]}"
TRAIN_NODE="${NODES[1]}"
ROLLOUT_HF_HOME="/scratch0/${USER}/principled-xfer-${SLURM_JOB_ID}/hf"
TRAIN_HF_HOME="/scratch0/${USER}/principled-xfer-${SLURM_JOB_ID}/hf"
ROLLOUT_URLS=(
    "http://${ROLLOUT_NODE}:8000/v1"
    "http://${ROLLOUT_NODE}:8010/v1"
    "http://${ROLLOUT_NODE}:8020/v1"
    "http://${ROLLOUT_NODE}:8030/v1"
)
PROPOSER_URL="http://${TRAIN_NODE}:8004/v1"

echo "experiment=${SPEC}"
echo "run_dir=${RUN_DIR}"
echo "rollout_node=${ROLLOUT_NODE} train_node=${TRAIN_NODE}"

# Freeze audit: the batch refuses to start if any preregistered byte drifted.
python -c 'import json,sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from scripts.aggregate_principled_alfworld import _verify_frozen_inputs; _verify_frozen_inputs(json.loads(Path(sys.argv[2]).read_text())); print("frozen inputs verified")' "${REPO_ROOT}" "${SPEC}"
python "${REPO_ROOT}/scripts/replay_source_programs.py" \
    --programs "${REPO_ROOT}/artifacts/source_evidence_index/source_programs.jsonl" \
    --output "${RUN_DIR}/source_replay_receipts.jsonl" \
    >"${RUN_DIR}/logs/source_replay.log" 2>&1

for node in "${ROLLOUT_NODE}" "${TRAIN_NODE}"; do
    srun --overlap --exact -N1 -n1 -c1 -w "${node}" nvidia-smi -q \
        >"${RUN_DIR}/nvidia_smi_start_${node}.txt" &
done
wait

BG_PIDS=()
cleanup() {
    local code=$?
    trap - EXIT INT TERM
    for pid in "${BG_PIDS[@]:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    exit "${code}"
}
trap cleanup EXIT INT TERM

echo "Staging 9B/35B caches to node-local NVMe..."
srun --overlap --exact -N1 -n1 -c4 -w "${ROLLOUT_NODE}" \
    env REPO_ROOT="${REPO_ROOT}" SOURCE_HF_HOME="${SHARED_HF_HOME}" \
        LOCAL_HF_HOME="${ROLLOUT_HF_HOME}" \
    bash -lc 'cd "$REPO_ROOT"; source scripts/stage_hf_model_cache.sh Qwen/Qwen3.5-9B' \
    >"${RUN_DIR}/logs/stage_rollout_9b.log" 2>&1 &
stage_rollout=$!
srun --overlap --exact -N1 -n1 -c4 -w "${TRAIN_NODE}" \
    env REPO_ROOT="${REPO_ROOT}" SOURCE_HF_HOME="${SHARED_HF_HOME}" \
        LOCAL_HF_HOME="${TRAIN_HF_HOME}" \
    bash -lc 'cd "$REPO_ROOT"; source scripts/stage_hf_model_cache.sh Qwen/Qwen3.5-9B' \
    >"${RUN_DIR}/logs/stage_train_9b.log" 2>&1 &
stage_train_9b=$!
srun --overlap --exact -N1 -n1 -c4 -w "${TRAIN_NODE}" \
    env REPO_ROOT="${REPO_ROOT}" SOURCE_HF_HOME="${SHARED_HF_HOME}" \
        LOCAL_HF_HOME="${TRAIN_HF_HOME}" \
    bash -lc 'cd "$REPO_ROOT"; source scripts/stage_hf_model_cache.sh Qwen/Qwen3.5-35B-A3B' \
    >"${RUN_DIR}/logs/stage_train_35b.log" 2>&1 &
stage_train_35b=$!
wait "${stage_rollout}" "${stage_train_9b}" "${stage_train_35b}"

wait_for_endpoint() {
    local url=$1
    local owner_pid=$2
    local label=$3
    local deadline=$((SECONDS + 1800))
    while (( SECONDS < deadline )); do
        if curl -fsS -m 3 "${url}/models" >/dev/null 2>&1; then
            echo "ready ${label}: ${url}"
            return 0
        fi
        if ! kill -0 "${owner_pid}" 2>/dev/null; then
            echo "ERROR: ${label} exited during startup" >&2
            return 1
        fi
        sleep 5
    done
    echo "ERROR: timeout waiting for ${label}" >&2
    return 1
}

start_rollout_pool() {
    local adapters=$1
    local phase=$2
    srun --overlap --exact -N1 -n1 -c16 -w "${ROLLOUT_NODE}" \
        env REPO_ROOT="${REPO_ROOT}" CONDA_ENV="${TRAIN_ENV}" \
            HF_HOME="${ROLLOUT_HF_HOME}" HF_HUB_OFFLINE=1 \
            VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_MAX_MODEL_LEN=8192 \
            VLLM_MAX_NUM_SEQS=32 VLLM_MAX_NUM_BATCHED_TOKENS=8192 \
            MODEL="${MODEL}" ADAPTER_DIR="${adapters}" \
            LOG_DIR="${RUN_DIR}/logs/rollout_${phase}" \
        bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; cd "$REPO_ROOT"; exec python scripts/serve_rollout_pool.py --model "$MODEL" --adapter-dir "$ADAPTER_DIR" --gpus 0 1 2 3 --base-port 8000 --gpu-util 0.82 --log-dir "$LOG_DIR"' \
        >"${RUN_DIR}/logs/rollout_${phase}_launcher.log" 2>&1 &
    ROLLOUT_PID=$!
    BG_PIDS+=("${ROLLOUT_PID}")
    for url in "${ROLLOUT_URLS[@]}"; do
        wait_for_endpoint "${url}" "${ROLLOUT_PID}" "rollout-${phase}"
    done
}

run_eval_group() {
    local group=$1
    shift
    local conditions=("$@")
    local eval_pids=()
    for condition in "${conditions[@]}"; do
        for split in eval_in_distribution eval_out_of_distribution; do
            local episodes=35
            if [[ "${split}" == "eval_out_of_distribution" ]]; then episodes=24; fi
            for shard in 0 1 2 3; do
                local out_dir="${RUN_DIR}/eval/${condition}/${split}"
                mkdir -p "${out_dir}"
                srun --overlap --exact -N1 -n1 -c1 -w "${ROLLOUT_NODE}" \
                    env REPO_ROOT="${REPO_ROOT}" ALFWORLD_ENV="${ALFWORLD_ENV}" \
                        ALFWORLD_DATA="${REPO_ROOT}/.cache/alfworld_data" \
                    bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ALFWORLD_ENV"; cd "$REPO_ROOT"; exec python scripts/eval_principled_alfworld.py "$@"' runner \
                        --condition "${condition}" --split "${split}" \
                        --episodes "${episodes}" --max-steps 50 --seed 91000 \
                        --num-shards 4 --shard-index "${shard}" \
                        --endpoints "${ROLLOUT_URLS[@]}" \
                        --base-model "${MODEL}" \
                        --admission-manifest "${ADMISSION_MANIFEST}" \
                        --binding-config "${BINDING_CONFIG}" --config "${TARGET_CONFIG}" \
                        --output "${out_dir}/shard_${shard}.json" \
                    >"${RUN_DIR}/logs/eval_${group}_${condition}_${split}_shard${shard}.log" 2>&1 &
                eval_pids+=("$!")
            done
        done
    done
    local failed=0
    for pid in "${eval_pids[@]}"; do
        wait "${pid}" || failed=1
    done
    if [[ "${failed}" != 0 ]]; then
        echo "ERROR: one or more ${group} eval shards failed" >&2
        return 1
    fi
}

echo "Phase A: base eval + source-only SFT + 35B proposals in parallel"
start_rollout_pool "${EMPTY_ADAPTER_DIR}" base

srun --overlap --exact -N1 -n1 -c12 -w "${TRAIN_NODE}" \
    env REPO_ROOT="${REPO_ROOT}" HF_HOME="${TRAIN_HF_HOME}" HF_HUB_OFFLINE=1 \
        CUDA_VISIBLE_DEVICES=0,1 CONDA_ENV="${TRAIN_ENV}" \
    bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; cd "$REPO_ROOT"; exec python -m trainer.SFT.train --model_name Qwen/Qwen3.5-9B --output_dir "$1" --decision_data_dir "$2" --skillbank_data_dir "$2" --require_source_only_manifest "$3" --games candy_crush sokoban super_mario tetris twenty_forty_eight --adapters skill_selection action_taking --parallel --gpus 0 1 --gpus_per_adapter 1 --max_seq_length 1024 --scale_effective_batch 0.0625 --optim adamw_torch_fused --dataloader_workers 2 --strict_lora_coverage' runner \
        "${ADAPTER_DIR}" "${SOURCE_SFT_DIR}" "${SOURCE_SFT_MANIFEST}" \
    >"${RUN_DIR}/logs/source_sft_launcher.log" 2>&1 &
SFT_PID=$!
BG_PIDS+=("${SFT_PID}")

srun --overlap --exact -N1 -n1 -c12 -w "${TRAIN_NODE}" \
    env REPO_ROOT="${REPO_ROOT}" CONDA_ENV="${TRAIN_ENV}" \
        HF_HOME="${TRAIN_HF_HOME}" HF_HUB_OFFLINE=1 \
        VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 \
        MODEL="${PROPOSER_MODEL}" GPUS=2,3 TENSOR_PARALLEL=2 EXPERT_PARALLEL=1 \
        GPU_UTIL=0.82 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=8 PORT=8004 \
        HOST=0.0.0.0 QUANTIZATION=fp8 SPECULATIVE=none \
    bash -lc 'cd "$REPO_ROOT"; exec bash inference/serve_qwen35_35b_a3b.sh' \
    >"${RUN_DIR}/logs/proposer_35b_server.log" 2>&1 &
PROPOSER_PID=$!
BG_PIDS+=("${PROPOSER_PID}")

wait_for_endpoint "${PROPOSER_URL}" "${PROPOSER_PID}" "35b-proposer"
srun --overlap --exact -N1 -n1 -c2 -w "${TRAIN_NODE}" \
    env REPO_ROOT="${REPO_ROOT}" CONDA_ENV="${TRAIN_ENV}" \
    bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; cd "$REPO_ROOT"; exec python scripts/propose_alfworld_bindings_35b.py --endpoint "$1" --model "$2" --output "$3"' runner \
        "${PROPOSER_URL}" "${PROPOSER_MODEL}" "${RUN_DIR}/proposals_35b.json" \
    >"${RUN_DIR}/logs/proposals_35b.log" 2>&1 &
PROPOSAL_AUDIT_PID=$!
BG_PIDS+=("${PROPOSAL_AUDIT_PID}")

run_eval_group base base base_harness
wait "${SFT_PID}"
wait "${PROPOSAL_AUDIT_PID}"
if [[ ! -f "${ADAPTER_DIR}/decision/action_taking/adapter_config.json" || ! -f "${ADAPTER_DIR}/decision/skill_selection/adapter_config.json" ]]; then
    echo "ERROR: source-only SFT did not produce production-layout adapters" >&2
    exit 3
fi
kill "${PROPOSER_PID}" 2>/dev/null || true
wait "${PROPOSER_PID}" 2>/dev/null || true
kill "${ROLLOUT_PID}" 2>/dev/null || true
wait "${ROLLOUT_PID}" 2>/dev/null || true

echo "Phase B: restart rollout replicas with source-only LoRA and evaluate"
start_rollout_pool "${ADAPTER_DIR}" game_sft
run_eval_group game_sft game_sft game_sft_harness
kill "${ROLLOUT_PID}" 2>/dev/null || true
wait "${ROLLOUT_PID}" 2>/dev/null || true

python "${REPO_ROOT}/scripts/aggregate_principled_alfworld.py" \
    --run-root "${RUN_DIR}" --spec "${SPEC}" --output "${RUN_DIR}/aggregate.json" \
    >"${RUN_DIR}/logs/aggregate.log" 2>&1
find "${ADAPTER_DIR}/decision" -type f \( -name 'adapter_config.json' -o -name 'adapter_model.safetensors' \) \
    -print0 | sort -z | xargs -0 sha256sum >"${RUN_DIR}/adapter_sha256.txt"
for node in "${ROLLOUT_NODE}" "${TRAIN_NODE}"; do
    srun --overlap --exact -N1 -n1 -c1 -w "${node}" nvidia-smi -q \
        >"${RUN_DIR}/nvidia_smi_end_${node}.txt" &
done
wait
scontrol show job "${SLURM_JOB_ID}" >"${RUN_DIR}/slurm_job_at_end.txt"

echo "COMPLETE ${RUN_DIR}"
cat "${RUN_DIR}/aggregate.md"
