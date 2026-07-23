#!/usr/bin/env bash
#SBATCH --job-name=causal-motif
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Local 9B target policy + GPT-5-mini diagnostic/adaptation Harness.
# GPT cannot emit target actions or confirm/refute a binding.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/causal_motif_pilot_${SLURM_JOB_ID}}"
CONDITION_DIR="${CONDITION_DIR:-${REPO_ROOT}/runs/causal_motif_audit_20260721/conditions}"
TRAIN_ENV="${TRAIN_ENV:-cosplay-candy-a100}"
ALFWORLD_ENV="${ALFWORLD_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/alfworld}"
HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
ALFWORLD_DATA_ROOT="${ALFWORLD_DATA_ROOT:-${REPO_ROOT}/.cache/alfworld_data}"
POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen3.5-9B}"
POLICY_PORT="${POLICY_PORT:-8016}"
POLICY_ENDPOINT="http://127.0.0.1:${POLICY_PORT}/v1"
HARNESS_ENDPOINT="${HARNESS_ENDPOINT:-https://us.api.openai.com/v1}"
HARNESS_MODEL="${HARNESS_MODEL:-gpt-5-mini}"
HARNESS_KEY_FILE="${HARNESS_KEY_FILE:-${REPO_ROOT}/../keys.py}"
EPISODES="${EPISODES:-2}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_HARNESS_CALLS="${MAX_HARNESS_CALLS:-6}"
AUDIT_INTERVAL="${AUDIT_INTERVAL:-5}"
SEED="${SEED:-91000}"
TREATMENTS="${TREATMENTS:-authentic generic_protocol shuffled_topology receipt_null other_source target_only}"
ADAPTATION_DEMO="${ADAPTATION_DEMO:-${REPO_ROOT}/artifacts/admission_demos/alfworld/pick_and_place/train_seed42_v3_shot0.json}"

[[ ! -e "${RUN_DIR}" ]] || { echo "ERROR refusing to overwrite ${RUN_DIR}" >&2; exit 2; }
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/results" "${RUN_DIR}/empty_adapters"
scontrol show job "${SLURM_JOB_ID}" >"${RUN_DIR}/slurm_job_at_start.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${TRAIN_ENV}"
cd "${REPO_ROOT}"

allocated_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
[[ "${allocated_gpu}" =~ ^[0-9]+$ ]] || {
    echo "ERROR invalid batch CUDA_VISIBLE_DEVICES" >&2
    exit 4
}
python scripts/verify_cuda_devices.py 1 >"${RUN_DIR}/cuda_receipt.json"
python scripts/serve_rollout_pool.py \
    --model "${POLICY_MODEL}" --adapter-dir "${RUN_DIR}/empty_adapters" \
    --gpus "${allocated_gpu}" --base-port "${POLICY_PORT}" --gpu-util 0.82 \
    --log-dir "${RUN_DIR}/logs/policy_server" \
    >"${RUN_DIR}/logs/policy_launcher.log" 2>&1 &
SERVER_PID=$!
cleanup() {
    local code=$?
    trap - EXIT INT TERM
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    exit "${code}"
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + 1200))
until curl -fsS -m 3 "${POLICY_ENDPOINT}/models" >/dev/null 2>&1; do
    kill -0 "${SERVER_PID}" 2>/dev/null || { echo "ERROR policy server exited" >&2; exit 3; }
    (( SECONDS < deadline )) || { echo "ERROR policy server timeout" >&2; exit 3; }
    sleep 5
done

result_args=()
for treatment in ${TREATMENTS}; do
    artifact="${CONDITION_DIR}/${treatment}.json"
    [[ -f "${artifact}" ]] || { echo "ERROR missing ${artifact}" >&2; exit 2; }
    result="${RUN_DIR}/results/${treatment}.json"
    ALFWORLD_DATA="${ALFWORLD_DATA_ROOT}" conda run -p "${ALFWORLD_ENV}" \
        python scripts/eval_advisory_reasoning_backbone.py \
        --treatment "${treatment}" --conditioning-artifact "${artifact}" \
        --adaptation-demo "${ADAPTATION_DEMO}" \
        --policy-endpoint "${POLICY_ENDPOINT}" --policy-model "${POLICY_MODEL}" \
        --harness-endpoint "${HARNESS_ENDPOINT}" --harness-model "${HARNESS_MODEL}" \
        --harness-key-file "${HARNESS_KEY_FILE}" \
        --episodes "${EPISODES}" --max-steps "${MAX_STEPS}" \
        --max-harness-calls "${MAX_HARNESS_CALLS}" \
        --audit-interval "${AUDIT_INTERVAL}" --seed "${SEED}" \
        --output "${result}" >"${RUN_DIR}/logs/${treatment}.log" 2>&1
    result_args+=(--result "${result}")
done

python scripts/evaluate_causal_motif_pilot.py \
    "${result_args[@]}" --output "${RUN_DIR}/matched_report.json"
echo "CAUSAL_MOTIF_PILOT_COMPLETE ${RUN_DIR}"
