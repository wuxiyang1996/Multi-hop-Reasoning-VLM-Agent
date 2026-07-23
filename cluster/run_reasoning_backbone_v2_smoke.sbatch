#!/usr/bin/env bash
#SBATCH --job-name=reasoning-v2-smoke
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00

# The GPU pins this short environment smoke to the same compute-node family as
# the six source-game collections. OpenRouter performs the model inference.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/reasoning_backbone_v2_smoke_${SLURM_JOB_ID}}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
GAME="${GAME:-candy_crush}"
EPISODES="${EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-3}"
SEED_BASE="${SEED_BASE:-3101}"
MODEL="${MODEL:-qwen/qwen3-max}"
ENDPOINT="${ENDPOINT:-https://openrouter.ai/api/v1}"
PROVIDER="${PROVIDER:-openrouter}"
API_KEY_ENV="${API_KEY_ENV:-OPENROUTER_API_KEY}"
API_KEY_FILE="${API_KEY_FILE:-}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"
OPENAI_REASONING_TOKEN_RESERVE="${OPENAI_REASONING_TOKEN_RESERVE:-512}"

[[ ! -e "${RUN_DIR}" ]] || {
    echo "ERROR refusing to overwrite ${RUN_DIR}" >&2
    exit 2
}
mkdir -p "${RUN_DIR}"
scontrol show job "${SLURM_JOB_ID}" >"${RUN_DIR}/slurm_job_at_start.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

CMD=(python scripts/run_instrumented_source_smoke.py \
    --output "${RUN_DIR}/evidence" \
    --game "${GAME}" \
    --episodes "${EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --seed-base "${SEED_BASE}" \
    --endpoint "${ENDPOINT}" \
    --model "${MODEL}" \
    --provider "${PROVIDER}" \
    --api-key-env "${API_KEY_ENV}" \
    --reasoning-effort "${REASONING_EFFORT}" \
    --openai-reasoning-token-reserve "${OPENAI_REASONING_TOKEN_RESERVE}" \
    --reasoning-backbone-harness)
if [[ -n "${API_KEY_FILE}" ]]; then
    CMD+=(--api-key-file "${API_KEY_FILE}")
fi
"${CMD[@]}" | tee "${RUN_DIR}/summary.json"

echo "REASONING_BACKBONE_V2_SMOKE_COMPLETE ${RUN_DIR}"
