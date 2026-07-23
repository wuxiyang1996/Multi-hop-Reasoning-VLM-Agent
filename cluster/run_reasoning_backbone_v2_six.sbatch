#!/usr/bin/env bash
#SBATCH --job-name=reasoning-v2-six
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-5%1

# Fixed-order, no-content-selection collection of six source games. The array
# throttle avoids changing Agent validity through provider-side rate limiting.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/reasoning_backbone_v2_six_20260721}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
GYMV_CONDA_ENV="${GYMV_CONDA_ENV:-gymv}"
EPISODES="${EPISODES:-2}"
MAX_STEPS="${MAX_STEPS:-6}"
MODEL="${MODEL:-qwen/qwen3-max}"
ENDPOINT="${ENDPOINT:-https://openrouter.ai/api/v1}"

GAMES=(
    candy_crush
    tetris
    gymv_columns
    gymv_streets_of_rage_2
    gymv_strider
    gymv_thunder_force_iii
)
GAME="${GAMES[${SLURM_ARRAY_TASK_ID}]}"
SEED_BASE=$((5101 + 100 * SLURM_ARRAY_TASK_ID))
RUN_DIR="${OUTPUT_ROOT}/${GAME}"

[[ ! -e "${RUN_DIR}" ]] || {
    echo "ERROR refusing to overwrite ${RUN_DIR}" >&2
    exit 2
}
mkdir -p "${RUN_DIR}"
scontrol show job "${SLURM_JOB_ID}" >"${RUN_DIR}/slurm_job_at_start.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

if [[ "${GAME}" == gymv_* ]]; then
    GYMV_PYTHON="$(conda run -n "${GYMV_CONDA_ENV}" python -c 'import sys; print(sys.executable)')"
    [[ -x "${GYMV_PYTHON}" ]] || {
        echo "ERROR invalid Gym-V interpreter: ${GYMV_PYTHON}" >&2
        exit 2
    }
    export GYMV_PYTHON
fi

python scripts/run_instrumented_source_smoke.py \
    --output "${RUN_DIR}/evidence" \
    --game "${GAME}" \
    --episodes "${EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --seed-base "${SEED_BASE}" \
    --endpoint "${ENDPOINT}" \
    --model "${MODEL}" \
    --reasoning-backbone-harness \
    | tee "${RUN_DIR}/summary.json"

echo "REASONING_BACKBONE_V2_GAME_COMPLETE ${GAME} ${RUN_DIR}"
