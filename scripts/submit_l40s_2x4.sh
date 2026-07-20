#!/usr/bin/env bash
# Submit the two-node L40S launcher from any working directory.
#
# Examples:
#   scripts/submit_l40s_2x4.sh
#   RUN_MODE=train TOTAL_STEPS=100 EPISODES=8 scripts/submit_l40s_2x4.sh
#   START_MODE=resume RUN_DIR=/shared/existing-run scripts/submit_l40s_2x4.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "${REPO_ROOT}/runs/slurm_logs"

SBATCH_ARGS=(
    --chdir="${REPO_ROOT}" \
    --output="${REPO_ROOT}/runs/slurm_logs/%x-%j.out" \
    --error="${REPO_ROOT}/runs/slurm_logs/%x-%j.err" \
    --export="ALL,REPO_ROOT=${REPO_ROOT}" \
    "${REPO_ROOT}/cluster/run_l40s_2x4.sbatch" "$@"
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'sbatch'
    printf ' %q' "${SBATCH_ARGS[@]}"
    printf '\n'
    exit 0
fi
exec sbatch "${SBATCH_ARGS[@]}"
