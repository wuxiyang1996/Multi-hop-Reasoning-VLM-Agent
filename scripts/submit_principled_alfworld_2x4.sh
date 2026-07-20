#!/usr/bin/env bash
# Submit the preregistered 2x4 L40S source-skill transfer experiment.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SBATCH_SCRIPT="${REPO_ROOT}/cluster/run_principled_alfworld_2x4.sbatch"
mkdir -p "${REPO_ROOT}/runs/slurm_logs"

ARGS=(
    --chdir="${REPO_ROOT}"
    --output="${REPO_ROOT}/runs/slurm_logs/%x-%j.out"
    --error="${REPO_ROOT}/runs/slurm_logs/%x-%j.err"
    --export="ALL,REPO_ROOT=${REPO_ROOT},SHARED_HF_HOME=${SHARED_HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache},SOURCE_EPISODE_ROOT=${SOURCE_EPISODE_ROOT:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main/labeling/gpt54_skill_labeled},ALFWORLD_DATA_ROOT=${ALFWORLD_DATA_ROOT:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main/.cache/alfworld_data}"
    "${SBATCH_SCRIPT}"
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'sbatch'
    printf ' %q' "${ARGS[@]}"
    printf '\n'
    exit 0
fi
exec sbatch "${ARGS[@]}"
