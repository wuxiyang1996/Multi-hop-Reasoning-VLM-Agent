#!/usr/bin/env bash

# Submit the formal 2x4 experiment to the validated RTX A6000 nodes.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NODELIST="${NODELIST:-gammagpu14,gammagpu15}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/fs/gamma-projects/vlm-robot/slurm_logs}"
mkdir -p "${SLURM_LOG_DIR}"

job_id=$(sbatch --parsable \
    --partition=gamma --account=gamma --qos=huge-long \
    --nodes=2 --nodelist="${NODELIST}" \
    --gres=gpu:rtxa6000:4 --cpus-per-task=16 --mem=120G \
    --time=1-00:00:00 \
    --output="${SLURM_LOG_DIR}/principled-alfworld-a6000-%j.out" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},GPU_GRES=gpu:rtxa6000" \
    "${REPO_ROOT}/cluster/run_principled_alfworld_2x4.sbatch")

echo "${job_id}"
