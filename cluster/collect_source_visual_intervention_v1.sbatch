#!/usr/bin/env bash
#SBATCH --job-name=thunder-visual-forks-v1
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=gammagpu10,gammagpu11,gammagpu13,gammagpu15

set -Eeuo pipefail

CLEAN_REPO="${CLEAN_REPO:-${SLURM_SUBMIT_DIR}}"
SOURCE_RUNTIME="${SOURCE_RUNTIME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-source-fresh-v1}"
CONFIG="${CONFIG:-${CLEAN_REPO}/configs/source_visual_intervention_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CLEAN_REPO}/runs/source_visual_intervention_v1}"
GYMV_CONDA_ENV="${GYMV_CONDA_ENV:-gymv}"
WORKERS="${WORKERS:-4}"
RUN_DIR="${OUTPUT_ROOT}/gymv_thunder_force_iii"

[[ ! -e "${RUN_DIR}" ]] || {
  echo "ERROR refusing to overwrite ${RUN_DIR}" >&2
  exit 2
}
mkdir -p "${RUN_DIR}"
scontrol show job "${SLURM_JOB_ID}" >"${RUN_DIR}/slurm_job_at_start.txt"
git -C "${CLEAN_REPO}" rev-parse HEAD >"${RUN_DIR}/clean_repo_commit.txt"
git -C "${SOURCE_RUNTIME}" rev-parse HEAD >"${RUN_DIR}/source_runtime_commit.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
GYMV_PYTHON="$(conda run -n "${GYMV_CONDA_ENV}" python -c 'import sys; print(sys.executable)')"
[[ -x "${GYMV_PYTHON}" ]] || {
  echo "ERROR invalid Gym-V interpreter: ${GYMV_PYTHON}" >&2
  exit 2
}
export GYMV_PYTHON

cd "${CLEAN_REPO}"
python scripts/collect_source_visual_interventions.py prepare \
  --config "${CONFIG}" \
  --output "${RUN_DIR}/frozen_plan.json"
python scripts/collect_source_visual_interventions.py collect \
  --plan "${RUN_DIR}/frozen_plan.json" \
  --output-dir "${RUN_DIR}/discovery" \
  --runtime-root "${SOURCE_RUNTIME}" \
  --split discovery \
  --workers "${WORKERS}"

echo "SOURCE_VISUAL_INTERVENTION_V1_COMPLETE ${RUN_DIR}"
