#!/usr/bin/env bash
#SBATCH --job-name=thunder-effect-gate-v1
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
RUN_ROOT="${RUN_ROOT:-${CLEAN_REPO}/runs/source_visual_intervention_v1/gymv_thunder_force_iii}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/qualification_episode_unit_v1}"
GYMV_CONDA_ENV="${GYMV_CONDA_ENV:-gymv}"
WORKERS="${WORKERS:-4}"

[[ -f "${RUN_ROOT}/frozen_plan.json" ]]
[[ -f "${RUN_ROOT}/frozen_effect_candidate.json" ]]
[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "ERROR refusing to overwrite ${OUTPUT_DIR}" >&2
  exit 2
}
source "$(conda info --base)/etc/profile.d/conda.sh"
GYMV_PYTHON="$(conda run -n "${GYMV_CONDA_ENV}" python -c 'import sys; print(sys.executable)')"
[[ -x "${GYMV_PYTHON}" ]]
export GYMV_PYTHON

cd "${CLEAN_REPO}"
python scripts/run_source_effect_qualification.py \
  --plan "${RUN_ROOT}/frozen_plan.json" \
  --artifact "${RUN_ROOT}/frozen_effect_candidate.json" \
  --output-dir "${OUTPUT_DIR}" \
  --runtime-root "${SOURCE_RUNTIME}" \
  --split qualification \
  --workers "${WORKERS}"

echo "SOURCE_VISUAL_EFFECT_QUALIFICATION_V1_COMPLETE ${OUTPUT_DIR}"
