#!/usr/bin/env bash
#SBATCH --job-name=motif-switch-replay
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00

set -Eeuo pipefail

CLEAN_REPO="${CLEAN_REPO:-${SLURM_SUBMIT_DIR}}"
SOURCE_REPO="${SOURCE_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
GYMV_CONDA_ENV="${GYMV_CONDA_ENV:-gymv}"
GAME="${GAME:-gymv_strider}"
EVIDENCE_DIR="${EVIDENCE_DIR:-${CLEAN_REPO}/runs/phase1_streets_strider_long_v1/authentic_skill_loaded/${GAME}/evidence}"
OUTPUT="${OUTPUT:-${CLEAN_REPO}/runs/phase1_switch_replays_v1/${GAME}}"
WORKERS="${WORKERS:-8}"

[[ ! -e "${OUTPUT}" ]] || { echo "ERROR refusing to overwrite ${OUTPUT}" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
GYMV_PYTHON="$(conda run -n "${GYMV_CONDA_ENV}" python -c 'import sys; print(sys.executable)')"
[[ -x "${GYMV_PYTHON}" ]] || { echo "ERROR invalid Gym-V interpreter" >&2; exit 2; }
export GYMV_PYTHON
export PYTHONPATH="${CLEAN_REPO}/src:${SOURCE_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${CLEAN_REPO}"
python scripts/collect_switch_replays.py "${EVIDENCE_DIR}" \
  --game "${GAME}" \
  --source-repo "${SOURCE_REPO}" \
  --workers "${WORKERS}" \
  --output "${OUTPUT}"

echo "SWITCH_REPLAY_COMPLETE ${GAME} ${OUTPUT}"
