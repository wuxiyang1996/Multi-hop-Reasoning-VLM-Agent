#!/usr/bin/env bash
#SBATCH --job-name=alf-task0-repl
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-9%10

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SOURCE_REPO="${SOURCE_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main}"
KEYS="${KEYS:-/fs/gamma-projects/vlm-robot/keys.py}"
CONDA_ENV="${CONDA_ENV:-alfworld}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/alfworld_task0_frozen_replication_v1}"
FROZEN_HYPOTHESES="${FROZEN_HYPOTHESES:-${REPO_ROOT}/runs/receipt_knowledge_teacher_pilot_v1/alfworld/full_v5_canonical/task_0.json}"
SEED_BASE="${SEED_BASE:-92000}"
CONDITIONS="${CONDITIONS:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

REPLICATE="${SLURM_ARRAY_TASK_ID}"
RUN_SEED="$((SEED_BASE + REPLICATE))"
mkdir -p "${RUN_ROOT}/cache/replicate_${REPLICATE}"
condition_args=()
if [[ -n "${CONDITIONS}" ]]; then
  read -r -a selected_conditions <<< "${CONDITIONS}"
  condition_args=(--conditions "${selected_conditions[@]}")
fi

python scripts/run_alfworld_weak_knowledge_teacher_pilot.py \
  --keys "${KEYS}" \
  --config "${SOURCE_REPO}/configs/alfworld_base_config.yaml" \
  --data "${SOURCE_REPO}/.cache/alfworld_data" \
  --manifest configs/target_rollout_collection_v1.json \
  --adaptation-dir runs/target_rollout_collection_v1/alfworld_valid_unseen/adaptation \
  --authentic-artifact runs/receipt_knowledge_teacher_pilot_v1/source/joint_v4.json \
  --other-artifact runs/receipt_knowledge_teacher_pilot_v1/source/other_four_games_v1.json \
  --source-evidence-dir runs/fresh_source_execution_motif_v1/gymv_thunder_force_iii/evidence \
  --source-evidence-dir runs/fresh_source_execution_motif_v1/gymv_strider/evidence \
  --task-offset 0 \
  --run-seed "${RUN_SEED}" \
  --environment-seed 81000 \
  --frozen-hypotheses-from "${FROZEN_HYPOTHESES}" \
  --output "${RUN_ROOT}/replicate_${REPLICATE}.json" \
  --cache-dir "${RUN_ROOT}/cache/replicate_${REPLICATE}" \
  --adaptation-count 4 \
  --max-steps 30 \
  --decision-max-tokens 1024 \
  "${condition_args[@]}"

echo "ALFWORLD_TASK0_FROZEN_REPLICATION_V1_COMPLETE ${REPLICATE} ${RUN_SEED}"
