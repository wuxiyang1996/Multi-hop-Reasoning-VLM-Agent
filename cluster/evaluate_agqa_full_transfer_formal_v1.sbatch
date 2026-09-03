#!/usr/bin/env bash
#SBATCH --job-name=agqa-full-formal
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
python scripts/evaluate_agqa_full_transfer_formal.py \
  --preregistration runs/agqa2_full_operator_transfer_v1/fresh_question_reserve_v1/preregistration.json \
  --cohort runs/agqa2_full_operator_transfer_v1/fresh_question_reserve_v1/public_cohort.json \
  --runtime-programs runs/agqa2_full_operator_transfer_v1/fresh_question_reserve_v1/compiler_runtime_v1.json \
  --neural-predictions runs/agqa2_full_operator_transfer_v1/fresh_question_reserve_v1/neural_qwen35_9b_runtime.json \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --stsg /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/scene_graphs/AGQA_scene_graphs/AGQA_test_stsgs.pkl \
  --ontology /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/scene_graphs/AGQA_scene_graphs/ENG.txt \
  --controls-dir runs/agqa2_full_operator_transfer_v1/controls_v2 \
  --output runs/agqa2_full_operator_transfer_v1/fresh_question_reserve_v1/formal_six_arm_result.json
