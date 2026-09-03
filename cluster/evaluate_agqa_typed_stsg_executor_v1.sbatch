#!/usr/bin/env bash
#SBATCH --job-name=agqa-stsg-v1
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
python scripts/evaluate_agqa_typed_stsg_executor.py \
  --compiler-data runs/agqa2_full_operator_transfer_v1/compiler_data \
  --archive /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip \
  --stsg /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/scene_graphs/AGQA_scene_graphs/AGQA_train_stsgs.pkl \
  --ontology /fs/gamma-projects/vlm-robot/datasets/AGQA2-official/scene_graphs/AGQA_scene_graphs/ENG.txt \
  --capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities.json \
  --output runs/agqa2_full_operator_transfer_v1/stsg_executor_development_2000_v1.json \
  --max-rows 2000
