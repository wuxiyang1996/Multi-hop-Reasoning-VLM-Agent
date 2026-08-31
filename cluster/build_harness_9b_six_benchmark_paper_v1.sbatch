#!/usr/bin/env bash
#SBATCH --job-name=ctrl9b-paper
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_mixed_v3/source_only_sft_seed20260901}"
PROTOCOL="${PROTOCOL:-${REPO_ROOT}/runs/harness_controller_qwen35_9b_mixed_v3_protocol/protocol.json}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/build_harness_9b_six_benchmark_paper_report_v1.py \
  --protocol "${PROTOCOL}" \
  --training-receipt "${RUN_DIR}/training_receipt.json" \
  --source-qualification "${RUN_DIR}/source_mixed_qualification.json" \
  --route-report "${RUN_DIR}/six_benchmark_route_report.json" \
  --action-audit "${RUN_DIR}/six_benchmark_action_equivalence.json" \
  --markdown "${REPO_ROOT}/docs/HARNESS_CONTROLLER_QWEN35_9B_SIX_BENCHMARK_V3_RESULTS.md" \
  --json "${REPO_ROOT}/docs/results/harness_controller_qwen35_9b_six_benchmark_v3.json"

echo "HARNESS_CONTROLLER_QWEN35_9B_SIX_BENCHMARK_PAPER_READY"
