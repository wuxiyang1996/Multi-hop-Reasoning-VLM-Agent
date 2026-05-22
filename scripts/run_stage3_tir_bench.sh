#!/usr/bin/env bash
# Stage 3 GRPO: TIR-Bench (image reasoning)
#   Train samples: 200 | GRPO steps: 10 | Episodes/step: 8
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python "$SCRIPT_DIR/run_stage3_grpo.py" \
    --task tir_bench \
    --total-steps 10 \
    --episodes-per-step 8 \
    --checkpoint-every 5 \
    --seed-bank-dir "$REPO_ROOT/frontier_data/output/stage3_seed_banks/tir_bench" \
    --adapter-dir "$REPO_ROOT/runs/stage3_adapters" \
    --output-dir "$REPO_ROOT/runs/stage3_grpo/tir_bench" \
    "$@"
