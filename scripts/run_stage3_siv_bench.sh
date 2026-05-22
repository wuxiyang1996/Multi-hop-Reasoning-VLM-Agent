#!/usr/bin/env bash
# Stage 3 GRPO: SIV-Bench (social video MCQ)
#   Train samples: 195 | GRPO steps: 10 | Episodes/step: 8
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python "$SCRIPT_DIR/run_stage3_grpo.py" \
    --task siv_bench \
    --total-steps 10 \
    --episodes-per-step 8 \
    --checkpoint-every 5 \
    --seed-bank-dir "$REPO_ROOT/frontier_data/output/stage3_seed_banks/siv_bench" \
    --adapter-dir "$REPO_ROOT/runs/stage3_adapters" \
    --output-dir "$REPO_ROOT/runs/stage3_grpo/siv_bench" \
    "$@"
