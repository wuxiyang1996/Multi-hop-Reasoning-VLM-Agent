#!/usr/bin/env bash
# Stage 3 GRPO: MiniWoB++ (browser tasks)
#   Train samples: 25 tasks | GRPO steps: 15 | Episodes/step: 8
#   Moderate step count: many deterministic tasks, higher overfit risk
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python "$SCRIPT_DIR/run_stage3_grpo.py" \
    --task miniwob \
    --total-steps 15 \
    --episodes-per-step 8 \
    --checkpoint-every 5 \
    --seed-bank-dir "$REPO_ROOT/frontier_data/output/stage3_seed_banks/miniwob" \
    --adapter-dir "$REPO_ROOT/runs/stage3_adapters" \
    --output-dir "$REPO_ROOT/runs/stage3_grpo/miniwob" \
    "$@"
