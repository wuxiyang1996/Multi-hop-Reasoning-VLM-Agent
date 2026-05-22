#!/usr/bin/env bash
# Stage 3 GRPO: WebShop (web shopping, continuous reward)
#   Train samples: 10 tasks | GRPO steps: 25 | Episodes/step: 8
#   Higher step count: combinatorial trajectory space, low overfit risk
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python "$SCRIPT_DIR/run_stage3_grpo.py" \
    --task webshop \
    --total-steps 25 \
    --episodes-per-step 8 \
    --checkpoint-every 5 \
    --seed-bank-dir "$REPO_ROOT/frontier_data/output/stage3_seed_banks/webshop" \
    --adapter-dir "$REPO_ROOT/runs/stage3_adapters" \
    --output-dir "$REPO_ROOT/runs/stage3_grpo/webshop" \
    "$@"
