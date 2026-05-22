#!/usr/bin/env bash
# Stage 3: Run GRPO domain adaptation for ALL 6 non-game benchmarks,
# then evaluate all conditions (baseline / seeds_only / seeds_grpo).
#
# Prerequisites:
#   1. Seed banks generated: python scripts/stage3_seeds_from_megaskills.py
#   2. Train/test splits exist: cold_start/task_samples/stage3_splits/
#
# Usage:
#   bash scripts/run_stage3_all.sh              # train + eval
#   bash scripts/run_stage3_all.sh --eval-only  # skip training, eval only
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

EVAL_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --eval-only) EVAL_ONLY=true ;;
    esac
done

echo "================================================================"
echo "  Stage 3: Non-Game GRPO Domain Adaptation"
echo "================================================================"

# ── Step 0: Generate seed banks (idempotent) ──
echo "[Step 0] Generating seed banks..."
python "$SCRIPT_DIR/stage3_seeds_from_megaskills.py" 2>&1 | tail -8

if [ "$EVAL_ONLY" = false ]; then
    # ── Step 1: GRPO training per task ──
    echo ""
    echo "================================================================"
    echo "  Step 1: GRPO Training"
    echo "================================================================"

    TASKS=(
        "visual_toolbench:10"
        "tir_bench:10"
        "video_holmes:10"
        "siv_bench:10"
        "webshop:25"
        "miniwob:15"
    )

    for entry in "${TASKS[@]}"; do
        IFS=':' read -r task steps <<< "$entry"
        echo ""
        echo "--- Training: $task ($steps steps) ---"
        bash "$SCRIPT_DIR/run_stage3_${task}.sh" || {
            echo "WARNING: $task training failed, continuing..."
        }
    done
fi

# ── Step 2: Evaluation (all conditions) ──
echo ""
echo "================================================================"
echo "  Step 2: Evaluation (3-way comparison)"
echo "================================================================"

python "$SCRIPT_DIR/run_stage3_eval.py" \
    --task all \
    --condition all \
    --adapter-dir "$REPO_ROOT/runs/stage3_adapters" \
    --output-dir "$REPO_ROOT/runs/stage3_eval"

echo ""
echo "================================================================"
echo "  Stage 3 complete!"
echo "  Results: $REPO_ROOT/runs/stage3_eval/"
echo "================================================================"
