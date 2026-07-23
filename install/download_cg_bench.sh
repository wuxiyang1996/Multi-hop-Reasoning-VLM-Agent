#!/usr/bin/env bash
# Download CG-Bench dataset (gated, ~411 GB total).
#
# Prerequisites:
#   pip install -U "huggingface_hub[cli]"
#   huggingface-cli login   # accept the gated dataset conditions first
#
# Usage:
#   bash install/download_cg_bench.sh [target_dir]
#
# Default target: /fs/gamma-projects/vlm-robot/datasets/CG-Bench

set -euo pipefail

TARGET="${1:-/fs/gamma-projects/vlm-robot/datasets/CG-Bench}"
echo "==> Downloading CG-Bench to: $TARGET"
echo "    NOTE: Dataset is gated (~411 GB). You must accept conditions at"
echo "    https://huggingface.co/datasets/CG-Bench/CG-Bench first."
echo ""

mkdir -p "$TARGET"

# Download annotations first (small)
echo "==> Downloading annotations..."
huggingface-cli download CG-Bench/CG-Bench \
    cgbench_mini.json \
    --repo-type dataset \
    --local-dir "$TARGET" \
    2>/dev/null || echo "    (cgbench_mini.json may not exist as a standalone file)"

# Full download (videos + annotations)
echo "==> Downloading full dataset (this will take a while)..."
huggingface-cli download CG-Bench/CG-Bench \
    --repo-type dataset \
    --local-dir "$TARGET"

# Unzip if needed
if [ -f "$TARGET/unzip_hf_zip.py" ]; then
    echo "==> Unzipping video files..."
    cd "$TARGET"
    python unzip_hf_zip.py
fi

# Process annotations into per-question JSONs
if [ -f "$TARGET/run/save_as_jsons.py" ]; then
    echo "==> Processing annotations..."
    cd "$TARGET"
    python run/save_as_jsons.py
fi

echo ""
echo "==> CG-Bench setup complete at: $TARGET"
echo "    Next: create eval splits with:"
echo "    python -c 'from visual_reasoning_wrapper.benchmarks.cg_bench import iter_cg_bench_samples; ...'"
