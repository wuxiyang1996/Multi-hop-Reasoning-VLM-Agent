#!/usr/bin/env bash
# ==============================================================================
# restart_one_model.sh — kill + restart 4 shards for ONE model in the existing
# /tmp/ab_test161 sweep, leaving the OTHER models' shards running untouched.
#
# Use after editing the cold-start driver / system prompt: a fresh Python
# process picks up the new code, but already-running shards still have the
# old logic in memory.
#
# Usage:
#     bash cold_start/restart_one_model.sh <tag>
# where <tag> is one of {claude, gemini, qwen}.
#
# What it does:
#   1. pkill the 4 shards belonging to <tag> (matched on "ab_test161/<tag>")
#   2. wipe /tmp/ab_test161/<tag>/ so --resume doesn't reuse stale rollouts
#   3. relaunch 4 shards in the background (nohup), one per shard file already
#      partitioned in /tmp/ab_test161/_shards/shard_*.txt
#
# Foreground exits immediately after dispatch; check progress with
# ``cold_start/ab_test161_status.sh``.
# ==============================================================================
set -uo pipefail

TAG="${1:-}"
case "$TAG" in
    claude) MODEL="anthropic/claude-4.6-sonnet-20260217" ;;
    gemini) MODEL="google/gemini-3.1-pro-preview" ;;
    qwen)   MODEL="qwen/qwen3-vl-235b-a22b-instruct" ;;
    *)
        echo "Usage: $0 {claude|gemini|qwen}" >&2
        exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_BASE="${OUT_BASE:-/tmp/ab_test161}"
SHARD_DIR="${OUT_BASE}/_shards"
OUT_MODEL="${OUT_BASE}/${TAG}"
NUM_SHARDS=4
MAX_STEPS=16

if [[ ! -d "$SHARD_DIR" ]]; then
    echo "[ERROR] shard dir $SHARD_DIR missing — has the main launcher run?" >&2
    exit 2
fi

# Project-local HF cache so this works even if ~/.bashrc isn't sourced.
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"

echo "[1/4] Killing existing ${TAG} shards..."
pkill -f "generate_cold_start_actor_browsergym.*${OUT_BASE}/${TAG}" 2>/dev/null || true
sleep 3
pkill -9 -f "generate_cold_start_actor_browsergym.*${OUT_BASE}/${TAG}" 2>/dev/null || true
# Don't pkill chrome wholesale — that would kill the OTHER models' chromiums.
sleep 1
# ``pgrep -fc PAT || echo 0`` is unsafe (when no match: prints "0" + exit-1
# → the ``|| echo 0`` appends another "0" → "0\n0"). Pipe to wc -l instead.
remaining=$(pgrep -f "generate_cold_start_actor_browsergym.*${OUT_BASE}/${TAG}" 2>/dev/null | wc -l)
echo "      remaining ${TAG} python procs: $remaining"

echo "[2/4] Wiping ${OUT_MODEL}/ ..."
rm -rf "$OUT_MODEL"
mkdir -p "$OUT_MODEL"

echo "[3/4] Conda-activate browsergym (set +u during qt-main activate.d)..."
set +u
source /fs/gamma-projects/vlm-robot/conda/etc/profile.d/conda.sh
conda activate "${BROWSERGYM_CONDA_ENV:-browsergym}"
set -u

echo "[4/4] Relaunching ${NUM_SHARDS} ${TAG} shards (model=$MODEL)..."
PIDS=()
for ((i = 0; i < NUM_SHARDS; i++)); do
    sf="${SHARD_DIR}/shard_$(printf '%02d' "$i").txt"
    if [[ ! -f "$sf" ]]; then
        echo "  [WARN] shard file $sf missing — skipping" >&2
        continue
    fi
    SHARD_TASKS=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && SHARD_TASKS+=("$line")
    done < "$sf"

    log="${OUT_MODEL}/_shard_$(printf '%02d' "$i").log"
    sentinel="${OUT_MODEL}/_shard_$(printf '%02d' "$i").done"
    rm -f "$sentinel"

    nohup bash -c "
        cd '$REPO_ROOT'
        python -u cold_start/generate_cold_start_actor_browsergym.py \\
            --tasks ${SHARD_TASKS[*]@Q} \\
            --model '$MODEL' \\
            --reasoning_effort low \\
            --max_steps $MAX_STEPS \\
            --episodes 1 \\
            --resume \\
            --output_dir '$OUT_MODEL' \\
            -v
        echo \"EXIT=\$? @ \$(date +%Y-%m-%dT%H:%M:%S)\" > '$sentinel'
    " > "$log" 2>&1 &
    pid=$!
    PIDS+=("$pid")
    printf "  ${TAG} sh%02d  pid=%-7d  log=%s\n" "$i" "$pid" "$log"
done

echo
echo "[OK] ${TAG} relaunched (${#PIDS[@]} shards, PIDs: ${PIDS[*]})"
echo "[OK] check progress:  bash cold_start/ab_test161_status.sh"
