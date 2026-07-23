#!/usr/bin/env bash
# ==============================================================================
# run_ab_test161_3models.sh — AssistantBench test_feasible (n=161) sweep across
# 3 OpenRouter teacher models, 4 round-robin shards per model, foreground.
#
# Usage (foreground; Ctrl-C kills all 12 shards):
#     bash cold_start/run_ab_test161_3models.sh [OUT_BASE]
#
# Defaults:
#     OUT_BASE = /tmp/ab_test161
#     models   = claude-4.6-sonnet / gemini-3.1-pro / qwen3-vl-235B-instruct
#     shards   = 4 per model (12 procs total, 12 chromiums, ~6 GB RAM)
#     reasoning_effort = low
#     max_steps = 16
#     episodes = 1
#     --resume so re-running a partial sweep skips finished tasks
#
# Wall-clock estimate:
#     161 tasks / 4 shards × ~4 min/task = ~160 min per model
#     all 3 models in parallel → wall ~160 min  (depends on rate limits)
#
# Outputs (per model):
#     ${OUT_BASE}/{claude,gemini,qwen}/
#         assistantbench.test.<id>/episode_000.json + rollout_summary.json
#         _shard_{00..03}.log
#         _shard_{00..03}.done   (sentinel; contains EXIT=<code> + timestamp)
#
# After the sweep, generate AB-leaderboard JSONLs with:
#     for m in claude gemini qwen; do
#       python cold_start/grade_assistantbench_eval.py \
#         --run_dir "${OUT_BASE}/${m}"
#     done
# ==============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKS_FILE="${REPO_ROOT}/cold_start/task_samples/browsergym_assistantbench_test_feasible.txt"
OUT_BASE="${1:-/tmp/ab_test161}"
NUM_SHARDS=4
MAX_STEPS=16
REASONING_EFFORT=low

# OpenRouter slugs — keep in sync with baselines/run_openrouter_transfer_baselines.sh
declare -A MODELS=(
    [claude]="${OPENROUTER_CLAUDE_MODEL:-anthropic/claude-4.6-sonnet-20260217}"
    [gemini]="${OPENROUTER_GEMINI_MODEL:-google/gemini-3.1-pro-preview}"
    [qwen]="${OPENROUTER_QWEN_MODEL:-qwen/qwen3-vl-235b-a22b-instruct}"
)

# Project-local HF cache (avoids dead /fs/cml-projects/FMPT path even if the
# launching shell hasn't sourced ~/.bashrc).
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"

# Conda activate with set -u temporarily relaxed (qt-main activate.d uses an
# unset var; bypass once, then re-enable strict mode).
set +u
source /fs/gamma-projects/vlm-robot/conda/etc/profile.d/conda.sh
conda activate "${BROWSERGYM_CONDA_ENV:-browsergym}"
set -u

if ! python3 -c 'import browsergym' 2>/dev/null; then
    echo "[ERROR] 'import browsergym' fails in active env." >&2
    echo "        Run: bash install/install_browsergym.sh" >&2
    exit 2
fi

mapfile -t ALL_TASKS < <(grep -vE '^\s*(#|$)' "$TASKS_FILE")
N=${#ALL_TASKS[@]}
echo "[info] tasks      : $N  (from $TASKS_FILE)"
echo "[info] models     : ${!MODELS[*]}"
echo "[info] shards/m   : $NUM_SHARDS"
echo "[info] OUT_BASE   : $OUT_BASE"
echo "[info] HF_HOME    : $HF_HOME"
echo

mkdir -p "$OUT_BASE"
SHARD_DIR="${OUT_BASE}/_shards"
mkdir -p "$SHARD_DIR"

for ((i = 0; i < NUM_SHARDS; i++)); do
    sf="${SHARD_DIR}/shard_$(printf '%02d' "$i").txt"
    : > "$sf"
    for ((j = i; j < N; j += NUM_SHARDS)); do
        echo "${ALL_TASKS[$j]}" >> "$sf"
    done
    echo "[info] shard $i : $(wc -l < "$sf") tasks  -> $sf"
done
echo

# ── Launch 3 models × 4 shards in parallel (12 procs total) ──
PIDS=()
LABELS=()
trap 'echo "[trap] caught signal — killing children: ${PIDS[*]}"; kill "${PIDS[@]}" 2>/dev/null; pkill -P $$ 2>/dev/null; pkill -9 chrome 2>/dev/null; exit 130' INT TERM

for tag in claude gemini qwen; do
    model="${MODELS[$tag]}"
    out_model="${OUT_BASE}/${tag}"
    mkdir -p "$out_model"
    for ((i = 0; i < NUM_SHARDS; i++)); do
        sf="${SHARD_DIR}/shard_$(printf '%02d' "$i").txt"
        SHARD_TASKS=()
        while IFS= read -r line; do
            [[ -n "$line" ]] && SHARD_TASKS+=("$line")
        done < "$sf"

        log="${out_model}/_shard_$(printf '%02d' "$i").log"
        sentinel="${out_model}/_shard_$(printf '%02d' "$i").done"
        rm -f "$sentinel"
        label="${tag}/sh${i}"
        (
            cd "$REPO_ROOT"
            python -u cold_start/generate_cold_start_actor_browsergym.py \
                --tasks "${SHARD_TASKS[@]}" \
                --model "$model" \
                --reasoning_effort "$REASONING_EFFORT" \
                --max_steps "$MAX_STEPS" \
                --episodes 1 \
                --resume \
                --output_dir "$out_model" \
                -v
            ec=$?
            echo "EXIT=$ec @ $(date +%Y-%m-%dT%H:%M:%S)" > "$sentinel"
        ) > "$log" 2>&1 &
        pid=$!
        PIDS+=("$pid")
        LABELS+=("$label")
        printf "[launch] %-12s pid=%-7d  log=%s\n" "$label" "$pid" "$log"
    done
done

echo
echo "[info] all 12 shards launched at $(date +%H:%M:%S) — waiting..."
echo "[info] live status anytime:  bash cold_start/ab_test161_status.sh $OUT_BASE"
echo

# ── Wait + per-shard rollup ──
FAILS=0
for k in "${!PIDS[@]}"; do
    pid="${PIDS[$k]}"; label="${LABELS[$k]}"
    if ! wait "$pid"; then
        FAILS=$((FAILS + 1))
        echo "[done] $label  FAILED  ($(date +%H:%M:%S))"
    else
        echo "[done] $label  OK      ($(date +%H:%M:%S))"
    fi
done

echo
echo "================================================================"
echo "  Sweep finished at $(date +%H:%M:%S)  failures=${FAILS}/12"
echo "  Next:"
for tag in claude gemini qwen; do
    echo "    python cold_start/grade_assistantbench_eval.py --run_dir ${OUT_BASE}/${tag}"
done
echo "================================================================"
