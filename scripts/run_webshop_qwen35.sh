#!/usr/bin/env bash
# ==============================================================================
# run_webshop_qwen35.sh — Qwen3.5-9B & 35B-A3B on WebShop via OpenRouter
#   with visual schema (screenshots sent to model).
#   Parallel workers per model to speed up evaluation.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONDA_BASE="$(conda info --base 2>/dev/null || echo /workspace/miniconda3)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate browsergym

# ── Config ────────────────────────────────────────────────────────────────
NUM_TASKS="${WEBSHOP_NUM_TASKS:-50}"
MAX_STEPS="${WEBSHOP_MAX_STEPS:-20}"
EPISODES="${WEBSHOP_EPISODES:-1}"
PARALLEL="${WEBSHOP_PARALLEL:-5}"
export WEBSHOP_BASE_URL="${WEBSHOP_BASE_URL:-http://127.0.0.1:3000}"
export WEBSHOP_NUM_GOALS="$NUM_TASKS"

unset OPENAI_API_KEY 2>/dev/null || true
unset VLLM_BASE_URL 2>/dev/null || true

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    OPENROUTER_API_KEY=$(python3 -c "
import sys; sys.path.insert(0, '$REPO_ROOT/..')
try:
    import keys; print(keys.openrouter_api_key)
except Exception:
    print('')
" 2>/dev/null)
fi
export OPENROUTER_API_KEY

MODEL_9B="qwen/qwen3.5-9b"
MODEL_35B="qwen/qwen3.5-35b-a3b"
OR_BASE="https://openrouter.ai/api/v1"
OUT_9B="Cold-start-out-browsergym/webshop_${NUM_TASKS}task_qwen35_9b"
OUT_35B="Cold-start-out-browsergym/webshop_${NUM_TASKS}task_qwen35_35b"

RUN_9B=true; RUN_35B=true
case "${1:-all}" in 9b) RUN_35B=false;; 35b) RUN_9B=false;; esac

# ── Checks ────────────────────────────────────────────────────────────────
echo "Checking WebShop server at $WEBSHOP_BASE_URL ..."
curl -sf --max-time 5 "${WEBSHOP_BASE_URL}/__bridge/session/fixed_0" >/dev/null || {
    echo "[ERROR] WebShop server not running."; exit 1; }
echo "  OK"
[ -z "$OPENROUTER_API_KEY" ] && { echo "[ERROR] OPENROUTER_API_KEY not set"; exit 1; }
echo "  Key: ${OPENROUTER_API_KEY:0:12}..."

# ── Split tasks into chunks for parallel workers ─────────────────────────
all_tasks=()
for i in $(seq 0 $((NUM_TASKS - 1))); do all_tasks+=("browsergym/webshop.$i"); done

split_tasks() {
    local n_workers="$1"
    local n_total="${#all_tasks[@]}"
    local chunk_size=$(( (n_total + n_workers - 1) / n_workers ))
    local w=0
    for (( start=0; start<n_total; start+=chunk_size )); do
        local end=$((start + chunk_size))
        [[ $end -gt $n_total ]] && end=$n_total
        WORKER_TASKS[$w]="${all_tasks[@]:$start:$((end - start))}"
        w=$((w + 1))
    done
}

declare -a WORKER_TASKS
split_tasks "$PARALLEL"
N_WORKERS=${#WORKER_TASKS[@]}

# ── Worker launcher ──────────────────────────────────────────────────────
run_worker() {
    local model="$1" out_base="$2" tag="$3" worker_id="$4" tasks="$5"
    local out="${out_base}"
    local log="/tmp/webshop_${tag}_w${worker_id}.log"

    echo "  [W${worker_id}] tasks: $tasks"
    echo "  [W${worker_id}] log: $log"

    nohup python cold_start/generate_cold_start_actor_browsergym.py \
        --tasks $tasks \
        --episodes "$EPISODES" \
        --max_steps "$MAX_STEPS" \
        --model "$model" \
        --api_key "$OPENROUTER_API_KEY" \
        --base_url "$OR_BASE" \
        --output_dir "$out" \
        --save_frames \
        --resume \
        -v \
        > "$log" 2>&1 &
    disown
    echo "  [W${worker_id}] PID=$!"
}

launch_model() {
    local model="$1" out="$2" tag="$3"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Launching $N_WORKERS parallel workers for: $model"
    echo "  Output dir: $out"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    mkdir -p "$out"
    for (( w=0; w<N_WORKERS; w++ )); do
        run_worker "$model" "$out" "$tag" "$w" "${WORKER_TASKS[$w]}"
    done
}

# ── Banner ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  WebShop Benchmark — Qwen3.5 (OpenRouter, vision ON)"
echo "  Parallel workers per model: $N_WORKERS"
echo "================================================================"
echo "  Tasks:       $NUM_TASKS"
echo "  Max steps:   $MAX_STEPS"
[[ "$RUN_9B"  == true ]] && echo "  Model 1:     $MODEL_9B → $OUT_9B"
[[ "$RUN_35B" == true ]] && echo "  Model 2:     $MODEL_35B → $OUT_35B"
echo "================================================================"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────
[[ "$RUN_9B"  == true ]] && launch_model "$MODEL_9B"  "$OUT_9B"  "qwen35_9b"
[[ "$RUN_35B" == true ]] && launch_model "$MODEL_35B" "$OUT_35B" "qwen35_35b"

echo ""
echo "All workers launched. Monitor:"
for (( w=0; w<N_WORKERS; w++ )); do
    [[ "$RUN_9B"  == true ]] && echo "  tail -f /tmp/webshop_qwen35_9b_w${w}.log"
done
for (( w=0; w<N_WORKERS; w++ )); do
    [[ "$RUN_35B" == true ]] && echo "  tail -f /tmp/webshop_qwen35_35b_w${w}.log"
done
echo ""
echo "Quick status:  ps aux | grep generate_cold_start"
echo "When done:     python -m webshop_wrapper._make_report"
