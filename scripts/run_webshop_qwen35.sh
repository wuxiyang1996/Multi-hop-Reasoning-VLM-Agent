#!/usr/bin/env bash
# ==============================================================================
# run_webshop_qwen35.sh — Qwen3.5-9B & 35B-A3B on WebShop via local vLLM
#   with visual schema (screenshots sent to model).
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONDA_BASE="$(conda info --base 2>/dev/null || echo /workspace/miniconda3)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── Config ────────────────────────────────────────────────────────────────
NUM_TASKS="${WEBSHOP_NUM_TASKS:-50}"
MAX_STEPS="${WEBSHOP_MAX_STEPS:-20}"
EPISODES="${WEBSHOP_EPISODES:-1}"
export WEBSHOP_BASE_URL="${WEBSHOP_BASE_URL:-http://127.0.0.1:3000}"
export WEBSHOP_NUM_GOALS="$NUM_TASKS"

# Prevent any OpenAI key from diverting calls
unset OPENAI_API_KEY 2>/dev/null || true
unset VLLM_BASE_URL 2>/dev/null || true

# cu13 runtime required by vLLM's bundled flash-attn .so
CU13_LIB="$CONDA_BASE/envs/game-ai-agent/lib/python3.11/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="${CU13_LIB}:${LD_LIBRARY_PATH:-}"

# HuggingFace cache
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"

# ── Model / vLLM config ──────────────────────────────────────────────────
MODEL_9B="Qwen/Qwen3.5-9B"
MODEL_35B="Qwen/Qwen3.5-35B-A3B"
PORT_9B=8001
PORT_35B=8002

OUT_9B="Cold-start-out-browsergym/webshop_${NUM_TASKS}task_qwen35_9b"
OUT_35B="Cold-start-out-browsergym/webshop_${NUM_TASKS}task_qwen35_35b"

TASKS=""
for i in $(seq 0 $((NUM_TASKS - 1))); do TASKS="$TASKS browsergym/webshop.$i"; done

RUN_9B=true; RUN_35B=true
case "${1:-all}" in 9b) RUN_35B=false;; 35b) RUN_9B=false;; esac

# ── Checks ────────────────────────────────────────────────────────────────
echo "Checking WebShop server at $WEBSHOP_BASE_URL ..."
curl -sf --max-time 5 "${WEBSHOP_BASE_URL}/__bridge/session/fixed_0" >/dev/null || {
    echo "[ERROR] WebShop server not running."; exit 1; }
echo "  OK"

# ── vLLM launcher ────────────────────────────────────────────────────────
start_vllm() {
    local model="$1" port="$2" gpus="$3" tp="$4" tag="$5"
    local log="/tmp/vllm_${tag}.log"
    echo "[vLLM] Starting $model on GPU $gpus (TP=$tp) → port $port"
    echo "  Log: $log"

    local extra_args=()
    if [[ "$tp" -ge 2 ]]; then
        extra_args+=(--disable-custom-all-reduce)
    fi

    conda activate game-ai-agent
    CUDA_VISIBLE_DEVICES="$gpus" nohup python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --port "$port" \
        --tensor-parallel-size "$tp" \
        --gpu-memory-utilization 0.92 \
        --max-model-len 16384 \
        --max-num-seqs 4 \
        --trust-remote-code \
        --tool-call-parser hermes \
        --enable-auto-tool-choice \
        "${extra_args[@]}" \
        > "$log" 2>&1 &
    disown
    echo "  PID=$!"
    conda activate browsergym
}

wait_vllm() {
    local port="$1" tag="$2" timeout="${3:-300}"
    echo "[vLLM] Waiting for $tag on port $port (timeout ${timeout}s) ..."
    local start=$SECONDS
    while ! curl -sf --max-time 2 "http://localhost:${port}/v1/models" >/dev/null 2>&1; do
        if (( SECONDS - start > timeout )); then
            echo "[ERROR] vLLM $tag did not start within ${timeout}s"
            echo "  Check: tail -50 /tmp/vllm_${tag}.log"
            exit 1
        fi
        sleep 5
    done
    echo "  $tag ready ($(( SECONDS - start ))s)"
}

# ── Eval runner ───────────────────────────────────────────────────────────
run_eval() {
    local model="$1" port="$2" out="$3" tag="$4"
    local log="/tmp/webshop_${tag}.log"
    local base="http://localhost:${port}/v1"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Launching eval: $model  →  $out"
    echo "  Base URL: $base"
    echo "  Log: $log"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    conda activate browsergym
    nohup python cold_start/generate_cold_start_actor_browsergym.py \
        --tasks $TASKS \
        --episodes "$EPISODES" \
        --max_steps "$MAX_STEPS" \
        --model "$model" \
        --api_key "dummy" \
        --base_url "$base" \
        --output_dir "$out" \
        --save_frames \
        -v \
        > "$log" 2>&1 &
    disown
    echo "  PID=$!"
}

# ── Banner ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  WebShop Benchmark — Qwen3.5 (local vLLM, vision ON)"
echo "================================================================"
echo "  Tasks:       $NUM_TASKS"
echo "  Max steps:   $MAX_STEPS"
[[ "$RUN_9B"  == true ]] && echo "  Model 1:     $MODEL_9B (GPU 1, port $PORT_9B)"
[[ "$RUN_35B" == true ]] && echo "  Model 2:     $MODEL_35B (GPU 2,3, port $PORT_35B)"
echo "================================================================"
echo ""

# ── Launch vLLM servers ──────────────────────────────────────────────────
[[ "$RUN_9B"  == true ]] && start_vllm "$MODEL_9B"  "$PORT_9B"  "1"   1 "qwen35_9b"
[[ "$RUN_35B" == true ]] && start_vllm "$MODEL_35B" "$PORT_35B" "2,3" 2 "qwen35_35b"
[[ "$RUN_9B"  == true ]] && wait_vllm "$PORT_9B"  "qwen35_9b"  300
[[ "$RUN_35B" == true ]] && wait_vllm "$PORT_35B" "qwen35_35b" 600

# ── Launch evals ─────────────────────────────────────────────────────────
[[ "$RUN_9B"  == true ]] && { rm -rf "$OUT_9B" 2>/dev/null; run_eval "$MODEL_9B"  "$PORT_9B"  "$OUT_9B"  "qwen35_9b"; }
[[ "$RUN_35B" == true ]] && { rm -rf "$OUT_35B" 2>/dev/null; run_eval "$MODEL_35B" "$PORT_35B" "$OUT_35B" "qwen35_35b"; }

echo ""
echo "Runs launched. Monitor:"
[[ "$RUN_9B"  == true ]] && echo "  tail -f /tmp/webshop_qwen35_9b.log"
[[ "$RUN_35B" == true ]] && echo "  tail -f /tmp/webshop_qwen35_35b.log"
echo ""
echo "vLLM logs:"
[[ "$RUN_9B"  == true ]] && echo "  tail -f /tmp/vllm_qwen35_9b.log"
[[ "$RUN_35B" == true ]] && echo "  tail -f /tmp/vllm_qwen35_35b.log"
echo ""
echo "When done: python -m webshop_wrapper._make_report"
