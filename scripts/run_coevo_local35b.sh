#!/usr/bin/env bash
# =============================================================================
# Co-evolution with LOCAL 35B judge + 4-GPU 9B rollout + 2-GPU GRPO
#
# Layout:
#   GPU 0-3 →  4× TP=1 vLLM (Qwen3.5-9B) for rollout (was 6 GPUs)
#   GPU 4-5 →  TP=2 vLLM (Qwen3.5-35B-A3B, MULTIMODAL) for vision judge
#   GPU 6-7 →  GRPO training
#
# Why local 35B beats API:
#   - API latency ~0.7s/call median (network + queue)
#   - Local on 2× RTX PRO 6000 Blackwell: ~0.1-0.2s/call
#   - 35B-A3B is MoE with only 3B active params → very fast
#
# Usage:
#   bash scripts/run_coevo_local35b.sh <game_slug> [total_steps]
# =============================================================================
set -euo pipefail

GAME="${1:?Usage: $0 <game_slug> [total_steps]}"
STEPS="${2:-10}"

cd /workspace/Multi-hop-Reasoning-VLM-Agent

export PYTHONPATH="/workspace/Multi-hop-Reasoning-VLM-Agent:/workspace/GamingAgent:/workspace/GamingAgent/gamingagent/envs/custom_03_candy_crush:${PYTHONPATH:-}"
export HF_HOME="/workspace/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export GRPO_FSDP_BATCH_SIZE=64

pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

# ── Per-game LoRA switching ──────────────────────────────────────────
declare -A GAME_TO_SFT_KEY=(
    [gymv_thunder_force_iii]="Temporal_ThunderForceIII-v0"
    [candy_crush]="candy_crush"
    [tetris]="tetris"
    [super_mario]="super_mario"
    [twenty_forty_eight]="twenty_forty_eight"
)

SFT_KEY="${GAME_TO_SFT_KEY[$GAME]:-}"
if [[ -z "$SFT_KEY" ]]; then
    echo "ERROR: Unknown game '$GAME'"
    exit 1
fi

ADAPTER_DIR="runs/lora_adapters/decision"

# skill_selection
SK_SRC="runs/sft_per_game_v3/${SFT_KEY}/skill_selection/${SFT_KEY}__skill_selection"
if [[ -d "$SK_SRC" ]]; then
    rm -f "${ADAPTER_DIR}/skill_selection/adapter_config.json" \
          "${ADAPTER_DIR}/skill_selection/adapter_model.safetensors"
    ln -s "$(pwd)/${SK_SRC}/adapter_config.json" \
          "${ADAPTER_DIR}/skill_selection/adapter_config.json"
    ln -s "$(pwd)/${SK_SRC}/adapter_model.safetensors" \
          "${ADAPTER_DIR}/skill_selection/adapter_model.safetensors"
    echo "✓ skill_selection → ${SFT_KEY}"
fi

# action_taking
AT_SRC="runs/sft_per_game/${SFT_KEY}/action_taking/${SFT_KEY}__action_taking"
if [[ -d "$AT_SRC" ]]; then
    rm -f "${ADAPTER_DIR}/action_taking/adapter_config.json" \
          "${ADAPTER_DIR}/action_taking/adapter_model.safetensors"
    ln -s "$(pwd)/${AT_SRC}/adapter_config.json" \
          "${ADAPTER_DIR}/action_taking/adapter_config.json"
    ln -s "$(pwd)/${AT_SRC}/adapter_model.safetensors" \
          "${ADAPTER_DIR}/action_taking/adapter_model.safetensors"
    echo "✓ action_taking → ${SFT_KEY}"
fi

# ── Start local 35B vLLM server on GPU 4-5 (MULTIMODAL for vision) ────
echo ""
echo "============================================"
echo "  Starting local 35B vLLM server (GPU 4-5)"
echo "============================================"
JUDGE_LOG="/tmp/qwen35_35b_a3b.log"

export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
CUDA_VISIBLE_DEVICES=4,5 nohup python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-35B-A3B \
    --host 127.0.0.1 \
    --port 8001 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --trust-remote-code \
    --dtype auto \
    --reasoning-parser qwen3 \
    --speculative_config '{"method":"mtp","num_speculative_tokens":3}' \
    > "$JUDGE_LOG" 2>&1 &
JUDGE_PID=$!
echo "[run_coevo_local35b] 35B server starting (PID=$JUDGE_PID, log=$JUDGE_LOG)"

# Wait for 35B server to become healthy
echo "[run_coevo_local35b] Waiting for 35B to become healthy (port 8001)..."
for i in $(seq 1 120); do
    if curl -s -m 2 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
        echo "✓ 35B server healthy after ${i}×10s"
        break
    fi
    if ! kill -0 $JUDGE_PID 2>/dev/null; then
        echo "ERROR: 35B server died. Tail of log:"
        tail -30 "$JUDGE_LOG"
        exit 1
    fi
    sleep 10
done

if ! curl -s -m 2 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
    echo "ERROR: 35B server failed to become healthy after 20 min"
    tail -50 "$JUDGE_LOG"
    kill $JUDGE_PID 2>/dev/null || true
    exit 1
fi

# ── Environment ──────────────────────────────────────────────────────
export VLM_AGENT_BACKBONE_JUDGE_MODEL="Qwen/Qwen3.5-35B-A3B"
export VLLM_BASE_URL_MAP="Qwen/Qwen3.5-9B=http://localhost:8000/v1,Qwen/Qwen3.5-35B-A3B=http://localhost:8001/v1"

# Source openrouter as last-resort fallback (in case local 35B has hiccups)
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    OPENROUTER_API_KEY=$(python3 -c "
import sys
for p in ['.', '..', '/workspace']:
    sys.path.insert(0, p)
try:
    from key import openrouter_api_key
    print(openrouter_api_key)
except Exception:
    pass
" 2>/dev/null || echo "")
fi
export OPENROUTER_API_KEY

RUN_DIR="runs/${GAME}_coevo_v4_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="runs/${GAME}_coevo_v4.log"

echo ""
echo "============================================"
echo "  Game:       $GAME"
echo "  Steps:      $STEPS"
echo "  9B vLLM:    4× TP=1 on GPU 0-3"
echo "  GRPO:       GPU 6-7"
echo "  35B local:  GPU 4-5 (TP=2, MULTIMODAL, port 8001)"
echo "  Run dir:    $RUN_DIR"
echo "============================================"

trap "echo 'Caught signal — stopping 35B server (PID=$JUDGE_PID)...'; kill $JUDGE_PID 2>/dev/null || true; exit 130" INT TERM

python3 scripts/run_coevolution.py \
    --games "$GAME" \
    --total-steps "$STEPS" \
    --episodes-per-game 12 \
    --model Qwen/Qwen3.5-9B \
    --vllm-gpus 0 1 2 3 \
    --grpo-devices 6 7 \
    --load-adapters-from runs/lora_adapters/decision \
    --from-scratch \
    --curriculum none \
    --seed-bank-dir /workspace/Multi-hop-Reasoning-VLM-Agent/frontier_data/output/per_task_banks \
    --run-dir "$RUN_DIR" \
    --vllm-gpu-util 0.90 \
    --temperature 0.3 \
    --max-tokens 512 \
    2>&1 | tee "$LOG_FILE"
