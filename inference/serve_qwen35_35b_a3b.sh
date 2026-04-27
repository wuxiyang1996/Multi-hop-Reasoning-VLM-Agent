#!/bin/bash
# =============================================================================
# Qwen3.5-35B-A3B inference-only vLLM server
# =============================================================================
# Spins up a single vLLM OpenAI-compatible endpoint for Qwen/Qwen3.5-35B-A3B
# (35B-total / 3B-active MoE) suitable for evaluation, baselines, or as a
# teacher / opponent.  This script does NOT load any LoRA adapters and is
# meant to run *outside* the GRPO training loop (Qwen3.5-9B + LoRA stays the
# trained model — see scripts/run_coevolution.py).
#
# When to use it:
#   • Post-training eval: kill the trainer, run this on the freed 8 H200s.
#   • Side-by-side baseline: launch on a separate machine and point your
#     eval harness at http://<host>:8001/v1.
#   • Teacher labeling: use as the oracle for `inference/run_*_eval.sh` by
#     setting `VLLM_BASE_URL` / `MODEL` in those scripts.
#
# GPU layout (defaults, 8x H200 / 141GB):
#   TENSOR_PARALLEL=8, EXPERT_PARALLEL=on  → ~5 GB weights/GPU + huge KV cache.
#   Override TENSOR_PARALLEL=4 if you only have 4 GPUs free during training.
#
# ======================== USAGE ==============================================
#
#   # Default: TP=8 + expert-parallel on GPUs 0-7, port 8001
#   bash inference/serve_qwen35_35b_a3b.sh
#
#   # Run on GPUs 4-7 only (e.g. while GRPO is training on 0-3):
#   GPUS="4,5,6,7" TENSOR_PARALLEL=4 \
#       bash inference/serve_qwen35_35b_a3b.sh
#
#   # Custom port + expose externally:
#   PORT=9000 HOST=0.0.0.0 bash inference/serve_qwen35_35b_a3b.sh
#
#   # FP8 weights (2x weight-memory savings, ~no quality loss):
#   QUANTIZATION=fp8 bash inference/serve_qwen35_35b_a3b.sh
#
#   # Multimodal mode (image+text inputs, vision tower enabled):
#   MULTIMODAL=1 bash inference/serve_qwen35_35b_a3b.sh
#
#   # Disable speculative decoding (debug perf):
#   SPECULATIVE=none bash inference/serve_qwen35_35b_a3b.sh
#
# =============================================================================

set -e

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Conda
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-game-ai-agent}"
CONDA_BASE="$(conda info --base 2>/dev/null || echo /workspace/miniconda3)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[serve-35b] Activated conda env: $CONDA_ENV - $(python --version 2>&1)"

# ---------------------------------------------------------------------------
# HF cache (shared with training)
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "$HF_HUB_CACHE"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-8}"
EXPERT_PARALLEL="${EXPERT_PARALLEL:-1}"   # 1=enable, 0=disable
GPU_UTIL="${GPU_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
PORT="${PORT:-8001}"
HOST="${HOST:-127.0.0.1}"
QUANTIZATION="${QUANTIZATION:-}"          # e.g. fp8, awq_marlin, gptq_marlin
MULTIMODAL="${MULTIMODAL:-0}"             # 1 to keep the vision tower
SPECULATIVE="${SPECULATIVE:-mtp}"         # mtp | none
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-1}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

# ---------------------------------------------------------------------------
# Build vllm command
# ---------------------------------------------------------------------------
VLLM_CMD=(
    python -m vllm.entrypoints.openai.api_server
    --model "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL"
    --gpu-memory-utilization "$GPU_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-prefix-caching
    --enable-chunked-prefill
    --trust-remote-code
    --dtype auto
)

# MoE expert-parallel reduces per-GPU weight memory: each rank holds a
# disjoint slice of experts instead of replicating them.  Highly
# recommended for Qwen3.5-MoE families when TP > 1.
if [ "$EXPERT_PARALLEL" = "1" ] && [ "$TENSOR_PARALLEL" -gt 1 ]; then
    VLLM_CMD+=( --enable-expert-parallel )
fi

if [ "$MULTIMODAL" != "1" ]; then
    VLLM_CMD+=( --language-model-only )
fi

if [ -n "$REASONING_PARSER" ]; then
    VLLM_CMD+=( --reasoning-parser "$REASONING_PARSER" )
fi

if [ -n "$QUANTIZATION" ]; then
    VLLM_CMD+=( --quantization "$QUANTIZATION" )
fi

if [ "$SPECULATIVE" = "mtp" ] && [ "$NUM_SPEC_TOKENS" -gt 0 ]; then
    SPEC_JSON=$(printf '{"method":"mtp","num_speculative_tokens":%d}' \
                       "$NUM_SPEC_TOKENS")
    VLLM_CMD+=( --speculative_config "$SPEC_JSON" )
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Qwen3.5-35B-A3B  (inference-only)"
echo "============================================"
echo "  Model:           $MODEL"
echo "  Bind:            http://${HOST}:${PORT}/v1"
echo "  GPUs:            $GPUS  (TP=${TENSOR_PARALLEL}, EP=${EXPERT_PARALLEL})"
echo "  Multimodal:      $MULTIMODAL  (0 = text-only / vision tower skipped)"
echo "  Quantization:    ${QUANTIZATION:-bf16}"
echo "  Speculative:     $SPECULATIVE  (${NUM_SPEC_TOKENS} tok)"
echo "  Reasoning:       $REASONING_PARSER"
echo "============================================"
echo ""
echo "[serve-35b] Command:"
printf '  %q ' "${VLLM_CMD[@]}"
echo
echo ""

CUDA_VISIBLE_DEVICES="$GPUS" exec "${VLLM_CMD[@]}"
