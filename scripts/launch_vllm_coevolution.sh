#!/usr/bin/env bash
# Launch vLLM server for co-evolution training.
#
# Uses GPUs 0-3 for inference with tensor parallelism.
# GPUs 4-7 are reserved for GRPO training.
#
# Headless: no display required.
#
# Usage:
#   conda activate game-ai-agent
#   bash scripts/launch_vllm_coevolution.sh
#
#   # Override defaults via env vars:
#   VLLM_MODEL=Qwen/Qwen3-8B VLLM_TP=2 bash scripts/launch_vllm_coevolution.sh
#
#   # Restrict to specific GPUs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/launch_vllm_coevolution.sh
#
#   # Point to a timestamped run directory for adapters:
#   ADAPTER_DIR=runs/Qwen3-14B_20260315_143022/lora_adapters bash scripts/launch_vllm_coevolution.sh

set -euo pipefail

# Headless rendering
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

MODEL="${VLLM_MODEL:-Qwen/Qwen3-14B}"
PORT="${VLLM_PORT:-8000}"
TP="${VLLM_TP:-4}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
ADAPTER_DIR="${ADAPTER_DIR:-runs/lora_adapters}"

echo "═══════════════════════════════════════════════════"
echo "  vLLM Co-Evolution Server"
echo "═══════════════════════════════════════════════════"
echo "  Model:    ${MODEL}"
echo "  TP:       ${TP}"
echo "  GPU Util: ${GPU_UTIL}"
echo "  Port:     ${PORT}"
echo "  Adapters: ${ADAPTER_DIR}"
echo "═══════════════════════════════════════════════════"

# Build --lora-modules args (only for adapters that exist)
LORA_ARGS=""
for adapter in skill_selection action_taking segment contract curator; do
    adapter_path="${ADAPTER_DIR}/${adapter}"
    if [ -d "${adapter_path}" ]; then
        LORA_ARGS="${LORA_ARGS} ${adapter}=${adapter_path}"
        echo "  LoRA: ${adapter} → ${adapter_path}"
    else
        echo "  LoRA: ${adapter} → (not found, will use base model)"
    fi
done

echo "═══════════════════════════════════════════════════"

# Always enable LoRA support so adapters can be hot-loaded after GRPO
# training creates them (cold start has no adapters yet).
LORA_FLAGS="--enable-lora --max-loras 5 --max-lora-rank 64"
if [ -n "${LORA_ARGS}" ]; then
    LORA_FLAGS="${LORA_FLAGS} --lora-modules ${LORA_ARGS}"
fi

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    ${LORA_FLAGS} \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-seqs 128 \
    --port "${PORT}" \
    --trust-remote-code \
    "$@"
