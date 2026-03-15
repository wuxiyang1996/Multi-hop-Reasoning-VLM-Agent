#!/usr/bin/env bash
# ======================================================================
#  Co-Evolution: Launch vLLM inference + training in one command.
#
#  Starts the vLLM server in the background (GPUs 0-3), waits for it
#  to become healthy, then runs the co-evolution training loop (GPUs 4-7).
#  On exit (or Ctrl-C) the vLLM server is automatically killed.
#
#  Prerequisites:
#    conda activate game-ai-agent
#    pip install wandb tensorboard peft   # one-time
#
#  Usage:
#    bash scripts/run_all.sh
#
#    # Override settings via env vars:
#    VLLM_MODEL=Qwen/Qwen3-8B VLLM_TP=2 TOTAL_STEPS=50 bash scripts/run_all.sh
#
#    # Option 1: Train from scratch (gaussian random LoRA init):
#    FROM_SCRATCH=1 bash scripts/run_all.sh
#
#    # Option 2: Load pre-trained adapters and continue GRPO training:
#    LOAD_ADAPTERS_FROM=runs/Qwen3-14B_20260310_120000/lora_adapters bash scripts/run_all.sh
#
#    # Resume a previous run from checkpoint:
#    RUN_DIR=runs/Qwen3-14B_20260315_143022 RESUME=1 bash scripts/run_all.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# ── PYTHONPATH ────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PYTHONPATH:-}"

# ── Configurable parameters ──────────────────────────────────────────
MODEL="${VLLM_MODEL:-Qwen/Qwen3-14B}"
PORT="${VLLM_PORT:-8000}"
TP="${VLLM_TP:-4}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"

TOTAL_STEPS="${TOTAL_STEPS:-100}"
EPISODES="${EPISODES_PER_GAME:-8}"
CKPT_INTERVAL="${CKPT_INTERVAL:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution}"
RUN_DIR="${RUN_DIR:-}"
RESUME="${RESUME:-}"
FROM_SCRATCH="${FROM_SCRATCH:-}"
LOAD_ADAPTERS_FROM="${LOAD_ADAPTERS_FROM:-}"

# ── Cleanup on exit ──────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
    echo ""
    echo "[run_all] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[run_all] Stopping vLLM server (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[run_all] Done."
}
trap cleanup EXIT INT TERM

# ======================================================================
# Phase 1: Launch vLLM inference server
# ======================================================================
echo "══════════════════════════════════════════════════════════════"
echo "  Co-Evolution: Starting vLLM + Training"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:         ${MODEL}"
echo "  vLLM TP:       ${TP} GPUs"
echo "  vLLM port:     ${PORT}"
echo "  Total steps:   ${TOTAL_STEPS}"
echo "  Eps/game:      ${EPISODES}"
echo "  Checkpoint:    every ${CKPT_INTERVAL} steps"
if [ -n "${RUN_DIR}" ]; then
    echo "  Run dir:       ${RUN_DIR}"
fi
if [ -n "${FROM_SCRATCH}" ]; then
    echo "  Start mode:    FROM SCRATCH"
elif [ -n "${RESUME}" ]; then
    echo "  Start mode:    RESUME"
else
    echo "  Start mode:    AUTO"
fi
echo "══════════════════════════════════════════════════════════════"

# ======================================================================
# Phase 0: Ensure LoRA adapters exist (cold-start init)
#
# Creates zero-initialised adapters if missing (identical to the base
# model until GRPO training updates them).  Also resolves the final
# RUN_DIR so vLLM and training share the same timestamped directory.
# ======================================================================
echo ""
echo "[run_all] Ensuring LoRA adapters exist..."

RESOLVED_RUN_DIR=$(python -c "
import sys, os
os.environ.setdefault('PYGLET_HEADLESS', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
from trainer.coevolution.config import CoEvolutionConfig, init_lora_adapters
cfg = CoEvolutionConfig(model_name='${MODEL}')
run_dir_override = '${RUN_DIR}'
if run_dir_override:
    cfg.run_dir = run_dir_override
force = bool('${FROM_SCRATCH}')
if force:
    cfg.start_mode = 'from_scratch'
cfg.resolve_paths()
created = init_lora_adapters(cfg, force=force)
if created:
    print(f'Created {len(created)} adapter(s) (gaussian init): {created}', file=sys.stderr)
else:
    print('All adapters already exist.', file=sys.stderr)
print(cfg.run_dir)
")

RUN_DIR="${RESOLVED_RUN_DIR}"
export ADAPTER_DIR="${RUN_DIR}/lora_adapters"
echo "[run_all] Run dir:     ${RUN_DIR}"
echo "[run_all] Adapter dir: ${ADAPTER_DIR}"

echo ""
echo "[run_all] Starting vLLM server..."
bash scripts/launch_vllm_coevolution.sh &
VLLM_PID=$!
echo "[run_all] vLLM server PID: ${VLLM_PID}"

# ── Wait for vLLM to become healthy ──────────────────────────────────
echo "[run_all] Waiting for vLLM to become ready at http://localhost:${PORT}..."
MAX_WAIT=300
WAITED=0
while [ ${WAITED} -lt ${MAX_WAIT} ]; do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || \
       curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[run_all] vLLM is ready! (waited ${WAITED}s)"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[run_all] ERROR: vLLM server exited unexpectedly."
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $((WAITED % 30)) -eq 0 ]; then
        echo "[run_all]   ... still waiting (${WAITED}s / ${MAX_WAIT}s)"
    fi
done

if [ ${WAITED} -ge ${MAX_WAIT} ]; then
    echo "[run_all] ERROR: vLLM server did not become ready within ${MAX_WAIT}s"
    exit 1
fi

# ======================================================================
# Phase 2: Run co-evolution training
# ======================================================================
echo ""
echo "[run_all] Starting co-evolution training..."

TRAIN_ARGS=(
    --total-steps "${TOTAL_STEPS}"
    --episodes-per-game "${EPISODES}"
    --checkpoint-interval "${CKPT_INTERVAL}"
    --model "${MODEL}"
    --vllm-url "http://localhost:${PORT}/v1"
    --wandb-project "${WANDB_PROJECT}"
    --run-dir "${RUN_DIR}"
)

if [ -n "${FROM_SCRATCH}" ]; then
    TRAIN_ARGS+=(--from-scratch)
elif [ -n "${RESUME}" ]; then
    TRAIN_ARGS+=(--resume)
fi

if [ -n "${LOAD_ADAPTERS_FROM}" ]; then
    TRAIN_ARGS+=(--load-adapters-from "${LOAD_ADAPTERS_FROM}")
fi

python scripts/run_coevolution.py "${TRAIN_ARGS[@]}"

echo ""
echo "[run_all] Training complete."
