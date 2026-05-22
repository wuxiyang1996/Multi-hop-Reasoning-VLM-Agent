#!/usr/bin/env bash
# =============================================================================
# Co-evolution with API-based 35B judge + 12-GPU 9B rollout + 2-GPU GRPO
#
# Layout:
#   GPU 0-11 →  12× TP=1 vLLM (Qwen3.5-9B) for rollout
#   GPU 12-13→  GRPO training
#   35B      →  OpenRouter API (no local GPU needed)
#
# Usage:
#   bash scripts/run_coevo_api35b.sh <game_slug> [total_steps]
#
# Examples:
#   bash scripts/run_coevo_api35b.sh gymv_thunder_force_iii 10
#   bash scripts/run_coevo_api35b.sh candy_crush 10
#   bash scripts/run_coevo_api35b.sh tetris 10
# =============================================================================
set -euo pipefail

GAME="${1:?Usage: $0 <game_slug> [total_steps]}"
STEPS="${2:-10}"

cd /workspace/Multi-hop-Reasoning-VLM-Agent

# ── PYTHONPATH ────────────────────────────────────────────────────────
export PYTHONPATH="/workspace/Multi-hop-Reasoning-VLM-Agent:/workspace/GamingAgent:/workspace/GamingAgent/gamingagent/envs/custom_03_candy_crush:${PYTHONPATH:-}"
export HF_HOME="/workspace/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
# T2.21 (2026-05-19): 64→48 on 80 GB A100 to leave OOM headroom for
# FSDP activations + KV when GRPO and 6× 9B vLLM share the box.  64
# triggered SIGABRT mid-step on the tetris 044258 run.  48 matches
# what run_coevo_local35b.sh uses with the local 35B judge layout.
export GRPO_FSDP_BATCH_SIZE=48

# ── Conda env with vLLM 0.20.2 + torch + peft ───────────────────────
# Use the explicit interpreter so that:
#   1. `python3 scripts/run_coevolution.py` resolves vllm/torch/peft
#   2. vLLM workers spawned by orchestrator (via sys.executable) inherit
#      the same interpreter, not /usr/bin/python (which lacks vllm).
CONDA_PY="/workspace/miniconda3/envs/game-ai-agent/bin/python"
export PATH="/workspace/miniconda3/envs/game-ai-agent/bin:${PATH}"

# ── Clean up stale GPU processes ─────────────────────────────────────
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 1

# ── Per-game LoRA switching ──────────────────────────────────────────
# Both skill_selection and action_taking are per-game SFT LoRAs.
# Map game slugs to their SFT directory names.
declare -A GAME_TO_SFT_KEY=(
    [gymv_thunder_force_iii]="Temporal_ThunderForceIII-v0"
    [candy_crush]="candy_crush"
    [tetris]="tetris"
    [super_mario]="super_mario"
    [twenty_forty_eight]="twenty_forty_eight"
    [gymv_altered_beast]="Temporal_AlteredBeast-v0"
    [gymv_columns]="Temporal_Columns-v0"
    [gymv_dynamite_headdy]="Temporal_DynamiteHeaddy-v0"
    [gymv_airstriker]="Temporal_Airstriker-v0"
    [gymv_space_harrier_ii]="Temporal_SpaceHarrierII-v0"
    [gymv_streets_of_rage_2]="Temporal_StreetsOfRage2-v0"
    [gymv_strider]="Temporal_Strider-v0"
)

SFT_KEY="${GAME_TO_SFT_KEY[$GAME]:-}"
if [[ -z "$SFT_KEY" ]]; then
    echo "ERROR: Unknown game '$GAME'. Valid games:"
    printf '  %s\n' "${!GAME_TO_SFT_KEY[@]}"
    exit 1
fi

ADAPTER_DIR="runs/lora_adapters/decision"

# skill_selection (v3 SFT)
SK_SRC="runs/sft_per_game_v3/${SFT_KEY}/skill_selection/${SFT_KEY}__skill_selection"
if [[ -d "$SK_SRC" ]]; then
    rm -f "${ADAPTER_DIR}/skill_selection/adapter_config.json" \
          "${ADAPTER_DIR}/skill_selection/adapter_model.safetensors"
    ln -s "$(pwd)/${SK_SRC}/adapter_config.json" \
          "${ADAPTER_DIR}/skill_selection/adapter_config.json"
    ln -s "$(pwd)/${SK_SRC}/adapter_model.safetensors" \
          "${ADAPTER_DIR}/skill_selection/adapter_model.safetensors"
    echo "✓ skill_selection → ${SFT_KEY}"
else
    echo "WARNING: No v3 skill_selection LoRA for ${SFT_KEY}, keeping current"
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
else
    echo "WARNING: No action_taking LoRA for ${SFT_KEY}, keeping current"
fi

# ── API keys ─────────────────────────────────────────────────────────
# Source from keys.py if env vars not set
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    OPENROUTER_API_KEY=$(python3 -c "
import sys
for p in ['.', '..', '/workspace']:
    sys.path.insert(0, p)
try:
    from keys import openrouter_api_key
    print(openrouter_api_key)
except Exception:
    pass
try:
    from key import openrouter_api_key
    print(openrouter_api_key)
except Exception:
    pass
")
fi
export OPENROUTER_API_KEY

if [[ -z "$OPENROUTER_API_KEY" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set and keys.py not found"
    exit 1
fi
echo "✓ OpenRouter API key loaded"

# ── Environment ──────────────────────────────────────────────────────
export VLM_AGENT_BACKBONE_JUDGE_MODEL="Qwen/Qwen3.5-35B-A3B"

# 9B → local vLLM (managed by orchestrator on GPU 0-11)
# 35B → OpenRouter API
export VLLM_BASE_URL_MAP="Qwen/Qwen3.5-9B=http://localhost:8000/v1,Qwen/Qwen3.5-35B-A3B=https://openrouter.ai/api/v1"

RUN_DIR="runs/${GAME}_coevo_v4_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="runs/${GAME}_coevo_v4.log"

echo "============================================"
echo "  Game:       $GAME"
echo "  SFT key:    $SFT_KEY"
echo "  Steps:      $STEPS"
echo "  9B vLLM:    6× TP=1 on GPU 0-5"
echo "  GRPO:       GPU 6-7"
echo "  35B:        OpenRouter API"
echo "  Run dir:    $RUN_DIR"
echo "  Log:        $LOG_FILE"
echo "============================================"

"$CONDA_PY" scripts/run_coevolution.py \
    --games "$GAME" \
    --total-steps "$STEPS" \
    --episodes-per-game 12 \
    --model Qwen/Qwen3.5-9B \
    --vllm-gpus 0 1 2 3 4 5 \
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
