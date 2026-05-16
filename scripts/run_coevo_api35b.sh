#!/usr/bin/env bash
# =============================================================================
# Co-evolution with API-based 35B judge + 6-GPU 9B rollout + 2-GPU GRPO
#
# Layout:
#   GPU 0-5  →  6× TP=1 vLLM (Qwen3.5-9B) for rollout
#   GPU 6-7  →  GRPO training
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
    print('')
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

# 9B → local vLLM (managed by orchestrator on GPU 0-5)
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

python scripts/run_coevolution.py \
    --games "$GAME" \
    --total-steps "$STEPS" \
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
