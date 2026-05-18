#!/usr/bin/env bash
# =============================================================================
# Sequential co-evolution chain runner.
#
# Waits for an in-flight run_coevo_local35b.sh (or candy_crush training) to
# finish, then runs each game in $GAMES in order with the same 12-episode /
# 10-step / local-35B-vision settings.
#
# Usage:
#   bash scripts/run_coevo_chain.sh [wait_for_pid] [game1 game2 ...]
#
# Defaults:
#   wait_for_pid : auto-detect from `pgrep -f run_coevo_local35b.sh`
#   games        : "gymv_strider gymv_columns"
# =============================================================================
set -uo pipefail

cd /workspace/Multi-hop-Reasoning-VLM-Agent

WAIT_PID="${1:-}"
shift 2>/dev/null || true
if [[ "$#" -gt 0 ]]; then
    GAMES=("$@")
else
    GAMES=(gymv_strider gymv_columns)
fi

if [[ -z "$WAIT_PID" ]]; then
    WAIT_PID=$(pgrep -fo "bash scripts/run_coevo_local35b.sh" || true)
fi

LOG="runs/coevo_chain_$(date +%Y%m%d_%H%M%S).log"
mkdir -p runs
echo "[chain] Logging to $LOG"
{
    echo "============================================"
    echo "  Chain runner started: $(date -u +%FT%TZ)"
    echo "  Games queued: ${GAMES[*]}"
    echo "  Waiting for PID: ${WAIT_PID:-<none, starting immediately>}"
    echo "============================================"
} | tee -a "$LOG"

# Phase 1 — wait for in-flight training
if [[ -n "$WAIT_PID" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
    echo "[chain] Waiting for PID $WAIT_PID ..." | tee -a "$LOG"
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 30
    done
    echo "[chain] PID $WAIT_PID exited at $(date -u +%FT%TZ)" | tee -a "$LOG"
else
    echo "[chain] No live PID to wait for; starting now" | tee -a "$LOG"
fi

# Phase 2 — give GPUs a moment to release contexts
echo "[chain] Cooling down 30s for GPU contexts to release..." | tee -a "$LOG"
sleep 30
# Belt-and-suspenders cleanup (idempotent)
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "Qwen3.5-35B-A3B" 2>/dev/null || true
pkill -9 -f "VLLM::" 2>/dev/null || true
sleep 10

# Phase 3 — run each game sequentially
for G in "${GAMES[@]}"; do
    echo "" | tee -a "$LOG"
    echo "============================================" | tee -a "$LOG"
    echo "[chain] >>> Starting game: $G  ($(date -u +%FT%TZ))" | tee -a "$LOG"
    echo "============================================" | tee -a "$LOG"

    bash scripts/run_coevo_local35b.sh "$G" 10
    rc=$?
    echo "[chain] <<< $G finished with rc=$rc at $(date -u +%FT%TZ)" | tee -a "$LOG"

    # Best-effort cleanup between games (script's own trap may have left
    # the local 35B server alive on clean exit — kill it so next launch
    # can claim port 8001 freshly).
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "Qwen3.5-35B-A3B" 2>/dev/null || true
    pkill -9 -f "VLLM::" 2>/dev/null || true
    sleep 20
done

echo "[chain] All games done at $(date -u +%FT%TZ)" | tee -a "$LOG"
