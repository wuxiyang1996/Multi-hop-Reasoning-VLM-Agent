#!/usr/bin/env bash
# Stage-2 Altered Beast v9 — PARALLEL variant (runs alongside AS v9).
# Uses GPU 4 (vLLM) + GPU 5 (GRPO) to avoid collision with AS on 0-3 + 6-7.
# CRITICAL FIX: uses --load-adapters-from to actually load the SFT adapter
# (--from-scratch alone random-inits, ignoring SFT).

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
SEED_BANK_DIR="${REPO_ROOT}/frontier_data/output/stage2_seeds_v3_grpo"
TS="$(date +%Y%m%d_%H%M%S)"

PRETRAINED_DIR="/tmp/ab_v9_pretrained_adapters"

DENSE_OVERRIDES='{"gymv_altered_beast":{"action_survival_bonus":0.0,"action_advance_bonus":0.0,"action_hit_penalty":0.0,"action_attack_bonus":0.0,"action_movement_bonus":0.0}}'

export VLM_AGENT_BACKBONE_JUDGE_MODEL="google/gemini-2.5-flash"
VISION_TIMEOUT_S=15
GRPO_LR="1e-5"
MIN_STEPS_BEFORE_STUCK=110

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

RUN_DIR="${REPO_ROOT}/runs/gymv_altered_beast_stage2_v9_${TS}"
mkdir -p "${RUN_DIR}"

log "============================================================"
log "ALTERED BEAST v9 — PARALLEL MODE (GPU 4 vLLM + GPU 5 GRPO)"
log "RUN DIR: ${RUN_DIR}"
log "SFT adapter: ${PRETRAINED_DIR}/action_taking"
log "VISION: ${VLM_AGENT_BACKBONE_JUDGE_MODEL}  (timeout=${VISION_TIMEOUT_S}s)"
log "GRPO LR: ${GRPO_LR}"
log "MIN_STEPS_BEFORE_STUCK: ${MIN_STEPS_BEFORE_STUCK}"
log "FIX: --load-adapters-from ensures SFT weights are loaded"
log "============================================================"

PYTHONUNBUFFERED=1 python3 scripts/run_coevolution.py \
    --games gymv_altered_beast \
    --total-steps 10 \
    --episodes-per-game 12 \
    --curriculum none \
    --seed-bank-dir "${SEED_BANK_DIR}" \
    --bank-mode per_game \
    --run-dir "${RUN_DIR}" \
    --vllm-gpus 4 \
    --vllm-base-port 8100 \
    --grpo-devices 5 \
    --grpo-lr "${GRPO_LR}" \
    --max-concurrent 16 \
    --no-wandb \
    --from-scratch \
    --load-adapters-from "${PRETRAINED_DIR}" \
    --dense-reward-overrides "${DENSE_OVERRIDES}" \
    --vision-state-perception-timeout-s "${VISION_TIMEOUT_S}" \
    --min-steps-before-stuck "${MIN_STEPS_BEFORE_STUCK}" \
    2>&1 | tee "${RUN_DIR}/launch.log"

log "Altered Beast v9 finished (rc=$?)"
