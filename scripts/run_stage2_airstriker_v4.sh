#!/usr/bin/env bash
# Stage-2 Airstriker v4 — stacks Tier-A (reward re-shape) + Tier-B
# (vision-perception fix).  Successor to v3 (id stage2_v3_20260521_011504).
#
# Why v4 (vs v3): comprehensive teacher-vs-agent rollout analysis showed
# two simultaneous root causes for the 42.5 mean_reward plateau ≪
# teachers (93.8-97.5):
#
#   1. VISION OOD — every single 35B vision-perception call was timing
#      out (default 6s vs measured 11-47s latency) so the
#      ``action_taking`` LoRA received the deterministic Python
#      ``state_to_markup`` fallback (HUD-blind, ``pos=null``,
#      ``label=genesis_viewport`` only).  Cold-start SFT was on
#      entity-rich markup (player ship + threat positions on the 12×16
#      grid), so this was a fundamental train/serve distribution
#      mismatch.  Fix: T2.17 timeout 6s → 45s + WARN-level fallback
#      surfacing.
#
#   2. REWARD MIS-CALIBRATION — v3's ``action_hit_penalty=5.0`` lowered
#      hit rate (good) but also collapsed aggressive scoring (bad).
#      Teachers strategically lose 1-2 lives per ep to maximise score;
#      the −5 penalty made the agent overly defensive.  v3 also had
#      LEFT+RIGHT = 3% (vs teachers' 21-29%) → almost no horizontal
#      evasion.
#
#      Fix (T2.19e):
#        * action_hit_penalty   5.0 → 1.0   (10% of one +10 score event;
#                                            mild risk-aversion, not
#                                            risk-suppression)
#        * action_attack_bonus  NEW → 0.05  (× B-press, encourages
#                                            firing — teachers fire 39-43%
#                                            of frames)
#        * action_movement_bonus NEW → 0.20 (× LEFT/RIGHT press,
#                                            encourages active evasion —
#                                            biggest behavior gap)
#
# Expected shape_ratio (with env_reward~50/ep):
#   bonus magnitude ≈ 1 (hit) + 2 (surv) + 4 (attack+move at teacher
#                       distribution)  =  ~7-10
#   → shape_ratio ≈ 0.12-0.18 — well below the 0.50 "GRPO advantage
#   dominated by shaping" threshold seen in v1 collapse.
#
# Same launch wiring as v1/v2/v3:
#   GPUs 0-3 : 4× Qwen3.5-9B  (managed by run_coevolution.py)
#   GPUs 4-5 : Qwen3.5-35B-A3B vision judge on port 8001 (KEEP, shared)
#   GPUs 6-7 : GRPO devices

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
SEED_BANK_DIR="${REPO_ROOT}/frontier_data/output/stage2_seeds_v3_grpo"
SFT_AT_ROOT="${REPO_ROOT}/runs/sft_per_game"          # action_taking
SFT_SS_ROOT="${REPO_ROOT}/runs/sft_per_game_v3"       # skill_selection (v3 balanced)
LORA_LINK_DIR="${REPO_ROOT}/runs/lora_adapters/decision"
TS="$(date +%Y%m%d_%H%M%S)"

# T2.19c (surv-fix) + T2.19d (hit_penalty, halved 5→1) + T2.19e
# (attack/movement bonuses).  Single JSON — bump per-knob here.
DENSE_OVERRIDES='{"gymv_airstriker":{"action_survival_bonus":0.05,"action_advance_bonus":0.0,"action_hit_penalty":1.0,"action_attack_bonus":0.05,"action_movement_bonus":0.2}}'

# Vision-perception knobs.  The 45s default matches the new
# _DEFAULT_TIMEOUT_S in _vision_state_perception.py — set explicitly
# here so we don't silently regress when defaults change.
VISION_TIMEOUT_S=45

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

kill_stale_9b_vllm() {
    local pids
    pids=$(pgrep -f "vllm.entrypoints.*Qwen3\.5-9B" || true)
    if [[ -n "${pids}" ]]; then
        log "Killing stale 9B vllm PIDs: ${pids//$'\n'/ }"
        kill ${pids} 2>/dev/null || true
        sleep 4
        local remaining
        remaining=$(pgrep -f "vllm.entrypoints.*Qwen3\.5-9B" || true)
        if [[ -n "${remaining}" ]]; then
            log "SIGKILL holdouts: ${remaining//$'\n'/ }"
            kill -9 ${remaining} 2>/dev/null || true
            sleep 2
        fi
    else
        log "No stale 9B vllm servers to kill."
    fi

    if ! curl -sf -m 3 http://localhost:8001/v1/models >/dev/null 2>&1; then
        log "WARN: 35B vision judge on :8001 not reachable."
    else
        log "35B vision judge :8001 still healthy."
    fi
}

repoint_decision_lora() {
    local key="$1"
    local at_src="${SFT_AT_ROOT}/${key}/action_taking/${key}__action_taking"
    local ss_src="${SFT_SS_ROOT}/${key}/skill_selection/${key}__skill_selection"

    if [[ ! -f "${at_src}/adapter_model.safetensors" ]]; then
        log "FATAL: action_taking adapter missing at ${at_src}"
        return 1
    fi
    if [[ ! -f "${ss_src}/adapter_model.safetensors" ]]; then
        log "FATAL: skill_selection adapter missing at ${ss_src}"
        return 1
    fi

    mkdir -p "${LORA_LINK_DIR}/action_taking" "${LORA_LINK_DIR}/skill_selection"
    for f in adapter_model.safetensors adapter_config.json; do
        ln -sfT "${at_src}/${f}" "${LORA_LINK_DIR}/action_taking/${f}"
        ln -sfT "${ss_src}/${f}" "${LORA_LINK_DIR}/skill_selection/${f}"
    done
    log "Repointed decision LoRA → ${key}"
}

launch_game() {
    local game_slug="$1"
    local sft_key="$2"

    local run_dir="${REPO_ROOT}/runs/${game_slug}_stage2_v4_${TS}"
    mkdir -p "${run_dir}"
    local launch_log="${run_dir}/launch.log"

    log "============================================================"
    log "GAME: ${game_slug}  (SFT key: ${sft_key})"
    log "RUN DIR: ${run_dir}"
    log "DENSE-REWARD OVERRIDES: ${DENSE_OVERRIDES}"
    log "VISION TIMEOUT: ${VISION_TIMEOUT_S}s  (was 6s in v3 → 100% silent fallback)"
    log "============================================================"

    kill_stale_9b_vllm
    repoint_decision_lora "${sft_key}"

    log "Launching run_coevolution.py …"
    PYTHONUNBUFFERED=1 python3 scripts/run_coevolution.py \
        --games "${game_slug}" \
        --total-steps 10 \
        --episodes-per-game 12 \
        --curriculum none \
        --seed-bank-dir "${SEED_BANK_DIR}" \
        --bank-mode per_game \
        --run-dir "${run_dir}" \
        --vllm-gpus 0 1 2 3 \
        --grpo-devices 6 7 \
        --max-concurrent 64 \
        --no-wandb \
        --from-scratch \
        --dense-reward-overrides "${DENSE_OVERRIDES}" \
        --vision-state-perception-timeout-s "${VISION_TIMEOUT_S}" \
        2>&1 | tee "${launch_log}"

    local rc=${PIPESTATUS[0]}
    log "Game ${game_slug} finished (rc=${rc})"

    kill_stale_9b_vllm
    return ${rc}
}

# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------

log "Stage-2 Airstriker v4 launcher started (ts=${TS})"
log "Seed bank: ${SEED_BANK_DIR}"
log "Fixes: Tier-B (vision timeout 6→45s) + Tier-A (hit 5→1, +attack 0.05, +movement 0.20)"

launch_game gymv_airstriker Temporal_Airstriker-v0

log "All games done."
