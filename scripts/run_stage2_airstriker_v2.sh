#!/usr/bin/env bash
# Stage-2 Airstriker re-run with the T2.19c surv/adv shaping fix.
#
# Background
# ----------
# The first AS run (runs/gymv_airstriker_stage2_20260520_185405) showed
# mean reward degrading 47.5 → 20.0 over 9 steps.  Diagnostics on
# reward_shaping_log + the hidden action_survival_bonus contribution
# (1.5 per zero-raw step × ~96% of steps ≈ 700/step) revealed the true
# shape_ratio was 0.69 → 0.83 — survival/format constants dominated raw
# env reward (range 0.33/dec avg), biasing GRPO towards "stay still"
# and collapsing horizontal movement from 20% → 1%.
#
# Fix (T2.19c)
# ------------
# Use the new --dense-reward-overrides CLI knob to scale AS's gymv
# defaults to its actual raw-reward magnitude:
#   * action_survival_bonus: 1.5 → 0.05  (≈ 15% of per-dec raw, healthy)
#   * action_advance_bonus:  2.0 → 0.0   (vertical shmup, RIGHT = dodge
#                                         not progress; was only firing
#                                         on 1% of decisions anyway)
#
# Other knobs unchanged:
#   * episode_return_redistribution_weight stays at 0.25 (gymv default)
#   * format_bonus / passive_penalty / intrinsic_bonus untouched
#   * survival_bonus still only fires when raw_env_reward == 0
#
# Same wiring as scripts/run_stage2_ab_airstriker.sh otherwise.
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

# T2.19c override JSON (single source of truth — easy to bump per run).
# action_survival_bonus=0.05 → 15% of avg per-decision raw (0.33).
# action_advance_bonus=0.0   → kill the RIGHT bonus (vertical shmup).
DENSE_OVERRIDES='{"gymv_airstriker":{"action_survival_bonus":0.05,"action_advance_bonus":0.0}}'

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

    local run_dir="${REPO_ROOT}/runs/${game_slug}_stage2_v2_${TS}"
    mkdir -p "${run_dir}"
    local launch_log="${run_dir}/launch.log"

    log "============================================================"
    log "GAME: ${game_slug}  (SFT key: ${sft_key})"
    log "RUN DIR: ${run_dir}"
    log "DENSE-REWARD OVERRIDES: ${DENSE_OVERRIDES}"
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
        2>&1 | tee "${launch_log}"

    local rc=${PIPESTATUS[0]}
    log "Game ${game_slug} finished (rc=${rc})"

    kill_stale_9b_vllm
    return ${rc}
}

# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------

log "Stage-2 Airstriker v2 launcher started (ts=${TS})"
log "Seed bank: ${SEED_BANK_DIR}"
log "Fix: T2.19c per-game dense-reward overrides"

launch_game gymv_airstriker Temporal_Airstriker-v0

log "All games done."
