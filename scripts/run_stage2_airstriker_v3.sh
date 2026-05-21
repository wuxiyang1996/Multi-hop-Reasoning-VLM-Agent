#!/usr/bin/env bash
# Stage-2 Airstriker v3 — stacks T2.19c (surv-fix) + T2.19d (hit_penalty).
#
# Background
# ----------
# * v1 (runs/gymv_airstriker_stage2_20260520_185405): collapsed
#   47.5 → 20.0 over 9 steps.  Root cause: ``action_survival_bonus=1.5``
#   per zero-raw-step ≈ 4.4× raw env reward → GRPO advantage
#   dominated by "stay still" signal.
# * v2 (run_stage2_airstriker_v2.sh, killed before step 0 finished):
#   would have only fixed the surv-bonus calibration.  User requested
#   stacking the new RAM-derived hit penalty.
#
# Fix (T2.19c + T2.19d)
# ---------------------
# Per-game overrides via --dense-reward-overrides:
#   * action_survival_bonus 1.5 → 0.05  (T2.19c, 30× ↓, lives surv as
#                                        a tiny cushion but doesn't
#                                        hijack the gradient)
#   * action_advance_bonus  2.0 → 0.0   (T2.19c, vertical shmup; RIGHT
#                                        is dodge, not progress)
#   * action_hit_penalty    0   → 5.0   (T2.19d, NEW — −5 per life lost,
#                                        from RAM-watch ``lives`` delta.
#                                        Replaces the "bullet-dodge
#                                        bonus" idea with a direct
#                                        ground-truth signal: the
#                                        Airstriker emulator's own life
#                                        counter at address 16712282 in
#                                        stable_retro/data.json.
#                                        Magnitude calibration: 50% of
#                                        one +10 score event = strong
#                                        enough to penalise reckless
#                                        flight, small enough to avoid
#                                        the v1 "do nothing" collapse.)
#
# Mechanics of the new hit_penalty:
#   * stable-retro env wrapper already plumbs ``info.structured_state.ram_watch``
#     every step.  Verified live 2026-05-20: ``{gameover:9, lives:3, score:0}``
#     populates after env.reset+step.
#   * episode_runner tracks ``_prev_lives`` per episode; computes
#     ``Δlives = curr - prev``; applies ``-action_hit_penalty * |Δ|``
#     only on negative delta (positive deltas from 1UP pickups are
#     ignored, never rewarded).
#   * The penalty is folded into the action_taking GRPO reward and the
#     reward_shaping_log ``constant_offset`` field so the diagnostic
#     shape_ratio sees it.
#
# Coverage in other gymv games (data.json watches surveyed 2026-05-20):
#   * lives:  Airstriker, SpaceHarrierII, AlteredBeast, StreetsOfRage2,
#             DynamiteHeaddy, ThunderForceIII, Strider (7/8)
#   * health: AlteredBeast, DynamiteHeaddy, Strider (additional finer
#             gradient via action_damage_penalty)
#   * Only Columns has neither (puzzle game — no lives concept).
#
# Same launch wiring as v1/v2:
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

# T2.19c (surv-fix) + T2.19d (hit_penalty) combined JSON.
# Single source of truth — bump per-knob here without editing the call.
DENSE_OVERRIDES='{"gymv_airstriker":{"action_survival_bonus":0.05,"action_advance_bonus":0.0,"action_hit_penalty":5.0}}'

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

    local run_dir="${REPO_ROOT}/runs/${game_slug}_stage2_v3_${TS}"
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

log "Stage-2 Airstriker v3 launcher started (ts=${TS})"
log "Seed bank: ${SEED_BANK_DIR}"
log "Fix: T2.19c (surv calibration) + T2.19d (RAM hit penalty)"

launch_game gymv_airstriker Temporal_Airstriker-v0

log "All games done."
