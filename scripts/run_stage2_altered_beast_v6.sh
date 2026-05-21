#!/usr/bin/env bash
# Stage-2 Altered Beast v6 — mirror of run_stage2_airstriker_v6.sh
# with altered_beast-specific reward shaping + the post-hotfix
# action / reasoning / state-flags pipeline (commit 4fbef06).
#
# Why this run
# ------------
# Run gymv_altered_beast_stage2_20260520_185405 (the previous, pre-hotfix
# attempt) caps every episode at raw_env_reward=200 (the stage-1
# life-bar bonus) with std=0 across 12 episodes × 10 outer steps —
# i.e. no enemy kills, no reward gradient, no GRPO learning signal.
# Root causes diagnosed in that run's debrief:
#
#   * "REASONING: Expert play." fabricated in 78-98% of action_taking
#     completions → policy reasoning collapsed.
#   * <actions> top-5 dominated by system buttons (MODE/START) →
#     skill-selector never saw RIGHT.
#   * action_taking picked B in 64% of steps and RIGHT in only ~4%,
#     vs gemini-teacher's 51% RIGHT / 22% B in 1300-pt episodes.
#   * GAME_MAX_STEPS=80 truncated just before the boss-spike at
#     step 72-76 (where the +1000 lands).
#
# Commit 4fbef06 ("fix(coevo): break the altered_beast reasoning-collapse
# + B-mash loop") fixed all four.  This launcher exercises those fixes
# from scratch — the SFT cold-start LoRA is reused (never GRPO-trained
# for altered_beast, so it can't have been collapsed by previous runs).
#
# Dense-reward overrides
# ----------------------
#   action_survival_bonus = 0.05  ── tiny per-step bonus for staying
#                                    alive without dying; matches v6 AS.
#   action_advance_bonus  = 0.2   ── per-step bonus when chosen action
#                                    is in ``action_advance_actions``.
#                                    Beatemups scroll only when RIGHT
#                                    is pressed, so this is the
#                                    movement-prior signal.  v6 AS used
#                                    0.0 because shmups auto-scroll.
#   action_advance_actions = RIGHT ── (default) — already correct.
#   action_hit_penalty    = 1.0   ── -1.0 reward per life lost (Δlives
#                                    < 0).  Discourages reckless approach.
#   action_attack_bonus   = 0.05  ── tiny B-press nudge; B is also a
#                                    GAME_CRITICAL_ACTIONS entry so the
#                                    dry-spell guard already enforces it.
#   action_movement_bonus = 0.2   ── nudges any movement (UP/DOWN/LEFT/
#                                    RIGHT) — useful for repositioning
#                                    between attacks.
#
# Vision-perception layer
# -----------------------
# Uses OpenRouter Gemini 2.5 Flash (same as v6 AS) for the 35B-judge
# state markup.  Latency 4-5s vs the local 35B's 22-45s, success rate
# ~100%, cost ~$3-5 for the full 10-step run.
#
# Launch wiring
# -------------
#   GPUs 0-3 : 4× Qwen3.5-9B  (managed by run_coevolution.py)
#   GPUs 4-5 : IDLE (35B kept as last-resort fallback, see KEEP_35B)
#   GPUs 6-7 : GRPO devices

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
SEED_BANK_DIR="${REPO_ROOT}/frontier_data/output/stage2_seeds_v3_grpo"
SFT_AT_ROOT="${REPO_ROOT}/runs/sft_per_game"
SFT_SS_ROOT="${REPO_ROOT}/runs/sft_per_game_v3"
LORA_LINK_DIR="${REPO_ROOT}/runs/lora_adapters/decision"
TS="$(date +%Y%m%d_%H%M%S)"

# Altered Beast (beatemup) — movement matters more than for the shmups
# Airstriker used as the v6 reference, so action_advance_bonus is bumped
# from 0.0 → 0.2 to encourage RIGHT-scrolling.
DENSE_OVERRIDES='{"gymv_altered_beast":{"action_survival_bonus":0.05,"action_advance_bonus":0.2,"action_hit_penalty":1.0,"action_attack_bonus":0.05,"action_movement_bonus":0.2}}'

export VLM_AGENT_BACKBONE_JUDGE_MODEL="google/gemini-2.5-flash"
VISION_TIMEOUT_S=15

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

kill_stale_9b_vllm() {
    local pids
    pids=$(pgrep -f "vllm.entrypoints.*Qwen3\.5-9B" || true)
    if [[ -n "${pids}" ]]; then
        log "Killing stale 9B vllm PIDs: ${pids//$'\n'/ }"
        for p in ${pids}; do kill ${p} 2>/dev/null || true; done
        sleep 4
        local engcore
        engcore=$(pgrep -f "VLLM::EngineCore" || true)
        # Keep EngineCore workers parented to the 35B server alive
        # (last-resort fallback should OpenRouter rate-limit).
        for p in ${engcore}; do
            ppid=$(ps -o ppid= -p ${p} 2>/dev/null | tr -d ' ')
            local parent_cmd
            parent_cmd=$(ps -p ${ppid} -o cmd= 2>/dev/null || true)
            if [[ "${parent_cmd}" != *"Qwen3.5-35B"* ]]; then
                kill -9 ${p} 2>/dev/null || true
            fi
        done
        sleep 3
        local remaining
        remaining=$(pgrep -f "vllm.entrypoints.*Qwen3\.5-9B" || true)
        if [[ -n "${remaining}" ]]; then
            log "SIGKILL holdouts: ${remaining//$'\n'/ }"
            for p in ${remaining}; do kill -9 ${p} 2>/dev/null || true; done
            sleep 2
        fi
    else
        log "No stale 9B vllm servers to kill."
    fi

    if ! curl -sf -m 3 http://localhost:8001/v1/models >/dev/null 2>&1; then
        log "INFO: 35B vision judge on :8001 not reachable (OK — using OpenRouter)."
    else
        log "35B vision judge :8001 still healthy (kept as last-resort fallback)."
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

    local run_dir="${REPO_ROOT}/runs/${game_slug}_stage2_v6_${TS}"
    mkdir -p "${run_dir}"
    local launch_log="${run_dir}/launch.log"

    log "============================================================"
    log "GAME: ${game_slug}  (SFT key: ${sft_key})"
    log "RUN DIR: ${run_dir}"
    log "VISION MODEL: ${VLM_AGENT_BACKBONE_JUDGE_MODEL}  (timeout=${VISION_TIMEOUT_S}s)"
    log "DENSE OVERRIDES: ${DENSE_OVERRIDES}"
    log "FIXES (commit 4fbef06): RIGHT/B critical + <actions> reorder"
    log "                       + ram_watch state_flags + no Expert play. fabrication"
    log "                       + GAME_MAX_STEPS 80→120 for altered_beast"
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

log "Stage-2 Altered Beast v6 launcher started (ts=${TS})"
log "Seed bank: ${SEED_BANK_DIR}"
log "Code commit baseline: 4fbef06 (reasoning-collapse + B-mash hotfix)"
log "Expected: 8-12 hr (gymv vision throttle); reward should break "
log "         the 200 ceiling with raw_env_reward > 300 in ≥1 episode."

launch_game gymv_altered_beast Temporal_AlteredBeast-v0

log "All games done."
