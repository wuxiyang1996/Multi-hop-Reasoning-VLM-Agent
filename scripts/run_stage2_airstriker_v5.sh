#!/usr/bin/env bash
# Stage-2 Airstriker v5 — v4 + visual-grounding round-2 improvements.
#
# What changed vs v4 (id stage2_v4_20260521_042144, killed at step 0):
#
#   * T2.17b smart-fallback: on any 35B-vision failure path
#     (timeout / parse_failure / request_error / build_failure),
#     reuse the previous step's last-good ``<state>`` markup with
#     the ``step=<N>`` field rewritten to the current step.  Was:
#     return the HUD-blind deterministic ``state_to_markup``, OOD
#     for the action_taking LoRA.
#
#   * T2.17b temperature 0.1 → 0.0 for the vision-perception 35B
#     call.  This is a *parser*-style task with a single
#     structurally-correct answer per frame — greedy decoding cuts
#     parse_failure rate and very-slightly trims tail latency.
#
#   * T2.17b AS-specific few-shot examples.  Pre-v5 the few-shot
#     loader fell through to the single Strider example in
#     ``gymv.txt`` for every gymv game (file naming used the
#     normalized task slug but every cold-start example file was
#     domain-wide).  Now: 2× real Airstriker cold-start examples at
#     ``vlm_wrapper/few_shot_examples/gymv.gymv_airstriker{,.2}.txt``
#     with 23-25 entities each including ``player ship`` + ``enemy
#     bullet`` * 12 — the exact distribution the LoRA was SFT'd on.
#     Anchors the 35B's labels on the Airstriker vocabulary instead
#     of the cross-domain Strider one.
#
# Everything else unchanged from v4:
#   * action_hit_penalty = 1.0  (down from v3's 5.0 — light
#     risk-aversion, lets the agent take strategic deaths like
#     teachers do)
#   * action_attack_bonus = 0.05  (× B-press)
#   * action_movement_bonus = 0.20  (× LEFT/RIGHT)
#   * action_survival_bonus = 0.05  (T2.19c fix)
#   * vision_state_perception_timeout_s = 45  (T2.17 fix; was 6s)
#
# Same launch wiring:
#   GPUs 0-3 : 4× Qwen3.5-9B  (managed by run_coevolution.py)
#   GPUs 4-5 : Qwen3.5-35B-A3B vision judge on port 8001 (KEEP, shared)
#   GPUs 6-7 : GRPO devices

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
SEED_BANK_DIR="${REPO_ROOT}/frontier_data/output/stage2_seeds_v3_grpo"
SFT_AT_ROOT="${REPO_ROOT}/runs/sft_per_game"
SFT_SS_ROOT="${REPO_ROOT}/runs/sft_per_game_v3"
LORA_LINK_DIR="${REPO_ROOT}/runs/lora_adapters/decision"
TS="$(date +%Y%m%d_%H%M%S)"

DENSE_OVERRIDES='{"gymv_airstriker":{"action_survival_bonus":0.05,"action_advance_bonus":0.0,"action_hit_penalty":1.0,"action_attack_bonus":0.05,"action_movement_bonus":0.2}}'
VISION_TIMEOUT_S=45

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

kill_stale_9b_vllm() {
    local pids
    pids=$(pgrep -f "vllm.entrypoints.*Qwen3\.5-9B" || true)
    if [[ -n "${pids}" ]]; then
        log "Killing stale 9B vllm PIDs: ${pids//$'\n'/ }"
        for p in ${pids}; do kill ${p} 2>/dev/null || true; done
        sleep 4
        # Kill any orphan EngineCore workers (vllm child subprocesses
        # sometimes outlive the parent api_server).
        local engcore
        engcore=$(pgrep -f "VLLM::EngineCore" || true)
        if [[ -n "${engcore}" ]]; then
            log "Killing orphan EngineCore workers: ${engcore//$'\n'/ }"
            for p in ${engcore}; do kill -9 ${p} 2>/dev/null || true; done
            sleep 3
        fi
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

    local run_dir="${REPO_ROOT}/runs/${game_slug}_stage2_v5_${TS}"
    mkdir -p "${run_dir}"
    local launch_log="${run_dir}/launch.log"

    log "============================================================"
    log "GAME: ${game_slug}  (SFT key: ${sft_key})"
    log "RUN DIR: ${run_dir}"
    log "DENSE-REWARD OVERRIDES: ${DENSE_OVERRIDES}"
    log "VISION: timeout=${VISION_TIMEOUT_S}s, smart-fallback, greedy, AS few-shot"
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

log "Stage-2 Airstriker v5 launcher started (ts=${TS})"
log "Seed bank: ${SEED_BANK_DIR}"
log "Fixes vs v4: smart-fallback + greedy (temp=0) + AS-specific few-shot"

launch_game gymv_airstriker Temporal_Airstriker-v0

log "All games done."
