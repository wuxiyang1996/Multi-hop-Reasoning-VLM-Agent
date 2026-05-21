#!/usr/bin/env bash
# Stage-2 Airstriker v6 — pivot vision-perception to OpenRouter
# (Gemini 2.5 Flash) instead of the local 35B.
#
# Why v6 (vs v5): even after T2.17 / T2.17b fixed the silent-fallback
# bug (timeout 6→45s + smart fallback + few-shot fix + greedy), the
# local 35B vision call was 22-45s/frame (highly variable under
# concurrent load) and capped step throughput at ~60 min/outer-step
# → ~10 hr for a 10-step AS run.  v5 was actually working but too slow
# to iterate on.
#
# Switching to OpenRouter Gemini 2.5 Flash (T2.17c, 2026-05-21):
#   * Latency 22-45s → 4-5s (6-10× faster)
#   * Success rate 70% → 100% (no parse failures or timeouts in
#     production-style probe of 8 concurrent calls)
#   * Quality on par or better — Gemini emits 1500-1800 chars w/ 8+
#     entities incl. player ship vs 35B's 1300 chars w/ 7 entities
#   * Cost ~$3-5 for the full 10-step run (Gemini 2.5 Flash:
#     $0.075/M input + $0.30/M output × ~9600 calls × ~5K tokens each
#     including images = ~$0.0005/call, with ~30% intra-ep cache hits
#     bringing effective spend down)
#   * Frees GPUs 4-5 (where 35B was) — left idle for now, could host
#     2× more 9B replicas in a future revision to also speed up the
#     agent-side rollout.
#
# Code path:
#   * VLM_AGENT_BACKBONE_JUDGE_MODEL env var overrides
#     common.models.BACKBONE_JUDGE_MODEL → resolves to
#     "google/gemini-2.5-flash".
#   * _vision_state_perception._ask_judge_blocking's URL loop:
#     - _candidate_vllm_urls(model) returns [] for external prefixes
#       (T2.17c API_func patch) → no wasted local probe.
#     - Appends OpenRouter URL → call routed to Gemini.
#     - OpenRouter API key picked up from API_func.open_router_api_key
#       (already configured).
#   * Inheriting T2.17 / T2.17b: timeout=15s (down from 45s — Gemini
#     is fast enough), smart fallback to _last_vision_markup on
#     failure, 2× AS-specific few-shot examples, greedy decoding.
#
# Everything else unchanged from v5:
#   * action_hit_penalty = 1.0  +  action_attack_bonus = 0.05  +
#     action_movement_bonus = 0.20  +  action_survival_bonus = 0.05
#     +  action_advance_bonus = 0.0
#
# Launch wiring:
#   GPUs 0-3 : 4× Qwen3.5-9B  (managed by run_coevolution.py)
#   GPUs 4-5 : IDLE (or KEEP 35B as last-resort fallback — see KEEP_35B)
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

# Vision-perception via Gemini 2.5 Flash.  Override defaults:
#   * timeout 45→15s (Gemini p50 ~4s, p95 <10s — 15s is plenty
#     and surfaces real failures fast)
#   * Model selected via env var; URL routing handled by
#     _candidate_vllm_urls returning [] for external prefixes.
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
        # NOTE: do NOT kill EngineCore workers whose parent is the
        # 35B server (PPID matches the 35B api_server PID).  In v6
        # we keep the 35B running as a *last-resort* fallback should
        # OpenRouter rate-limit or go down for the entire run.
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

log "Stage-2 Airstriker v6 launcher started (ts=${TS})"
log "Seed bank: ${SEED_BANK_DIR}"
log "Fixes vs v5: vision pivoted to ${VLM_AGENT_BACKBONE_JUDGE_MODEL}"
log "Expected: 6-10× faster step iteration, full run ~2-3 hr, cost ~\$3-5"

launch_game gymv_airstriker Temporal_Airstriker-v0

log "All games done."
