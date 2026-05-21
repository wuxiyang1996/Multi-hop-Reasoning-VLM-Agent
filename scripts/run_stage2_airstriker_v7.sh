#!/usr/bin/env bash
# Stage-2 Airstriker v7 — pure raw-env reward, no REASONING in output.
#
# v7 diagnosis (from v6 step 1→3 trajectory analysis):
#   * Vision pipeline (Gemini 2.5 Flash) works perfectly — entity-rich
#     markup at 100% success rate.
#   * BUT the model's reasoning collapsed from real CoT (p50=424 chars)
#     to literal "Expert play." (p50=104 chars) within 3 GRPO steps.
#   * Root cause: action_movement_bonus=0.2 gave free reward for
#     outputting RIGHT, so GRPO learned the shortcut:
#       "Expert play." + ACTION:5(RIGHT) → guaranteed +0.2/step
#     which dominates the GRPO advantage over actual game rewards.
#
# v7 changes vs v6:
#   1. ALL dense reward shaping zeroed out (survival, advance, hit,
#      attack, movement all = 0).  GRPO reward = raw env reward ONLY.
#   2. REASONING removed from output format — model outputs only
#      "ACTION: <number>".  Eliminates the CoT-collapse vector entirely.
#   3. Same Gemini 2.5 Flash vision pipeline (proven in v6 step 0-1).
#
# Expected:
#   * No reward hacking — only way to get reward is to actually score
#   * Simpler output → faster inference, no collapse risk
#   * Learning may be slower (sparse reward) but signal is clean

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
SEED_BANK_DIR="${REPO_ROOT}/frontier_data/output/stage2_seeds_v3_grpo"
SFT_AT_ROOT="${REPO_ROOT}/runs/sft_per_game"
SFT_SS_ROOT="${REPO_ROOT}/runs/sft_per_game_v3"
LORA_LINK_DIR="${REPO_ROOT}/runs/lora_adapters/decision"
TS="$(date +%Y%m%d_%H%M%S)"

# All shaping zeroed — pure raw env reward
DENSE_OVERRIDES='{"gymv_airstriker":{"action_survival_bonus":0.0,"action_advance_bonus":0.0,"action_hit_penalty":0.0,"action_attack_bonus":0.0,"action_movement_bonus":0.0}}'

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

    local run_dir="${REPO_ROOT}/runs/${game_slug}_stage2_v7_${TS}"
    mkdir -p "${run_dir}"
    local launch_log="${run_dir}/launch.log"

    log "============================================================"
    log "GAME: ${game_slug}  (SFT key: ${sft_key})"
    log "RUN DIR: ${run_dir}"
    log "VISION MODEL: ${VLM_AGENT_BACKBONE_JUDGE_MODEL}  (timeout=${VISION_TIMEOUT_S}s)"
    log "DENSE OVERRIDES: ${DENSE_OVERRIDES}"
    log "v7: NO REASONING in output, pure raw env reward"
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

log "Stage-2 Airstriker v7 launcher started (ts=${TS})"
log "Seed bank: ${SEED_BANK_DIR}"
log "v7: pure raw env reward (all shaping=0) + no REASONING output"
log "Vision: ${VLM_AGENT_BACKBONE_JUDGE_MODEL}"

launch_game gymv_airstriker Temporal_Airstriker-v0

log "All games done."
