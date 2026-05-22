#!/usr/bin/env bash
# Stage-2 Airstriker v9 — Gemini+Claude SFT + full 80-step episodes.
#
# v9 changes vs v8:
#   1. SFT retrained on Gemini+Claude-only data (mean teacher raw=102)
#      instead of GPT-5.4-only (mean=49). Expect step-0 baseline ≈70-80.
#   2. --min-steps-before-stuck 75 (was 60) → episodes reach full 80 steps.
#      Teacher episodes were 80 steps; v8 terminated at 60 due to stuck detection.
#   3. Keeps all v8 anti-collapse fixes:
#      - REASONING+ACTION format (SFT-native)
#      - Reasoning quality gate (< 50 chars → reward=0)
#      - GRPO LR 1e-5 (halved)
#      - Symmetric format signal +0.05/-0.05
#      - All dense shaping zeroed (pure raw env reward)
#      - Stop tokens for <think>/<thinking>

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
SEED_BANK_DIR="${REPO_ROOT}/frontier_data/output/stage2_seeds_v3_grpo"
SFT_SS_ROOT="${REPO_ROOT}/runs/sft_per_game_v3"
LORA_LINK_DIR="${REPO_ROOT}/runs/lora_adapters/decision"
TS="$(date +%Y%m%d_%H%M%S)"

# v9: new SFT adapter from Gemini+Claude only
SFT_V9_AT="/workspace/Multi-hop-Reasoning-VLM-Agent/runs/sft_v9_gemini_claude/decision/action_taking/hf_trainer/checkpoint-200/action_taking"

DENSE_OVERRIDES='{"gymv_airstriker":{"action_survival_bonus":0.0,"action_advance_bonus":0.0,"action_hit_penalty":0.0,"action_attack_bonus":0.0,"action_movement_bonus":0.0}}'

export VLM_AGENT_BACKBONE_JUDGE_MODEL="google/gemini-2.5-flash"
VISION_TIMEOUT_S=15
GRPO_LR="1e-5"
MIN_STEPS_BEFORE_STUCK=75

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
}

repoint_decision_lora_v9() {
    local sft_key="$1"
    local ss_src="${SFT_SS_ROOT}/${sft_key}/skill_selection/${sft_key}__skill_selection"

    if [[ ! -d "${SFT_V9_AT}" ]]; then
        log "FATAL: v9 action_taking adapter missing at ${SFT_V9_AT}"
        return 1
    fi
    if [[ ! -f "${ss_src}/adapter_model.safetensors" ]]; then
        log "FATAL: skill_selection adapter missing at ${ss_src}"
        return 1
    fi

    mkdir -p "${LORA_LINK_DIR}/action_taking" "${LORA_LINK_DIR}/skill_selection"

    # action_taking → v9 Gemini+Claude SFT
    for f in adapter_model.safetensors adapter_config.json; do
        ln -sfT "${SFT_V9_AT}/${f}" "${LORA_LINK_DIR}/action_taking/${f}"
    done
    # skill_selection → keep v3 (unchanged)
    for f in adapter_model.safetensors adapter_config.json; do
        ln -sfT "${ss_src}/${f}" "${LORA_LINK_DIR}/skill_selection/${f}"
    done
    log "Repointed action_taking → v9 Gemini+Claude SFT"
    log "Repointed skill_selection → ${sft_key}"
}

launch_game() {
    local game_slug="$1"
    local sft_key="$2"

    local run_dir="${REPO_ROOT}/runs/${game_slug}_stage2_v9_${TS}"
    mkdir -p "${run_dir}"
    local launch_log="${run_dir}/launch.log"

    log "============================================================"
    log "GAME: ${game_slug}  (SFT: v9 Gemini+Claude)"
    log "RUN DIR: ${run_dir}"
    log "VISION: ${VLM_AGENT_BACKBONE_JUDGE_MODEL}  (timeout=${VISION_TIMEOUT_S}s)"
    log "GRPO LR: ${GRPO_LR}"
    log "MIN_STEPS_BEFORE_STUCK: ${MIN_STEPS_BEFORE_STUCK} (was 60)"
    log "FORMAT: REASONING+ACTION + quality gate ≥50 chars"
    log "SHAPING: all zeroed (pure raw env reward)"
    log "============================================================"

    kill_stale_9b_vllm
    repoint_decision_lora_v9 "${sft_key}"

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
        --grpo-lr "${GRPO_LR}" \
        --max-concurrent 64 \
        --no-wandb \
        --from-scratch \
        --dense-reward-overrides "${DENSE_OVERRIDES}" \
        --vision-state-perception-timeout-s "${VISION_TIMEOUT_S}" \
        --min-steps-before-stuck "${MIN_STEPS_BEFORE_STUCK}" \
        2>&1 | tee "${launch_log}"

    local rc=${PIPESTATUS[0]}
    log "Game ${game_slug} finished (rc=${rc})"

    kill_stale_9b_vllm
    return ${rc}
}

log "Stage-2 Airstriker v9 launcher started (ts=${TS})"
log "Key changes: Gemini+Claude SFT (teacher mean=102) + 80-step episodes"

launch_game gymv_airstriker Temporal_Airstriker-v0

log "All games done."
