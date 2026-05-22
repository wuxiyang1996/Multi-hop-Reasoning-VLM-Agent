#!/usr/bin/env bash
# Stage-2 Airstriker v8 — three-pronged anti-collapse fix.
#
# Root cause from v6-v7c iterations:
#   1. GRPO LR too aggressive → destroys LoRA format adherence in 2-4 steps
#   2. Removing REASONING made it worse (LoRA was SFT-trained on that format)
#   3. Dense shaping bonuses (movement, attack) created reward-hack shortcuts
#
# v8 fixes:
#   1. REASONING+ACTION format restored (SFT-native, most stable)
#   2. Reasoning quality gate: reasoning < 50 chars → reward=0
#      (prevents "Expert play." degenerate shortcut from v6)
#   3. GRPO LR halved: 2e-5→1e-5 steady (was 5e-5→2e-5)
#   4. All game shaping zeroed (pure raw env reward)
#   5. Stop tokens for <think>/<thinking> retained
#   6. Symmetric format signal: +0.05 correct / -0.05 failed
#
# Expected:
#   * Stable format across 10+ GRPO steps
#   * No reward hacking (pure env reward)
#   * Slower but more reliable learning (lower LR)

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
SEED_BANK_DIR="${REPO_ROOT}/frontier_data/output/stage2_seeds_v3_grpo"
SFT_AT_ROOT="${REPO_ROOT}/runs/sft_per_game"
SFT_SS_ROOT="${REPO_ROOT}/runs/sft_per_game_v3"
LORA_LINK_DIR="${REPO_ROOT}/runs/lora_adapters/decision"
TS="$(date +%Y%m%d_%H%M%S)"

DENSE_OVERRIDES='{"gymv_airstriker":{"action_survival_bonus":0.0,"action_advance_bonus":0.0,"action_hit_penalty":0.0,"action_attack_bonus":0.0,"action_movement_bonus":0.0}}'

export VLM_AGENT_BACKBONE_JUDGE_MODEL="google/gemini-2.5-flash"
VISION_TIMEOUT_S=15
GRPO_LR="1e-5"

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

    local run_dir="${REPO_ROOT}/runs/${game_slug}_stage2_v8_${TS}"
    mkdir -p "${run_dir}"
    local launch_log="${run_dir}/launch.log"

    log "============================================================"
    log "GAME: ${game_slug}  (SFT key: ${sft_key})"
    log "RUN DIR: ${run_dir}"
    log "VISION: ${VLM_AGENT_BACKBONE_JUDGE_MODEL}  (timeout=${VISION_TIMEOUT_S}s)"
    log "GRPO LR: ${GRPO_LR} (halved from default)"
    log "FORMAT: REASONING+ACTION (SFT-native) + quality gate ≥50 chars"
    log "SHAPING: all zeroed (pure raw env reward)"
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
        --grpo-lr "${GRPO_LR}" \
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

log "Stage-2 Airstriker v8 launcher started (ts=${TS})"
log "Three-pronged fix: REASONING restored + quality gate + LR halved"

launch_game gymv_airstriker Temporal_Airstriker-v0

log "All games done."
