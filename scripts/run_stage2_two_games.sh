#!/usr/bin/env bash
# Stage-2 sequential co-evolution: SpaceHarrierII → DynamiteHeaddy
#
# Layout:
#   GPUs 0-3 : 4× Qwen3.5-9B  (managed by run_coevolution.py per game)
#   GPUs 4-5 : Qwen3.5-35B-A3B vision judge on port 8001 (KEEP, shared)
#   GPUs 6-7 : GRPO devices
#
# Per game:
#   - kill stale 9B vllm servers (spare the 35B on port 8001)
#   - repoint runs/lora_adapters/decision/{action_taking,skill_selection}
#     to the per-game SFT LoRA (Temporal_<Game>-v0)
#   - launch run_coevolution.py with seed_bank_dir = stage2_seeds_v3_grpo
#   - wait for completion (--block until log shows "Run summary" or process exits)

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
SEED_BANK_DIR="${REPO_ROOT}/frontier_data/output/stage2_seeds_v3_grpo"
SFT_AT_ROOT="${REPO_ROOT}/runs/sft_per_game"          # action_taking
SFT_SS_ROOT="${REPO_ROOT}/runs/sft_per_game_v3"       # skill_selection (v3 balanced)
LORA_LINK_DIR="${REPO_ROOT}/runs/lora_adapters/decision"
TS="$(date +%Y%m%d_%H%M%S)"

# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

kill_stale_9b_vllm() {
    # vllm 9B servers spawned previously (e.g. by a prior SoR2 run) hold
    # GPUs 0-3 even though the orchestrator already exited. Kill them
    # but spare the Qwen3.5-35B-A3B vision judge (port 8001, TP=2,
    # GPUs 4-5) which the new run reuses for vision_state_perception.
    local pids
    pids=$(pgrep -f "vllm.entrypoints.*Qwen3\.5-9B" || true)
    if [[ -n "${pids}" ]]; then
        log "Killing stale 9B vllm PIDs: ${pids//$'\n'/ }"
        kill ${pids} 2>/dev/null || true
        sleep 4
        # SIGKILL holdouts
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

    # Sanity: confirm 35B vision judge still up on :8001.
    if ! curl -sf -m 3 http://localhost:8001/v1/models >/dev/null 2>&1; then
        log "WARN: 35B vision judge on :8001 not reachable. Vision-state-perception may fail until it comes back."
    else
        log "35B vision judge :8001 still healthy."
    fi
}

repoint_decision_lora() {
    # Symlink the per-game SFT LoRA into the global decision/ dir that
    # the orchestrator's vllm --lora-modules flag will pick up.
    local key="$1"  # e.g. Temporal_SpaceHarrierII-v0
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
    log "  action_taking → ${at_src}"
    log "  skill_selection → ${ss_src}"
}

launch_game() {
    local game_slug="$1"      # e.g. gymv_space_harrier_ii
    local sft_key="$2"        # e.g. Temporal_SpaceHarrierII-v0
    local run_tag="$3"        # e.g. spaceharrier2_stage2

    local run_dir="${REPO_ROOT}/runs/${game_slug}_stage2_${TS}"
    mkdir -p "${run_dir}"
    local launch_log="${run_dir}/launch.log"

    log "============================================================"
    log "GAME: ${game_slug}  (SFT key: ${sft_key})"
    log "RUN DIR: ${run_dir}"
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
        2>&1 | tee "${launch_log}"

    local rc=${PIPESTATUS[0]}
    log "Game ${game_slug} finished (rc=${rc})"

    # cleanup: kill the 9B vllms this run spawned so next game starts clean
    kill_stale_9b_vllm
    return ${rc}
}

# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------

log "Stage-2 two-game sequential launcher started (ts=${TS})"
log "Seed bank: ${SEED_BANK_DIR}"

# Game 1 — SpaceHarrierII (shmup, source TF3 same-genre, top-1 affinity 0.82)
launch_game gymv_space_harrier_ii Temporal_SpaceHarrierII-v0 spaceharrier2

# Game 2 — DynamiteHeaddy (platformer, source Strider __EXPLORE__, top-1 affinity 0.88)
launch_game gymv_dynamite_headdy Temporal_DynamiteHeaddy-v0 dynamiteheaddy

log "All games done."
