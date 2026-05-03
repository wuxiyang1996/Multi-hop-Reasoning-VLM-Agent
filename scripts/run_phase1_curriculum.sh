#!/usr/bin/env bash
# ======================================================================
#  Phase-1 source-GRPO curriculum — 6 games × 15 steps, sequential, with
#  bank + LoRA carry-over between games.
#
#  Implements the plan locked in
#    training_notes/coevo-3phase-cross-game-ood-transfer-plan.md §4.1
#  (game roster + curriculum order, 2026-05-03 PM).
#
#  Curriculum order (high-density rewards first, then puzzle/action,
#  finishing on the two paper Table-3 anchors):
#    Phase 1: gymv_space_harrier_ii   (shmup,       baseline 100%)
#    Phase 2: gymv_streets_of_rage_2  (beat-em-up,  baseline 100%)
#    Phase 3: gymv_columns            (puzzle,      baseline  89%)
#    Phase 4: gymv_strider            (action,      baseline  78%)
#    Phase 5: candy_crush             (paper Table 3, match-3)
#    Phase 6: tetris                  (paper Table 3, spatial puzzle)
#
#  Total: 90 GRPO steps. Wall-clock ~54 h sequential at ~36 min/step.
#
#  Bank + LoRA carry over phase-to-phase by design (Option C in §11.2).
#  After each phase a snapshot lands at
#    <run_dir>/phase_snapshots/phase_<N>_<game>/{lora_adapters,
#    skillbank, checkpoint, phase_meta.json}
#  so any per-game-best evaluation can re-load that snapshot independently
#  of the post-curriculum state.
#
#  Prerequisites:
#    conda activate game-ai-agent
#    pip install wandb tensorboard peft         # one-time
#    bash install/install_gymv.sh \
#         && bash install/gymv_temporal_patch/apply_patch.sh   # one-time
#                                                              (gym_v + Mega Drive ROMs)
#    Phase-1-source SFT cold-start adapters present at
#      runs/sft_coldstart/{decision,skillbank}/{skill_selection,
#      action_taking,segment,contract,curator}
#
#  Usage:
#    bash scripts/run_phase1_curriculum.sh
#
#    # Override per-phase iteration count (default 15 per §4.1):
#    ITERS_PER_PHASE=10 bash scripts/run_phase1_curriculum.sh
#
#    # Override episodes per step (default 8 — matches paper Table 3):
#    EPISODES=4 bash scripts/run_phase1_curriculum.sh
#
#    # Resume from a specific phase (e.g. phase 4 = Strider):
#    RESUME_PHASE=4 RUN_DIR=runs/Qwen3.5-9B_<ts>_phase1 \
#      bash scripts/run_phase1_curriculum.sh
#
#    # Use external vLLM instead of MANAGED dual-stack:
#    MANAGE_VLLM=0 VLLM_URL=http://localhost:8000/v1 \
#      bash scripts/run_phase1_curriculum.sh
#
#  Cross-refs:
#    - scripts/run_all.sh                    (5-phase Stage-0 script this is patterned on)
#    - env_wrappers/gymv_temporal_nl_wrapper.py  (the 4 gymv_* slugs)
#    - trainer/coevolution/episode_runner.py     (GYMV_TEMPORAL_GAMES_SET dispatch)
#    - baselines/README.md § "Gym-V benchmark scope" (per-game baseline anchors for §4.3 sanity bar)
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# ── HuggingFace cache ────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

# ── PYTHONPATH (mirrors run_all.sh) ──────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── Configurable parameters ──────────────────────────────────────────
MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-9B}"
PORT="${VLLM_PORT:-8000}"
TP="${VLLM_TP:-4}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
MANAGE_VLLM="${MANAGE_VLLM:-1}"

# §4.1 lock: 15 steps per phase, 8 episodes per step, snapshot every 5
# steps so a mid-phase failure doesn't lose more than ~3 h of work.
ITERS_PER_PHASE="${ITERS_PER_PHASE:-15}"
EPISODES="${EPISODES:-8}"
CKPT_INTERVAL="${CKPT_INTERVAL:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution-phase1}"
RUN_DIR="${RUN_DIR:-}"
DEBUG="${DEBUG:-}"
RESUME_PHASE="${RESUME_PHASE:-1}"
VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"

# Cold-start adapter paths (SFT-pretrained); same layout as run_all.sh
COLDSTART_DIR="${COLDSTART_DIR:-runs/sft_coldstart}"
COLDSTART_DECISION="${COLDSTART_DIR}/decision"
COLDSTART_SKILLBANK="${COLDSTART_DIR}/skillbank"

# ── Locked Phase-1 curriculum (training_notes/coevo-3phase-cross-game-ood-transfer-plan.md §4.1)
# Format: "phase_number:game_slug:display_name"
# Slugs match env_wrappers/gymv_temporal_nl_wrapper.GYMV_TEMPORAL_GAMES
# for the 4 gymv games and env_wrappers/game_configs.py for the 2 paper games.
PHASES=(
    "1:gymv_space_harrier_ii:Space Harrier II"
    "2:gymv_streets_of_rage_2:Streets of Rage 2"
    "3:gymv_columns:Columns"
    "4:gymv_strider:Strider"
    "5:candy_crush:Candy Crush"
    "6:tetris:Tetris"
)
NUM_PHASES=${#PHASES[@]}

# Per-game baseline anchor for §4.3 sanity bar — recorded here so the
# operator can spot-check end-of-phase reward against the 4-backbone
# pooled per-episode-success rate from baselines/README.md § "Gym-V
# benchmark scope". n/a for the 2 paper games (anchor is paper Figure 4).
declare -A BASELINE_ANCHOR=(
    ["gymv_space_harrier_ii"]="100% (4/4 backbones)"
    ["gymv_streets_of_rage_2"]="100% (4/4 backbones)"
    ["gymv_columns"]="89% (Claude/Q9 100%)"
    ["gymv_strider"]="78% (Q9 100%)"
    ["candy_crush"]="paper Figure 4 (±30%)"
    ["tetris"]="paper Figure 4 (±30%)"
)

# ── Cleanup on exit (only relevant in legacy mode) ───────────────────
VLLM_PID=""
cleanup() {
    echo ""
    echo "[run_phase1] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[run_phase1] Stopping vLLM server (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[run_phase1] Done."
}
trap cleanup EXIT INT TERM

# ======================================================================
# Phase 0: Resolve run directory + ensure LoRA adapters
# ======================================================================
echo "══════════════════════════════════════════════════════════════"
echo "  Phase-1 Source Co-Evolution Curriculum"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:            ${MODEL}"
echo "  Phases:           ${NUM_PHASES} sequential games × ${ITERS_PER_PHASE} GRPO steps"
echo "  Total iterations: $((NUM_PHASES * ITERS_PER_PHASE))"
echo "  Episodes/step:    ${EPISODES}"
echo "  Snapshot every:   ${CKPT_INTERVAL} steps within each phase"
echo "  Debug I/O:        ${DEBUG:-disabled}"
echo "  Resume phase:     ${RESUME_PHASE}"
echo "  Cold-start:       ${COLDSTART_DIR}"
if [ "${MANAGE_VLLM}" = "1" ]; then
    echo "  GPU mode:         MANAGED (persistent vLLM + FSDP)"
    echo "  vLLM GPUs:        ${VLLM_GPUS}"
    echo "  GRPO GPUs:        ${GRPO_GPUS}"
    echo "  Spec decode:      ${SPEC_MODEL} (${SPEC_TOKENS} tokens)"
else
    echo "  GPU mode:         LEGACY (external vLLM at port ${PORT})"
fi
echo ""
echo "  Curriculum schedule:"
for phase_def in "${PHASES[@]}"; do
    IFS=':' read -r pnum game display <<< "${phase_def}"
    step_start=$(( (pnum - 1) * ITERS_PER_PHASE ))
    step_end=$(( pnum * ITERS_PER_PHASE - 1 ))
    anchor="${BASELINE_ANCHOR[${game}]:-no anchor}"
    echo "    Phase ${pnum}: ${display} (steps ${step_start}–${step_end}; baseline ${anchor})"
done
echo "══════════════════════════════════════════════════════════════"

echo ""
echo "[run_phase1] Ensuring LoRA adapters exist (from SFT cold-start)..."

if [ ! -d "${COLDSTART_DECISION}" ]; then
    echo "[run_phase1] ERROR: Cold-start decision adapters not found: ${COLDSTART_DECISION}"
    echo "[run_phase1]   Run scripts/run_sft_coldstart.sh first, or set COLDSTART_DIR"
    exit 1
fi
if [ ! -d "${COLDSTART_SKILLBANK}" ]; then
    echo "[run_phase1] ERROR: Cold-start skillbank adapters not found: ${COLDSTART_SKILLBANK}"
    echo "[run_phase1]   Run scripts/run_sft_coldstart.sh first, or set COLDSTART_DIR"
    exit 1
fi

# Pre-flight: confirm the 4 gymv slugs are wired in episode_runner.
# A missing slug would surface at runtime as a confusing
# "GAME_CONFIGS not found" error far inside the rollout loop, so
# fail loudly here.
python -c "
from trainer.coevolution.episode_runner import _lazy_imports, GYMV_TEMPORAL_GAMES_SET
_lazy_imports()
required = {'gymv_space_harrier_ii', 'gymv_streets_of_rage_2',
            'gymv_columns', 'gymv_strider'}
missing = required - GYMV_TEMPORAL_GAMES_SET
if missing:
    raise SystemExit(
        f'[run_phase1] FATAL: gymv slugs not wired in episode_runner: {sorted(missing)}. '
        f'Run install/install_gymv.sh + install/gymv_temporal_patch/apply_patch.sh.'
    )
print(f'[run_phase1] Gym-V slugs wired: {sorted(GYMV_TEMPORAL_GAMES_SET & required)}')
"

RESOLVED_RUN_DIR=$(python -c "
import os, sys
os.environ.setdefault('PYGLET_HEADLESS', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
from pathlib import Path

from trainer.coevolution.config import CoEvolutionConfig, prepare_adapters

decision_dir = '${COLDSTART_DECISION}'
skillbank_dir = '${COLDSTART_SKILLBANK}'

pretrained = {}
for name in ['skill_selection', 'action_taking']:
    p = Path(decision_dir) / name
    if p.exists():
        pretrained[name] = str(p)
for name in ['segment', 'contract', 'curator']:
    p = Path(skillbank_dir) / name
    if p.exists():
        pretrained[name] = str(p)

cfg = CoEvolutionConfig(
    model_name='${MODEL}',
    pretrained_adapter_paths=pretrained,
)
run_dir_override = '${RUN_DIR}'
if run_dir_override:
    cfg.run_dir = run_dir_override
cfg.resolve_paths()

result = prepare_adapters(cfg)
loaded = [n for n in result if n in pretrained]
inited = [n for n in result if n not in pretrained]
if loaded:
    print(f'Loaded {len(loaded)} cold-start adapter(s): {loaded}', file=sys.stderr)
if inited:
    print(f'Random-init {len(inited)} adapter(s): {inited}', file=sys.stderr)
print(cfg.run_dir)
")

RUN_DIR="${RESOLVED_RUN_DIR}"
export ADAPTER_DIR="${RUN_DIR}/lora_adapters"
SNAPSHOT_DIR="${RUN_DIR}/phase_snapshots"
mkdir -p "${SNAPSHOT_DIR}"

echo "[run_phase1] Run dir:      ${RUN_DIR}"
echo "[run_phase1] Adapter dir:  ${ADAPTER_DIR}"
echo "[run_phase1] Snapshot dir: ${SNAPSHOT_DIR}"

# ======================================================================
# Helper: save a phase snapshot (LoRA + bank + checkpoint + metadata)
# ======================================================================
save_phase_snapshot() {
    local phase_num="$1"
    local game="$2"
    local display="$3"
    local step_end="$4"

    local snap_name
    snap_name=$(printf "phase_%02d_%s" "${phase_num}" "${game}")
    local snap_path="${SNAPSHOT_DIR}/${snap_name}"

    echo ""
    echo "[run_phase1] ── Saving phase snapshot: ${snap_name} ──"
    mkdir -p "${snap_path}"

    if [ -d "${RUN_DIR}/lora_adapters" ]; then
        echo "[run_phase1]   Copying LoRA adapters..."
        cp -r "${RUN_DIR}/lora_adapters" "${snap_path}/lora_adapters"
    fi

    if [ -d "${RUN_DIR}/skillbank" ]; then
        echo "[run_phase1]   Copying skill bank (rolling, shared across phases)..."
        cp -r "${RUN_DIR}/skillbank" "${snap_path}/skillbank"
    fi

    local latest_ckpt=""
    if [ -d "${RUN_DIR}/checkpoints" ]; then
        latest_ckpt=$(ls -d "${RUN_DIR}/checkpoints"/step_* 2>/dev/null | sort -V | tail -1 || true)
        if [ -n "${latest_ckpt}" ]; then
            echo "[run_phase1]   Copying latest checkpoint: $(basename "${latest_ckpt}")"
            cp -r "${latest_ckpt}" "${snap_path}/checkpoint"
        fi
    fi

    cat > "${snap_path}/phase_meta.json" <<METAEOF
{
    "phase": ${phase_num},
    "game": "${game}",
    "display_name": "${display}",
    "step_end": ${step_end},
    "iters_per_phase": ${ITERS_PER_PHASE},
    "episodes_per_step": ${EPISODES},
    "model": "${MODEL}",
    "timestamp": "$(date -Iseconds)",
    "run_dir": "${RUN_DIR}",
    "latest_checkpoint": "${latest_ckpt:-none}",
    "baseline_anchor": "${BASELINE_ANCHOR[${game}]:-n/a}"
}
METAEOF

    local skill_summary=""
    if [ -d "${snap_path}/skillbank" ]; then
        for bank_file in "${snap_path}"/skillbank/*/skill_bank.jsonl; do
            if [ -f "${bank_file}" ]; then
                local gname
                gname=$(basename "$(dirname "${bank_file}")")
                local count
                count=$(wc -l < "${bank_file}" 2>/dev/null || echo 0)
                skill_summary="${skill_summary}  ${gname}=${count}"
            fi
        done
    fi

    echo "[run_phase1]   Phase ${phase_num} snapshot saved to: ${snap_path}"
    if [ -n "${skill_summary}" ]; then
        echo "[run_phase1]   Skill bank sizes:${skill_summary}"
    fi
    echo "[run_phase1] ── Snapshot complete ──"
    echo ""
}

# ======================================================================
# Build common training args
# ======================================================================
build_train_args() {
    local game="$1"
    local total_steps="$2"
    local is_first_phase="$3"

    local args=(
        --games "${game}"
        --total-steps "${total_steps}"
        --curriculum "none"
        --episodes-per-game "${EPISODES}"
        --checkpoint-interval "${CKPT_INTERVAL}"
        --model "${MODEL}"
        --wandb-project "${WANDB_PROJECT}"
        --run-dir "${RUN_DIR}"
    )

    if [ "${is_first_phase}" = "true" ]; then
        # First phase only: load the cold-start SFT adapters.
        # Subsequent phases inherit the rolling LoRA from --run-dir
        # (Option C carry-over).
        if [ -d "${COLDSTART_DECISION}" ]; then
            args+=(--load-decision-adapters "${COLDSTART_DECISION}")
        fi
        if [ -d "${COLDSTART_SKILLBANK}" ]; then
            args+=(--load-skillbank-adapters "${COLDSTART_SKILLBANK}")
        fi
    else
        args+=(--resume)
    fi

    if [ -n "${DEBUG}" ]; then
        args+=(--debug-io)
    fi

    echo "${args[@]}"
}

# ======================================================================
# Curriculum training loop
# ======================================================================
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Starting Phase-1 curriculum (${NUM_PHASES} phases, sequential)"
echo "══════════════════════════════════════════════════════════════"

PHASE_FAILED=""

for phase_def in "${PHASES[@]}"; do
    IFS=':' read -r phase_num game display <<< "${phase_def}"

    if [ "${phase_num}" -lt "${RESUME_PHASE}" ]; then
        echo ""
        echo "[run_phase1] Skipping phase ${phase_num} (${display}) — resuming from phase ${RESUME_PHASE}"
        continue
    fi

    step_start=$(( (phase_num - 1) * ITERS_PER_PHASE ))
    step_end=$(( phase_num * ITERS_PER_PHASE ))
    is_first=$([ "${phase_num}" -eq "${RESUME_PHASE}" ] && [ "${RESUME_PHASE}" -eq 1 ] && echo "true" || echo "false")

    echo ""
    echo "┌──────────────────────────────────────────────────────────┐"
    echo "│  Phase ${phase_num}/${NUM_PHASES}: ${display}"
    echo "│  Slug:     ${game}"
    echo "│  Steps:    ${step_start} → ${step_end} (${ITERS_PER_PHASE} iterations)"
    echo "│  Episodes: ${EPISODES} rollouts per step"
    echo "│  Mode:     $([ "${is_first}" = "true" ] && echo "COLD-START" || echo "RESUME (carry-over LoRA + bank)")"
    echo "│  Anchor:   ${BASELINE_ANCHOR[${game}]:-n/a}"
    echo "└──────────────────────────────────────────────────────────┘"

    PHASE_ARGS=()
    read -ra PHASE_ARGS <<< "$(build_train_args "${game}" "${step_end}" "${is_first}")"

    if [ "${MANAGE_VLLM}" = "1" ]; then
        # shellcheck disable=SC2086
        PHASE_ARGS+=(--vllm-gpus ${VLLM_GPUS})
        PHASE_ARGS+=(--grpo-devices ${GRPO_GPUS})
        PHASE_ARGS+=(--vllm-base-port "${PORT}")
        PHASE_ARGS+=(--vllm-gpu-util "${GPU_UTIL}")
        PHASE_ARGS+=(--speculative-model "${SPEC_MODEL}")
        PHASE_ARGS+=(--num-speculative-tokens "${SPEC_TOKENS}")
    else
        PHASE_ARGS+=(--no-manage-vllm)
        PHASE_ARGS+=(--vllm-url "http://localhost:${PORT}/v1")
    fi

    echo "[run_phase1] Training args: ${PHASE_ARGS[*]}"
    echo ""

    if python scripts/run_coevolution.py "${PHASE_ARGS[@]}"; then
        echo ""
        echo "[run_phase1] Phase ${phase_num} (${display}) completed successfully."
        save_phase_snapshot "${phase_num}" "${game}" "${display}" "${step_end}"
    else
        PHASE_FAILED="${phase_num}"
        echo ""
        echo "[run_phase1] ERROR: Phase ${phase_num} (${display}) FAILED."
        echo "[run_phase1] Saving partial snapshot before aborting..."
        save_phase_snapshot "${phase_num}" "${game}" "${display}_FAILED" "${step_end}"
        break
    fi
done

# ======================================================================
# Summary
# ======================================================================
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ -n "${PHASE_FAILED}" ]; then
    echo "  Phase-1 curriculum STOPPED at phase ${PHASE_FAILED}"
    echo "  Resume with: RESUME_PHASE=${PHASE_FAILED} RUN_DIR=${RUN_DIR} bash scripts/run_phase1_curriculum.sh"
else
    echo "  Phase-1 curriculum COMPLETE"
    echo "  All ${NUM_PHASES} phases finished successfully."
    echo ""
    echo "  Next: bash scripts/run_phase2_holdout.sh \\"
    echo "          PHASE1_SNAPSHOT=${SNAPSHOT_DIR}/phase_06_tetris"
fi
echo ""
echo "  Run dir:    ${RUN_DIR}"
echo "  Snapshots:  ${SNAPSHOT_DIR}/"
if [ -d "${SNAPSHOT_DIR}" ]; then
    echo ""
    echo "  Phase snapshots (sorted by phase order):"
    for d in "${SNAPSHOT_DIR}"/phase_*; do
        if [ -d "$d" ]; then
            local_name=$(basename "$d")
            skill_count=0
            if [ -d "$d/skillbank" ]; then
                skill_count=$(find "$d/skillbank" -name "skill_bank.jsonl" -exec cat {} + 2>/dev/null | wc -l || echo 0)
            fi
            echo "    ${local_name}/ (${skill_count} skills)"
        fi
    done
fi
echo "══════════════════════════════════════════════════════════════"

if [ -n "${PHASE_FAILED}" ]; then
    exit 1
fi

echo ""
echo "[run_phase1] Phase-1 curriculum complete. Post-curriculum state:"
echo "    LoRA:  ${SNAPSHOT_DIR}/phase_06_tetris/lora_adapters/"
echo "    Bank:  ${SNAPSHOT_DIR}/phase_06_tetris/skillbank/"
