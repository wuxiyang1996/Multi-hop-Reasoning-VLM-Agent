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
#  ── 8×H200 dual-stack layout (default, 2026-05-03) ──────────────────
#  Hosts the 9B actor AND the 35B-A3B judge simultaneously so the live
#  promotion gates (E0/E1/E2) and crafter / harness control plane hit
#  the 35B endpoint instead of silently falling back to 9B-self-judging.
#
#    GPUs 0–3 → 4× Qwen3.5-9B vLLM (TP=1 each, ports 8000–8003)
#    GPUs 4–5 → 1× Qwen3.5-35B-A3B vLLM (TP=2 + expert-parallel,
#               port 8004) — judge / crafter / harness / orchestrator.
#               Auto-launched in the background by this script when
#               JUDGE_MODE=auto (the default).
#    GPUs 6–7 → FSDP=2 GRPO trainer.
#
#  Why FSDP=2 not FSDP=4: GRPO trains LoRA only (~250 M params across
#  5 adapters); the frozen 9B base is FSDP-sharded but does no
#  backward, so per-GPU memory ≈ 12 GB on H200 141 GB.  The
#  orchestrator pipelines GRPO(N) with rollout(N+1)
#  (orchestrator.py L559/742/1093) so train_time hides inside the
#  ~25-30 min/step rollout window — FSDP=2 ends up FREE wall-clock-wise
#  while freeing 2 GPUs for the live 35B judge.
#
#  Layout overrides (env vars):
#    LAYOUT=dual_stack       → default; 4×9B + 1×35B(TP=2) + FSDP=2
#    LAYOUT=dual_stack_fsdp4 → 2×9B + 1×35B(TP=2) + FSDP=4 + EPISODES=16
#                              (use only when train_time would exceed
#                              rollout even after pipelining)
#    LAYOUT=actor_only       → 4×9B + FSDP=4, no 35B server (legacy /
#                              ablations; 9B-self-judges)
#    JUDGE_MODE=auto|external|off
#                            → control whether this script launches the
#                              35B server itself.  external=already
#                              running elsewhere (set JUDGE_URL).
#                              off=no 35B routing wired (NOT recommended).
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
#    # 35B server already running on a 2nd box / port:
#    JUDGE_MODE=external JUDGE_URL=http://otherhost:8001/v1 \
#      bash scripts/run_phase1_curriculum.sh
#
#    # Single-stack (skip the 35B judge, max GRPO throughput):
#    LAYOUT=actor_only JUDGE_MODE=off bash scripts/run_phase1_curriculum.sh
#
#    # Use external vLLM instead of MANAGED 9B (legacy):
#    MANAGE_VLLM=0 VLLM_URL=http://localhost:8000/v1 \
#      bash scripts/run_phase1_curriculum.sh
#
#  Cross-refs:
#    - scripts/run_2048.sh                       (single-game variant of this layout)
#    - scripts/use_35b_judge.sh                  (manual VLLM_BASE_URL_MAP wiring)
#    - inference/serve_qwen35_35b_a3b.sh         (the 35B serve script this calls)
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

# §4.1 lock: 15 steps per phase, snapshot every 5 steps so a mid-phase
# failure doesn't lose more than ~3 h of work.  ``EPISODES`` default
# is set per LAYOUT below (8 for FSDP=2, 16 for FSDP=4).
ITERS_PER_PHASE="${ITERS_PER_PHASE:-15}"
CKPT_INTERVAL="${CKPT_INTERVAL:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution-phase1}"
RUN_DIR="${RUN_DIR:-}"
DEBUG="${DEBUG:-}"
RESUME_PHASE="${RESUME_PHASE:-1}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"

# 8×H200 layout selector (see banner above for full table).
LAYOUT="${LAYOUT:-dual_stack}"
JUDGE_MODE="${JUDGE_MODE:-auto}"

case "${LAYOUT}" in
    dual_stack)
        VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
        JUDGE_GPUS="${JUDGE_GPUS:-4,5}"
        JUDGE_TP="${JUDGE_TP:-2}"
        GRPO_GPUS="${GRPO_GPUS:-6 7}"
        EPISODES="${EPISODES:-8}"
        ;;
    dual_stack_fsdp4)
        VLLM_GPUS="${VLLM_GPUS:-0 1}"
        JUDGE_GPUS="${JUDGE_GPUS:-2,3}"
        JUDGE_TP="${JUDGE_TP:-2}"
        GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
        EPISODES="${EPISODES:-16}"
        ;;
    actor_only)
        VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
        JUDGE_GPUS=""
        JUDGE_TP="0"
        GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
        EPISODES="${EPISODES:-8}"
        if [ "${JUDGE_MODE}" = "auto" ]; then
            JUDGE_MODE="off"
        fi
        ;;
    *)
        echo "ERROR: unknown LAYOUT=${LAYOUT}" \
             "(expected dual_stack|dual_stack_fsdp4|actor_only)"
        exit 1
        ;;
esac

# 35B-A3B judge endpoint (used by the orchestrator + skill_eval gates).
JUDGE_PORT="${JUDGE_PORT:-8004}"
JUDGE_URL="${JUDGE_URL:-http://localhost:${JUDGE_PORT}/v1}"
JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.92}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.5-35B-A3B}"

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

# ── Cleanup on exit ───────────────────────────────────────────────────
VLLM_PID=""
JUDGE_PID=""
cleanup() {
    echo ""
    echo "[run_phase1] Shutting down..."
    if [ -n "${JUDGE_PID}" ] && kill -0 "${JUDGE_PID}" 2>/dev/null; then
        echo "[run_phase1] Stopping 35B judge (PID ${JUDGE_PID})..."
        kill "${JUDGE_PID}" 2>/dev/null || true
        for _ in $(seq 1 10); do
            kill -0 "${JUDGE_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -9 "${JUDGE_PID}" 2>/dev/null || true
    fi
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[run_phase1] Stopping vLLM server (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[run_phase1] Done."
}
trap cleanup EXIT INT TERM

# ── 35B-A3B judge server (auto-launched in dual_stack*) ──────────────
JUDGE_LOG=""

start_35b_judge() {
    if [ -z "${JUDGE_GPUS}" ] || [ "${JUDGE_TP}" = "0" ]; then
        echo "[run_phase1] LAYOUT=${LAYOUT}: 35B server not part of layout — skipping."
        return 0
    fi
    JUDGE_LOG="${RUN_DIR:-runs}/judge_35b.log"
    mkdir -p "$(dirname "${JUDGE_LOG}")"
    echo "[run_phase1] Auto-launching 35B-A3B judge:"
    echo "  GPUs:   ${JUDGE_GPUS}  (TP=${JUDGE_TP}, expert-parallel ON)"
    echo "  Port:   ${JUDGE_PORT}"
    echo "  Model:  ${JUDGE_MODEL}"
    echo "  Log:    ${JUDGE_LOG}"
    GPUS="${JUDGE_GPUS}" \
    TENSOR_PARALLEL="${JUDGE_TP}" \
    EXPERT_PARALLEL=1 \
    GPU_UTIL="${JUDGE_GPU_UTIL}" \
    PORT="${JUDGE_PORT}" \
    HOST="127.0.0.1" \
    MODEL="${JUDGE_MODEL}" \
        bash "${PROJECT_ROOT}/inference/serve_qwen35_35b_a3b.sh" \
            > "${JUDGE_LOG}" 2>&1 &
    JUDGE_PID=$!
    echo "[run_phase1] 35B server PID: ${JUDGE_PID}"

    echo "[run_phase1] Waiting for 35B server health on ${JUDGE_URL} ..."
    for _ in $(seq 1 120); do
        if curl -fs -m 3 "${JUDGE_URL}/models" >/dev/null 2>&1; then
            echo "[run_phase1] 35B server is healthy."
            return 0
        fi
        if ! kill -0 "${JUDGE_PID}" 2>/dev/null; then
            echo "[run_phase1] 35B server died during startup — see ${JUDGE_LOG}"
            return 1
        fi
        sleep 5
    done
    echo "[run_phase1] 35B server did not become healthy within 600s — see ${JUDGE_LOG}"
    return 1
}

# Resolved later (after RUN_DIR is set) — wire JUDGE_MODE here so the
# pre-flight banner can show the final URL/model values.
case "${JUDGE_MODE}" in
    auto|external)
        export VLLM_BASE_URL_MAP="${MODEL}=http://localhost:${PORT}/v1,${JUDGE_MODEL}=${JUDGE_URL}"
        export VLM_AGENT_BACKBONE_JUDGE_MODEL="${JUDGE_MODEL}"
        ;;
    off)
        echo "[run_phase1] JUDGE_MODE=off → 35B routing NOT wired" \
             "(judge will fall back to 9B; only safe for ablations)."
        ;;
    *)
        echo "ERROR: unknown JUDGE_MODE=${JUDGE_MODE}" \
             "(expected auto|external|off)"
        exit 1
        ;;
esac

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
    echo "  GPU layout:       LAYOUT=${LAYOUT}"
    echo "    9B vLLM GPUs:   ${VLLM_GPUS}        (TP=1 each, ports ${PORT}+)"
    if [ -n "${JUDGE_GPUS}" ]; then
        echo "    35B judge GPUs: ${JUDGE_GPUS}      (TP=${JUDGE_TP}+EP, port ${JUDGE_PORT})"
    else
        echo "    35B judge:      OFF (LAYOUT=actor_only)"
    fi
    echo "    GRPO trainer:   ${GRPO_GPUS}        (FSDP=$(echo "${GRPO_GPUS}" | wc -w))"
    echo "  Judge mode:       ${JUDGE_MODE}"
    if [ "${JUDGE_MODE}" != "off" ]; then
        echo "  Judge URL:        ${JUDGE_URL}"
        echo "  Judge model:      ${JUDGE_MODEL}"
    fi
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

# Start the 35B judge BEFORE the curriculum loop so its KV cache
# allocation lands while no FSDP / vLLM-9B processes are competing
# for HBM yet.  In external mode this is a no-op; in off mode it's
# skipped entirely.
if [ "${JUDGE_MODE}" = "auto" ]; then
    if ! start_35b_judge; then
        echo "[run_phase1] FATAL: cannot start 35B judge.  Re-run with" \
             "JUDGE_MODE=external (point JUDGE_URL elsewhere) or" \
             "JUDGE_MODE=off (disable 35B routing — ablation only)."
        exit 1
    fi
elif [ "${JUDGE_MODE}" = "external" ]; then
    echo "[run_phase1] JUDGE_MODE=external → assuming 35B server at ${JUDGE_URL}"
    if command -v curl >/dev/null 2>&1; then
        if curl -fs -m 3 "${JUDGE_URL}/models" >/dev/null 2>&1; then
            echo "[run_phase1] 35B server is reachable."
        else
            echo "[run_phase1] WARNING: ${JUDGE_URL} not reachable — judge calls" \
                 "will fail at runtime.  Start the 35B server before running."
        fi
    fi
fi

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
