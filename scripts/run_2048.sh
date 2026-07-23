#!/usr/bin/env bash
# ======================================================================
#  Train 2048 with LoRA adapters warm-started from SFT cold-start.
#
#  Loads pre-trained LoRA adapters (skill_selection, action_taking,
#  segment, contract, curator) from the SFT cold-start output and
#  begins co-evolution on twenty_forty_eight.  The 2048 skill bank
#  starts empty and bootstraps from scratch.
#
#  All GRPO stability fixes are already baked into the shared training
#  code (game-agnostic):
#    - Format penalty: OOB actions get reward=0 with raw LLM output
#    - Format bonus: +1.0 for valid action outputs
#    - Constrained decoding: max_tokens=48, extra stop sequences
#    - LR cosine decay + stronger KL ramp
#    - Replay buffer capped at 1:1 fresh:stale
#    - Contract adapter stabilization (threshold=32, lr×0.5, kl×1.5)
#    - Best-checkpoint rollback after 3 consecutive declines
#    - Reduced GRPO epochs
#
#  ── GPU layouts (see scripts/gpu_layouts.sh) ─────────────────────────
#  Default LAYOUT=dual_stack targets 8×H200 (141 GB).
#  For GAMMA / Nexus L40S (48 GB) use:
#    LAYOUT=l40s_8            → 8×L40S, 4×9B + fp8 35B(TP=2) + FSDP=2
#    LAYOUT=l40s_8_tp4        → 8×L40S, 2×9B + bf16 35B(TP=4) + FSDP=2
#    LAYOUT=l40s_4            → 4×L40S actor+GRPO (judge external/off)
#    LAYOUT=l40s_4_compact    → 4×L40S + live fp8 35B judge (tight)
#
#  Usage:
#    conda activate game-ai-agent
#    bash scripts/run_2048.sh
#
#    # 8×L40S (e.g. cml34 / two gammagpu L40S nodes via scavenger):
#    LAYOUT=l40s_8 bash scripts/run_2048.sh
#
#    # 4×L40S actor + external 35B on a second node:
#    LAYOUT=l40s_4 JUDGE_MODE=external JUDGE_URL=http://otherhost:8004/v1 \
#        bash scripts/run_2048.sh
#
#    # Skip launching the 35B server (e.g. run it yourself on a 2nd box):
#    JUDGE_MODE=external JUDGE_URL=http://otherhost:8001/v1 \
#        bash scripts/run_2048.sh
#
#    # Legacy single-stack (no 35B judge; faster GRPO at the cost of
#    # 9B-self-judging during promotion gates):
#    LAYOUT=actor_only JUDGE_MODE=off bash scripts/run_2048.sh
#
#    # Or with batch overrides:
#    TOTAL_STEPS=20 EPISODES=12 bash scripts/run_2048.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=gpu_layouts.sh
source "${SCRIPT_DIR}/gpu_layouts.sh"

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# ── 2048-specific segmentation tuning ────────────────────────────────
# 2048 episodes are 200 steps → ~33 segments with default duration prior,
# generating 250+ LLM ranking calls that overwhelm vLLM and cause 600s
# segmentation timeouts.  These overrides:
#   - max_tokens 1000→300: actual responses are <250 tokens; reduces KV
#     cache pressure under concurrent load (~3x vLLM throughput)
#   - timeout 600→120: segmentation either finishes in <120s or is stuck
export SKILLBANK_LLM_TEACHER_MAX_TOKENS="${SKILLBANK_LLM_TEACHER_MAX_TOKENS:-300}"
export SKILLBANK_SEGMENT_TIMEOUT_S="${SKILLBANK_SEGMENT_TIMEOUT_S:-120}"

# ── HuggingFace cache ────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

# ── PYTHONPATH ────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── Configurable parameters ──────────────────────────────────────────
MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-9B}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"

TOTAL_STEPS="${TOTAL_STEPS:-25}"
CKPT_INTERVAL="${CKPT_INTERVAL:-3}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"

# Layout selector (H200 + L40S presets — see scripts/gpu_layouts.sh).
LAYOUT="${LAYOUT:-dual_stack}"
JUDGE_MODE="${JUDGE_MODE:-auto}"
apply_gpu_layout || exit 1

# 35B-A3B judge endpoint (used by the orchestrator + skill_eval gates).
JUDGE_PORT="${JUDGE_PORT:-8004}"
JUDGE_URL="${JUDGE_URL:-http://localhost:${JUDGE_PORT}/v1}"
JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.92}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-65536}"
JUDGE_MAX_NUM_SEQS="${JUDGE_MAX_NUM_SEQS:-128}"
JUDGE_QUANTIZATION="${JUDGE_QUANTIZATION:-}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.5-35B-A3B}"

# ── SFT cold-start adapter paths ─────────────────────────────────────
SFT_DIR="${SFT_DIR:-${PROJECT_ROOT}/runs/sft_coldstart}"
DECISION_ADAPTERS="${SFT_DIR}/decision"
SKILLBANK_ADAPTERS="${SFT_DIR}/skillbank"

if [ ! -d "${DECISION_ADAPTERS}" ]; then
    echo "ERROR: Decision adapters not found: ${DECISION_ADAPTERS}"
    echo "Run SFT cold-start first:  bash scripts/run_sft_coldstart.sh"
    exit 1
fi
if [ ! -d "${SKILLBANK_ADAPTERS}" ]; then
    echo "ERROR: Skill-bank adapters not found: ${SKILLBANK_ADAPTERS}"
    echo "Run SFT cold-start first:  bash scripts/run_sft_coldstart.sh"
    exit 1
fi

# ── New run directory for 2048 ───────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-/workspace/game_agent/Game-AI-Agent/runs/Qwen3.5-9B_2048_${TIMESTAMP}}"
mkdir -p "${RUN_DIR}/checkpoints"

# ── 35B-A3B judge server (auto-launched in dual_stack) ───────────────
JUDGE_PID=""
JUDGE_LOG="${RUN_DIR}/judge_35b.log"

start_35b_judge() {
    if [ -z "${JUDGE_GPUS}" ] || [ "${JUDGE_TP}" = "0" ]; then
        echo "[2048] LAYOUT=${LAYOUT}: 35B server not part of layout — skipping."
        return 0
    fi
    echo "[2048] Auto-launching 35B-A3B judge:"
    echo "  GPUs:   ${JUDGE_GPUS}  (TP=${JUDGE_TP}, expert-parallel ON)"
    echo "  Port:   ${JUDGE_PORT}"
    echo "  Model:  ${JUDGE_MODEL}"
    echo "  Log:    ${JUDGE_LOG}"
    GPUS="${JUDGE_GPUS}" \
    TENSOR_PARALLEL="${JUDGE_TP}" \
    EXPERT_PARALLEL=1 \
    GPU_UTIL="${JUDGE_GPU_UTIL}" \
    MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN}" \
    MAX_NUM_SEQS="${JUDGE_MAX_NUM_SEQS}" \
    QUANTIZATION="${JUDGE_QUANTIZATION}" \
    PORT="${JUDGE_PORT}" \
    HOST="127.0.0.1" \
    MODEL="${JUDGE_MODEL}" \
        bash "${PROJECT_ROOT}/inference/serve_qwen35_35b_a3b.sh" \
            > "${JUDGE_LOG}" 2>&1 &
    JUDGE_PID=$!
    echo "[2048] 35B server PID: ${JUDGE_PID}"

    # Block until /v1/models responds (or timeout 600s).
    echo "[2048] Waiting for 35B server health on ${JUDGE_URL} ..."
    for _ in $(seq 1 120); do
        if curl -fs -m 3 "${JUDGE_URL}/models" >/dev/null 2>&1; then
            echo "[2048] 35B server is healthy."
            return 0
        fi
        if ! kill -0 "${JUDGE_PID}" 2>/dev/null; then
            echo "[2048] 35B server died during startup — see ${JUDGE_LOG}"
            return 1
        fi
        sleep 5
    done
    echo "[2048] 35B server did not become healthy within 600s — see ${JUDGE_LOG}"
    return 1
}

case "${JUDGE_MODE}" in
    auto)
        if ! start_35b_judge; then
            echo "[2048] FATAL: cannot start 35B judge.  Re-run with" \
                 "JUDGE_MODE=external (point JUDGE_URL elsewhere) or" \
                 "JUDGE_MODE=off (disable 35B routing)."
            exit 1
        fi
        # Only singleton services belong in the map. Actor requests that are
        # unmapped round-robin across the full VLLM_BASE_URLS pool.
        export VLLM_BASE_URL_MAP="${JUDGE_MODEL}=${JUDGE_URL}"
        export VLM_AGENT_BACKBONE_JUDGE_MODEL="${JUDGE_MODEL}"
        ;;
    external)
        echo "[2048] JUDGE_MODE=external → assuming 35B server at ${JUDGE_URL}"
        export VLLM_BASE_URL_MAP="${JUDGE_MODEL}=${JUDGE_URL}"
        export VLM_AGENT_BACKBONE_JUDGE_MODEL="${JUDGE_MODEL}"
        ;;
    off)
        echo "[2048] JUDGE_MODE=off → 35B routing NOT wired (judge will" \
             "fall back to 9B; only safe for ablations)."
        ;;
    *)
        echo "ERROR: unknown JUDGE_MODE=${JUDGE_MODE}" \
             "(expected auto|external|off)"
        exit 1
        ;;
esac

# ── Cleanup on exit ──────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[2048] Shutting down..."
    if [ -n "${JUDGE_PID}" ] && kill -0 "${JUDGE_PID}" 2>/dev/null; then
        echo "[2048] Stopping 35B judge (PID ${JUDGE_PID})..."
        kill "${JUDGE_PID}" 2>/dev/null || true
        # Give vLLM up to 10s to exit cleanly before SIGKILL.
        for _ in $(seq 1 10); do
            kill -0 "${JUDGE_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -9 "${JUDGE_PID}" 2>/dev/null || true
    fi
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    echo "[2048] Done."
}
trap cleanup EXIT INT TERM

# ── Print banner ─────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  2048: Warm-start from SFT cold-start LoRA adapters"
echo "══════════════════════════════════════════════════════════════"
echo "  SFT dir:        ${SFT_DIR}"
echo "  Decision LoRA:  ${DECISION_ADAPTERS}"
echo "  SkillBank LoRA: ${SKILLBANK_ADAPTERS}"
echo "  New run dir:    ${RUN_DIR}"
echo "  Model:          ${MODEL}"
echo "  Total steps:    ${TOTAL_STEPS}"
echo "  Episodes/step:  ${EPISODES}"
echo "  Checkpoint:     every ${CKPT_INTERVAL} steps"
echo ""
echo "  GPU layout (LAYOUT=${LAYOUT}):"
echo "    9B vLLM GPUs:   ${VLLM_GPUS}        (TP=1 each, ports 8000+)"
if [ -n "${JUDGE_GPUS}" ]; then
    echo "    35B judge GPUs: ${JUDGE_GPUS}      (TP=${JUDGE_TP}+EP, port ${JUDGE_PORT})"
else
    echo "    35B judge:      OFF (LAYOUT=actor_only)"
fi
echo "    GRPO trainer:   ${GRPO_GPUS}        (FSDP=$(echo "${GRPO_GPUS}" | wc -w))"
echo "  Judge mode:     ${JUDGE_MODE}"
if [ "${JUDGE_MODE}" != "off" ]; then
    echo "  Judge URL:      ${JUDGE_URL}"
    echo "  Judge model:    ${JUDGE_MODEL}"
fi
echo "  Spec decode:    ${SPEC_MODEL} (${SPEC_TOKENS} tokens)"
echo ""
echo "  2048 game profile:"
echo "    - 4 string actions (up/down/left/right)"
echo "    - 200 max steps/episode"
echo "    - Sparse rewards (many 0-reward steps from non-merge moves)"
echo "    - Skill bank starts empty; LoRA adapters warm-started from SFT"
echo ""
echo "  2048 segmentation tuning:"
echo "    - Duration prior: mean=40 (→ ~5 segments/episode vs 33 default)"
echo "    - LLM teacher max_tokens: ${SKILLBANK_LLM_TEACHER_MAX_TOKENS} (was 1000)"
echo "    - Segment timeout: ${SKILLBANK_SEGMENT_TIMEOUT_S}s (was 600s)"
echo ""
echo "  Stability fixes baked in:"
echo "    - OOB action format penalty (reward=0)"
echo "    - Format bonus (+1.0) for valid action outputs"
echo "    - Constrained decoding: max_tokens=48, extra stop seq"
echo "    - LR cosine decay + stronger KL ramp"
echo "    - Replay buffer capped at 1:1 fresh:stale"
echo "    - Contract adapter: threshold=32, lr×0.5, kl×1.5"
echo "    - Best-checkpoint rollback after 3 consecutive declines"
echo "    - Reduced GRPO epochs"
echo "══════════════════════════════════════════════════════════════"
echo ""

# Show SFT adapter info
echo "[2048] SFT cold-start adapters:"
for adapter_dir in "${DECISION_ADAPTERS}"/* "${SKILLBANK_ADAPTERS}"/*; do
    if [ -d "${adapter_dir}" ]; then
        name="$(basename "${adapter_dir}")"
        if [ -f "${adapter_dir}/adapter_config.json" ]; then
            echo "  ✓ ${name}"
        else
            echo "  ✗ ${name} (missing adapter_config.json)"
        fi
    fi
done
echo ""

# ── Build training command ───────────────────────────────────────────
TRAIN_ARGS=(
    --games twenty_forty_eight
    --total-steps "${TOTAL_STEPS}"
    --curriculum none
    --episodes-per-game "${EPISODES}"
    --checkpoint-interval "${CKPT_INTERVAL}"
    --model "${MODEL}"
    --wandb-project "${WANDB_PROJECT}"
    --run-dir "${RUN_DIR}"
    --load-decision-adapters "${DECISION_ADAPTERS}"
    --load-skillbank-adapters "${SKILLBANK_ADAPTERS}"
    --debug-io
    # shellcheck disable=SC2086
    --vllm-gpus ${VLLM_GPUS}
    --grpo-devices ${GRPO_GPUS}
    --vllm-base-port "${PORT}"
    --vllm-gpu-util "${GPU_UTIL}"
    --speculative-model "${SPEC_MODEL}"
    --num-speculative-tokens "${SPEC_TOKENS}"
)

echo "[2048] Command:"
echo "  python scripts/run_coevolution.py ${TRAIN_ARGS[*]}"
echo ""

# ── Run training ─────────────────────────────────────────────────────
python scripts/run_coevolution.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  2048 training COMPLETE"

    echo ""
    echo "  Step log:"
    if [ -f "${RUN_DIR}/step_log.jsonl" ]; then
        python -c "
import json
with open('${RUN_DIR}/step_log.jsonl') as f:
    rows = [json.loads(l) for l in f if l.strip()]
for r in rows[-5:]:
    step = r['step']
    mr = r['mean_reward']
    ns = r.get('n_skills', '?')
    wt = r['wall_time_s'] / 60
    print(f'  Step {step:2d}: mean_reward={mr:6.1f}  skills={ns}  time={wt:.1f}m')
"
    fi
else
    echo "  2048 training FAILED (exit code ${EXIT_CODE})"
    echo "  Check logs: ${RUN_DIR}/coevolution.log"
fi
echo ""
echo "  Run dir: ${RUN_DIR}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
