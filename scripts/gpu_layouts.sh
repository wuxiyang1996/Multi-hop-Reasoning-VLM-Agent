#!/usr/bin/env bash
# =============================================================================
# Shared GPU layout presets for coevolution launchers.
#
# Sourced by scripts/run_2048.sh, run_phase1_curriculum.sh, run_phase2_holdout.sh.
# Sets (or defaults):
#   VLLM_GPUS  JUDGE_GPUS  JUDGE_TP  GRPO_GPUS  EPISODES
#   GPU_UTIL   JUDGE_GPU_UTIL  JUDGE_MAX_MODEL_LEN  JUDGE_MAX_NUM_SEQS
#   JUDGE_QUANTIZATION (optional)
# and may export VLLM_MAX_* knobs consumed by trainer/coevolution/vllm_server.py.
#
# Layouts
# -------
#   dual_stack / dual_stack_fsdp4 / actor_only
#       Original 8×H200 (141 GB) presets.
#
#   l40s_8  (aliases: l40s_8_dual, l40s_8_dp4)  ← default / high throughput
#       8×L40S (48 GB). Same shape as H200 dual_stack, 4×9B rollout DP:
#         GPUs 0–3 → 4× Qwen3.5-9B vLLM (TP=1)
#         GPUs 4–5 → 1× Qwen3.5-35B-A3B (TP=2 + expert-parallel, fp8)
#         GPUs 6–7 → FSDP=2 GRPO
#       Judge uses fp8 so TP=2 fits in 48 GB (bf16 TP=2 is ~35 GB
#       weights alone and OOMs).  Override JUDGE_QUANTIZATION="" only
#       if you've verified headroom.
#
#   l40s_8_tp4  (alias: l40s_8_safe)
#       8×L40S, safer bf16 judge, half the rollout DP:
#         GPUs 0–1 → 2×9B vLLM
#         GPUs 2–5 → 35B TP=4 bf16 + expert-parallel
#         GPUs 6–7 → FSDP=2 GRPO
#
#   l40s_4  (alias: l40s_4_actor)
#       4×L40S actor+train only (GAMMA gammagpu18–21):
#         GPUs 0–1 → 2×9B vLLM
#         GPUs 2–3 → FSDP=2 GRPO
#       Cannot host 4×9B and GRPO at once (vLLM and FSDP need
#       disjoint GPUs).  Pair with JUDGE_MODE=external on a second
#       4×L40S node for the 35B judge.
#
#   l40s_4_compact
#       4×L40S with a live 35B judge on the same node (tight):
#         GPU 0     → 1×9B vLLM
#         GPUs 1–2  → 35B TP=2 (defaults to QUANTIZATION=fp8)
#         GPU 3     → FSDP=1 GRPO
#
# Usage:
#   LAYOUT=l40s_8 bash scripts/run_phase1_curriculum.sh          # 4×9B
#   LAYOUT=l40s_8_tp4 bash scripts/run_phase1_curriculum.sh     # 2×9B, bf16 judge
#   LAYOUT=l40s_4 JUDGE_MODE=external JUDGE_URL=http://other:8004/v1 \
#       bash scripts/run_2048.sh
# =============================================================================

apply_gpu_layout() {
    local layout="${LAYOUT:-dual_stack}"

    case "${layout}" in
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
            if [ "${JUDGE_MODE:-auto}" = "auto" ]; then
                JUDGE_MODE="off"
            fi
            ;;
        l40s_8|l40s_8_dual|l40s_8_dp4)
            # 8×L40S — 4×9B rollout DP (H200 dual_stack shape).
            # 35B judge is TP=2 + fp8 so it fits beside 4 actor replicas.
            VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
            JUDGE_GPUS="${JUDGE_GPUS:-4,5}"
            JUDGE_TP="${JUDGE_TP:-2}"
            GRPO_GPUS="${GRPO_GPUS:-6 7}"
            EPISODES="${EPISODES:-16}"
            GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
            VLLM_GPU_UTIL="${GPU_UTIL}"
            JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.90}"
            JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-16384}"
            JUDGE_MAX_NUM_SEQS="${JUDGE_MAX_NUM_SEQS:-16}"
            # Use ${VAR-default} (no colon) so JUDGE_QUANTIZATION="" can
            # disable fp8; :- would treat empty as unset and force fp8.
            JUDGE_QUANTIZATION="${JUDGE_QUANTIZATION-fp8}"
            export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
            export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
            export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
            ;;
        l40s_8_tp4|l40s_8_safe)
            # 8×L40S — 2×9B + bf16 35B TP=4 (more judge headroom, less DP).
            VLLM_GPUS="${VLLM_GPUS:-0 1}"
            JUDGE_GPUS="${JUDGE_GPUS:-2,3,4,5}"
            JUDGE_TP="${JUDGE_TP:-4}"
            GRPO_GPUS="${GRPO_GPUS:-6 7}"
            EPISODES="${EPISODES:-8}"
            GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
            VLLM_GPU_UTIL="${GPU_UTIL}"
            JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.88}"
            JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-32768}"
            JUDGE_MAX_NUM_SEQS="${JUDGE_MAX_NUM_SEQS:-32}"
            export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
            export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
            export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
            ;;
        l40s_4|l40s_4_actor)
            # 4×L40S — actor + GRPO; 35B judge off-box or disabled.
            VLLM_GPUS="${VLLM_GPUS:-0 1}"
            JUDGE_GPUS=""
            JUDGE_TP="0"
            GRPO_GPUS="${GRPO_GPUS:-2 3}"
            EPISODES="${EPISODES:-8}"
            GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
            VLLM_GPU_UTIL="${GPU_UTIL}"
            export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
            export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
            export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
            if [ "${JUDGE_MODE:-auto}" = "auto" ]; then
                JUDGE_MODE="off"
                echo "[gpu_layouts] LAYOUT=${layout}: no spare GPUs for a live" \
                     "35B judge — JUDGE_MODE=off." \
                     "Use JUDGE_MODE=external + JUDGE_URL=... on a 2nd node," \
                     "or LAYOUT=l40s_4_compact for same-node (fp8) judge."
            fi
            ;;
        l40s_4_compact)
            # 4×L40S with live 35B — fp8 judge by default so TP=2 fits.
            VLLM_GPUS="${VLLM_GPUS:-0}"
            JUDGE_GPUS="${JUDGE_GPUS:-1,2}"
            JUDGE_TP="${JUDGE_TP:-2}"
            GRPO_GPUS="${GRPO_GPUS:-3}"
            EPISODES="${EPISODES:-4}"
            GPU_UTIL="${VLLM_GPU_UTIL:-0.82}"
            VLLM_GPU_UTIL="${GPU_UTIL}"
            JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.90}"
            JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-16384}"
            JUDGE_MAX_NUM_SEQS="${JUDGE_MAX_NUM_SEQS:-16}"
            JUDGE_QUANTIZATION="${JUDGE_QUANTIZATION-fp8}"
            export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
            export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
            export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
            ;;
        *)
            echo "ERROR: unknown LAYOUT=${layout}" \
                 "(expected dual_stack|dual_stack_fsdp4|actor_only|" \
                 "l40s_8|l40s_8_tp4|l40s_4|l40s_4_compact)"
            return 1
            ;;
    esac

    # Export so nested python / serve scripts see the resolved values.
    export LAYOUT="${layout}"
    export VLLM_GPUS JUDGE_GPUS JUDGE_TP GRPO_GPUS EPISODES
    export GPU_UTIL VLLM_GPU_UTIL
    export JUDGE_GPU_UTIL JUDGE_MAX_MODEL_LEN JUDGE_MAX_NUM_SEQS
    export JUDGE_QUANTIZATION="${JUDGE_QUANTIZATION:-}"
    export JUDGE_MODE="${JUDGE_MODE:-auto}"
    return 0
}
