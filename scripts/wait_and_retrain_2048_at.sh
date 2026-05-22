#!/usr/bin/env bash
# Wait for the tetris GRPO run (PID 1814898 = scripts/run_coevolution.py
# under runs/tetris_coevo_v4_20260520_063432) to exit, then retrain the
# twenty_forty_eight/action_taking adapter that failed during the original
# sft_per_game_xml run on 2026-05-19 (GPU 2 was killed at step 950/1137,
# zero checkpoints saved).
#
# After tetris GRPO finishes, this also waits ~60s for VLLM workers
# spawned by run_coevo_local35b.sh to release their GPU memory, then
# launches a single-GPU training job (GPU 0).
#
# Usage:
#   nohup bash scripts/wait_and_retrain_2048_at.sh \
#       > runs/sft_per_game_xml/_wait_and_retrain.log 2>&1 &

set -u

GRPO_PID=1814898
GRPO_BASH_PID=1812967
PARENT_RUN_DIR="runs/tetris_coevo_v4_20260520_063432"

REPO_DIR="/workspace/Multi-hop-Reasoning-VLM-Agent"
OUTPUT_DIR="runs/sft_per_game_xml"
LOG_DIR="${REPO_DIR}/${OUTPUT_DIR}"
WATCHER_LOG="${LOG_DIR}/_wait_and_retrain.log"

mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date -u +'%Y-%m-%d %H:%M:%SZ')] $*"
}

log "watcher started (pid=$$)"
log "monitoring tetris GRPO pid=${GRPO_PID} (bash parent ${GRPO_BASH_PID})"
log "target run: ${PARENT_RUN_DIR}"

while kill -0 "${GRPO_PID}" 2>/dev/null; do
    sleep 30
done
log "tetris GRPO process ${GRPO_PID} exited"

# Give bash parent + VLLM workers time to clean up GPU memory
log "waiting 60s for VLLM workers to release GPU memory"
sleep 60

# Best-effort check: warn if any VLLM workers still occupy GPU 0
if nvidia-smi --id=0 --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
        | grep -q .; then
    log "WARNING: GPU 0 still has compute apps; sleeping another 90s"
    nvidia-smi --id=0 --query-compute-apps=pid,process_name,used_memory --format=csv \
        2>&1 | tee -a "${WATCHER_LOG}" >/dev/null
    sleep 90
fi

cd "${REPO_DIR}"

# Preserve the original 8-GPU summary so we know which GPU originally
# failed and can audit the retrain.
if [[ -f "${OUTPUT_DIR}/training_summary.json" \
        && ! -f "${OUTPUT_DIR}/training_summary.original_8gpu.json" ]]; then
    cp "${OUTPUT_DIR}/training_summary.json" \
       "${OUTPUT_DIR}/training_summary.original_8gpu.json"
    log "saved original training summary as training_summary.original_8gpu.json"
fi

# Clean the empty hf_trainer directory from the failed run so Trainer
# starts fresh without complaining about a stale output dir.
TARGET_HF_OUT="${OUTPUT_DIR}/twenty_forty_eight/action_taking/hf_trainer"
if [[ -d "${TARGET_HF_OUT}" ]] && [[ -z "$(ls -A "${TARGET_HF_OUT}" 2>/dev/null)" ]]; then
    rmdir "${TARGET_HF_OUT}"
    log "removed empty ${TARGET_HF_OUT}"
fi

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate game-ai-agent

log "launching twenty_forty_eight/action_taking retrain on GPU 0"
log "(save_strategy=epoch is now active so a mid-training crash will leave checkpoints)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m trainer.SFT.train_per_game \
    --decision_data_dir /workspace/SFT_Data/xml_retrain/decision_sft \
    --v2_data_dir /workspace/SFT_Data/xml_retrain/decision_sft_v2 \
    --output_dir "${OUTPUT_DIR}" \
    --gpus 0 \
    --epochs 3 \
    --batch_size 8 \
    --games twenty_forty_eight \
    --adapters action_taking \
    2>&1 | tee -a "${LOG_DIR}/_retrain_2048_at.log"

RC=${PIPESTATUS[0]}
log "retrain exit code: ${RC}"

if [[ ${RC} -eq 0 ]]; then
    log "DONE — twenty_forty_eight/action_taking adapter rebuilt"
    log "check ${OUTPUT_DIR}/twenty_forty_eight/action_taking/ for adapter files"
else
    log "FAILED — see ${LOG_DIR}/gpu0.log and ${LOG_DIR}/_retrain_2048_at.log"
fi

exit ${RC}
