#!/usr/bin/env bash

# Start the experiment's 35B proposer on the two GPUs assigned by Slurm,
# require a successful OpenAI-compatible completion, then shut it down.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-cosplay-candy-a100}"
HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/hf_cache}"
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
PORT="${PORT:-18004}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
QUANTIZATION="${QUANTIZATION:-}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: Slurm did not assign GPUs" >&2
    exit 2
fi

nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv

server_log="${SMOKE_LOG:-/tmp/skill_xfer_35b_smoke_${SLURM_JOB_ID:-manual}.log}"
env REPO_ROOT="${REPO_ROOT}" CONDA_ENV="${CONDA_ENV}" HF_HOME="${HF_HOME}" \
    HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 \
    MODEL="${MODEL}" GPUS="${CUDA_VISIBLE_DEVICES}" TENSOR_PARALLEL=2 \
    EXPERT_PARALLEL=1 GPU_UTIL="${GPU_UTIL}" MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS}" PORT="${PORT}" HOST=127.0.0.1 \
    QUANTIZATION="${QUANTIZATION}" SPECULATIVE=none \
    bash "${REPO_ROOT}/inference/serve_qwen35_35b_a3b.sh" >"${server_log}" 2>&1 &
server_pid=$!
cleanup() {
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl -fsS -m 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "ERROR: 35B server exited during startup; log follows" >&2
        tail -n 200 "${server_log}" >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "ERROR: 35B server startup timed out; log follows" >&2
        tail -n 200 "${server_log}" >&2
        exit 1
    fi
    sleep 5
done

python - "${PORT}" "${MODEL}" <<'PY'
import json
import sys
import urllib.request

port, model = sys.argv[1:]
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "Reply with exactly OK."}],
    "temperature": 0,
    "max_tokens": 64,
}).encode()
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=180) as response:
    result = json.load(response)
message = result["choices"][0]["message"]
content = message.get("content") or message.get("reasoning")
if not isinstance(content, str) or not content.strip():
    raise SystemExit(f"empty completion: {result!r}")
print(json.dumps({"status": "ok", "content": content}, ensure_ascii=False))
PY

echo "SMOKE_OK model=${MODEL} quantization=${QUANTIZATION:-bf16} log=${server_log}"
