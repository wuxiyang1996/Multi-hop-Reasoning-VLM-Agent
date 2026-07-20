#!/usr/bin/env python3
"""Run a persistent pool of one-GPU vLLM rollout servers."""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import signal
import time

from trainer.coevolution.vllm_server import VLLMServerManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args()

    # vLLM may JIT-build FlashInfer's sampler. Some compute images expose the
    # CUDA runtime but not nvcc, which otherwise fails on the first request.
    if shutil.which("nvcc") is None:
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    manager = VLLMServerManager(
        model_name=args.model,
        adapter_dir=args.adapter_dir,
        gpu_ids=args.gpus,
        base_port=args.base_port,
        gpu_util=args.gpu_util,
        log_dir=args.log_dir,
        speculative_method="none",
        num_speculative_tokens=0,
    )
    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    manager.start()
    if not asyncio.run(manager.wait_healthy(timeout=1800)):
        manager.stop()
        raise SystemExit("rollout server pool failed to become healthy")
    manager.start_health_monitor()
    print("rollout server pool ready", flush=True)
    try:
        while not stopping:
            time.sleep(2)
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
