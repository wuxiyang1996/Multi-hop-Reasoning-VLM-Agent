#!/usr/bin/env python3
"""Create the five random LoRA adapters needed before external vLLM starts."""

from __future__ import annotations

import argparse
from pathlib import Path

from trainer.coevolution.config import (
    ADAPTER_NAMES,
    CoEvolutionConfig,
    prepare_adapters,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    adapter_dir = Path(args.adapter_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    config = CoEvolutionConfig(
        model_name=args.model,
        run_dir=str(work_dir),
        adapter_dir=str(adapter_dir),
        start_mode="from_scratch",
        wandb_enabled=False,
    )
    config.resolve_paths()
    adapters = prepare_adapters(config)
    missing = sorted(set(ADAPTER_NAMES) - set(adapters))
    if missing:
        raise SystemExit(f"failed to prepare adapters: {missing}")
    print(f"Prepared {len(adapters)} random adapters under {adapter_dir}")


if __name__ == "__main__":
    main()
