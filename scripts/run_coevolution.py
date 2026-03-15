#!/usr/bin/env python
"""Launch the co-evolution training loop.

Usage (from Game-AI-Agent root):

    # 1. Start vLLM server (on GPUs 0-3 with 5 LoRA adapters):
    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3-14B \\
        --tensor-parallel-size 4 \\
        --gpu-memory-utilization 0.90 \\
        --enable-lora \\
        --max-loras 5 \\
        --max-lora-rank 64 \\
        --lora-modules \\
            skill_selection=runs/lora_adapters/skill_selection \\
            action_taking=runs/lora_adapters/action_taking \\
            segment=runs/lora_adapters/segment \\
            contract=runs/lora_adapters/contract \\
            curator=runs/lora_adapters/curator \\
        --enable-prefix-caching \\
        --enable-chunked-prefill \\
        --max-num-seqs 128 \\
        --port 8000

    # 2. Run co-evolution (GRPO on GPUs 4-7):
    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
    python scripts/run_coevolution.py

    # Or with custom settings:
    python scripts/run_coevolution.py \\
        --total-steps 100 \\
        --episodes-per-game 8 \\
        --checkpoint-interval 5 \\
        --wandb-project game-ai-coevolution \\
        --resume

    # Specific games only:
    python scripts/run_coevolution.py \\
        --games tetris twenty_forty_eight sokoban \\
        --total-steps 10

    # Resume from specific step:
    python scripts/run_coevolution.py --resume-from-step 25
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent

for p in [str(CODEBASE_ROOT), str(CODEBASE_ROOT.parent / "GamingAgent")]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

from trainer.coevolution.config import CoEvolutionConfig, SKILL_BANK_GAMES
from trainer.coevolution.orchestrator import co_evolution_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Co-Evolution Training: Decision Agent + Skill Bank Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core
    parser.add_argument(
        "--total-steps", type=int, default=30,
        help="Total co-evolution steps (default: 30)",
    )
    parser.add_argument(
        "--games", nargs="+", default=None,
        help=f"Games to train on (default: all {len(SKILL_BANK_GAMES)})",
    )
    parser.add_argument(
        "--episodes-per-game", type=int, default=8,
        help="Episodes per game per step (default: 8)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=40,
        help="Max concurrent episodes (default: 40)",
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-14B",
        help="Base model name (default: Qwen/Qwen3-14B)",
    )
    parser.add_argument(
        "--vllm-url", type=str, default="http://localhost:8000/v1",
        help="vLLM server URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Sampling temperature (default: 0.3)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max generation tokens (default: 512)",
    )

    # GRPO
    parser.add_argument(
        "--no-grpo", action="store_true",
        help="Disable GRPO training (rollout + skill bank only)",
    )
    parser.add_argument(
        "--grpo-decision-devices", nargs="+", type=int, default=[4, 5],
        help="GPU devices for decision agent GRPO (default: 4 5)",
    )
    parser.add_argument(
        "--grpo-skillbank-devices", nargs="+", type=int, default=[6, 7],
        help="GPU devices for skill bank GRPO (default: 6 7)",
    )

    # Directories
    parser.add_argument(
        "--bank-dir", type=str, default="runs/skillbank",
        help="Skill bank directory (default: runs/skillbank)",
    )
    parser.add_argument(
        "--adapter-dir", type=str, default="runs/lora_adapters",
        help="LoRA adapter directory (default: runs/lora_adapters)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="runs/coevolution/checkpoints",
        help="Checkpoint directory (default: runs/coevolution/checkpoints)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="runs/coevolution",
        help="Log directory (default: runs/coevolution)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-interval", type=int, default=5,
        help="Save checkpoint every N steps (default: 5)",
    )

    # Resume
    parser.add_argument(
        "--resume", action="store_true",
        help="Auto-resume from latest checkpoint",
    )
    parser.add_argument(
        "--resume-from-step", type=int, default=None,
        help="Resume from specific checkpoint step",
    )

    # W&B
    parser.add_argument(
        "--wandb-project", type=str, default="game-ai-coevolution",
        help="W&B project name (default: game-ai-coevolution)",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None,
        help="W&B run name (auto-generated if not set)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging",
    )

    # Workers
    parser.add_argument(
        "--thread-workers", type=int, default=20,
        help="Thread pool size (default: 20)",
    )
    parser.add_argument(
        "--process-workers", type=int, default=8,
        help="Process pool size (default: 8)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                Path(args.log_dir) / "coevolution.log",
                mode="a",
            ) if Path(args.log_dir).parent.exists() else logging.StreamHandler(),
        ],
    )

    games = args.games if args.games else list(SKILL_BANK_GAMES)
    for g in games:
        if g not in SKILL_BANK_GAMES:
            logging.warning("Unknown game '%s', skipping", g)
    games = [g for g in games if g in SKILL_BANK_GAMES]

    if not games:
        logging.error("No valid games specified")
        sys.exit(1)

    resume_step = args.resume_from_step
    if args.resume and resume_step is None:
        resume_step = -1  # sentinel: auto-detect latest

    config = CoEvolutionConfig(
        games=games,
        episodes_per_game=args.episodes_per_game,
        max_concurrent_episodes=args.max_concurrent,
        total_steps=args.total_steps,
        vllm_base_url=args.vllm_url,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        grpo_enabled=not args.no_grpo,
        grpo_decision_devices=args.grpo_decision_devices,
        grpo_skillbank_devices=args.grpo_skillbank_devices,
        bank_dir=args.bank_dir,
        adapter_dir=args.adapter_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        checkpoint_interval=args.checkpoint_interval,
        wandb_enabled=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        resume_from_step=resume_step if resume_step != -1 else None,
        thread_workers=args.thread_workers,
        process_workers=args.process_workers,
    )

    print("=" * 70)
    print("  CO-EVOLUTION TRAINING")
    print("=" * 70)
    print(f"  Games:        {', '.join(games)}")
    print(f"  Steps:        {config.total_steps}")
    print(f"  Eps/game:     {config.episodes_per_game}")
    print(f"  Concurrent:   {config.max_concurrent_episodes}")
    print(f"  Model:        {config.model_name}")
    print(f"  vLLM:         {config.vllm_base_url}")
    print(f"  GRPO:         {'enabled' if config.grpo_enabled else 'disabled'}")
    if config.grpo_enabled:
        print(f"    Decision:   GPUs {config.grpo_decision_devices}")
        print(f"    SkillBank:  GPUs {config.grpo_skillbank_devices}")
    print(f"  Checkpoint:   every {config.checkpoint_interval} steps → {config.checkpoint_dir}")
    print(f"  W&B:          {'enabled' if config.wandb_enabled else 'disabled'}")
    if config.resume_from_step is not None:
        print(f"  Resume:       from step {config.resume_from_step}")
    elif args.resume:
        print("  Resume:       auto-detect latest checkpoint")
    print("=" * 70)

    # Handle auto-resume
    if args.resume and config.resume_from_step is None:
        from trainer.coevolution.checkpoint import find_latest_checkpoint
        latest = find_latest_checkpoint(config.checkpoint_dir)
        if latest is not None:
            config.resume_from_step = latest
            print(f"  Auto-resuming from checkpoint step {latest}")
        else:
            print("  No checkpoint found, starting from step 0")

    asyncio.run(co_evolution_loop(config))


if __name__ == "__main__":
    main()
