"""Configuration for the co-evolution training loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


SKILL_BANK_GAMES = [
    "super_mario",
    "pokemon_red",
    "diplomacy",
    "twenty_forty_eight",
    "tetris",
    "avalon",
    "sokoban",
    "candy_crush",
]

GAME_MAX_STEPS: Dict[str, int] = {
    "super_mario": 500,
    "pokemon_red": 200,
    "diplomacy": 200,
    "twenty_forty_eight": 200,
    "tetris": 200,
    "avalon": 200,
    "sokoban": 100,
    "candy_crush": 50,
}

EMULATOR_GAMES = {"pokemon_red", "super_mario", "diplomacy"}

GAME_DURATION_ORDER = [
    "super_mario",
    "pokemon_red",
    "diplomacy",
    "twenty_forty_eight",
    "tetris",
    "avalon",
    "sokoban",
    "candy_crush",
]

ADAPTER_NAMES = [
    "skill_selection",
    "action_taking",
    "segment",
    "contract",
    "curator",
]


@dataclass
class CoEvolutionConfig:
    """Top-level configuration for the co-evolution loop."""

    games: List[str] = field(default_factory=lambda: list(SKILL_BANK_GAMES))
    episodes_per_game: int = 8
    max_concurrent_episodes: int = 40
    total_steps: int = 30

    # vLLM
    vllm_base_url: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen3-14B"
    temperature: float = 0.3
    max_tokens: int = 512

    # Skill bank EM
    em_max_iterations: int = 3
    em_micro_batch_size: int = 8

    # GRPO
    grpo_enabled: bool = True
    grpo_decision_devices: List[int] = field(default_factory=lambda: [4, 5])
    grpo_skillbank_devices: List[int] = field(default_factory=lambda: [6, 7])

    # Directories
    bank_dir: str = "runs/skillbank"
    adapter_dir: str = "runs/lora_adapters"
    checkpoint_dir: str = "runs/coevolution/checkpoints"
    log_dir: str = "runs/coevolution"

    # Checkpointing
    checkpoint_interval: int = 5

    # W&B
    wandb_enabled: bool = True
    wandb_project: str = "game-ai-coevolution"
    wandb_run_name: Optional[str] = None

    # Resume
    resume_from_step: Optional[int] = None

    # Thread/process executors
    thread_workers: int = 20
    process_workers: int = 8

    # Early episode termination
    stuck_window: int = 15
    min_steps_before_stuck_check: int = 20

    def adapter_path(self, name: str) -> str:
        return str(Path(self.adapter_dir) / name)
