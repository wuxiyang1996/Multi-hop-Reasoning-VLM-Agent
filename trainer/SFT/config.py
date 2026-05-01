"""Configuration for SFT cold-start training.

LoRA hyper-parameters and target modules are kept identical to
``trainer.coevolution.config`` so the resulting adapters can be loaded
directly by the GRPO / FSDP trainer without any conversion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HOME", "/workspace/huggingface")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DECISION_ADAPTERS = ["skill_selection", "action_taking"]
SKILLBANK_ADAPTERS = ["segment", "contract", "curator"]
ALL_ADAPTERS = DECISION_ADAPTERS + SKILLBANK_ADAPTERS

# ── Default data sources (current dual-axis SFT corpus) ──────────────
#
# Decision adapters consume the per-game JSONLs produced by
# ``labeling/build_decision_sft_jsonl.py`` (output dir is timestamped;
# the auto-resolver below picks the most recent run).
DECISION_DATA_DIR = REPO_ROOT / "labeling" / "decision_sft_jsonl"

# Skill-bank adapters consume the per-game artefacts produced by
# ``labeling/run_skill_discovery.sh`` (skill_bank.jsonl,
# teacher_io_coldstart.jsonl, coldstart_io_all.jsonl).  The new layout
# inserts a ``<corpus>`` level (``gym_v`` / ``env_wrappers``) between
# the run dir and the game dir; ``data_loader.py`` walks it.
SKILLBANK_DATA_DIR = REPO_ROOT / "labeling" / "skill_bank_out"


def _latest_run(parent: Path, prefix: str = "run_") -> Path | None:
    """Return the most-recently-modified ``run_*`` dir under *parent*."""
    if not parent.is_dir():
        return None
    runs = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


# The two SFT corpora are timestamped (``run_<ts>``).  When the user
# does not pin a specific run via ``--decision_data_dir`` /
# ``--skillbank_data_dir`` we resolve to the latest one available.
DEFAULT_DECISION_DATA_DIR = _latest_run(DECISION_DATA_DIR) or DECISION_DATA_DIR
DEFAULT_SKILLBANK_DATA_DIR = _latest_run(SKILLBANK_DATA_DIR) or SKILLBANK_DATA_DIR


# All 17 games covered by the dual-axis SFT corpus
# (13 Gym-V Temporal envs + 4 env_wrappers games).  data_loader.py
# auto-skips games missing data so this list is forward-compatible
# with future adds.
COLDSTART_GAMES = [
    # Gym-V (stable-retro / Genesis ROMs)
    "Temporal_Airstriker-v0",
    "Temporal_AlteredBeast-v0",
    "Temporal_CastleOfIllusion-v0",
    "Temporal_CastlevaniaBloodlines-v0",
    "Temporal_Columns-v0",
    "Temporal_DynamiteHeaddy-v0",
    "Temporal_GoldenAxe-v0",
    "Temporal_KidChameleon-v0",
    "Temporal_MortalKombatII-v0",
    "Temporal_SpaceHarrierII-v0",
    "Temporal_StreetsOfRage2-v0",
    "Temporal_Strider-v0",
    "Temporal_ThunderForceIII-v0",
    # env_wrappers (game-ai-agent + orak-mario)
    "candy_crush",
    "super_mario",
    "tetris",
    "twenty_forty_eight",
]

# module → adapter mapping for coldstart_io_all.jsonl
#
# The cold-start extraction didn't run Stage 3 (contract learning) or
# Stage 4 (bank maintenance/curation), so there are no exact-match
# records for the ``contract`` or ``curator`` adapters.  We map the
# closest available proxy data instead:
#
#   contract ← boundary_proposal (predicate analysis, ~1.4k examples)
#              + pipeline (predicate extraction + protocol synthesis,
#                ~200 examples).  These teach the model to analyze
#              trajectory states and produce structured JSON — the same
#              domain knowledge needed for effect summarization.
#
#   curator  ← skill_naming (skill name + description generation,
#              ~216 examples).  Shares domain overlap (evaluating
#              skills) but the task format differs from the co-evolution
#              approve/veto/defer prompt.  With only 216 examples this
#              adapter benefits most from the higher epoch count.
#
# GRPO training during co-evolution refines these approximations to
# the actual task-specific prompts.
COLDSTART_IO_MODULE_MAP: Dict[str, str] = {
    "boundary_proposal": "contract",
    "pipeline": "contract",
    "skill_naming": "curator",
}


@dataclass
class SFTConfig:
    """Configuration for SFT cold-start training of all 5 LoRA adapters."""

    # Base model — must match what co-evolution uses
    model_name: str = "Qwen/Qwen3.5-9B"

    # Data sources — point at the latest dual-axis SFT corpus by default.
    # Both paths can be overridden on the CLI; ``data_loader.py`` walks
    # both legacy ``<root>/<game>/`` and the new ``<root>/<corpus>/<game>/``
    # layout so old corpora remain consumable.
    decision_data_dir: str = str(DEFAULT_DECISION_DATA_DIR)
    skillbank_data_dir: str = str(DEFAULT_SKILLBANK_DATA_DIR)
    games: List[str] = field(default_factory=lambda: list(COLDSTART_GAMES))

    # Output — adapters written to decision/ and skillbank/ subdirs
    output_dir: str = str(REPO_ROOT / "runs" / "sft_coldstart")

    # LoRA — matches trainer.coevolution.config exactly (no down_proj)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # Training
    lr: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.05
    eval_fraction: float = 0.05
    bf16: bool = True

    # Logging / saving
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2

    # Which adapters to train (subset of ALL_ADAPTERS; None = all)
    adapters: Optional[List[str]] = None

    # Per-adapter overrides (adapter_name → {param: value}).
    # Tuned for an ~12 h H200 budget: bigger micro-batch + fewer
    # accumulation steps to amortise data-loader / kernel-launch
    # overhead, with epoch counts scaled to the dataset size.
    # Effective batch size is held at 16 across all decision/skill-bank
    # adapters so loss curves stay comparable to the previous schedule.
    adapter_overrides: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            # Big decision LoRAs (~31 k samples): 2 epochs is still a
            # standard SFT budget for cold-start; bs=16, ga=1 saturates
            # the H200 BF16 path with the 9 B base model.
            "skill_selection": {"epochs": 2, "batch_size": 16, "grad_accum": 1},
            "action_taking":   {"epochs": 2, "batch_size": 16, "grad_accum": 1},
            # Segment has the longest sequences in the corpus; keep
            # bs=4 to stay well under H200 memory and bump ga=4 to
            # preserve the effective-16 batch.  Halve epochs (8 → 4).
            "segment":         {"epochs": 4, "batch_size": 4, "grad_accum": 4},
            # Contract / boundary-proposal data is short.  bs=8 ga=2
            # saturates compute without OOM risk.  Halve epochs (10 → 5).
            "contract":        {"epochs": 5, "batch_size": 8, "grad_accum": 2},
            # curator: already finished in the previous run; keep the
            # historical schedule for reproducibility if it gets retrained.
            "curator":         {"epochs": 15, "lr": 1e-4},
        }
    )

    def resolve_target_modules(self) -> List[str]:
        """Return target_modules, auto-detecting for Qwen if unset.

        Mirrors :func:`trainer.coevolution.config.prepare_adapters` so SFT
        cold-start adapters share the same shape as the GRPO loop reloads.
        """
        if self.lora_target_modules is not None:
            return self.lora_target_modules
        from transformers import AutoConfig
        model_cfg = AutoConfig.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        text_cfg = getattr(model_cfg, "text_config", model_cfg)
        text_arch = (getattr(text_cfg, "model_type", "") or "").lower()
        if "qwen3_5_moe" in text_arch:
            return [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "in_proj_qkv", "out_proj",
            ]
        if "qwen3_5" in text_arch:
            return [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "in_proj_qkv", "out_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        if "qwen" in text_arch:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
        return ["q_proj", "v_proj"]

    def adapter_output_path(self, name: str) -> Path:
        """Return the output directory for a given adapter.

        Layout matches co-evolution's ``config.adapter_path()``:
          ``<output_dir>/decision/<name>``  or
          ``<output_dir>/skillbank/<name>``
        """
        if name in DECISION_ADAPTERS:
            return Path(self.output_dir) / "decision" / name
        return Path(self.output_dir) / "skillbank" / name

    @property
    def adapters_to_train(self) -> List[str]:
        if self.adapters:
            return [a for a in self.adapters if a in ALL_ADAPTERS]
        return list(ALL_ADAPTERS)

    def effective_params(self, adapter_name: str) -> Dict[str, Any]:
        """Merge per-adapter overrides on top of the global defaults."""
        base = {
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "max_seq_length": self.max_seq_length,
        }
        overrides = self.adapter_overrides.get(adapter_name, {})
        base.update(overrides)
        return base
