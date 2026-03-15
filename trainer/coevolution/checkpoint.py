"""Checkpointing and resume for the co-evolution loop.

Saves a full snapshot every ``checkpoint_interval`` steps:
  - Skill bank (``skill_bank.jsonl``)
  - All 5 LoRA adapter weights
  - Step metadata (step number, bank version, metrics)

Supports resuming from any checkpoint.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ADAPTER_NAMES = [
    "skill_selection",
    "action_taking",
    "segment",
    "contract",
    "curator",
]


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    *,
    bank_agent: Any = None,
    adapter_dir: str = "runs/lora_adapters",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a co-evolution checkpoint.

    Creates ``{checkpoint_dir}/step_{step:04d}/`` with:
      - ``skill_bank.jsonl`` — current bank state
      - ``adapters/{name}/`` — LoRA adapter weights for each adapter
      - ``metadata.json`` — step info, bank version, metrics
    """
    ckpt_path = Path(checkpoint_dir) / f"step_{step:04d}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Save skill bank
    if bank_agent is not None:
        bank_path = ckpt_path / "skill_bank.jsonl"
        try:
            bank = getattr(bank_agent, "bank", bank_agent)
            if hasattr(bank, "save"):
                bank.save(str(bank_path))
                logger.info("Saved bank to %s", bank_path)
        except Exception as exc:
            logger.warning("Bank save failed: %s", exc)

    # Copy LoRA adapters
    adapters_dir = ckpt_path / "adapters"
    adapters_dir.mkdir(exist_ok=True)
    src_dir = Path(adapter_dir)
    for name in ADAPTER_NAMES:
        src = src_dir / name
        if src.exists() and src.is_dir():
            dst = adapters_dir / name
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            except Exception as exc:
                logger.warning("Adapter copy failed for '%s': %s", name, exc)

    # Save metadata
    meta = {
        "step": step,
        "timestamp": time.time(),
        "adapter_names": ADAPTER_NAMES,
    }
    if metadata:
        meta.update(metadata)
    meta_path = ckpt_path / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info("Checkpoint saved: step %d → %s", step, ckpt_path)
    return ckpt_path


def load_checkpoint(
    checkpoint_dir: str,
    step: int,
    *,
    adapter_dir: str = "runs/lora_adapters",
    bank_agent: Any = None,
) -> Dict[str, Any]:
    """Load a co-evolution checkpoint.

    Restores adapter weights to ``adapter_dir`` and optionally reloads
    the skill bank.

    Returns the metadata dict.
    """
    ckpt_path = Path(checkpoint_dir) / f"step_{step:04d}"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load metadata
    meta_path = ckpt_path / "metadata.json"
    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    # Restore adapters
    adapters_dir = ckpt_path / "adapters"
    dst_dir = Path(adapter_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_NAMES:
        src = adapters_dir / name
        if src.exists() and src.is_dir():
            dst = dst_dir / name
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                logger.info("Restored adapter '%s' from checkpoint", name)
            except Exception as exc:
                logger.warning("Adapter restore failed for '%s': %s", name, exc)

    # Restore skill bank
    bank_path = ckpt_path / "skill_bank.jsonl"
    if bank_path.exists() and bank_agent is not None:
        try:
            bank = getattr(bank_agent, "bank", bank_agent)
            if hasattr(bank, "load"):
                bank.load(str(bank_path))
                logger.info("Restored bank from checkpoint: %s", bank_path)
        except Exception as exc:
            logger.warning("Bank restore failed: %s", exc)

    logger.info("Checkpoint loaded: step %d from %s", step, ckpt_path)
    return metadata


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[int]:
    """Find the latest checkpoint step number, or None if no checkpoints exist."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    steps = []
    for d in ckpt_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                step = int(d.name.split("_")[1])
                meta = d / "metadata.json"
                if meta.exists():
                    steps.append(step)
            except (ValueError, IndexError):
                pass

    return max(steps) if steps else None


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last: int = 5,
) -> List[int]:
    """Remove old checkpoints, keeping the most recent ``keep_last``."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return []

    steps = []
    for d in ckpt_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                step = int(d.name.split("_")[1])
                steps.append(step)
            except (ValueError, IndexError):
                pass

    if len(steps) <= keep_last:
        return []

    steps.sort()
    to_remove = steps[:-keep_last]
    removed = []
    for step in to_remove:
        p = ckpt_dir / f"step_{step:04d}"
        try:
            shutil.rmtree(p)
            removed.append(step)
        except Exception as exc:
            logger.warning("Failed to remove checkpoint step_%04d: %s", step, exc)

    return removed
