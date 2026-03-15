"""GRPO training wrappers for the co-evolution loop.

Provides two independent training paths that can run concurrently on
separate GPU groups:

1. **Decision Agent GRPO** — updates ``skill_selection`` and
   ``action_taking`` LoRA adapters using per-step environment rewards.
2. **Skill Bank GRPO** — updates ``segment``, ``contract``, and
   ``curator`` LoRA adapters using stage-specific reward signals.

Both wrap the existing ``GRPOOrchestrator`` / ``GRPOLoRATrainer`` from
``skill_agents_grpo.grpo``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from trainer.coevolution.episode_runner import EpisodeResult, GRPORecord

logger = logging.getLogger(__name__)


@dataclass
class GRPOTrainStats:
    adapter: str
    n_samples: int = 0
    n_tokens: int = 0
    mean_loss: float = 0.0
    epochs: int = 0
    wall_time_s: float = 0.0


@dataclass
class GRPOStepResult:
    decision_stats: Dict[str, GRPOTrainStats] = field(default_factory=dict)
    skillbank_stats: Dict[str, GRPOTrainStats] = field(default_factory=dict)
    wall_time_s: float = 0.0
    records: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


def _collect_grpo_records(results: List[EpisodeResult]) -> Dict[str, List[GRPORecord]]:
    """Group GRPO records by adapter from all episode results."""
    records: Dict[str, List[GRPORecord]] = {
        "action_taking": [],
        "skill_selection": [],
    }
    for r in results:
        for rec in r.grpo_records:
            adapter = rec.adapter
            if adapter in records:
                records[adapter].append(rec)
    return records


class DecisionGRPOTrainer:
    """GRPO trainer for decision agent LoRAs (skill_selection + action_taking).

    Uses the ``GRPOOrchestrator`` from ``skill_agents_grpo.grpo`` with
    custom stage configs for the two decision adapters.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        adapter_dir: str = "runs/lora_adapters",
        devices: Optional[List[int]] = None,
        group_size: int = 8,
        lr: float = 5e-5,
        temperature: float = 0.7,
        kl_coeff: float = 0.05,
    ):
        self.model_name = model_name
        self.adapter_dir = adapter_dir
        self.devices = devices or [4, 5]
        self.group_size = group_size
        self.lr = lr
        self.temperature = temperature
        self.kl_coeff = kl_coeff
        self._orchestrator: Any = None
        self._llm: Any = None

    def _ensure_orchestrator(self) -> Any:
        if self._orchestrator is not None:
            return self._orchestrator

        from skill_agents_grpo.lora.config import MultiLoraConfig
        from skill_agents_grpo.lora.model import MultiLoraSkillBankLLM
        from skill_agents_grpo.grpo.orchestrator import GRPOOrchestrator
        from skill_agents_grpo.grpo.config import GRPOConfig, StageGRPOConfig

        adapter_paths = {}
        ad = Path(self.adapter_dir)
        for name in ("skill_selection", "action_taking"):
            p = ad / name
            if p.exists():
                adapter_paths[name] = str(p)

        device_str = f"cuda:{self.devices[0]}" if self.devices else "auto"
        cfg = MultiLoraConfig(
            base_model_name_or_path=self.model_name,
            adapter_paths=adapter_paths,
            allow_fallback_to_base_model=True,
            device=device_str,
        )
        self._llm = MultiLoraSkillBankLLM(cfg)

        grpo_cfg = GRPOConfig(stage_configs={
            "skill_selection": StageGRPOConfig(
                group_size=self.group_size,
                kl_coeff=min(self.kl_coeff, 0.02),
                lr=self.lr * 0.6,
                epochs_per_batch=3,
                temperature=self.temperature,
            ),
            "action_taking": StageGRPOConfig(
                group_size=self.group_size,
                kl_coeff=self.kl_coeff,
                lr=self.lr,
                epochs_per_batch=2,
                temperature=self.temperature,
            ),
        })
        self._orchestrator = GRPOOrchestrator(self._llm, grpo_cfg)
        logger.info(
            "Decision GRPO orchestrator initialized on %s "
            "(lr=%.2e, temp=%.2f, kl=%.3f)",
            device_str, self.lr, self.temperature, self.kl_coeff,
        )
        return self._orchestrator

    def train_step(
        self,
        records: Dict[str, List[GRPORecord]],
    ) -> Dict[str, GRPOTrainStats]:
        """Run one GRPO update for decision agent adapters.

        Parameters
        ----------
        records : dict
            Maps adapter name → list of ``GRPORecord`` from rollouts.
        """
        orchestrator = self._ensure_orchestrator()
        from skill_agents_grpo.grpo.buffer import GRPOSample, GRPOBuffer
        from skill_agents_grpo.lora.skill_function import SkillFunction

        adapter_map = {
            "action_taking": SkillFunction.ACTION_TAKING
            if hasattr(SkillFunction, "ACTION_TAKING")
            else "action_taking",
            "skill_selection": SkillFunction.SKILL_SELECTION
            if hasattr(SkillFunction, "SKILL_SELECTION")
            else "skill_selection",
        }

        for adapter_name, recs in records.items():
            if not recs:
                continue
            sf = adapter_map.get(adapter_name, adapter_name)

            groups: Dict[str, List[GRPORecord]] = {}
            for rec in recs:
                key = f"{rec.episode_id}_{rec.step}"
                groups.setdefault(key, []).append(rec)

            for key, group in groups.items():
                sample = GRPOSample(
                    adapter=sf,
                    prompt=group[0].prompt,
                    completions=[r.completion for r in group],
                    rewards=[r.reward for r in group],
                    metadata=group[0].metadata,
                )
                orchestrator.buffer.add(sample)

        t0 = time.monotonic()
        raw_stats = orchestrator.train_step()
        elapsed = time.monotonic() - t0

        result: Dict[str, GRPOTrainStats] = {}
        for adapter_key, stats in raw_stats.items():
            result[str(adapter_key)] = GRPOTrainStats(
                adapter=str(adapter_key),
                n_samples=stats.get("n_samples", 0),
                n_tokens=stats.get("n_tokens", 0),
                mean_loss=stats.get("mean_loss", 0.0),
                epochs=stats.get("epochs", 0),
                wall_time_s=elapsed / max(len(raw_stats), 1),
            )

        return result

    def save_adapters(self) -> None:
        """Save updated LoRA adapters to disk."""
        if self._llm is None:
            return
        ad = Path(self.adapter_dir)
        for name in ("skill_selection", "action_taking"):
            save_path = ad / name
            save_path.mkdir(parents=True, exist_ok=True)
            try:
                adapter_name = f"lora_{name}"
                loaded = getattr(self._llm, "_loaded_adapters", {})
                if adapter_name in loaded:
                    self._llm._model.set_adapter(adapter_name)
                    self._llm._model.save_pretrained(str(save_path))
                    logger.info("Saved decision adapter '%s' → %s", name, save_path)
            except Exception as exc:
                logger.warning("Save failed for '%s': %s", name, exc)


class SkillBankGRPOTrainer:
    """GRPO trainer for skill bank LoRAs (segment, contract, curator).

    Wraps the existing ``GRPOOrchestrator`` from ``skill_agents_grpo.grpo``.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        adapter_dir: str = "runs/lora_adapters",
        devices: Optional[List[int]] = None,
        lr: float = 5e-5,
        temperature: float = 0.7,
        kl_coeff: float = 0.05,
    ):
        self.model_name = model_name
        self.adapter_dir = adapter_dir
        self.devices = devices or [6, 7]
        self.lr = lr
        self.temperature = temperature
        self.kl_coeff = kl_coeff
        self._orchestrator: Any = None
        self._llm: Any = None

    def _ensure_orchestrator(self) -> Any:
        if self._orchestrator is not None:
            return self._orchestrator

        from skill_agents_grpo.lora.config import MultiLoraConfig
        from skill_agents_grpo.lora.model import MultiLoraSkillBankLLM
        from skill_agents_grpo.grpo.orchestrator import GRPOOrchestrator
        from skill_agents_grpo.grpo.config import GRPOConfig, StageGRPOConfig

        adapter_paths = {}
        ad = Path(self.adapter_dir)
        for name in ("segment", "contract", "curator"):
            p = ad / name
            if p.exists():
                adapter_paths[name] = str(p)

        device_str = f"cuda:{self.devices[0]}" if self.devices else "auto"
        cfg = MultiLoraConfig(
            base_model_name_or_path=self.model_name,
            adapter_paths=adapter_paths,
            allow_fallback_to_base_model=True,
            device=device_str,
        )
        self._llm = MultiLoraSkillBankLLM(cfg)

        grpo_cfg = GRPOConfig(stage_configs={
            "segment": StageGRPOConfig(
                group_size=4,
                kl_coeff=min(self.kl_coeff, 0.02),
                lr=self.lr * 0.6,
                epochs_per_batch=3,
                temperature=self.temperature,
            ),
            "contract": StageGRPOConfig(
                group_size=4,
                kl_coeff=self.kl_coeff,
                lr=self.lr,
                epochs_per_batch=2,
                temperature=self.temperature,
            ),
            "curator": StageGRPOConfig(
                group_size=4,
                kl_coeff=self.kl_coeff,
                lr=self.lr,
                epochs_per_batch=2,
                temperature=self.temperature,
            ),
        })
        self._orchestrator = GRPOOrchestrator(self._llm, grpo_cfg)
        logger.info(
            "Skill bank GRPO orchestrator initialized on %s "
            "(lr=%.2e, temp=%.2f, kl=%.3f)",
            device_str, self.lr, self.temperature, self.kl_coeff,
        )
        return self._orchestrator

    def train_step(
        self,
        grpo_data: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, GRPOTrainStats]:
        """Run one GRPO update for skill bank adapters.

        Parameters
        ----------
        grpo_data : dict
            Maps adapter name (segment/contract/curator) → list of
            GRPO training samples from the skill bank pipeline.
        """
        orchestrator = self._ensure_orchestrator()
        from skill_agents_grpo.grpo.buffer import GRPOSample
        from skill_agents_grpo.lora.skill_function import SkillFunction

        sf_map = {
            "segment": SkillFunction.SEGMENT,
            "contract": SkillFunction.CONTRACT,
            "curator": SkillFunction.CURATOR,
        }

        for adapter_name, samples in grpo_data.items():
            sf = sf_map.get(adapter_name)
            if sf is None or not samples:
                continue
            for s in samples:
                sample = GRPOSample(
                    adapter=sf,
                    prompt=s.get("prompt", ""),
                    completions=s.get("completions", []),
                    rewards=s.get("rewards", []),
                    metadata=s.get("metadata", {}),
                )
                orchestrator.buffer.add(sample)

        t0 = time.monotonic()
        raw_stats = orchestrator.train_step()
        elapsed = time.monotonic() - t0

        result: Dict[str, GRPOTrainStats] = {}
        for adapter_key, stats in raw_stats.items():
            result[str(adapter_key)] = GRPOTrainStats(
                adapter=str(adapter_key),
                n_samples=stats.get("n_samples", 0),
                n_tokens=stats.get("n_tokens", 0),
                mean_loss=stats.get("mean_loss", 0.0),
                epochs=stats.get("epochs", 0),
                wall_time_s=elapsed / max(len(raw_stats), 1),
            )

        return result

    def save_adapters(self) -> None:
        """Save updated skill bank LoRA adapters to disk."""
        if self._llm is None:
            return
        ad = Path(self.adapter_dir)
        for name in ("segment", "contract", "curator"):
            save_path = ad / name
            save_path.mkdir(parents=True, exist_ok=True)
            try:
                adapter_name = f"lora_{name}"
                loaded = getattr(self._llm, "_loaded_adapters", {})
                if adapter_name in loaded:
                    self._llm._model.set_adapter(adapter_name)
                    self._llm._model.save_pretrained(str(save_path))
                    logger.info("Saved skillbank adapter '%s' → %s", name, save_path)
            except Exception as exc:
                logger.warning("Save failed for '%s': %s", name, exc)


async def run_grpo_training(
    rollout_results: List[EpisodeResult],
    skillbank_grpo_data: Dict[str, List[Dict[str, Any]]],
    config: Any,
    *,
    step: int = 0,
    executor: Optional[ThreadPoolExecutor] = None,
) -> GRPOStepResult:
    """Run GRPO training for both decision agent and skill bank.

    Decision agent and skill bank GRPO are independent and can run
    concurrently on separate GPU groups.

    Parameters
    ----------
    step : int
        Current co-evolution step (used for from-scratch learning rate /
        temperature schedule).
    """
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    # Get per-step hyperparameters from the from-scratch schedule
    sched = config.grpo_schedule(step)
    lr = sched["lr"]
    temperature = sched["temperature"]
    kl_coeff = sched["kl_coeff"]
    logger.info(
        "GRPO step %d schedule: lr=%.2e, temp=%.2f, kl=%.3f",
        step, lr, temperature, kl_coeff,
    )

    decision_records = _collect_grpo_records(rollout_results)
    has_decision_data = any(len(v) > 0 for v in decision_records.values())
    has_skillbank_data = any(len(v) > 0 for v in skillbank_grpo_data.values())

    decision_stats: Dict[str, GRPOTrainStats] = {}
    skillbank_stats: Dict[str, GRPOTrainStats] = {}

    async def _train_decision():
        nonlocal decision_stats
        if not has_decision_data:
            return
        trainer = DecisionGRPOTrainer(
            model_name=config.model_name,
            adapter_dir=config.adapter_dir,
            devices=config.grpo_decision_devices,
            lr=lr,
            temperature=temperature,
            kl_coeff=kl_coeff,
        )
        decision_stats = await loop.run_in_executor(
            executor, trainer.train_step, decision_records,
        )
        await loop.run_in_executor(executor, trainer.save_adapters)

    async def _train_skillbank():
        nonlocal skillbank_stats
        if not has_skillbank_data:
            return
        trainer = SkillBankGRPOTrainer(
            model_name=config.model_name,
            adapter_dir=config.adapter_dir,
            devices=config.grpo_skillbank_devices,
            lr=lr,
            temperature=temperature,
            kl_coeff=kl_coeff,
        )
        skillbank_stats = await loop.run_in_executor(
            executor, trainer.train_step, skillbank_grpo_data,
        )
        await loop.run_in_executor(executor, trainer.save_adapters)

    await asyncio.gather(_train_decision(), _train_skillbank())

    # Collect serializable records for disk export
    all_records: Dict[str, List[Dict[str, Any]]] = {}
    for adapter_name, recs in decision_records.items():
        all_records[adapter_name] = [
            {"prompt": r.prompt, "completion": r.completion,
             "reward": r.reward, "episode_id": r.episode_id,
             "step": r.step, "adapter": r.adapter}
            for r in recs
        ]
    for adapter_name, samples in skillbank_grpo_data.items():
        all_records[adapter_name] = list(samples)

    elapsed = time.monotonic() - t0
    return GRPOStepResult(
        decision_stats=decision_stats,
        skillbank_stats=skillbank_stats,
        wall_time_s=elapsed,
        records=all_records,
    )
