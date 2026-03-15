"""Async skill bank pipeline wrapper for the co-evolution loop.

Wraps the synchronous ``SkillBankAgent`` pipeline (Stage 1+2 segmentation,
Stage 3 contract learning, Stage 4 bank maintenance) to run concurrently
with rollout collection.  Uses ``asyncio.Queue`` to receive completed
episodes and processes them in micro-batches through the pipeline stages.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from trainer.coevolution.episode_runner import EpisodeResult

logger = logging.getLogger(__name__)


@dataclass
class SkillBankUpdateResult:
    accepted: bool = False
    bank_version: int = 0
    n_skills: int = 0
    n_new_skills: int = 0
    n_episodes_processed: int = 0
    wall_time_s: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    grpo_data: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


class AsyncSkillBankPipeline:
    """Manages the skill bank update lifecycle across a co-evolution step.

    Receives completed episodes (via ``ingest_episode()`` or
    ``process_batch_async()``), processes them through the SkillBankAgent
    pipeline, and produces an updated bank.
    """

    def __init__(
        self,
        bank_dir: str = "runs/skillbank",
        model_name: str = "Qwen/Qwen3-14B",
        executor: Optional[ThreadPoolExecutor] = None,
        report_dir: Optional[str] = None,
    ):
        self.bank_dir = bank_dir
        self.model_name = model_name
        self._executor = executor
        self.report_dir = report_dir or str(Path(bank_dir) / "reports")
        self._agent: Any = None
        self._pending_episodes: List[Any] = []
        self._grpo_data: Dict[str, List[Dict[str, Any]]] = {
            "segment": [],
            "contract": [],
            "curator": [],
        }
        self._update_result: Optional[SkillBankUpdateResult] = None

    def _ensure_agent(self) -> Any:
        """Lazily create the SkillBankAgent."""
        if self._agent is not None:
            return self._agent

        from skill_agents_grpo.pipeline import SkillBankAgent, PipelineConfig

        bank_path = str(Path(self.bank_dir) / "skill_bank.jsonl")
        config = PipelineConfig(
            bank_path=bank_path,
            env_name="llm",
            llm_model=self.model_name,
            extractor_model=self.model_name,
            segmentation_method="dp",
            preference_iterations=1,
            new_skill_penalty=2.0,
            eff_freq=0.5,
            min_instances_per_skill=1,
            start_end_window=3,
            new_pool_min_cluster_size=1,
            new_pool_min_consistency=0.3,
            new_pool_min_distinctiveness=0.15,
            min_new_cluster_size=1,
            report_dir=self.report_dir,
        )
        self._agent = SkillBankAgent(config=config)

        if Path(bank_path).exists():
            try:
                self._agent.load()
                n = len(self._agent.skill_ids)
                logger.info("Loaded existing skill bank: %d skills", n)
            except Exception as exc:
                logger.warning("Failed to load skill bank: %s", exc)

        return self._agent

    def load_bank(self, bank: Any) -> None:
        """Inject a pre-loaded bank into the pipeline agent."""
        agent = self._ensure_agent()
        if hasattr(agent, "bank") and bank is not None:
            agent.bank = bank

    def _convert_episode_result(self, result: EpisodeResult) -> Any:
        """Convert ``EpisodeResult`` to the ``Episode`` format for the pipeline."""
        from data_structure.experience import Experience, Episode

        experiences = []
        for exp_dict in result.experiences:
            exp = Experience(
                state=exp_dict.get("state", ""),
                action=exp_dict.get("action", ""),
                reward=exp_dict.get("reward", 0.0),
                next_state=exp_dict.get("next_state", ""),
                done=exp_dict.get("done", False),
                intentions=exp_dict.get("intention"),
            )
            exp.idx = exp_dict.get("step", 0)
            exp.summary_state = exp_dict.get("summary_state", "")
            exp.action_type = "primitive"
            exp.interface = {"env_name": "gamingagent", "game_name": result.game}
            experiences.append(exp)

        episode = Episode(
            experiences=experiences,
            task=f"Play {result.game}",
            env_name="gamingagent",
            game_name=result.game,
            episode_id=result.episode_id,
            metadata={
                "done": result.terminated or result.truncated,
                "steps": result.steps,
                "total_reward": result.total_reward,
            },
        )
        episode.set_outcome()
        return episode

    async def ingest_episode(self, result: EpisodeResult) -> None:
        """Convert and queue a completed episode for processing."""
        if result.steps == 0:
            return
        episode = self._convert_episode_result(result)
        self._pending_episodes.append(episode)

    async def process_batch_async(
        self,
        results: List[EpisodeResult],
    ) -> None:
        """Process a micro-batch of completed episodes through Stages 1+2.

        Runs synchronous pipeline code in a thread executor to avoid
        blocking the event loop.
        """
        episodes = []
        for r in results:
            if r.steps > 0:
                episodes.append(self._convert_episode_result(r))

        if not episodes:
            return

        agent = self._ensure_agent()
        loop = asyncio.get_running_loop()

        def _segment_batch():
            for ep in episodes:
                try:
                    result, sub_eps = agent.segment_episode(ep, env_name="llm")
                    n_segs = len(result.segments) if hasattr(result, "segments") else 0
                    logger.debug(
                        "Segmented %s: %d steps → %d segments",
                        ep.episode_id, len(ep.experiences), n_segs,
                    )
                except Exception as exc:
                    logger.warning("Segmentation failed for %s: %s", ep.episode_id, exc)

        executor = self._executor
        await loop.run_in_executor(executor, _segment_batch)
        self._pending_episodes.extend(episodes)

    async def finalize_update(self) -> SkillBankUpdateResult:
        """Run contract learning + bank maintenance after all episodes ingested.

        Returns the update result with bank metrics.
        """
        agent = self._ensure_agent()
        loop = asyncio.get_running_loop()
        executor = self._executor
        t0 = time.monotonic()
        stage_times: Dict[str, float] = {}

        n_episodes = len(self._pending_episodes)
        n_skills_before = len(agent.skill_ids)

        # Stage 3: Contract learning
        t_s3 = time.monotonic()

        def _run_contracts():
            if agent._all_segments:
                try:
                    return agent.run_contract_learning()
                except Exception as exc:
                    logger.warning("Contract learning failed: %s", exc)
            return None

        s3_result = await loop.run_in_executor(executor, _run_contracts)
        stage_times["contract_learning"] = time.monotonic() - t_s3

        # Stage 4: Bank maintenance
        t_s4 = time.monotonic()

        def _run_maintenance():
            if agent._all_segments and len(agent.skill_ids) > 0:
                try:
                    return agent.run_bank_maintenance()
                except Exception as exc:
                    logger.warning("Bank maintenance failed: %s", exc)
            return None

        s4_result = await loop.run_in_executor(executor, _run_maintenance)
        stage_times["bank_maintenance"] = time.monotonic() - t_s4

        # Proto-skill materialization
        t_mat = time.monotonic()

        def _materialize():
            try:
                n_formed = agent.form_proto_skills()
                n_verified = agent.verify_proto_skills()
                n_promoted = agent.promote_proto_skills()
                n_materialized = agent.materialize_new_skills()
                return {
                    "formed": n_formed, "verified": n_verified,
                    "promoted": n_promoted, "materialized": n_materialized,
                }
            except Exception as exc:
                logger.warning("Proto-skill processing failed: %s", exc)
                return {}

        mat_result = await loop.run_in_executor(executor, _materialize)
        stage_times["materialization"] = time.monotonic() - t_mat

        # Save bank
        def _save_bank():
            try:
                agent.save()
            except Exception as exc:
                logger.warning("Bank save failed: %s", exc)

        await loop.run_in_executor(executor, _save_bank)

        n_skills_after = len(agent.skill_ids)
        elapsed = time.monotonic() - t0

        self._update_result = SkillBankUpdateResult(
            accepted=True,
            bank_version=getattr(agent, "_iteration_count", 0),
            n_skills=n_skills_after,
            n_new_skills=max(0, n_skills_after - n_skills_before),
            n_episodes_processed=n_episodes,
            wall_time_s=elapsed,
            stage_times=stage_times,
            grpo_data=self._grpo_data,
        )

        logger.info(
            "Skill bank update: %d→%d skills (+%d), %d episodes, %.1fs",
            n_skills_before, n_skills_after,
            self._update_result.n_new_skills, n_episodes, elapsed,
        )

        return self._update_result

    def get_bank(self) -> Any:
        """Return the current skill bank object."""
        if self._agent is not None:
            return self._agent.bank
        return None

    def get_agent(self) -> Any:
        """Return the SkillBankAgent instance."""
        return self._agent

    @property
    def grpo_data(self) -> Dict[str, List[Dict[str, Any]]]:
        return self._grpo_data

    def reset_for_step(self) -> None:
        """Clear per-step state (pending episodes, GRPO data)."""
        self._pending_episodes.clear()
        self._grpo_data = {"segment": [], "contract": [], "curator": []}
        self._update_result = None
        if self._agent is not None:
            self._agent._all_segments = []
            self._agent._new_pool = []
