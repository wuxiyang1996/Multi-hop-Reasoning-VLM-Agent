"""`SkillHarness` — the actual runtime.

Public API (PLAN-HARNESS §5.2 & §8 unified skill interface):

    h = SkillHarness(registry, reward_logger=..., config=...)

    # 1. Narrow the bank-supplied candidate list to runnable skills.
    eligible = h.select_eligible_skills(candidates, state, skill_type_hint=...)

    # 2. Execute one chosen skill (chosen by the Actor, not by us).
    episode = h.run_skill(skill, state, parent_run_id=..., bindings=...)

    # 3. Replay-validate a proposed skill against stored seeds (gate Stage 1).
    result = h.replay_validate(skill, seeds=...)

The harness writes nothing to the bank. It writes SkillEpisodes to the
RewardLogger (which is also the orchestrator's artifact source) and
returns failure traces when execution aborts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from common.enums import SkillStatus, SkillType
from common.ids import new_episode_id
from common.state_schema import StateSchema
from data_structure.extensions.failure_trace import FailureTrace
from data_structure.extensions.skill_episode import (
    SkillEpisode,
    SkillEpisodeOutcome,
    SkillEpisodeStep,
)
from data_structure.extensions.skill_record import SkillRecord
from harness.adapter_registry import AdapterRegistry
from harness.eligibility import EligibilityFilter, EligibleSkill
from harness.replay_validator import ReplayResult, ReplayValidator
from harness.reward_logger import RewardLogger
from harness.skill_adapter import AdapterRunContext, AdapterRunResult


@dataclass
class HarnessConfig:
    allow_shadow: bool = True
    default_budget_tokens: float = 4096.0
    default_budget_hops: float = 8.0
    default_budget_ms: float = 30_000.0
    fail_on_missing_adapter: bool = True
    seed: Optional[int] = None


class SkillHarness:
    """Per-invocation runtime for skill execution and verification."""

    def __init__(
        self,
        registry: AdapterRegistry,
        *,
        reward_logger: Optional[RewardLogger] = None,
        config: Optional[HarnessConfig] = None,
    ) -> None:
        self._registry = registry
        self._logger = reward_logger or RewardLogger()
        self._config = config or HarnessConfig()
        self._eligibility = EligibilityFilter(registry, allow_shadow=self._config.allow_shadow)
        self._replay = ReplayValidator(registry)
        self._failure_sink: List[FailureTrace] = []

    # ---------------------------------------------------------------- API

    def select_eligible_skills(
        self,
        candidates: Iterable[SkillRecord],
        state: StateSchema,
        *,
        skill_type_hint: Optional[SkillType] = None,
    ) -> List[EligibleSkill]:
        return self._eligibility.filter(candidates, state, skill_type_hint=skill_type_hint)

    def run_skill(
        self,
        skill: SkillRecord,
        state: StateSchema,
        *,
        parent_run_id: Optional[str],
        bindings: Optional[Dict[str, Any]] = None,
        budget: Optional[Dict[str, float]] = None,
        parent_episode_id: Optional[str] = None,
    ) -> SkillEpisode:
        adapter = self._registry.get(state.domain, skill.skill_type)
        episode = SkillEpisode.begin(
            skill_id=skill.skill_id,
            skill_version=skill.version,
            skill_type=skill.skill_type,
            domain=state.domain,
            parent_run_id=parent_run_id,
            initial_state=state,
            parent_episode_id=parent_episode_id,
        )
        episode.started_at = time.time()

        if adapter is None:
            outcome = SkillEpisodeOutcome(
                success=False,
                contract_satisfied=False,
                abort_reason=f"no_adapter_for({state.domain},{skill.skill_type.value})",
            )
            episode.finalize(outcome=outcome, final_state=state)
            episode.finished_at = time.time()
            self._logger.log_episode(episode)
            self._record_failure(episode, "MISSING_ADAPTER")
            if self._config.fail_on_missing_adapter:
                return episode
            return episode

        ctx = AdapterRunContext(
            state=state,
            bindings=dict(bindings or {}),
            parent_run_id=parent_run_id,
            parent_episode_id=parent_episode_id,
            budget=self._effective_budget(budget),
            seed=self._config.seed,
            dry_run=False,
        )
        try:
            result = adapter.run(skill, ctx)
        except Exception as exc:                          # noqa: BLE001
            outcome = SkillEpisodeOutcome(
                success=False,
                contract_satisfied=False,
                abort_reason=f"adapter_exception: {exc!r}",
            )
            episode.finalize(outcome=outcome, final_state=state)
            episode.finished_at = time.time()
            self._logger.log_episode(episode)
            self._record_failure(episode, "ADAPTER_EXCEPTION")
            return episode

        # Translate adapter steps -> SkillEpisodeSteps (lossy but
        # invariant-preserving).
        for i, raw in enumerate(result.steps):
            step = SkillEpisodeStep(
                step_index=i,
                action_type=str(raw.get("action_type", "STEP")),
                action_payload=dict(raw.get("payload", {})),
                pre_state=raw.get("pre_state"),
                post_state=raw.get("post_state"),
                evidence=list(raw.get("evidence", [])),
                notes=str(raw.get("notes", "")),
            )
            episode.add_step(step)

        evidence_role = sorted({e.role for e in result.new_evidence}) or [
            r for r in skill.contract.expected_evidence_roles if r
        ]
        outcome = SkillEpisodeOutcome(
            success=bool(result.success),
            contract_satisfied=bool(result.contract_satisfied),
            abort_reason=result.abort_reason,
            evidence_role=evidence_role,
            answer=result.answer,
            score=result.score,
            extra=dict(result.extra),
        )
        episode.cost.update(result.cost)
        episode.transfer_label = result.diagnostic_label
        try:
            episode.finalize(outcome=outcome, final_state=result.final_state or state)
        except ValueError as exc:
            # G0 violation — record as failure & flip success false rather
            # than corrupt the episode.
            outcome.success = False
            outcome.contract_satisfied = False
            outcome.abort_reason = (outcome.abort_reason or "") + f"|invariant:{exc}"
            episode.finalize(outcome=outcome, final_state=result.final_state or state)
        episode.finished_at = time.time()
        self._logger.log_episode(episode)
        if not outcome.success:
            self._record_failure(episode, _classify_abort(outcome.abort_reason))
        return episode

    def replay_validate(
        self,
        skill: SkillRecord,
        seeds: List[SkillEpisode],
        *,
        budget: Optional[Dict[str, float]] = None,
    ) -> ReplayResult:
        return self._replay.validate(skill=skill, seeds=seeds, budget=budget)

    # -- accessors ---------------------------------------------------------

    @property
    def reward_logger(self) -> RewardLogger:
        return self._logger

    @property
    def adapter_registry(self) -> AdapterRegistry:
        return self._registry

    def drain_failures(self) -> List[FailureTrace]:
        out, self._failure_sink = self._failure_sink, []
        return out

    # -- helpers -----------------------------------------------------------

    def _effective_budget(self, override: Optional[Dict[str, float]]) -> Dict[str, float]:
        budget = {
            "tokens": self._config.default_budget_tokens,
            "hops": self._config.default_budget_hops,
            "ms": self._config.default_budget_ms,
        }
        if override:
            budget.update(override)
        return budget

    def _record_failure(self, episode: SkillEpisode, failure_class: str) -> None:
        last_step = episode.steps[-1] if episode.steps else None
        trace = FailureTrace(
            skill_id=episode.skill_id,
            skill_episode_id=episode.episode_id,
            domain=episode.domain,
            failed_step_index=last_step.step_index if last_step else None,
            failure_class=failure_class,
            abort_reason=episode.outcome.abort_reason if episode.outcome else None,
            pre_state=episode.initial_state,
            failed_step=last_step.to_json() if last_step else None,
            observed_evidence_roles=(
                list(episode.outcome.evidence_role) if episode.outcome else []
            ),
        )
        self._failure_sink.append(trace)


def _classify_abort(reason: Optional[str]) -> str:
    if not reason:
        return "UNKNOWN"
    r = reason.lower()
    if "precondition" in r:
        return "PRECONDITION_VIOLATION"
    if "invariant" in r:
        return "INVARIANT_VIOLATION"
    if "budget" in r:
        return "BUDGET_EXCEEDED"
    if "no_adapter" in r:
        return "MISSING_ADAPTER"
    if "exception" in r:
        return "ADAPTER_EXCEPTION"
    return "OTHER"


__all__ = ["HarnessConfig", "SkillHarness"]
