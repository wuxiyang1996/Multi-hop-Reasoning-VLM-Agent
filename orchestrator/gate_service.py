"""`GateService` — composes the seven canonical gate stages.

Spec: PLAN-UNIFIED-SKILL-GATE §7.

  Stage 0 — Static          (this module, syntax / contract sanity)
  Stage 1 — Replay          (delegates to harness.ReplayValidator)
  Stage 2 — Shadow          (read-only check — orchestrator runs SHADOW
                             skills in production and we read their stats
                             from the reward log)
  Stage 3 — Transfer        (verify ≥ N domains pass)
  Stage 4 — Non-regression  (compare new bank against last release)
  Stage 5 — Promotion       (handed to PromotionOrchestrator on PASS)
  Stage 6 — Rollback/depr.  (handed to PromotionOrchestrator on regression)

This module owns Stages 0–4. Stages 5 and 6 are *promotion actions*, not
verdicts, and live in `promotion_orchestrator`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from common.enums import DOMAINS, GateStage, GateVerdict
from common.ids import new_proposal_id, schema_hash
from data_structure.extensions.bank_mutation_proposal import (
    BankMutationProposal,
    ComposeProposal,
    GeneralizeProposal,
    HypothesisProposal,
    PatchProposal,
    RetireProposal,
)
from data_structure.extensions.gate_verdict import GateVerdictPayload, StageVerdict
from data_structure.extensions.skill_episode import SkillEpisode
from data_structure.extensions.skill_evaluation import SkillEvaluationRecord
from data_structure.extensions.skill_record import SkillRecord
from harness import SkillHarness
from harness.reward_logger import RewardLogger
from orchestrator.config import GateThresholds


@dataclass
class NonRegressionResult:
    delta: float
    pre: float
    post: float
    passed: bool

    def as_stage_verdict(self, *, max_delta: float) -> StageVerdict:
        verdict = GateVerdict.PASS if self.passed else GateVerdict.FAIL
        return StageVerdict(
            stage=GateStage.NON_REGRESSION,
            verdict=verdict,
            metrics={"delta": self.delta, "pre": self.pre, "post": self.post, "max_delta": max_delta},
        )


class GateService:
    """Runs gate stages against a candidate proposal+skill."""

    def __init__(
        self,
        *,
        harness: SkillHarness,
        thresholds: Optional[GateThresholds] = None,
    ) -> None:
        self._harness = harness
        self._thresholds = thresholds or GateThresholds()

    # -- public API --------------------------------------------------------

    def evaluate(
        self,
        *,
        proposal: BankMutationProposal,
        skill: SkillRecord,
        replay_seeds: Iterable[SkillEpisode] = (),
        shadow_log: Optional[RewardLogger] = None,
        baseline_score: Optional[float] = None,
        post_score: Optional[float] = None,
    ) -> SkillEvaluationRecord:
        evaluation_id = f"eval-{new_proposal_id().split('-', 1)[1]}"
        verdicts: List[StageVerdict] = []

        verdicts.append(self._run_static(skill, proposal))
        verdicts.append(self._run_replay(skill, list(replay_seeds)))
        verdicts.append(self._run_shadow(skill, shadow_log))
        verdicts.append(self._run_transfer(skill))
        verdicts.append(self._run_non_regression(baseline_score, post_score))

        rationale, final_verdict, eligible = self._aggregate(verdicts, skill)

        verdict_payload = GateVerdictPayload(
            proposal_id=proposal.proposal_id,
            skill_id=skill.skill_id,
            skill_content_hash=skill.content_hash(),
            stages=verdicts,
            final_verdict=final_verdict,
            rationale=rationale,
            eligible_domains=eligible,
        )
        ev = SkillEvaluationRecord(
            evaluation_id=evaluation_id,
            proposal_id=proposal.proposal_id,
            skill_id=skill.skill_id,
            skill_content_hash=skill.content_hash(),
            episode_ids=[ep.episode_id for ep in replay_seeds],
            verdict=verdict_payload,
            metrics={
                f"{v.stage.value}.{k}": float(val)
                for v in verdicts
                for k, val in v.metrics.items()
            },
            started_at=time.time(),
            finished_at=time.time(),
        )
        return ev

    # -- stage 0 -----------------------------------------------------------

    def _run_static(
        self, skill: SkillRecord, proposal: BankMutationProposal
    ) -> StageVerdict:
        failures: List[str] = []
        # General-protocol invariant
        if len(set(skill.feasible_domains)) < 2:
            failures.append("feasible_domains < 2 (general-protocol invariant)")
        for d in skill.feasible_domains:
            if d not in DOMAINS:
                failures.append(f"unknown_domain={d!r}")
        # Evidence-driven invariant
        if not skill.contract.expected_evidence_roles and skill.skill_type.value != "action":
            failures.append("contract.expected_evidence_roles empty (G0)")
        # Protocol non-empty (except retirements)
        if not isinstance(proposal, RetireProposal) and not skill.protocol:
            failures.append("skill.protocol is empty")
        # Source-type sanity
        if proposal.source_type != skill.source_type:
            failures.append(
                f"source_type mismatch: proposal={proposal.source_type.value}, "
                f"skill={skill.source_type.value}"
            )
        # Lineage check (composition / repair must reference parents)
        if isinstance(proposal, (ComposeProposal,)) and not proposal.component_skill_ids:
            failures.append("ComposeProposal.component_skill_ids is empty")
        if isinstance(proposal, (GeneralizeProposal, PatchProposal)) and not getattr(proposal, "base_skill_id", ""):
            failures.append("base_skill_id is empty")
        verdict = GateVerdict.PASS if not failures else GateVerdict.FAIL
        return StageVerdict(stage=GateStage.STATIC, verdict=verdict, failures=failures)

    # -- stage 1 -----------------------------------------------------------

    def _run_replay(
        self, skill: SkillRecord, seeds: List[SkillEpisode]
    ) -> StageVerdict:
        if not seeds:
            return StageVerdict(
                stage=GateStage.REPLAY,
                verdict=GateVerdict.LIMITED_PASS,
                metrics={"n_seeds": 0.0},
                notes="no replay seeds provided",
            )
        result = self._harness.replay_validate(skill, seeds=seeds)
        return result.as_stage_verdict(threshold=self._thresholds.replay_pass_rate)

    # -- stage 2 -----------------------------------------------------------

    def _run_shadow(
        self, skill: SkillRecord, log: Optional[RewardLogger]
    ) -> StageVerdict:
        if log is None:
            return StageVerdict(
                stage=GateStage.SHADOW,
                verdict=GateVerdict.LIMITED_PASS,
                notes="no shadow log",
            )
        entries = list(log.filter(skill_id=skill.skill_id))
        if not entries:
            return StageVerdict(
                stage=GateStage.SHADOW,
                verdict=GateVerdict.LIMITED_PASS,
                notes="no shadow data yet",
            )
        rate = sum(1 for e in entries if e.success) / len(entries)
        verdict = GateVerdict.PASS if rate >= self._thresholds.shadow_pass_rate else GateVerdict.FAIL
        return StageVerdict(
            stage=GateStage.SHADOW,
            verdict=verdict,
            metrics={"shadow_pass_rate": rate, "n_shadow_episodes": float(len(entries))},
        )

    # -- stage 3 -----------------------------------------------------------

    def _run_transfer(self, skill: SkillRecord) -> StageVerdict:
        n = len(set(skill.feasible_domains))
        verdict = (
            GateVerdict.PASS
            if n >= self._thresholds.transfer_min_domains
            else GateVerdict.LIMITED_PASS
        )
        return StageVerdict(
            stage=GateStage.TRANSFER,
            verdict=verdict,
            metrics={
                "n_domains": float(n),
                "min_domains": float(self._thresholds.transfer_min_domains),
            },
        )

    # -- stage 4 -----------------------------------------------------------

    def _run_non_regression(
        self, baseline: Optional[float], post: Optional[float]
    ) -> StageVerdict:
        if baseline is None or post is None:
            return StageVerdict(
                stage=GateStage.NON_REGRESSION,
                verdict=GateVerdict.LIMITED_PASS,
                notes="no baseline / post score provided",
            )
        delta = post - baseline
        passed = delta >= -self._thresholds.non_regression_max_delta
        return NonRegressionResult(
            delta=delta, pre=baseline, post=post, passed=passed
        ).as_stage_verdict(max_delta=self._thresholds.non_regression_max_delta)

    # -- aggregate ---------------------------------------------------------

    def _aggregate(
        self, verdicts: List[StageVerdict], skill: SkillRecord
    ) -> tuple[str, GateVerdict, List[str]]:
        any_fail = any(v.verdict == GateVerdict.FAIL for v in verdicts)
        any_limited = any(v.verdict == GateVerdict.LIMITED_PASS for v in verdicts)
        if any_fail:
            failing = [v.stage.value for v in verdicts if v.verdict == GateVerdict.FAIL]
            return f"failed_stages={failing}", GateVerdict.FAIL, []
        if any_limited:
            return "promotion_to_provisional_only", GateVerdict.LIMITED_PASS, list(skill.feasible_domains)
        return "all_stages_pass", GateVerdict.PASS, list(skill.feasible_domains)


__all__ = ["GateService", "NonRegressionResult"]
