"""`Hypothesizer` — propose net-new skills from failure patterns or rules.

Spec: PLAN-SKILL-CRAFTER §5.3.

The MVP rule: for each `FailurePattern` whose recommended strategy is
`HOP_INSERTION` or `PROTOCOL_PATCH`, propose a new skill whose body is
the *minimal* counterfactual fix (e.g. a single VERIFY hop). The teacher
LLM hook (`set_llm_proposer`) can replace this with richer proposals.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

from common.enums import DOMAINS, RecoveryStrategy
from data_structure.extensions.bank_mutation_proposal import HypothesisProposal
from data_structure.extensions.failure_trace import FailureDiagnosis
from data_structure.extensions.skill_record import SkillContract
from crafter.failure_memory import FailurePattern


# Fix-B: widen the LLM-proposer surface so the Hypothesizer can pass
# a "concepts already covered" list down — the v3 mode-collapse
# diagnosis is that the LLM had no signal pushing it to diversify
# AWAY from previously-minted hypotheses.  We use ``Callable[..., …]``
# (rather than a tighter callable signature) so legacy hooks that
# only accept ``(pattern, diagnosis)`` keep working — see
# ``Hypothesizer.propose`` for the TypeError-fallback path.
_LLMProposer = Callable[..., Optional[HypothesisProposal]]


class Hypothesizer:
    def __init__(self, llm: Optional[_LLMProposer] = None) -> None:
        self._llm = llm

    def set_llm_proposer(self, llm: _LLMProposer) -> None:
        self._llm = llm

    def propose(
        self,
        *,
        pattern: FailurePattern,
        diagnosis: FailureDiagnosis,
        target_domains: Optional[List[str]] = None,
        teacher_model: Optional[str] = None,
        existing_concepts: Optional[List[str]] = None,
    ) -> Optional[HypothesisProposal]:
        if self._llm is not None:
            try:
                proposal = self._call_llm_with_concepts(
                    pattern, diagnosis, existing_concepts,
                )
                if proposal is not None:
                    return proposal
            except Exception:                                  # noqa: BLE001
                pass
        return self._rule_propose(pattern, diagnosis, target_domains, teacher_model)

    def _call_llm_with_concepts(
        self,
        pattern: FailurePattern,
        diagnosis: FailureDiagnosis,
        existing_concepts: Optional[List[str]],
    ) -> Optional[HypothesisProposal]:
        """Two-level fallback for legacy hooks.

        New hooks (``crafter._llm_runtime.LLMHypothesizer``) accept
        the ``existing_concepts`` kw.  Legacy hooks installed by
        2026-04 callers only accept positional ``(pattern,
        diagnosis)``.  We probe the new signature first, fall back
        to the old one on ``TypeError`` — and we only swallow
        ``TypeError``, NOT generic exceptions, because the outer
        ``except Exception`` in :meth:`propose` is already there to
        absorb provider failures and we don't want to mask those by
        retrying without the kw.
        """
        if self._llm is None:
            return None
        try:
            return self._llm(
                pattern, diagnosis, existing_concepts=existing_concepts,
            )
        except TypeError:
            return self._llm(pattern, diagnosis)

    # -- rule path --------------------------------------------------------

    def _rule_propose(
        self,
        pattern: FailurePattern,
        diagnosis: FailureDiagnosis,
        target_domains: Optional[List[str]],
        teacher_model: Optional[str],
    ) -> Optional[HypothesisProposal]:
        domains = sorted({d for d in (target_domains or pattern.domains) if d in DOMAINS})
        if not domains:
            domains = list(DOMAINS[:2])  # at least 2 to satisfy general-protocol invariant

        if diagnosis.recommended_strategy == RecoveryStrategy.HOP_INSERTION:
            novel_protocol = [
                {"action": "VERIFY", "payload": {"target": "${commit_target}"}, "notes": "auto-inserted by hypothesizer"},
                {"action": "COMMIT", "payload": {"target": "${commit_target}"}},
            ]
            contract = SkillContract(
                preconditions=["have_commit_target"],
                belief_progress=["narrows(open_question)"],
                grounding_progress=["confirms(target)"],
                expected_evidence_roles=["VERIFY", "COMMIT"],
                success_criteria=["verify_passed AND commit_succeeded"],
                abort_criteria=["verify_failed"],
            )
        elif diagnosis.recommended_strategy == RecoveryStrategy.PROTOCOL_PATCH:
            novel_protocol = [
                {"action": "GROUND", "payload": {"target": "${commit_target}"}, "notes": "ensure visible"},
                {"action": "EXECUTE", "payload": {"target": "${commit_target}"}},
            ]
            contract = SkillContract(
                preconditions=["have_commit_target"],
                grounding_progress=["confirms(target)"],
                expected_evidence_roles=["GATHER"],
                success_criteria=["execute_succeeded"],
                abort_criteria=["target_not_visible"],
            )
        else:
            return None

        return HypothesisProposal(
            name=f"hyp_for_{pattern.failure_class.lower()}",
            rationale=f"hypothesis for pattern={pattern.pattern_id}: {diagnosis.root_cause}",
            parent_skill_ids=[pattern.skill_id] if pattern.skill_id else [],
            seed_failure_ids=list(pattern.failure_ids),
            target_domains=domains,
            teacher_model=teacher_model,
            novel_protocol=novel_protocol,
            contract=contract,
            source_failure_pattern_ids=[pattern.pattern_id],
            proposed_at=time.time(),
        )


__all__ = ["Hypothesizer"]
