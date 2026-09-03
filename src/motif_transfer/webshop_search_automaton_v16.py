"""WebShop target-native coverage binding for the V16 search automaton."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import stable_hash
from .search_automaton_transfer_v16 import (
    SourceSearchAutomaton,
    bind_native_action,
    ground_target_event,
)
from .sokoban_search_automaton_v16 import BACKTRACK, COMMIT, EXPLORE
from .webshop_coverage_transfer_v14 import (
    TARGET_COVERAGE,
    TARGET_ONLY,
    CoverageTransferController,
)
from .webshop_sokoban_effect_transfer import (
    EffectTransferDecision,
    choose_sokoban_effect_action,
)


RAW = "raw_target_only"
AUTHENTIC = "authentic_search_automaton_plus_target"
PERMUTED = "event_binding_permuted_control"
LEDGER_BLIND = "ledger_blind_control"
CEILING = "target_native_search_ceiling"
CONDITIONS = (RAW, AUTHENTIC, PERMUTED, LEDGER_BLIND, CEILING)


@dataclass
class WebShopSearchAutomatonController:
    condition: str
    source: SourceSearchAutomaton
    episode_id: str
    goal_options: Mapping[str, Any] = field(default_factory=dict)
    maximum_steps: int = 12
    source_trace: list[dict[str, Any]] = field(default_factory=list)
    target_fallbacks: int = 0

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown WebShop V16 condition: {self.condition}")
        self.coverage = CoverageTransferController(
            TARGET_ONLY if self.condition in {RAW, PERMUTED} else TARGET_COVERAGE,
            goal_options=self.goal_options,
            anytime_reward_salvage=self.condition in {
                AUTHENTIC, LEDGER_BLIND, CEILING,
            },
            maximum_steps=self.maximum_steps,
        )

    def _target_fallback(self, **kwargs: Any) -> EffectTransferDecision:
        return choose_sokoban_effect_action(
            condition="target_only",
            predictions=kwargs["predictions"],
            semantics=kwargs["semantics"],
            source_models=kwargs["source_models"],
            visible_satisfied=kwargs["visible_satisfied"],
            visible_unsatisfied=kwargs["visible_unsatisfied"],
            prior_no_effect=kwargs["prior_no_effect"],
            remaining_fraction=kwargs["remaining_fraction"],
            previous_action=kwargs["previous_action"],
            candidates=kwargs["candidates"],
            uncertainty_scale=kwargs["uncertainty_scale"],
            decision_margin=kwargs["decision_margin"],
        )

    @staticmethod
    def _event_name(
        decision: EffectTransferDecision,
        semantics: Sequence[Mapping[str, Any]],
        *,
        coverage_ready: bool,
    ) -> tuple[str, str]:
        selected = semantics[decision.selected_index]
        if decision.reason == "target_coverage_reject_incomplete_product":
            return "REFUTED", "target_product_constraint_set_refuted"
        if selected.get("is_commit"):
            if not coverage_ready:
                raise RuntimeError("target coverage proposed an unverified commit")
            return "VERIFIED", "target_constraint_set_coverage_verified"
        return "UNBOUND", "target_untried_product_or_constraint_candidate"

    def __call__(
        self,
        *,
        condition: str,
        predictions: np.ndarray,
        semantics: Sequence[Mapping[str, Any]],
        source_models: Mapping[str, Any],
        visible_satisfied: bool,
        visible_unsatisfied: bool,
        prior_no_effect: bool,
        remaining_fraction: float,
        previous_action: str | None,
        candidates: Sequence[str],
        uncertainty_scale: float,
        decision_margin: float,
    ) -> EffectTransferDecision:
        if condition != self.condition:
            raise ValueError("WebShop V16 controller condition mismatch")
        kwargs = {
            "predictions": predictions,
            "semantics": semantics,
            "source_models": source_models,
            "visible_satisfied": visible_satisfied,
            "visible_unsatisfied": visible_unsatisfied,
            "prior_no_effect": prior_no_effect,
            "remaining_fraction": remaining_fraction,
            "previous_action": previous_action,
            "candidates": candidates,
            "uncertainty_scale": uncertainty_scale,
            "decision_margin": decision_margin,
        }
        if self.condition == LEDGER_BLIND:
            self.coverage.ledger.verified.clear()
            self.coverage.ledger.pending_signature = None
            self.coverage.rejected_product_ids.clear()
            self.coverage.last_selected_product_id = None
        proposal = self.coverage(
            condition=self.coverage.condition,
            **kwargs,
        )
        if self.condition in {RAW, CEILING}:
            return proposal
        if proposal.reason.startswith(
            "target_budget_infeasible_immediate_reward_salvage:"
        ):
            # The source automaton has no authority to relax a verified commit
            # predicate.  When exact success is no longer reachable, it
            # explicitly abstains and the target-native outcome model chooses
            # the best immediate-reward terminal action.
            self.target_fallbacks += 1
            return proposal
        if self.condition == PERMUTED:
            # The permuted source event cannot safely authorize a WebShop
            # action.  Fail closed to the raw target policy while preserving a
            # receipt that source authority was denied.
            fallback = self._target_fallback(**kwargs)
            event = ground_target_event(
                domain="webshop",
                episode_id=self.episode_id,
                decision_index=len(self.source_trace),
                untried_candidate_available=False,
                active_candidate_refuted=True,
                terminal_commit_verified=False,
                evidence_kind="permuted_unbound_as_refuted_control",
                evidence_payload={
                    "target_state_sha256": stable_hash([
                        dict(row) for row in semantics
                    ]),
                },
                grounding_confidence=1.0,
            )
            assert event is not None
            binding = bind_native_action(
                event,
                abstract_action=BACKTRACK,
                native_action_id="abstain_to_raw_target",
                native_action={"operation": "target_fallback"},
                grounding_confidence=1.0,
            )
            self.source_trace.append(asdict(self.source.route(
                event, {BACKTRACK: binding},
            )))
            self.target_fallbacks += 1
            return EffectTransferDecision(
                selected_index=fallback.selected_index,
                abstract_kind=fallback.abstract_kind,
                source_abstained=True,
                source_test_value=None,
                source_commit_value=None,
                reason="event_permuted_source_abstain_to_raw_target",
            )

        event_name, evidence_kind = self._event_name(
            proposal, semantics, coverage_ready=self.coverage.coverage_ready,
        )
        abstract_action = {
            "UNBOUND": EXPLORE,
            "REFUTED": BACKTRACK,
            "VERIFIED": COMMIT,
        }[event_name]
        selected_index = proposal.selected_index
        event = ground_target_event(
            domain="webshop",
            episode_id=self.episode_id,
            decision_index=len(self.source_trace),
            untried_candidate_available=event_name == "UNBOUND",
            active_candidate_refuted=event_name == "REFUTED",
            terminal_commit_verified=event_name == "VERIFIED",
            evidence_kind=evidence_kind,
            evidence_payload={
                "target_candidate_id": stable_hash(str(candidates[selected_index])),
                "target_semantics_sha256": stable_hash(dict(semantics[selected_index])),
                "coverage_ledger_sha256": stable_hash(self.coverage.ledger.as_dict()),
            },
            grounding_confidence=1.0,
        )
        assert event is not None
        binding = bind_native_action(
            event,
            abstract_action=abstract_action,
            native_action_id=stable_hash(str(candidates[selected_index])),
            native_action=str(candidates[selected_index]),
            grounding_confidence=1.0,
        )
        routed = asdict(self.source.route(event, {abstract_action: binding}))
        self.source_trace.append(routed)
        if not routed["admitted"]:
            fallback = self._target_fallback(**kwargs)
            self.target_fallbacks += 1
            return EffectTransferDecision(
                selected_index=fallback.selected_index,
                abstract_kind=fallback.abstract_kind,
                source_abstained=True,
                source_test_value=None,
                source_commit_value=None,
                reason="source_abstain_to_raw_target",
            )
        return EffectTransferDecision(
            selected_index=selected_index,
            abstract_kind={
                EXPLORE: "POSITION",
                BACKTRACK: "REPLAN",
                COMMIT: "COMMIT",
            }[abstract_action],
            source_abstained=False,
            source_test_value=None,
            source_commit_value=None,
            reason=f"v16:{event_name}->{abstract_action}:{proposal.reason}",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "source_artifact_sha256": self.source.artifact_sha256,
            "source_decisions": len(self.source_trace),
            "source_action_counts": dict(sorted(Counter(
                row["source_action"] for row in self.source_trace
                if row["admitted"]
            ).items())),
            "target_fallbacks": self.target_fallbacks,
            "coverage_controller": self.coverage.as_dict(),
            "source_trace": list(self.source_trace),
        }

__all__ = [
    "AUTHENTIC",
    "CEILING",
    "CONDITIONS",
    "LEDGER_BLIND",
    "PERMUTED",
    "RAW",
    "WebShopSearchAutomatonController",
]
