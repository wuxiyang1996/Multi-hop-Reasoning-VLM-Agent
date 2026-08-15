"""Stateful target coverage gate for Sokoban-to-WebShop transfer V14.

The source program remains a domain-agnostic PREPARE/COMMIT controller.  This
module supplies the target-native grounding that the V12 experiment lacked:
WebShop option readiness is a set-coverage predicate, and an option counts as
covered only after its action produced an observed state transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .webshop_constraint_coverage_v14 import (
    ConstraintCoverage,
    constraint_signature,
    goal_option_signatures,
    ground_structured_goal_constraints,
)
from .webshop_sokoban_effect_transfer import (
    EffectTransferDecision,
    choose_sokoban_effect_action,
)


TARGET_ONLY = "target_only"
TARGET_COVERAGE = "target_native_coverage_only"
AUTHENTIC_COVERAGE = "authentic_sokoban_effect_plus_coverage_target"
COMMIT_AVAILABILITY_COVERAGE = (
    "commit_availability_control_plus_coverage_target"
)
INVERTED_COVERAGE = "inverted_effect_control_plus_coverage_target"
POSITION_PRIOR_COVERAGE = "position_prior_control_plus_coverage_target"

CONDITION_TO_BASE = {
    TARGET_ONLY: "target_only",
    TARGET_COVERAGE: "target_only",
    AUTHENTIC_COVERAGE: "authentic_sokoban_effect_plus_target",
    COMMIT_AVAILABILITY_COVERAGE: "commit_availability_control_plus_target",
    INVERTED_COVERAGE: "inverted_effect_control_plus_target",
    POSITION_PRIOR_COVERAGE: "position_prior_control_plus_target",
}
CONDITIONS = tuple(CONDITION_TO_BASE)


@dataclass
class CoverageTransferController:
    """One episode's coverage state plus a frozen source selector."""

    condition: str
    goal_options: Mapping[str, Any] = field(default_factory=dict)
    ledger: ConstraintCoverage = field(default_factory=ConstraintCoverage)
    coverage_interventions: int = 0
    source_calls_after_coverage: int = 0
    rejected_product_ids: set[str] = field(default_factory=set)
    last_selected_product_id: str | None = None
    anytime_reward_salvage: bool = False
    maximum_steps: int | None = None
    budget_abstentions: int = 0
    product_ledger_resets: int = 0

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown V14 condition: {self.condition}")
        if self.maximum_steps is not None and self.maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive")
        self.ledger.required.update(goal_option_signatures(self.goal_options))

    @property
    def coverage_enabled(self) -> bool:
        return self.condition != TARGET_ONLY

    @property
    def coverage_ready(self) -> bool:
        return not self.goal_options or self.ledger.commit_authorized

    def _record_product_selection(self, row: Mapping[str, Any]) -> None:
        """Start a product-local option ledger when search changes product.

        WebShop option clicks apply only to the currently open product.  The
        required signatures are episode-level, but evidence that an option was
        selected must never leak from one product to the next.
        """

        product_id = self._product_id(row)
        if product_id is None:
            return
        if (
            self.last_selected_product_id is not None
            and product_id != self.last_selected_product_id
        ):
            self.ledger.verified.clear()
            self.ledger.pending_signature = None
            self.product_ledger_resets += 1
        self.last_selected_product_id = product_id

    def _budget_salvage(
        self,
        *,
        semantics: Sequence[Mapping[str, Any]],
        predictions: np.ndarray,
        remaining_fraction: float,
    ) -> EffectTransferDecision | None:
        """Return a target-native partial-reward commit iff strict success is unreachable.

        The lower bound is derived from the WebShop action graph.  If a needed
        option is absent on the current product, strict completion needs at
        least: back to results, open another product, select every required
        option on that product, and commit.  No task identity or outcome is
        consulted.  A visible missing option is still selected first whenever
        there is time for one subsequent commit.
        """

        if (
            not self.anytime_reward_salvage
            or self.maximum_steps is None
            or not self.ledger.missing
        ):
            return None
        commit_indices = [
            index for index, row in enumerate(semantics) if row.get("is_commit")
        ]
        if not commit_indices:
            return None
        remaining_steps = max(
            1, int(round(float(remaining_fraction) * self.maximum_steps)),
        )
        visible_missing = {
            signature
            for row in semantics
            if (signature := constraint_signature(row)) in set(self.ledger.missing)
            and not row.get("is_selected")
        }
        missing_available_here = set(self.ledger.missing).issubset(visible_missing)
        if missing_available_here:
            strict_lower_bound = len(self.ledger.missing) + 1
        else:
            # Verified choices are product-local and must be made again after
            # opening another product.
            strict_lower_bound = len(self.ledger.required) + 3
        if remaining_steps >= strict_lower_bound:
            return None

        prepare = self.ledger.preferred_missing_index(semantics)
        if prepare is not None and remaining_steps > 1:
            return None
        reward_index = 2
        selected = max(
            commit_indices,
            key=lambda index: (float(predictions[index, reward_index]), -index),
        )
        self.budget_abstentions += 1
        return EffectTransferDecision(
            selected_index=selected,
            abstract_kind="TARGET",
            source_abstained=True,
            source_test_value=None,
            source_commit_value=None,
            reason=(
                "target_budget_infeasible_immediate_reward_salvage:"
                f"remaining={remaining_steps}:strict_lb={strict_lower_bound}"
            ),
        )

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
            raise ValueError(
                f"controller condition mismatch: {condition} != {self.condition}"
            )
        ground_structured_goal_constraints(semantics, self.goal_options)
        self.ledger.begin_decision(
            semantics, prior_action_had_no_effect=prior_no_effect,
        )

        salvage = self._budget_salvage(
            semantics=semantics,
            predictions=predictions,
            remaining_fraction=remaining_fraction,
        )
        if salvage is not None:
            self.coverage_interventions += 1
            return salvage

        if self.coverage_enabled and self.ledger.missing:
            visible_missing = {
                signature for row in semantics
                if (signature := row.get("goal_constraint_signature")) is not None
                and not row.get("is_selected")
            }
            on_product_page = any(row.get("is_commit") for row in semantics)
            if on_product_page and not set(self.ledger.missing).issubset(visible_missing):
                back = next(
                    (
                        index for index, action in enumerate(candidates)
                        if action == "go_back()"
                    ),
                    None,
                )
                if back is not None:
                    if self.last_selected_product_id is not None:
                        self.rejected_product_ids.add(self.last_selected_product_id)
                    self.coverage_interventions += 1
                    return EffectTransferDecision(
                        selected_index=back,
                        abstract_kind="POSITION",
                        source_abstained=True,
                        source_test_value=None,
                        source_commit_value=None,
                        reason="target_coverage_reject_incomplete_product",
                    )

            if self.rejected_product_ids:
                untried = next(
                    (
                        index for index, row in enumerate(semantics)
                        if (product_id := self._product_id(row)) is not None
                        and product_id not in self.rejected_product_ids
                    ),
                    None,
                )
                if untried is not None:
                    self._record_product_selection(semantics[untried])
                    self.coverage_interventions += 1
                    return EffectTransferDecision(
                        selected_index=untried,
                        abstract_kind="POSITION",
                        source_abstained=True,
                        source_test_value=None,
                        source_commit_value=None,
                        reason="target_coverage_explore_untried_product",
                    )

            selected = self.ledger.preferred_missing_index(semantics)
            if selected is not None:
                missing_before = ",".join(self.ledger.missing)
                self.ledger.record_selected(semantics[selected])
                self.coverage_interventions += 1
                return EffectTransferDecision(
                    selected_index=selected,
                    abstract_kind="POSITION",
                    source_abstained=True,
                    source_test_value=None,
                    source_commit_value=None,
                    reason=f"target_coverage_prepare:{missing_before}",
                )

        base_condition = CONDITION_TO_BASE[self.condition]
        decision = choose_sokoban_effect_action(
            condition=base_condition,
            predictions=predictions,
            semantics=semantics,
            source_models=source_models,
            visible_satisfied=(
                self.coverage_ready if self.coverage_enabled else visible_satisfied
            ),
            visible_unsatisfied=(
                not self.coverage_ready if self.coverage_enabled else visible_unsatisfied
            ),
            prior_no_effect=prior_no_effect,
            remaining_fraction=remaining_fraction,
            previous_action=previous_action,
            candidates=candidates,
            uncertainty_scale=uncertainty_scale,
            decision_margin=decision_margin,
        )

        # Fail closed: a commit may not pass merely because the candidate
        # generator omitted an executable missing constraint on this step.
        selected_semantics = semantics[decision.selected_index]
        if (
            self.coverage_enabled
            and selected_semantics.get("is_commit")
            and not self.coverage_ready
        ):
            safe = next(
                (
                    index for index, row in enumerate(semantics)
                    if not row.get("is_commit") and not row.get("is_noop")
                ),
                None,
            )
            if safe is None:
                safe = next(
                    (index for index, row in enumerate(semantics)
                     if not row.get("is_commit")),
                    None,
                )
            if safe is None:
                raise RuntimeError(
                    "coverage gate cannot realize a safe non-commit action"
                )
            decision = EffectTransferDecision(
                selected_index=safe,
                abstract_kind="POSITION",
                source_abstained=True,
                source_test_value=None,
                source_commit_value=None,
                reason="target_coverage_blocked_unverified_commit",
            )
            self.coverage_interventions += 1
        elif (
            self.condition not in {TARGET_ONLY, TARGET_COVERAGE}
            and not decision.source_abstained
        ):
            self.source_calls_after_coverage += 1

        self.ledger.record_selected(semantics[decision.selected_index])
        self._record_product_selection(semantics[decision.selected_index])
        return decision

    @staticmethod
    def _product_id(row: Mapping[str, Any]) -> str | None:
        if row.get("element_role") != "link":
            return None
        match = re.search(
            r"\blink\s+['\"]([A-Z0-9]{8,16})['\"]",
            str(row.get("element_text") or ""),
        )
        return match.group(1) if match else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "coverage_enabled": self.coverage_enabled,
            "coverage_ready": self.coverage_ready,
            "goal_option_signatures": list(goal_option_signatures(self.goal_options)),
            "coverage_interventions": self.coverage_interventions,
            "source_calls_after_coverage": self.source_calls_after_coverage,
            "rejected_product_ids": sorted(self.rejected_product_ids),
            "last_selected_product_id": self.last_selected_product_id,
            "anytime_reward_salvage": self.anytime_reward_salvage,
            "maximum_steps": self.maximum_steps,
            "budget_abstentions": self.budget_abstentions,
            "product_ledger_resets": self.product_ledger_resets,
            "ledger": self.ledger.as_dict(),
        }


__all__ = [
    "AUTHENTIC_COVERAGE",
    "COMMIT_AVAILABILITY_COVERAGE",
    "CONDITIONS",
    "CONDITION_TO_BASE",
    "CoverageTransferController",
    "INVERTED_COVERAGE",
    "POSITION_PRIOR_COVERAGE",
    "TARGET_COVERAGE",
    "TARGET_ONLY",
]
