"""Risk-first induction of AGQA temporal-binding applicability.

The calibrator observes only provenance and interval geometry already frozen by
the target-native neural grounder.  It may revoke a symbolic binding, never
move an endpoint, change a relation, or create a binding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from .agqa_aggregate_temporal_transfer import AGQAAggregateTemporalBindingReceipt
from .agqa_interval_reliability_calibrator import (
    GAP_THRESHOLDS,
    SPREAD_THRESHOLDS,
    binding_geometry,
)
from .agqa_view_reliability_calibrator import VIEW_KINDS, singleton_view_kind
from .contracts import stable_hash


# Fractions of the 48-frame proxy timeline: none, 1/16, 1/8, and 1/4.
MINIMUM_MAX_INTERVAL_SPANS = (0, 3, 6, 12)


@dataclass(frozen=True)
class TemporalSupportExample:
    split: str
    task_id: str
    aggregate_authorized: bool
    singleton_view: str | None
    minimum_cross_pair_gap: int
    maximum_within_operand_endpoint_spread: int
    maximum_interval_span: int
    source_correct: bool
    target_native_correct: bool


@dataclass(frozen=True)
class TemporalSupportRule:
    allowed_singleton_views: tuple[str, ...]
    minimum_cross_pair_gap: int
    maximum_within_operand_endpoint_spread: int
    minimum_max_interval_span: int
    training_examples: int
    training_authorizations: int
    training_wins: int
    training_losses: int
    training_ties: int
    selection_authority: str
    rule_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TemporalSupportRule":
        payload = dict(value)
        payload["allowed_singleton_views"] = tuple(
            str(item) for item in payload["allowed_singleton_views"]
        )
        rule = cls(**payload)
        rule.validate()
        return rule

    def validate(self) -> None:
        body = asdict(self)
        claimed = body.pop("rule_sha256")
        if stable_hash(body) != claimed:
            raise ValueError("AGQA temporal-support rule hash mismatch")
        if tuple(sorted(self.allowed_singleton_views)) != self.allowed_singleton_views:
            raise ValueError("allowed singleton views are not canonical")
        if not set(self.allowed_singleton_views).issubset(VIEW_KINDS):
            raise ValueError("unknown view in temporal-support rule")
        if self.minimum_cross_pair_gap not in GAP_THRESHOLDS:
            raise ValueError("gap threshold escaped finite class")
        if self.maximum_within_operand_endpoint_spread not in SPREAD_THRESHOLDS:
            raise ValueError("spread threshold escaped finite class")
        if self.minimum_max_interval_span not in MINIMUM_MAX_INTERVAL_SPANS:
            raise ValueError("temporal-support threshold escaped finite class")


def maximum_interval_span(binding: AGQAAggregateTemporalBindingReceipt) -> int:
    hypotheses = binding.operand_a_hypotheses + binding.operand_b_hypotheses
    return max(
        (row.end_frame - row.start_frame for row in hypotheses), default=-1,
    )


def _evaluate(
    rows: Sequence[TemporalSupportExample], allowed: tuple[str, ...],
    gap: int, spread: int, support: int,
) -> dict[str, object]:
    allowed_set = set(allowed)
    decisions = []
    for row in rows:
        selected = (
            row.aggregate_authorized
            and (row.singleton_view is None or row.singleton_view in allowed_set)
            and row.minimum_cross_pair_gap >= gap
            and row.maximum_within_operand_endpoint_spread <= spread
            and row.maximum_interval_span >= support
        )
        left = row.source_correct if selected else row.target_native_correct
        decisions.append((selected, left, row.target_native_correct))
    wins = sum(left and not right for _, left, right in decisions)
    losses = sum(right and not left for _, left, right in decisions)
    authorizations = sum(selected for selected, _, _ in decisions)
    return {
        "allowed_singleton_views": list(allowed),
        "minimum_cross_pair_gap": gap,
        "maximum_within_operand_endpoint_spread": spread,
        "minimum_max_interval_span": support,
        "authorizations": authorizations,
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - wins - losses,
        "net_gain": wins - losses,
        "zero_observed_negative_transfer": losses == 0,
        "rule_description_length": len(allowed) + 3,
    }


def induce_temporal_support_rule(
    examples: Iterable[TemporalSupportExample],
) -> tuple[TemporalSupportRule, tuple[Mapping[str, object], ...]]:
    rows = tuple(examples)
    if not rows:
        raise ValueError("temporal-support induction requires examples")
    candidates = []
    for size in range(len(VIEW_KINDS) + 1):
        for allowed in combinations(VIEW_KINDS, size):
            for gap in GAP_THRESHOLDS:
                for spread in SPREAD_THRESHOLDS:
                    for support in MINIMUM_MAX_INTERVAL_SPANS:
                        candidates.append(
                            _evaluate(rows, allowed, gap, spread, support)
                        )
    # Negative transfer is the primary endpoint.  Among equally safe rules,
    # maximize paired gain and then coverage; the final terms are fixed MDL
    # tie-breaks, not per-task choices.
    chosen = min(
        candidates,
        key=lambda row: (
            int(row["losses"]),
            -int(row["net_gain"]),
            -int(row["wins"]),
            -int(row["authorizations"]),
            int(row["rule_description_length"]),
            int(row["minimum_cross_pair_gap"]),
            -int(row["maximum_within_operand_endpoint_spread"]),
            int(row["minimum_max_interval_span"]),
            tuple(row["allowed_singleton_views"]),
        ),
    )
    body = {
        "allowed_singleton_views": tuple(sorted(
            str(value) for value in chosen["allowed_singleton_views"]
        )),
        "minimum_cross_pair_gap": int(chosen["minimum_cross_pair_gap"]),
        "maximum_within_operand_endpoint_spread": int(
            chosen["maximum_within_operand_endpoint_spread"]
        ),
        "minimum_max_interval_span": int(chosen["minimum_max_interval_span"]),
        "training_examples": len(rows),
        "training_authorizations": int(chosen["authorizations"]),
        "training_wins": int(chosen["wins"]),
        "training_losses": int(chosen["losses"]),
        "training_ties": int(chosen["ties"]),
        "selection_authority": (
            "EXHAUSTIVE_768_RULE_FINITE_CLASS;MINIMIZE_NEGATIVE_TRANSFER_"
            "THEN_MAXIMIZE_NET_GAIN_WINS_COVERAGE;MDL_TIE_BREAK"
        ),
    }
    rule = TemporalSupportRule(**body, rule_sha256=stable_hash(body))
    rule.validate()
    return rule, tuple(candidates)


def apply_temporal_support_rule(
    binding: AGQAAggregateTemporalBindingReceipt,
    rule: TemporalSupportRule,
) -> AGQAAggregateTemporalBindingReceipt:
    binding.validate()
    rule.validate()
    if binding.authorized_relation is None:
        return binding
    singleton = singleton_view_kind(binding)
    gap, spread = binding_geometry(binding)
    support = maximum_interval_span(binding)
    reason = None
    if singleton is not None and singleton not in set(rule.allowed_singleton_views):
        reason = f"SOURCE_ABSTAIN_SINGLETON_VIEW_UNQUALIFIED:{singleton}"
    elif gap < rule.minimum_cross_pair_gap:
        reason = f"SOURCE_ABSTAIN_CROSS_PAIR_GAP_UNQUALIFIED:{gap}"
    elif spread > rule.maximum_within_operand_endpoint_spread:
        reason = f"SOURCE_ABSTAIN_ENDPOINT_SPREAD_UNQUALIFIED:{spread}"
    elif support < rule.minimum_max_interval_span:
        reason = f"SOURCE_ABSTAIN_TEMPORAL_SUPPORT_UNQUALIFIED:{support}"
    if reason is None:
        return binding
    interim = replace(
        binding, authorized_relation=None, reason=reason, receipt_sha256="",
    )
    body = asdict(interim)
    body.pop("receipt_sha256")
    result = replace(interim, receipt_sha256=stable_hash(body))
    result.validate()
    return result


def temporal_support_target_grounder_sha256(
    *, parent_grounder_sha256: str, aggregate_adapter_sha256: str,
    normalization_module_sha256: str, acquisition_collector_sha256: str,
    calibrator_module_sha256: str, calibration_artifact_sha256: str,
) -> str:
    return stable_hash({
        "schema_version": "agqa2-temporal-support-grounder-v49",
        "parent_grounder_sha256": parent_grounder_sha256,
        "aggregate_adapter_sha256": aggregate_adapter_sha256,
        "normalization_module_sha256": normalization_module_sha256,
        "acquisition_collector_sha256": acquisition_collector_sha256,
        "calibrator_module_sha256": calibrator_module_sha256,
        "calibration_artifact_sha256": calibration_artifact_sha256,
        "runtime_authority": "ABSTENTION_ONLY;CANNOT_INVENT_OR_EDIT_BINDING",
        "runtime_features": (
            "SINGLETON_VIEW;MINIMUM_CROSS_PAIR_GAP;MAXIMUM_WITHIN_OPERAND_"
            "ENDPOINT_SPREAD;MAXIMUM_INTERVAL_SPAN"
        ),
        "outcome_or_label_runtime_input": False,
    })


__all__ = [
    "MINIMUM_MAX_INTERVAL_SPANS", "TemporalSupportExample",
    "TemporalSupportRule", "apply_temporal_support_rule",
    "induce_temporal_support_rule", "maximum_interval_span",
    "temporal_support_target_grounder_sha256",
]
