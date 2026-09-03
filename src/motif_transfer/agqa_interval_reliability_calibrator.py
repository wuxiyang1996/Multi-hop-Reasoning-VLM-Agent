"""Induce an abstention-only interval reliability rule for AGQA.

The finite hypothesis class combines neural-view provenance with two geometric
quantities already present in a frozen binding: minimum strict cross-operand
gap and maximum within-operand endpoint spread.  It cannot move an endpoint,
change a relation, or create a binding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from .agqa_aggregate_temporal_transfer import (
    AGQAAggregateTemporalBindingReceipt,
)
from .agqa_view_reliability_calibrator import VIEW_KINDS, singleton_view_kind
from .contracts import stable_hash


GAP_THRESHOLDS = (1, 2, 4, 8)
SPREAD_THRESHOLDS = (8, 16, 24, 32, 40, 47)


@dataclass(frozen=True)
class IntervalReliabilityExample:
    task_id: str
    aggregate_authorized: bool
    singleton_view: str | None
    minimum_cross_pair_gap: int
    maximum_within_operand_endpoint_spread: int
    source_correct: bool
    target_native_correct: bool


@dataclass(frozen=True)
class IntervalReliabilityRule:
    allowed_singleton_views: tuple[str, ...]
    minimum_cross_pair_gap: int
    maximum_within_operand_endpoint_spread: int
    training_examples: int
    training_authorizations: int
    training_wins: int
    training_losses: int
    training_ties: int
    selection_authority: str
    rule_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "IntervalReliabilityRule":
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
            raise ValueError("AGQA interval-reliability rule hash mismatch")
        if tuple(sorted(self.allowed_singleton_views)) != (
            self.allowed_singleton_views
        ):
            raise ValueError("allowed singleton views are not canonical")
        if not set(self.allowed_singleton_views).issubset(VIEW_KINDS):
            raise ValueError("unknown view in interval-reliability rule")
        if self.minimum_cross_pair_gap not in GAP_THRESHOLDS:
            raise ValueError("gap threshold escaped finite class")
        if self.maximum_within_operand_endpoint_spread not in SPREAD_THRESHOLDS:
            raise ValueError("spread threshold escaped finite class")


def binding_geometry(
    binding: AGQAAggregateTemporalBindingReceipt,
) -> tuple[int, int]:
    if not binding.cross_view_relations:
        return -1, 99
    gaps = []
    for left in binding.operand_a_hypotheses:
        for right in binding.operand_b_hypotheses:
            if left.end_frame < right.start_frame:
                gaps.append(right.start_frame - left.end_frame)
            elif right.end_frame < left.start_frame:
                gaps.append(left.start_frame - right.end_frame)
            else:
                return -1, 99

    def spread(rows) -> int:
        if not rows:
            return 99
        return max(
            max(row.start_frame for row in rows)
            - min(row.start_frame for row in rows),
            max(row.end_frame for row in rows)
            - min(row.end_frame for row in rows),
        )

    return min(gaps), max(
        spread(binding.operand_a_hypotheses),
        spread(binding.operand_b_hypotheses),
    )


def _evaluate(
    rows: Sequence[IntervalReliabilityExample],
    allowed: tuple[str, ...], gap: int, spread: int,
) -> dict[str, object]:
    allowed_set = set(allowed)
    decisions = []
    for row in rows:
        selected = (
            row.aggregate_authorized
            and (row.singleton_view is None or row.singleton_view in allowed_set)
            and row.minimum_cross_pair_gap >= gap
            and row.maximum_within_operand_endpoint_spread <= spread
        )
        left = row.source_correct if selected else row.target_native_correct
        right = row.target_native_correct
        decisions.append((selected, left, right))
    wins = sum(left and not right for _, left, right in decisions)
    losses = sum(right and not left for _, left, right in decisions)
    authorizations = sum(selected for selected, _, _ in decisions)
    return {
        "allowed_singleton_views": list(allowed),
        "minimum_cross_pair_gap": gap,
        "maximum_within_operand_endpoint_spread": spread,
        "authorizations": authorizations,
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - wins - losses,
        "net_gain": wins - losses,
        "negative_transfer_gate": losses <= 1,
        "rule_description_length": len(allowed) + 2,
    }


def induce_interval_reliability_rule(
    examples: Iterable[IntervalReliabilityExample],
) -> tuple[IntervalReliabilityRule, tuple[Mapping[str, object], ...]]:
    rows = tuple(examples)
    if not rows:
        raise ValueError("interval-reliability induction requires examples")
    candidates = []
    for size in range(len(VIEW_KINDS) + 1):
        for allowed in combinations(VIEW_KINDS, size):
            for gap in GAP_THRESHOLDS:
                for spread in SPREAD_THRESHOLDS:
                    candidates.append(_evaluate(rows, allowed, gap, spread))
    eligible = [row for row in candidates if row["negative_transfer_gate"]]
    if not eligible:
        raise ValueError("no finite interval-reliability rule passed loss gate")
    chosen = min(
        eligible,
        key=lambda row: (
            -int(row["net_gain"]),
            -int(row["wins"]),
            int(row["losses"]),
            -int(row["authorizations"]),
            int(row["rule_description_length"]),
            int(row["minimum_cross_pair_gap"]),
            -int(row["maximum_within_operand_endpoint_spread"]),
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
        "training_examples": len(rows),
        "training_authorizations": int(chosen["authorizations"]),
        "training_wins": int(chosen["wins"]),
        "training_losses": int(chosen["losses"]),
        "training_ties": int(chosen["ties"]),
        "selection_authority": (
            "EXHAUSTIVE_192_RULE_FINITE_CLASS;MAX_ONE_NEGATIVE_TRANSFER;"
            "MAX_NET_GAIN_THEN_WINS_AND_COVERAGE;MDL_TIE_BREAK"
        ),
    }
    rule = IntervalReliabilityRule(**body, rule_sha256=stable_hash(body))
    rule.validate()
    return rule, tuple(candidates)


def apply_interval_reliability_rule(
    binding: AGQAAggregateTemporalBindingReceipt,
    rule: IntervalReliabilityRule,
) -> AGQAAggregateTemporalBindingReceipt:
    binding.validate()
    rule.validate()
    if binding.authorized_relation is None:
        return binding
    singleton = singleton_view_kind(binding)
    gap, spread = binding_geometry(binding)
    reason = None
    if singleton is not None and singleton not in set(rule.allowed_singleton_views):
        reason = f"SOURCE_ABSTAIN_SINGLETON_VIEW_UNQUALIFIED:{singleton}"
    elif gap < rule.minimum_cross_pair_gap:
        reason = f"SOURCE_ABSTAIN_CROSS_PAIR_GAP_UNQUALIFIED:{gap}"
    elif spread > rule.maximum_within_operand_endpoint_spread:
        reason = f"SOURCE_ABSTAIN_ENDPOINT_SPREAD_UNQUALIFIED:{spread}"
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


def interval_calibrated_target_grounder_sha256(
    *, parent_grounder_sha256: str, aggregate_adapter_sha256: str,
    normalization_module_sha256: str, acquisition_collector_sha256: str,
    calibrator_module_sha256: str, calibration_artifact_sha256: str,
) -> str:
    return stable_hash({
        "schema_version": "agqa2-interval-reliability-grounder-v46",
        "parent_grounder_sha256": parent_grounder_sha256,
        "aggregate_adapter_sha256": aggregate_adapter_sha256,
        "normalization_module_sha256": normalization_module_sha256,
        "acquisition_collector_sha256": acquisition_collector_sha256,
        "calibrator_module_sha256": calibrator_module_sha256,
        "calibration_artifact_sha256": calibration_artifact_sha256,
        "runtime_authority": "ABSTENTION_ONLY;CANNOT_INVENT_OR_EDIT_BINDING",
        "runtime_features": (
            "SINGLETON_VIEW;MINIMUM_CROSS_PAIR_GAP;MAXIMUM_WITHIN_OPERAND_"
            "ENDPOINT_SPREAD"
        ),
        "outcome_or_label_runtime_input": False,
    })


__all__ = [
    "GAP_THRESHOLDS", "SPREAD_THRESHOLDS", "IntervalReliabilityExample",
    "IntervalReliabilityRule", "apply_interval_reliability_rule",
    "binding_geometry", "induce_interval_reliability_rule",
    "interval_calibrated_target_grounder_sha256",
]
