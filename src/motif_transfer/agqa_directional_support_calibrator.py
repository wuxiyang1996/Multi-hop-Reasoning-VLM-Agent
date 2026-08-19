"""Risk-first directional applicability for AGQA temporal transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from .agqa_aggregate_temporal_transfer import AGQAAggregateTemporalBindingReceipt
from .agqa_interval_reliability_calibrator import (
    GAP_THRESHOLDS, SPREAD_THRESHOLDS, binding_geometry,
)
from .agqa_temporal_support_calibrator import (
    MINIMUM_MAX_INTERVAL_SPANS, maximum_interval_span,
)
from .agqa_view_reliability_calibrator import VIEW_KINDS, singleton_view_kind
from .contracts import stable_hash


RELATIONS = ("after", "before")


@dataclass(frozen=True)
class DirectionalSupportExample:
    split: str
    task_id: str
    aggregate_authorized: bool
    resolved_relation: str | None
    singleton_view: str | None
    minimum_cross_pair_gap: int
    maximum_within_operand_endpoint_spread: int
    maximum_interval_span: int
    source_correct: bool
    target_native_correct: bool


@dataclass(frozen=True)
class DirectionalSupportRule:
    allowed_relations: tuple[str, ...]
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
    def from_mapping(cls, value: Mapping[str, object]) -> "DirectionalSupportRule":
        payload = dict(value)
        for field in ("allowed_relations", "allowed_singleton_views"):
            payload[field] = tuple(str(item) for item in payload[field])
        rule = cls(**payload)
        rule.validate()
        return rule

    def validate(self) -> None:
        body = asdict(self); claimed = body.pop("rule_sha256")
        if stable_hash(body) != claimed:
            raise ValueError("AGQA directional-support rule hash mismatch")
        if tuple(sorted(self.allowed_relations)) != self.allowed_relations:
            raise ValueError("allowed relations are not canonical")
        if not self.allowed_relations or not set(self.allowed_relations).issubset(RELATIONS):
            raise ValueError("invalid directional-support relations")
        if tuple(sorted(self.allowed_singleton_views)) != self.allowed_singleton_views:
            raise ValueError("allowed singleton views are not canonical")
        if not set(self.allowed_singleton_views).issubset(VIEW_KINDS):
            raise ValueError("unknown directional-support view")
        if self.minimum_cross_pair_gap not in GAP_THRESHOLDS:
            raise ValueError("gap escaped finite class")
        if self.maximum_within_operand_endpoint_spread not in SPREAD_THRESHOLDS:
            raise ValueError("spread escaped finite class")
        if self.minimum_max_interval_span not in MINIMUM_MAX_INTERVAL_SPANS:
            raise ValueError("support escaped finite class")


def _evaluate(rows: Sequence[DirectionalSupportExample], relations, views,
              gap, spread, support):
    selected = [row for row in rows if row.aggregate_authorized
                and row.resolved_relation in set(relations)
                and (row.singleton_view is None or row.singleton_view in set(views))
                and row.minimum_cross_pair_gap >= gap
                and row.maximum_within_operand_endpoint_spread <= spread
                and row.maximum_interval_span >= support]
    wins = sum(row.source_correct and not row.target_native_correct for row in selected)
    losses = sum(row.target_native_correct and not row.source_correct for row in selected)
    return {"allowed_relations": list(relations),
            "allowed_singleton_views": list(views),
            "minimum_cross_pair_gap": gap,
            "maximum_within_operand_endpoint_spread": spread,
            "minimum_max_interval_span": support,
            "authorizations": len(selected), "wins": wins, "losses": losses,
            "ties": len(rows)-wins-losses, "net_gain": wins-losses,
            "rule_description_length": len(relations)+len(views)+3}


def induce_directional_support_rule(examples: Iterable[DirectionalSupportExample]):
    rows = tuple(examples)
    if not rows:
        raise ValueError("directional-support induction requires examples")
    candidates = []
    for relation_size in range(1, len(RELATIONS)+1):
        for relations in combinations(RELATIONS, relation_size):
            for view_size in range(len(VIEW_KINDS)+1):
                for views in combinations(VIEW_KINDS, view_size):
                    for gap in GAP_THRESHOLDS:
                        for spread in SPREAD_THRESHOLDS:
                            for support in MINIMUM_MAX_INTERVAL_SPANS:
                                candidates.append(_evaluate(
                                    rows, relations, views, gap, spread, support,
                                ))
    chosen = min(candidates, key=lambda row: (
        int(row["losses"]), -int(row["net_gain"]), -int(row["wins"]),
        -int(row["authorizations"]), int(row["rule_description_length"]),
        int(row["minimum_cross_pair_gap"]),
        -int(row["maximum_within_operand_endpoint_spread"]),
        int(row["minimum_max_interval_span"]),
        tuple(row["allowed_relations"]), tuple(row["allowed_singleton_views"]),
    ))
    body = {
        "allowed_relations": tuple(sorted(chosen["allowed_relations"])),
        "allowed_singleton_views": tuple(sorted(chosen["allowed_singleton_views"])),
        "minimum_cross_pair_gap": int(chosen["minimum_cross_pair_gap"]),
        "maximum_within_operand_endpoint_spread": int(chosen["maximum_within_operand_endpoint_spread"]),
        "minimum_max_interval_span": int(chosen["minimum_max_interval_span"]),
        "training_examples": len(rows),
        "training_authorizations": int(chosen["authorizations"]),
        "training_wins": int(chosen["wins"]),
        "training_losses": int(chosen["losses"]),
        "training_ties": int(chosen["ties"]),
        "selection_authority": (
            "EXHAUSTIVE_2304_RULE_FINITE_CLASS;MINIMIZE_NEGATIVE_TRANSFER_"
            "THEN_MAXIMIZE_NET_GAIN_WINS_COVERAGE;MDL_TIE_BREAK"
        ),
    }
    rule = DirectionalSupportRule(**body, rule_sha256=stable_hash(body))
    rule.validate()
    return rule, tuple(candidates)


def apply_directional_support_rule(binding, rule: DirectionalSupportRule):
    binding.validate(); rule.validate()
    if binding.authorized_relation is None:
        return binding
    singleton = singleton_view_kind(binding)
    gap, spread = binding_geometry(binding)
    support = maximum_interval_span(binding)
    reason = None
    if binding.authorized_relation not in set(rule.allowed_relations):
        reason = f"SOURCE_ABSTAIN_RELATION_UNQUALIFIED:{binding.authorized_relation}"
    elif singleton is not None and singleton not in set(rule.allowed_singleton_views):
        reason = f"SOURCE_ABSTAIN_SINGLETON_VIEW_UNQUALIFIED:{singleton}"
    elif gap < rule.minimum_cross_pair_gap:
        reason = f"SOURCE_ABSTAIN_CROSS_PAIR_GAP_UNQUALIFIED:{gap}"
    elif spread > rule.maximum_within_operand_endpoint_spread:
        reason = f"SOURCE_ABSTAIN_ENDPOINT_SPREAD_UNQUALIFIED:{spread}"
    elif support < rule.minimum_max_interval_span:
        reason = f"SOURCE_ABSTAIN_TEMPORAL_SUPPORT_UNQUALIFIED:{support}"
    if reason is None:
        return binding
    interim = replace(binding, authorized_relation=None, reason=reason, receipt_sha256="")
    body = asdict(interim); body.pop("receipt_sha256")
    result = replace(interim, receipt_sha256=stable_hash(body)); result.validate()
    return result


def directional_support_target_grounder_sha256(**kwargs):
    return stable_hash({"schema_version": "agqa2-directional-support-grounder-v52",
                        **kwargs,
                        "runtime_authority": "ABSTENTION_ONLY;CANNOT_INVENT_OR_EDIT_BINDING",
                        "runtime_features": "RELATION;VIEW;GAP;SPREAD;INTERVAL_SPAN",
                        "outcome_or_label_runtime_input": False})


__all__ = ["DirectionalSupportExample", "DirectionalSupportRule",
           "apply_directional_support_rule", "directional_support_target_grounder_sha256",
           "induce_directional_support_rule"]
