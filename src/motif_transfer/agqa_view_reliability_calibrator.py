"""Learn a small symbolic abstention rule over neural grounding views.

The calibrator cannot change an interval or temporal relation.  It can only
preserve or revoke an otherwise valid aggregate temporal binding according to
which independently configured neural view is the sole support for one typed
argument.  The finite rule class is enumerated exhaustively and selected from
consumed target-development outcomes before any future qualification split is
opened.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from .agqa_aggregate_temporal_transfer import (
    AGQAAggregateTemporalBindingReceipt,
)
from .contracts import stable_hash


VIEW_KINDS = ("primary", "rescan", "tiebreak")


@dataclass(frozen=True)
class ViewReliabilityExample:
    task_id: str
    aggregate_authorized: bool
    singleton_view: str | None
    source_correct: bool
    target_native_correct: bool


@dataclass(frozen=True)
class ViewReliabilityRule:
    allowed_singleton_views: tuple[str, ...]
    strict_two_per_operand_always_allowed: bool
    training_examples: int
    training_authorizations: int
    training_wins: int
    training_losses: int
    training_ties: int
    selection_authority: str
    rule_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ViewReliabilityRule":
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
            raise ValueError("AGQA view-reliability rule hash mismatch")
        if tuple(sorted(self.allowed_singleton_views)) != (
            self.allowed_singleton_views
        ):
            raise ValueError("allowed singleton views are not canonical")
        if not set(self.allowed_singleton_views).issubset(VIEW_KINDS):
            raise ValueError("unknown neural view escaped calibration")


def singleton_view_kind(
    binding: AGQAAggregateTemporalBindingReceipt,
) -> str | None:
    a = binding.operand_a_hypotheses
    b = binding.operand_b_hypotheses
    if len(a) == 1 and len(b) >= 2:
        return a[0].view
    if len(b) == 1 and len(a) >= 2:
        return b[0].view
    return None


def _evaluate_candidate(
    examples: Sequence[ViewReliabilityExample],
    allowed: tuple[str, ...],
) -> dict[str, object]:
    allowed_set = set(allowed)
    decisions = []
    for row in examples:
        selected = row.aggregate_authorized and (
            row.singleton_view is None
            or row.singleton_view in allowed_set
        )
        left = row.source_correct if selected else row.target_native_correct
        right = row.target_native_correct
        decisions.append((selected, left, right))
    wins = sum(left and not right for _, left, right in decisions)
    losses = sum(right and not left for _, left, right in decisions)
    authorizations = sum(selected for selected, _, _ in decisions)
    return {
        "allowed_singleton_views": list(allowed),
        "authorizations": authorizations,
        "wins": wins,
        "losses": losses,
        "ties": len(decisions) - wins - losses,
        "net_gain": wins - losses,
        "zero_negative_transfer": losses == 0,
        "rule_description_length": len(allowed),
    }


def induce_view_reliability_rule(
    examples: Iterable[ViewReliabilityExample],
) -> tuple[ViewReliabilityRule, tuple[Mapping[str, object], ...]]:
    rows = tuple(examples)
    if not rows:
        raise ValueError("view-reliability induction requires examples")
    candidates = []
    for size in range(len(VIEW_KINDS) + 1):
        for allowed in combinations(VIEW_KINDS, size):
            candidates.append(_evaluate_candidate(rows, allowed))
    # Constrained empirical risk minimization over the complete finite class:
    # first forbid observed negative transfer, then maximize paired wins and
    # coverage, with MDL and lexical ordering as deterministic tie-breaks.
    eligible = [row for row in candidates if row["zero_negative_transfer"]]
    pool = eligible or candidates
    chosen = min(
        pool,
        key=lambda row: (
            int(row["losses"]),
            -int(row["wins"]),
            -int(row["authorizations"]),
            int(row["rule_description_length"]),
            tuple(row["allowed_singleton_views"]),
        ),
    )
    body = {
        "allowed_singleton_views": tuple(sorted(
            str(value) for value in chosen["allowed_singleton_views"]
        )),
        "strict_two_per_operand_always_allowed": True,
        "training_examples": len(rows),
        "training_authorizations": int(chosen["authorizations"]),
        "training_wins": int(chosen["wins"]),
        "training_losses": int(chosen["losses"]),
        "training_ties": int(chosen["ties"]),
        "selection_authority": (
            "EXHAUSTIVE_FINITE_VIEW_SUBSET_CLASS;ZERO_NEGATIVE_TRANSFER_"
            "CONSTRAINT;MAX_WINS_THEN_COVERAGE;MDL_TIE_BREAK"
        ),
    }
    rule = ViewReliabilityRule(
        **body, rule_sha256=stable_hash(body),
    )
    rule.validate()
    return rule, tuple(candidates)


def apply_view_reliability_rule(
    binding: AGQAAggregateTemporalBindingReceipt,
    rule: ViewReliabilityRule,
) -> AGQAAggregateTemporalBindingReceipt:
    binding.validate()
    rule.validate()
    if binding.authorized_relation is None:
        return binding
    singleton = singleton_view_kind(binding)
    if singleton is None or singleton in set(rule.allowed_singleton_views):
        return binding
    interim = replace(
        binding,
        authorized_relation=None,
        reason=f"SOURCE_ABSTAIN_SINGLETON_VIEW_UNQUALIFIED:{singleton}",
        receipt_sha256="",
    )
    body = asdict(interim)
    body.pop("receipt_sha256")
    result = replace(interim, receipt_sha256=stable_hash(body))
    result.validate()
    return result


def calibrated_target_grounder_sha256(
    *, parent_grounder_sha256: str, aggregate_adapter_sha256: str,
    normalization_module_sha256: str, acquisition_collector_sha256: str,
    calibrator_module_sha256: str, calibration_artifact_sha256: str,
) -> str:
    return stable_hash({
        "schema_version": "agqa2-view-reliability-grounder-v43",
        "parent_grounder_sha256": parent_grounder_sha256,
        "aggregate_adapter_sha256": aggregate_adapter_sha256,
        "normalization_module_sha256": normalization_module_sha256,
        "acquisition_collector_sha256": acquisition_collector_sha256,
        "calibrator_module_sha256": calibrator_module_sha256,
        "calibration_artifact_sha256": calibration_artifact_sha256,
        "runtime_authority": "ABSTENTION_ONLY;CANNOT_INVENT_OR_EDIT_BINDING",
        "outcome_or_label_runtime_input": False,
    })


__all__ = [
    "VIEW_KINDS", "ViewReliabilityExample", "ViewReliabilityRule",
    "apply_view_reliability_rule", "calibrated_target_grounder_sha256",
    "induce_view_reliability_rule",
    "singleton_view_kind",
]
