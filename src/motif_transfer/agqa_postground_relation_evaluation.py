"""Outcome-blind AGQA composition after target-native binding resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .agqa_goal_relation_evaluation import runtime_integrity_qualified
from .agqa_goal_relation_transfer import (
    decide_source_candidate,
    unified_target_grounding,
)
from .agqa_postground_relation_transfer import bind_postground_source_program
from .agqa_query_object_source_specific import target_only_ontology_decision
from .contracts import stable_hash
from .unified_neurosymbolic_harness import UnifiedNeurosymbolicHarness


@dataclass(frozen=True)
class FrozenPostgroundTransferPredictions:
    task_id: str
    target_native_prediction: str
    source_harness_prediction: str
    effect_shuffled_prediction: str
    generic_scaffold_prediction: str
    target_written_equivalent_prediction: str
    target_only_ontology_decision: str | None
    resolved_source_candidate: str | None
    source_executor_authorized: bool
    effect_shuffled_executor_authorized: bool
    source_binding_receipt_sha256: str
    effect_shuffled_binding_receipt_sha256: str
    source_phase7_reason: str
    source_utility_reason: str
    source_utility_lower_bound: float
    source_authenticity_lower_bound: float
    raw_neural_votes_used_as_symbolic_bindings: bool
    target_grounder_resolves_binding_before_symbolic_execution: bool
    runtime_integrity_qualified: bool
    current_outcome_read: bool
    receipt_sha256: str

    def validate(self) -> None:
        body = asdict(self)
        claimed = body.pop("receipt_sha256")
        if stable_hash(body) != claimed:
            raise ValueError("postground AGQA prediction hash mismatch")
        if self.current_outcome_read:
            raise ValueError("postground AGQA prediction read current outcome")
        if self.raw_neural_votes_used_as_symbolic_bindings:
            raise ValueError("raw perception votes leaked into symbolic bindings")
        if not self.target_grounder_resolves_binding_before_symbolic_execution:
            raise ValueError("target grounding did not precede symbolic execution")
        if self.effect_shuffled_executor_authorized:
            raise ValueError("effect-shuffled source program executed")


def freeze_postground_predictions(
    *, row: Mapping[str, Any], artifact: Mapping[str, Any],
    confirmation: Mapping[str, Any], harness: UnifiedNeurosymbolicHarness,
    target_grounder_sha256: str, target_executor_sha256: str,
    minimum_ontology_confidences: tuple[float, float] = (0.8, 0.8),
) -> FrozenPostgroundTransferPredictions:
    """Freeze every target arm without accepting outcome-bearing inputs."""

    integrity = runtime_integrity_qualified(row)
    task_id = str(row["task_id"])
    target_only = target_only_ontology_decision(
        row["object_ontology_receipts"], minimum_ontology_confidences,
    )
    target_prediction = target_only or str(row["direct_response"])
    binding = bind_postground_source_program(
        artifact=artifact,
        confirmation=confirmation,
        task_id=task_id,
        target_state_sha256=str(row["runtime_receipt_sha256"]),
        target_grounder_sha256=target_grounder_sha256,
        calibrated_execution=row["calibrated_target_native_execution"],
        grounder_qualified=integrity,
        formal_outcome_read=False,
    )
    target = unified_target_grounding(
        artifact=artifact, confirmation=confirmation, binding=binding,
    )
    source = decide_source_candidate(
        harness=harness, target=target, binding=binding,
        target_executor_sha256=target_executor_sha256,
    )
    source_prediction = source.source_candidate or target_prediction

    shuffled_binding = bind_postground_source_program(
        artifact=artifact,
        confirmation=confirmation,
        task_id=task_id,
        target_state_sha256=str(row["runtime_receipt_sha256"]),
        target_grounder_sha256=target_grounder_sha256,
        calibrated_execution=row["calibrated_target_native_execution"],
        grounder_qualified=integrity,
        effect_binding_authenticated=False,
        formal_outcome_read=False,
    )
    shuffled_target = unified_target_grounding(
        artifact=artifact, confirmation=confirmation,
        binding=shuffled_binding,
    )
    shuffled = decide_source_candidate(
        harness=harness, target=shuffled_target, binding=shuffled_binding,
        target_executor_sha256=target_executor_sha256,
    )
    if shuffled.source_candidate is not None:
        raise AssertionError("effect-shuffled source unexpectedly executed")
    generic_candidate = row["calibrated_target_native_execution"].get(
        "decision"
    )
    generic_prediction = generic_candidate or target_prediction
    if source_prediction != generic_prediction:
        raise AssertionError(
            "source program drifted from the matched target-written ceiling"
        )
    body = {
        "task_id": task_id,
        "target_native_prediction": target_prediction,
        "source_harness_prediction": source_prediction,
        "effect_shuffled_prediction": target_prediction,
        "generic_scaffold_prediction": generic_prediction,
        "target_written_equivalent_prediction": generic_prediction,
        "target_only_ontology_decision": target_only,
        "resolved_source_candidate": source.source_candidate,
        "source_executor_authorized": source.source_candidate is not None,
        "effect_shuffled_executor_authorized": False,
        "source_binding_receipt_sha256": binding.receipt_sha256,
        "effect_shuffled_binding_receipt_sha256": (
            shuffled_binding.receipt_sha256
        ),
        "source_phase7_reason": source.phase7.reason,
        "source_utility_reason": source.utility.reason,
        "source_utility_lower_bound": source.utility.utility_lower_bound,
        "source_authenticity_lower_bound": (
            source.utility.authenticity_lower_bound
        ),
        "raw_neural_votes_used_as_symbolic_bindings": False,
        "target_grounder_resolves_binding_before_symbolic_execution": True,
        "runtime_integrity_qualified": integrity,
        "current_outcome_read": False,
    }
    receipt = FrozenPostgroundTransferPredictions(
        **body, receipt_sha256=stable_hash(body),
    )
    receipt.validate()
    return receipt


__all__ = [
    "FrozenPostgroundTransferPredictions", "freeze_postground_predictions",
]
