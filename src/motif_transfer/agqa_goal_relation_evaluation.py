"""Outcome-blind prediction composition for AGQA goal-relation transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .agqa_goal_relation_transfer import (
    bind_source_goal_relation_program,
    decide_source_candidate,
    unified_target_grounding,
)
from .agqa_query_object_source_specific import target_only_ontology_decision
from .contracts import stable_hash
from .unified_neurosymbolic_harness import UnifiedNeurosymbolicHarness


def runtime_integrity_qualified(row: Mapping[str, Any]) -> bool:
    forbidden = (
        "runtime_answer_read",
        "runtime_functional_program_read",
        "runtime_scene_graph_read",
        "runtime_source_identity_read",
        "operand_grounder_question_read",
        "operand_grounder_competing_operand_read",
        "object_ontology_original_question_read",
        "object_ontology_answer_candidates_read",
    )
    return all(row.get(key) is False for key in forbidden)


@dataclass(frozen=True)
class FrozenAGQATransferPredictions:
    task_id: str
    target_native_prediction: str
    source_harness_prediction: str
    effect_shuffled_prediction: str
    generic_scaffold_prediction: str
    target_written_equivalent_prediction: str
    target_only_ontology_decision: str | None
    source_candidate: str | None
    generic_candidate: str | None
    source_executor_authorized: bool
    effect_shuffled_executor_authorized: bool
    source_binding_receipt_sha256: str
    effect_shuffled_binding_receipt_sha256: str
    source_phase7_reason: str
    source_utility_reason: str
    source_utility_lower_bound: float
    source_authenticity_lower_bound: float
    runtime_integrity_qualified: bool
    current_outcome_read: bool
    receipt_sha256: str

    def validate(self) -> None:
        body = asdict(self)
        claimed = body.pop("receipt_sha256")
        if stable_hash(body) != claimed:
            raise ValueError("frozen AGQA transfer prediction hash mismatch")
        if self.current_outcome_read:
            raise ValueError("frozen AGQA transfer prediction read current outcome")
        if not self.source_executor_authorized and self.source_candidate is not None:
            raise ValueError("unauthorized AGQA source candidate escaped")
        if self.effect_shuffled_executor_authorized:
            raise ValueError("effect-shuffled source program executed")


def freeze_transfer_predictions(
    *, row: Mapping[str, Any], artifact: Mapping[str, Any],
    confirmation: Mapping[str, Any], harness: UnifiedNeurosymbolicHarness,
    target_grounder_sha256: str, target_executor_sha256: str,
    minimum_ontology_confidences: tuple[float, float] = (0.8, 0.8),
) -> FrozenAGQATransferPredictions:
    """Freeze all arms without accepting a gold answer or outcome."""

    integrity = runtime_integrity_qualified(row)
    task_id = str(row["task_id"])
    target_only = target_only_ontology_decision(
        row["object_ontology_receipts"], minimum_ontology_confidences,
    )
    target_prediction = target_only or str(row["direct_response"])
    binding = bind_source_goal_relation_program(
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

    shuffled_binding = bind_source_goal_relation_program(
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
    body = {
        "task_id": task_id,
        "target_native_prediction": target_prediction,
        "source_harness_prediction": source_prediction,
        "effect_shuffled_prediction": target_prediction,
        "generic_scaffold_prediction": generic_prediction,
        "target_written_equivalent_prediction": source_prediction,
        "target_only_ontology_decision": target_only,
        "source_candidate": source.source_candidate,
        "generic_candidate": generic_candidate,
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
        "runtime_integrity_qualified": integrity,
        "current_outcome_read": False,
    }
    receipt = FrozenAGQATransferPredictions(
        **body, receipt_sha256=stable_hash(body),
    )
    receipt.validate()
    return receipt


__all__ = [
    "FrozenAGQATransferPredictions", "freeze_transfer_predictions",
    "runtime_integrity_qualified",
]
