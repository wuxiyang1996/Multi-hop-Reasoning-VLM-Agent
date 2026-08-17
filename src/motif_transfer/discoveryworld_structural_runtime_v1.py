"""Execute a target-induced DiscoveryWorld function with source subgraph bias."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .target_structural_induction import (
    ADD_ENTITY_SLOT,
    ADD_OBSERVATION_RELATION,
    REMOVE_ENTITY_SLOT,
    UPDATE_CONTROL_POSITION,
    grounded_operator_ids,
    predict_operator_probabilities,
    validate_mlp_grounder,
    validate_target_program,
)


@dataclass(frozen=True)
class StructuralRuntimeDecision:
    kind: str
    action: Mapping[str, Any] | None
    expected_operator_type_id: str | None
    reason: str
    source_program_sha256: str | None
    receipt_sha256: str

    @classmethod
    def create(
        cls, *, kind: str, action: Mapping[str, Any] | None,
        expected_operator_type_id: str | None, reason: str,
        source_program_sha256: str | None,
    ) -> "StructuralRuntimeDecision":
        body = {
            "kind": str(kind),
            "action": dict(action) if action is not None else None,
            "expected_operator_type_id": expected_operator_type_id,
            "reason": str(reason),
            "source_program_sha256": source_program_sha256,
        }
        return cls(**body, receipt_sha256=stable_hash(body))


def _candidate_step(
    facts: Mapping[str, Any], action: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "action": dict(action),
        "before_target_native_facts": dict(facts),
    }


def grounded_prefix_counts(
    steps: Sequence[Mapping[str, Any]], grounder: Mapping[str, Any],
) -> dict[str, int]:
    """Count target events from neural predictions, deduplicating observations."""

    validate_mlp_grounder(grounder)
    counts: Counter[str] = Counter()
    observed_subjects = set()
    for step in steps:
        predicted = grounded_operator_ids(
            grounder,
            _candidate_step(
                step.get("before_target_native_facts") or {},
                step.get("action") or {},
            ),
        )
        for type_id in predicted:
            if type_id == ADD_OBSERVATION_RELATION["operator_type_id"]:
                message = str(
                    (step.get("after_target_native_facts") or {}).get(
                        "last_action_message"
                    ) or ""
                )
                # The target-native acquisition runner already ensures one
                # measurement per subject. Hashing the message prevents an
                # identical retry from inflating symbolic progress without
                # exporting the subject name.
                identity = stable_hash(message)
                if identity in observed_subjects:
                    continue
                observed_subjects.add(identity)
            if type_id != UPDATE_CONTROL_POSITION["operator_type_id"]:
                counts[type_id] += 1
    return dict(counts)


def target_prerequisites_satisfied(
    target_program: Mapping[str, Any], prefix_counts: Mapping[str, int],
) -> bool:
    validate_target_program(target_program)
    remove = REMOVE_ENTITY_SLOT["operator_type_id"]
    return all(
        str(row["operator_type_id"]) == remove
        or int(prefix_counts.get(str(row["operator_type_id"]), 0))
        >= int(row["minimum_count"])
        for row in target_program["operator_requirements"]
    )


def target_commit_guard_satisfied(
    target_program: Mapping[str, Any], facts: Mapping[str, Any],
    *, target_uuid: int,
) -> bool:
    validate_target_program(target_program)
    remove = REMOVE_ENTITY_SLOT["operator_type_id"]
    guards = [
        row for row in target_program.get("learned_target_guards") or ()
        if row.get("operator_type_id") == remove
    ]
    if len(guards) != 1:
        return False
    guard = guards[0]
    inventory = [
        row for row in facts.get("inventory") or () if isinstance(row, Mapping)
    ]
    targets = [
        row for row in facts.get("salient_relative_objects") or ()
        if isinstance(row, Mapping) and row.get("uuid") == int(target_uuid)
    ]
    return (
        len(targets) == 1
        and str(targets[0].get("relation_from_agent"))
        == str(guard["target_relation_from_agent"])
        and int(targets[0].get("distance", -1)) == int(guard["target_distance"])
        and len(inventory) >= int(guard["minimum_inventory_cardinality"])
    )


def _source_next_operator(
    source_program: Mapping[str, Any], prefix_counts: Mapping[str, int],
) -> str | None:
    sequence = tuple(map(str, source_program.get("induced_sequence") or ()))
    position = 0
    consumed: Counter[str] = Counter()
    for expected in sequence:
        if int(prefix_counts.get(expected, 0)) > consumed[expected]:
            consumed[expected] += 1
            position += 1
        else:
            break
    return sequence[position] if position < len(sequence) else None


def choose_structural_action(
    *, condition: str, facts: Mapping[str, Any], target_uuid: int,
    commit_action: Mapping[str, Any], position_action: Mapping[str, Any],
    grounder: Mapping[str, Any], target_program: Mapping[str, Any],
    source_program: Mapping[str, Any] | None,
    prefix_counts: Mapping[str, int],
) -> tuple[StructuralRuntimeDecision, dict[str, Any]]:
    """Choose one native candidate or fail closed under a frozen condition."""

    validate_mlp_grounder(grounder)
    validate_target_program(target_program)
    commit_step = _candidate_step(facts, commit_action)
    position_step = _candidate_step(facts, position_action)
    scores = {
        "commit": predict_operator_probabilities(grounder, commit_step),
        "position": predict_operator_probabilities(grounder, position_step),
    }
    guard = target_commit_guard_satisfied(
        target_program, facts, target_uuid=target_uuid,
    )
    prerequisites = target_prerequisites_satisfied(target_program, prefix_counts)
    source_sha = (
        str(source_program.get("program_sha256"))
        if source_program is not None else None
    )
    expected = None
    if condition in {"source_induced", "source_permuted"}:
        if source_program is None:
            decision = StructuralRuntimeDecision.create(
                kind="ABSTAIN", action=None, expected_operator_type_id=None,
                reason="SOURCE_PROGRAM_MISSING", source_program_sha256=None,
            )
            return decision, {"scores": scores, "guard": guard, "prerequisites": prerequisites}
        expected = _source_next_operator(source_program, prefix_counts)
        if not prerequisites:
            decision = StructuralRuntimeDecision.create(
                kind="ABSTAIN", action=None, expected_operator_type_id=expected,
                reason="TARGET_DOMAIN_PREREQUISITES_NOT_OBSERVED",
                source_program_sha256=source_sha,
            )
            return decision, {"scores": scores, "guard": guard, "prerequisites": prerequisites}
        if expected != REMOVE_ENTITY_SLOT["operator_type_id"]:
            decision = StructuralRuntimeDecision.create(
                kind="ABSTAIN", action=None, expected_operator_type_id=expected,
                reason="SOURCE_SUBGRAPH_HAS_NO_UNIQUE_TARGET_NATIVE_BINDING",
                source_program_sha256=source_sha,
            )
            return decision, {"scores": scores, "guard": guard, "prerequisites": prerequisites}
        if scores["commit"][expected] < float(grounder["threshold"]):
            decision = StructuralRuntimeDecision.create(
                kind="ABSTAIN", action=None, expected_operator_type_id=expected,
                reason="NEURAL_COMMIT_DELTA_BELOW_FROZEN_THRESHOLD",
                source_program_sha256=source_sha,
            )
            return decision, {"scores": scores, "guard": guard, "prerequisites": prerequisites}
        action = commit_action if guard else position_action
        reason = (
            "SOURCE_SUBGRAPH_BOUND_TO_TARGET_COMMIT"
            if guard else "TARGET_NATIVE_GUARD_REQUIRES_POSITION_REALIZATION"
        )
    elif condition == "generic_scaffold":
        expected = REMOVE_ENTITY_SLOT["operator_type_id"]
        if scores["commit"][expected] < float(grounder["threshold"]):
            decision = StructuralRuntimeDecision.create(
                kind="ABSTAIN", action=None, expected_operator_type_id=expected,
                reason="GENERIC_SCAFFOLD_HAS_NO_SUBSTANTIVE_BINDING",
                source_program_sha256=None,
            )
            return decision, {"scores": scores, "guard": guard, "prerequisites": prerequisites}
        # This capacity-matched control has the same neural delta grounder but
        # no learned source ordering or target guard.
        action = commit_action
        reason = "UNORDERED_GENERIC_SUBSTANTIVE_OPERATOR"
    elif condition == "target_native_ceiling":
        expected = REMOVE_ENTITY_SLOT["operator_type_id"]
        if not prerequisites:
            decision = StructuralRuntimeDecision.create(
                kind="ABSTAIN", action=None, expected_operator_type_id=expected,
                reason="EXACT_TARGET_PROGRAM_PREREQUISITES_NOT_OBSERVED",
                source_program_sha256=None,
            )
            return decision, {"scores": scores, "guard": guard, "prerequisites": prerequisites}
        action = commit_action if guard else position_action
        reason = (
            "EXACT_TARGET_PROGRAM_COMMIT"
            if guard else "EXACT_TARGET_PROGRAM_POSITION_REALIZATION"
        )
    else:
        raise ValueError(f"unsupported structural condition: {condition}")
    decision = StructuralRuntimeDecision.create(
        kind="EXECUTE", action=action,
        expected_operator_type_id=expected, reason=reason,
        source_program_sha256=source_sha,
    )
    audit = {
        "scores": scores,
        "grounder_threshold": float(grounder["threshold"]),
        "target_commit_guard_satisfied": guard,
        "target_prerequisites_satisfied": prerequisites,
        "target_outcome_read": False,
    }
    return decision, audit


__all__ = [
    "StructuralRuntimeDecision", "choose_structural_action",
    "grounded_prefix_counts", "target_commit_guard_satisfied",
    "target_prerequisites_satisfied",
]
