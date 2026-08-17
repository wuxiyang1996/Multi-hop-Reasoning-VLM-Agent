"""Shared structural IR with target-native DiscoveryWorld grounding.

Source programs and target programs share anonymous graph-edit operator types,
but they do not have to share a controller body.  A target function is induced
as a partially ordered multiset of target transitions.  Source programs are
then tested as reusable subgraphs of that target function; target-only event
types and guards remain target-native.

The grounder consumes only current target state/action features.  Transition
outcomes are labels during target development and are never inference inputs.
"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Mapping, MutableSet, Sequence

import numpy as np

from .contracts import stable_hash


ACTION_VOCABULARY = (
    "DISCOVERY_FEED_GET_UPDATES",
    "MOVE_DIRECTION",
    "ROTATE_DIRECTION",
    "TELEPORT_TO_LOCATION",
    "TELEPORT_TO_OBJECT",
    "PICKUP",
    "DROP",
    "PUT",
    "USE",
    "OPEN",
    "CLOSE",
    "ACTIVATE",
    "DEACTIVATE",
    "OTHER",
)


def anonymous_operator_descriptor(
    operation: str, predicate_family: str, arity: int, value_kind: str,
) -> dict[str, Any]:
    core = {
        "schema_version": "anonymous-structural-operator-type-v1",
        "operation": str(operation),
        "predicate_family": str(predicate_family),
        "arity": int(arity),
        "value_kind": str(value_kind),
    }
    return core | {"operator_type_id": f"ATYPE_{stable_hash(core)[:16]}"}


ADD_ENTITY_SLOT = anonymous_operator_descriptor(
    "ADD", "ENTITY_SLOT", 1, "ENTITY_REFERENCE",
)
REMOVE_ENTITY_SLOT = anonymous_operator_descriptor(
    "REMOVE", "ENTITY_SLOT", 1, "ENTITY_REFERENCE",
)
UPDATE_CONTROL_POSITION = anonymous_operator_descriptor(
    "UPDATE", "CONTROL_STATE", 1, "POSITION",
)
ADD_OBSERVATION_RELATION = anonymous_operator_descriptor(
    "ADD", "OBSERVATION_RELATION", 2, "NUMERIC_VECTOR",
)
DISCOVERYWORLD_OPERATOR_TYPES = (
    UPDATE_CONTROL_POSITION,
    ADD_ENTITY_SLOT,
    REMOVE_ENTITY_SLOT,
    ADD_OBSERVATION_RELATION,
)
DISCOVERYWORLD_OPERATOR_IDS = tuple(
    row["operator_type_id"] for row in DISCOVERYWORLD_OPERATOR_TYPES
)

_SUBJECT = re.compile(r"investigate the (?P<subject>[^.\n]+)", re.IGNORECASE)
_FORBIDDEN_OUTCOME_FIELDS = frozenset({
    "official_success", "evaluation", "scorecard", "scoreCard", "score",
    "maxScore", "completed", "completedSuccessfully", "terminal",
})


def _inventory_ids(facts: Mapping[str, Any]) -> set[int]:
    return {
        int(row["uuid"])
        for row in facts.get("inventory") or ()
        if isinstance(row, Mapping)
        and isinstance(row.get("uuid"), int)
        and not isinstance(row.get("uuid"), bool)
    }


def _action(step: Mapping[str, Any]) -> Mapping[str, Any]:
    value = step.get("action")
    if isinstance(value, Mapping):
        return value
    selection = step.get("selection")
    if isinstance(selection, Mapping):
        value = selection.get("selected_action")
        if isinstance(value, Mapping):
            return value
    return {}


def _facts(step: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = step.get(key)
    return value if isinstance(value, Mapping) else {}


def observed_subject(step: Mapping[str, Any]) -> str | None:
    if str(_action(step).get("action") or "") != "USE":
        return None
    message = str(
        _facts(step, "after_target_native_facts").get("last_action_message")
        or ""
    )
    match = _SUBJECT.search(message)
    return match.group("subject").strip().lower() if match else None


def discoveryworld_transition_operator_ids(
    step: Mapping[str, Any], *, seen_subjects: MutableSet[str] | None = None,
) -> tuple[str, ...]:
    """Label one target transition without reading evaluator/outcome fields."""

    before = _facts(step, "before_target_native_facts")
    after = _facts(step, "after_target_native_facts")
    before_location = before.get("agent_location") or {}
    after_location = after.get("agent_location") or {}
    output: list[str] = []
    if (
        before_location.get("x") != after_location.get("x")
        or before_location.get("y") != after_location.get("y")
    ):
        output.append(UPDATE_CONTROL_POSITION["operator_type_id"])
    before_inventory = _inventory_ids(before)
    after_inventory = _inventory_ids(after)
    output.extend(
        [ADD_ENTITY_SLOT["operator_type_id"]]
        * len(after_inventory - before_inventory)
    )
    output.extend(
        [REMOVE_ENTITY_SLOT["operator_type_id"]]
        * len(before_inventory - after_inventory)
    )
    subject = observed_subject(step)
    if subject is not None and (seen_subjects is None or subject not in seen_subjects):
        output.append(ADD_OBSERVATION_RELATION["operator_type_id"])
        if seen_subjects is not None:
            seen_subjects.add(subject)
    return tuple(output)


def discoveryworld_core_sequence(
    steps: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return target substantive events, retaining learned multiplicity."""

    seen_subjects: set[str] = set()
    output = []
    for step in steps:
        output.extend(
            type_id for type_id in discoveryworld_transition_operator_ids(
                step, seen_subjects=seen_subjects,
            )
            if type_id != UPDATE_CONTROL_POSITION["operator_type_id"]
        )
    return tuple(output)


def target_action_features(step: Mapping[str, Any]) -> tuple[float, ...]:
    """Compile outcome-blind target-native inputs for a neural grounder."""

    before = _facts(step, "before_target_native_facts")
    action = _action(step)
    name = str(action.get("action") or "OTHER")
    if name not in ACTION_VOCABULARY:
        name = "OTHER"
    one_hot = [float(name == value) for value in ACTION_VOCABULARY]
    inventory = _inventory_ids(before)
    accessible = {
        int(row["uuid"])
        for row in before.get("accessible_objects") or ()
        if isinstance(row, Mapping)
        and isinstance(row.get("uuid"), int)
        and not isinstance(row.get("uuid"), bool)
    }
    salient = [
        row for row in before.get("salient_relative_objects") or ()
        if isinstance(row, Mapping)
    ]
    arg1 = action.get("arg1")
    arg2 = action.get("arg2")
    location = before.get("agent_location") or {}
    numeric = [
        min(len(inventory), 8) / 8.0,
        min(len(accessible), 32) / 32.0,
        min(len(salient), 48) / 48.0,
        min(len(location.get("directions_you_can_move") or ()), 4) / 4.0,
        float(isinstance(arg1, int) and not isinstance(arg1, bool)),
        float(isinstance(arg2, int) and not isinstance(arg2, bool)),
        float(isinstance(arg1, str)),
        float(isinstance(arg2, str)),
        float(arg1 in inventory),
        float(arg2 in inventory),
        float(arg1 in accessible),
        float(arg2 in accessible),
    ]
    return tuple(one_hot + numeric)


def target_action_labels(step: Mapping[str, Any]) -> tuple[int, ...]:
    observed = set(discoveryworld_transition_operator_ids(step))
    return tuple(int(type_id in observed) for type_id in DISCOVERYWORLD_OPERATOR_IDS)


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def export_mlp_grounder(model: Any, *, threshold: float) -> dict[str, Any]:
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("grounding threshold must be inside (0, 1)")
    body = {
        "schema_version": "target-native-structural-mlp-grounder-v1",
        "input_schema": {
            "action_vocabulary": list(ACTION_VOCABULARY),
            "feature_count": len(ACTION_VOCABULARY) + 12,
            "outcome_fields_used_at_inference": False,
        },
        "output_operator_type_ids": list(DISCOVERYWORLD_OPERATOR_IDS),
        "activation": "RELU",
        "output_activation": "SIGMOID",
        "coefs": [np.asarray(row).tolist() for row in model.coefs_],
        "intercepts": [np.asarray(row).tolist() for row in model.intercepts_],
        "threshold": float(threshold),
        "training_authority": "CONSUMED_TARGET_DEVELOPMENT_TRANSITIONS_ONLY",
        "formal_target_outcome_read": False,
    }
    return body | {"grounder_sha256": stable_hash(body)}


def validate_mlp_grounder(artifact: Mapping[str, Any]) -> None:
    body = dict(artifact)
    claimed = body.pop("grounder_sha256", None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError("target structural grounder hash mismatch")
    if artifact.get("schema_version") != "target-native-structural-mlp-grounder-v1":
        raise ValueError("unsupported target structural grounder")
    if artifact.get("formal_target_outcome_read") is not False:
        raise ValueError("formal outcome leaked into target structural grounder")
    if artifact.get("input_schema", {}).get("outcome_fields_used_at_inference") is not False:
        raise ValueError("outcome fields are enabled at inference")


def predict_operator_probabilities(
    artifact: Mapping[str, Any], step: Mapping[str, Any],
) -> dict[str, float]:
    validate_mlp_grounder(artifact)
    serialized = str(step)
    # This audit catches accidental callers that pass an evaluation object as
    # the candidate state. Nested target facts legitimately contain task
    # progress booleans in historical receipts, but features never inspect it.
    if any(f"'{field}':" in serialized for field in ("evaluation", "scorecard")):
        raise ValueError("evaluation fields passed to target grounder")
    hidden = np.asarray(target_action_features(step), dtype=float)[None, :]
    coefs = [np.asarray(row, dtype=float) for row in artifact["coefs"]]
    intercepts = [np.asarray(row, dtype=float) for row in artifact["intercepts"]]
    for index, (weights, bias) in enumerate(zip(coefs, intercepts)):
        hidden = hidden @ weights + bias
        hidden = _sigmoid(hidden) if index + 1 == len(coefs) else _relu(hidden)
    return {
        type_id: float(hidden[0, index])
        for index, type_id in enumerate(artifact["output_operator_type_ids"])
    }


def grounded_operator_ids(
    artifact: Mapping[str, Any], step: Mapping[str, Any],
) -> tuple[str, ...]:
    scores = predict_operator_probabilities(artifact, step)
    threshold = float(artifact["threshold"])
    return tuple(sorted(key for key, value in scores.items() if value >= threshold))


def _positions(sequence: Sequence[str], type_id: str) -> list[int]:
    return [index for index, value in enumerate(sequence) if value == type_id]


def induce_target_partial_order_program(
    paths: Sequence[Sequence[str]], *, development_receipts_sha256: str,
    learned_target_guards: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Induce a domain-specific counted DAG, not a copied source controller."""

    normalized = [tuple(map(str, row)) for row in paths if row]
    if not normalized:
        raise ValueError("target function induction requires nonempty paths")
    shared_types = sorted(set.intersection(*(set(row) for row in normalized)))
    minimum_counts = {
        type_id: min(row.count(type_id) for row in normalized)
        for type_id in shared_types
    }
    edges = []
    for left in shared_types:
        for right in shared_types:
            if left == right:
                continue
            if all(max(_positions(row, left)) < min(_positions(row, right)) for row in normalized):
                edges.append({"before": left, "after": right})
    # Transitive edges are safe but noisy. Remove A->C whenever A->B and B->C
    # are present; the remaining relation is the learned cover graph.
    pairs = {(row["before"], row["after"]) for row in edges}
    cover = [
        row for row in edges
        if not any(
            (row["before"], middle) in pairs
            and (middle, row["after"]) in pairs
            for middle in shared_types
            if middle not in {row["before"], row["after"]}
        )
    ]
    requirements = [
        {"operator_type_id": type_id, "minimum_count": minimum_counts[type_id]}
        for type_id in shared_types
    ]
    body = {
        "schema_version": "target-induced-structural-partial-order-program-v1",
        "development_receipts_sha256": str(development_receipts_sha256),
        "operator_requirements": requirements,
        "precedence_edges": sorted(cover, key=lambda row: (row["before"], row["after"])),
        "learned_target_guards": [dict(row) for row in learned_target_guards],
        "terminal_rule": "ALL_REQUIRED_COUNTS_AND_PRECEDENCE_EDGES_OBSERVED",
        "abstention_rule": {
            "missing_target_grounding": "ABSTAIN",
            "ambiguous_native_binding": "ABSTAIN",
            "precedence_violation": "ABSTAIN",
        },
        "source_program_copied_as_target_body": False,
        "formal_target_data_read": False,
    }
    return body | {"program_sha256": stable_hash(body)}


def validate_target_program(program: Mapping[str, Any]) -> None:
    body = dict(program)
    claimed = body.pop("program_sha256", None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError("target structural program hash mismatch")
    if program.get("source_program_copied_as_target_body") is not False:
        raise ValueError("target program is a copied source controller")
    if program.get("formal_target_data_read") is not False:
        raise ValueError("formal data leaked into target program")


def target_program_supports(
    program: Mapping[str, Any], sequence: Sequence[str],
) -> bool:
    validate_target_program(program)
    counts = Counter(map(str, sequence))
    if any(
        counts[str(row["operator_type_id"])] < int(row["minimum_count"])
        for row in program["operator_requirements"]
    ):
        return False
    for edge in program["precedence_edges"]:
        left = _positions(sequence, str(edge["before"]))
        right = _positions(sequence, str(edge["after"]))
        if not left or not right or max(left) >= min(right):
            return False
    return True


def source_sequence_support(
    source_sequence: Sequence[str], target_sequence: Sequence[str],
) -> bool:
    position = 0
    for value in target_sequence:
        if position < len(source_sequence) and value == source_sequence[position]:
            position += 1
    return position == len(source_sequence)


__all__ = [
    "ACTION_VOCABULARY", "ADD_ENTITY_SLOT", "ADD_OBSERVATION_RELATION",
    "DISCOVERYWORLD_OPERATOR_IDS", "DISCOVERYWORLD_OPERATOR_TYPES",
    "REMOVE_ENTITY_SLOT", "UPDATE_CONTROL_POSITION",
    "anonymous_operator_descriptor", "discoveryworld_core_sequence",
    "discoveryworld_transition_operator_ids", "export_mlp_grounder",
    "grounded_operator_ids", "induce_target_partial_order_program",
    "observed_subject", "predict_operator_probabilities",
    "source_sequence_support", "target_action_features",
    "target_action_labels", "target_program_supports",
    "validate_mlp_grounder", "validate_target_program",
]
