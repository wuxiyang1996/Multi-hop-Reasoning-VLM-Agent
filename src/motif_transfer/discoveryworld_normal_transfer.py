"""Target-native grounding of a source-induced acquisition program.

The source artifact supplies four anonymous operator roles: two repeatable
acquisition updates, a positive-binding onset, and a terminal goal-relation
update.  This module binds those roles to Proteomics Normal using only public
target observations, candidate actions, and memory produced by the target
surveyor.  Evaluator scorecards are deliberately outside the feature schema.

The important distinction is between *grounding* and *program induction*.
Local target transitions may train the neural grounder.  They do not provide
an ordered successful target program to the source condition.  The matched
target-only inducer receives an explicitly counted number of complete ordered
target trajectories and must abstain at budget zero.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import stable_hash
from .source_goal_acquisition_induction import (
    validate_goal_acquisition_program,
)


SCHEMA_VERSION = "discoveryworld-normal-source-grounding-v1"
GROUNDING_CLASSES = ("ABSTAIN", "ACQUISITION_ENTITY", "ACQUISITION_CONTROL", "BINDING", "RELATION")
ACTION_CLASSES = (
    "TELEPORT_TO_LOCATION", "TELEPORT_TO_OBJECT", "PICKUP", "USE", "DROP", "OTHER",
)
CARDINAL_RELATIONS = frozenset({"north", "south", "east", "west"})
FORBIDDEN_FEATURE_FIELDS = frozenset({
    "evaluation", "official_success", "scorecard", "scoreCard", "score",
    "completed", "completedSuccessfully", "terminal",
})


class SourceProgramMonitor:
    """Fail-closed phase monitor for authentic and destructive controls."""

    def __init__(self, condition: str) -> None:
        if condition not in {"authentic_source", "source_permuted", "generic_scaffold", "neural_only"}:
            raise ValueError(f"unsupported monitor condition: {condition}")
        self.condition = condition
        self.phase = "ACQUISITION"
        self.authorized = 0
        self.abstentions = 0

    def authorize(self, grounded_role: str) -> tuple[bool, str]:
        role = str(grounded_role)
        if self.condition == "neural_only":
            self.authorized += 1
            return True, "NO_SYMBOLIC_MONITOR"
        if self.condition == "generic_scaffold":
            allowed = role != "ABSTAIN"
            self.authorized += int(allowed)
            self.abstentions += int(not allowed)
            return allowed, "GENERIC_NON_ABSTAIN" if allowed else "GROUNDER_ABSTAIN"
        if self.condition == "source_permuted":
            role = {"BINDING": "RELATION", "RELATION": "BINDING"}.get(role, role)
        if self.phase == "ACQUISITION":
            if role in {"ACQUISITION_ENTITY", "ACQUISITION_CONTROL"}:
                allowed, reason = True, "SOURCE_ACQUISITION_LOOP"
            elif role == "BINDING":
                self.phase = "BOUND"
                allowed, reason = True, "SOURCE_POSITIVE_BINDING_ONSET"
            else:
                allowed, reason = False, "SOURCE_REQUIRES_POSITIVE_BINDING"
        elif self.phase == "BOUND":
            if role == "RELATION":
                self.phase = "DONE"
                allowed, reason = True, "SOURCE_BINDING_TO_RELATION"
            else:
                allowed, reason = False, "SOURCE_REQUIRES_TERMINAL_RELATION"
        else:
            allowed, reason = False, "SOURCE_PROGRAM_COMPLETE"
        self.authorized += int(allowed)
        self.abstentions += int(not allowed)
        return allowed, reason


def _action(step: Mapping[str, Any]) -> Mapping[str, Any]:
    value = step.get("action")
    return value if isinstance(value, Mapping) else {}


def _facts(step: Mapping[str, Any], which: str = "before") -> Mapping[str, Any]:
    value = step.get(f"{which}_target_native_facts")
    return value if isinstance(value, Mapping) else {}


def _memory(step: Mapping[str, Any]) -> Mapping[str, Any]:
    value = step.get("memory") or "{}"
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _inventory(facts: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in facts.get("inventory") or () if isinstance(row, Mapping)]


def _has_named_inventory(facts: Mapping[str, Any], needle: str) -> bool:
    return any(needle in str(row.get("name") or "").lower() for row in _inventory(facts))


def _anomaly(step: Mapping[str, Any]) -> str | None:
    value = _memory(step).get("anomaly")
    return str(value).strip().lower() if value else None


def _measurement_count(step: Mapping[str, Any]) -> int:
    measured = _memory(step).get("measured") or {}
    return len(measured) if isinstance(measured, Mapping) else 0


def _target_statue_rows(step: Mapping[str, Any], which: str = "before") -> list[Mapping[str, Any]]:
    anomaly = _anomaly(step)
    if anomaly is None:
        return []
    return [
        row for row in _facts(step, which).get("salient_relative_objects") or ()
        if isinstance(row, Mapping)
        and str(row.get("name") or "").lower().startswith("statue of ")
        and str(row.get("name") or "").lower().endswith(anomaly)
    ]


def positive_binding_count(step: Mapping[str, Any], which: str = "before") -> int:
    """Count executable positive goal-relation interventions from public facts."""

    facts = _facts(step, which)
    if _measurement_count(step) != 5 or not _has_named_inventory(facts, "flag"):
        return 0
    rows = _target_statue_rows(step, which)
    return sum(
        int(row.get("distance", -1)) == 1
        and str(row.get("relation_from_agent") or "") in CARDINAL_RELATIONS
        for row in rows
    )


def _candidate_targets_anomaly(step: Mapping[str, Any]) -> bool:
    action = _action(step)
    anomaly = _anomaly(step)
    return (
        anomaly is not None
        and str(action.get("action") or "") == "TELEPORT_TO_LOCATION"
        and str(action.get("arg1") or "").strip().lower() == f"statue of a {anomaly}"
    )


def target_grounding_features(step: Mapping[str, Any]) -> tuple[float, ...]:
    """Outcome-blind neural features for one target-native candidate action."""

    if any(key in step for key in FORBIDDEN_FEATURE_FIELDS):
        raise ValueError("evaluator outcome field passed to Normal grounder")
    action = _action(step)
    name = str(action.get("action") or "OTHER")
    if name not in ACTION_CLASSES:
        name = "OTHER"
    before = _facts(step)
    arg1 = action.get("arg1")
    inventory_ids = {
        int(row["uuid"]) for row in _inventory(before)
        if isinstance(row.get("uuid"), int) and not isinstance(row.get("uuid"), bool)
    }
    one_hot = [float(name == value) for value in ACTION_CLASSES]
    numeric = [
        _measurement_count(step) / 5.0,
        float(_has_named_inventory(before, "proteomics meter")),
        float(_has_named_inventory(before, "flag")),
        float(_anomaly(step) is not None),
        float(_candidate_targets_anomaly(step)),
        float(positive_binding_count(step) == 1),
        float(isinstance(arg1, int) and not isinstance(arg1, bool) and arg1 in inventory_ids),
        min(len(_inventory(before)), 4) / 4.0,
    ]
    return tuple(one_hot + numeric)


def target_grounding_label(step: Mapping[str, Any]) -> str:
    """Development label from public action/effect facts, never formal success."""

    action = _action(step)
    name = str(action.get("action") or "")
    succeeded = bool(step.get("action_succeeded"))
    if not succeeded:
        return "ABSTAIN"
    if name == "DROP" and positive_binding_count(step) == 1:
        return "RELATION"
    if (
        positive_binding_count(step) == 0
        and positive_binding_count(step, "after") == 1
        and _candidate_targets_anomaly(step)
    ):
        return "BINDING"
    if name in {"TELEPORT_TO_LOCATION", "TELEPORT_TO_OBJECT"}:
        return "ACQUISITION_CONTROL"
    if name in {"PICKUP", "USE"}:
        return "ACQUISITION_ENTITY"
    return "ABSTAIN"


def source_role_operator_ids(source: Mapping[str, Any]) -> dict[str, Any]:
    validate_goal_acquisition_program(source)
    program = source["program"]
    acquisition = list(map(str, program["acquisition_operator_type_ids"]))
    descriptors = {
        str(row["operator_type_id"]): row for row in source["operator_types"]
    }
    entity = [
        operator_id for operator_id in acquisition
        if descriptors[operator_id]["predicate_family"] == "ENTITY_RELATION"
    ]
    control = [
        operator_id for operator_id in acquisition
        if descriptors[operator_id]["predicate_family"] == "CONTROL_STATE"
    ]
    if len(entity) != 1 or len(control) != 1:
        raise ValueError("source artifact lacks unique entity/control acquisition roles")
    return {
        "ACQUISITION_ENTITY": entity[0],
        "ACQUISITION_CONTROL": control[0],
        "BINDING": str(program["binding_operator_type_id"]),
        "RELATION": str(program["relation_operator_type_id"]),
        "ABSTAIN": None,
    }


def typed_trace(steps: Sequence[Mapping[str, Any]], source: Mapping[str, Any]) -> tuple[str, ...]:
    roles = source_role_operator_ids(source)
    return tuple(
        str(roles[label]) for step in steps
        if (label := target_grounding_label(step)) != "ABSTAIN"
    )


def trace_conforms(sequence: Sequence[str], source: Mapping[str, Any]) -> bool:
    """Check the induced phase program, including its fail-closed terminal edge."""

    roles = source_role_operator_ids(source)
    values = tuple(map(str, sequence))
    if len(values) < 3:
        return False
    binding = str(roles["BINDING"])
    relation = str(roles["RELATION"])
    acquisition = {
        str(roles["ACQUISITION_ENTITY"]), str(roles["ACQUISITION_CONTROL"]),
    }
    if values.count(binding) != 1 or values.count(relation) != 1:
        return False
    binding_at = values.index(binding)
    return (
        binding_at == len(values) - 2
        and values[-1] == relation
        and all(value in acquisition for value in values[:binding_at])
    )


def induce_target_only_program(
    complete_successful_paths: Sequence[Sequence[str]], *, budget: int,
) -> dict[str, Any]:
    """Matched target-only program induction with an explicit demo budget."""

    if budget < 0 or budget > len(complete_successful_paths):
        raise ValueError("invalid target-only trajectory budget")
    paths = [tuple(map(str, row)) for row in complete_successful_paths[:budget]]
    if not paths:
        body = {
            "schema_version": "target-only-normal-program-v1",
            "status": "ABSTAIN_NO_COMPLETE_TARGET_TRAJECTORY",
            "complete_target_trajectory_budget": 0,
            "program": None,
        }
        return body | {"program_sha256": stable_hash(body)}
    binding_candidates = set.intersection(*(set((row[-2],)) for row in paths if len(row) >= 2))
    relation_candidates = set.intersection(*(set((row[-1],)) for row in paths if row))
    if len(binding_candidates) != 1 or len(relation_candidates) != 1:
        status = "ABSTAIN_AMBIGUOUS_TERMINAL_PROGRAM"
        program = None
    else:
        binding = binding_candidates.pop()
        relation = relation_candidates.pop()
        acquisition_sets = [set(row[:-2]) for row in paths]
        acquisition = sorted(set.intersection(*acquisition_sets))
        status = "TARGET_ONLY_PROGRAM_INDUCED"
        program = {
            "acquisition_operator_type_ids": acquisition,
            "binding_operator_type_id": binding,
            "relation_operator_type_id": relation,
            "binding_to_relation": True,
        }
    body = {
        "schema_version": "target-only-normal-program-v1",
        "status": status,
        "complete_target_trajectory_budget": budget,
        "program": program,
    }
    return body | {"program_sha256": stable_hash(body)}


def export_neural_grounder(model: Any) -> dict[str, Any]:
    body = {
        "schema_version": SCHEMA_VERSION,
        "feature_count": len(ACTION_CLASSES) + 8,
        "feature_authority": "TARGET_PUBLIC_FACTS_AND_CANDIDATE_ACTION_ONLY",
        "outcome_fields_used_at_inference": False,
        "classes": [str(value) for value in model.classes_],
        "coefs": [np.asarray(value).tolist() for value in model.coefs_],
        "intercepts": [np.asarray(value).tolist() for value in model.intercepts_],
        "activation": "RELU",
        "output_activation": "SOFTMAX",
    }
    return body | {"grounder_sha256": stable_hash(body)}


def predict_grounding(artifact: Mapping[str, Any], step: Mapping[str, Any]) -> tuple[str, dict[str, float]]:
    body = dict(artifact)
    claimed = body.pop("grounder_sha256", None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError("Normal neural grounder hash mismatch")
    hidden = np.asarray(target_grounding_features(step), dtype=float)[None, :]
    coefs = [np.asarray(value, dtype=float) for value in artifact["coefs"]]
    intercepts = [np.asarray(value, dtype=float) for value in artifact["intercepts"]]
    for index, (weights, bias) in enumerate(zip(coefs, intercepts)):
        hidden = hidden @ weights + bias
        if index + 1 < len(coefs):
            hidden = np.maximum(hidden, 0.0)
    logits = hidden[0] - float(np.max(hidden[0]))
    probabilities = np.exp(logits) / np.exp(logits).sum()
    scores = {
        str(label): float(probabilities[index])
        for index, label in enumerate(artifact["classes"])
    }
    return max(scores, key=scores.get), scores


__all__ = [
    "GROUNDING_CLASSES", "export_neural_grounder", "induce_target_only_program",
    "positive_binding_count", "predict_grounding", "source_role_operator_ids",
    "target_grounding_features", "target_grounding_label", "trace_conforms",
    "typed_trace",
]
