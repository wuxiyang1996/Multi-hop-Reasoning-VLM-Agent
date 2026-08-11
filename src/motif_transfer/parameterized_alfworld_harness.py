"""Role-aware target grounding for a parameterized real-source effect graph."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .alfworld_hierarchical_grounder import action_verb, tokens
from .contracts import stable_hash
from .typed_alfworld_harness import target_effect, validate_typed_effect_ir


CONDITIONS = (
    "target_only",
    "authentic_parameterized_ir",
    "edge_permuted_ir",
    "property_permuted_router",
)
PROPERTY_CLASSES = ("NONE", "CLEAN", "HEAT", "COOL", "LIGHT")
PROPERTY_BY_VERB = {
    "clean": "CLEAN",
    "heat": "HEAT",
    "cool": "COOL",
    "toggle": "LIGHT",
    "use": "LIGHT",
}
RECEIPTS = ("IGNORE", "BIND_ROLE", "MUTATE_REQUIRED_PROPERTY", "RELATE_ROLE")


def parameterize_source_ir(source_ir: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve source edge lineage while replacing unbound coarse node names."""
    validate_typed_effect_ir(source_ir)
    edges = {
        (str(row["from"]), str(row["to"])): row
        for row in source_ir["edges"]
    }
    mutate = edges[("BIND", "MUTATE")]
    relate = edges[("BIND", "RELATE")]
    body = {
        "schema_version": "parameterized-real-source-effect-ir-v7",
        "parent_ir_sha256": str(source_ir["ir_sha256"]),
        "induction_split": str(source_ir["induction_split"]),
        "validation_splits": list(source_ir["validation_splits"]),
        "nodes": [
            {"effect": "BIND", "roles": ["goal_object"]},
            {
                "effect": "ACHIEVE_UNARY_GOAL",
                "roles": ["goal_object", "required_property"],
            },
            {
                "effect": "RELATE",
                "roles": ["goal_object", "goal_receptacle"],
            },
        ],
        "edges": [
            {
                "from": "BIND(goal_object)",
                "to": "ACHIEVE_UNARY_GOAL(goal_object, required_property)",
                "guard": "TARGET_NEURAL_UNARY_GOAL_UNSATISFIED",
                "supporting_source_tasks": list(
                    mutate["supporting_source_tasks"]
                ),
            },
            {
                "from": "BIND(goal_object)",
                "to": "RELATE(goal_object, goal_receptacle)",
                "guard": "TARGET_NEURAL_NO_UNARY_GOAL_OR_SATISFIED",
                "supporting_source_tasks": list(
                    relate["supporting_source_tasks"]
                ),
            },
            {
                "from": "ACHIEVE_UNARY_GOAL(goal_object, required_property)",
                "to": "RELATE(goal_object, goal_receptacle)",
                "guard": "TARGET_NEURAL_UNARY_GOAL_SATISFIED",
                "supporting_source_tasks": list(
                    mutate["supporting_source_tasks"]
                ),
            },
        ],
        "execution_authority": "SYMBOLIC_EFFECT_ROUTING_ONLY",
        "target_grounding": (
            "TARGET_NATIVE_NEURAL_PROPERTY_ROUTER_AND_ACTION_PROBES"
        ),
        "prohibited_runtime_fields": sorted(set(map(
            str, source_ir["prohibited_runtime_fields"]
        )) | {
            "source_action_ordinal",
            "environment_id",
            "source_task_id",
        }),
        "source_lineage": list(source_ir["source_lineage"]),
    }
    return body | {"ir_sha256": stable_hash(body)}


def validate_parameterized_source_ir(source_ir: Mapping[str, Any]) -> None:
    body = dict(source_ir)
    claimed = str(body.pop("ir_sha256", ""))
    if stable_hash(body) != claimed:
        raise ValueError("parameterized source IR hash mismatch")
    if source_ir.get("schema_version") != "parameterized-real-source-effect-ir-v7":
        raise ValueError("unsupported parameterized source IR")
    effects = {str(row["effect"]) for row in source_ir.get("nodes", ())}
    if effects != {"BIND", "ACHIEVE_UNARY_GOAL", "RELATE"}:
        raise ValueError("parameterized source IR has wrong node set")
    guards = {str(row["guard"]) for row in source_ir.get("edges", ())}
    required_guards = {
        "TARGET_NEURAL_UNARY_GOAL_UNSATISFIED",
        "TARGET_NEURAL_NO_UNARY_GOAL_OR_SATISFIED",
        "TARGET_NEURAL_UNARY_GOAL_SATISFIED",
    }
    if not required_guards.issubset(guards):
        raise ValueError("parameterized source IR lacks goal-conditioned guards")
    prohibited = set(map(str, source_ir.get("prohibited_runtime_fields", ())))
    if not {"source_action_ordinal", "environment_id", "source_task_id"}.issubset(
        prohibited
    ):
        raise ValueError("parameterized source IR permits source-native runtime data")


def property_router_features(goal: str, *, feature_bins: int) -> np.ndarray:
    if feature_bins <= 0:
        raise ValueError("property-router feature bins must be positive")
    values = tokens(goal)
    vector = np.zeros(feature_bins, dtype=np.float64)
    features = [f"unigram:{value}" for value in values]
    features.extend(
        f"bigram:{left}:{right}" for left, right in zip(values, values[1:])
    )
    for feature in features:
        index = int(hashlib.sha256(feature.encode()).hexdigest()[:16], 16)
        vector[index % feature_bins] += 1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential)


def property_router_probabilities(
    goal: str, artifact: Mapping[str, Any]
) -> dict[str, float]:
    value = property_router_features(
        goal, feature_bins=int(artifact["feature_bins"])
    )
    for layer_index, layer in enumerate(artifact["layers"]):
        value = (
            value @ np.asarray(layer["weights"], dtype=np.float64)
            + np.asarray(layer["bias"], dtype=np.float64)
        )
        if layer_index < len(artifact["layers"]) - 1:
            activation = str(artifact["hidden_activation"])
            value = np.tanh(value) if activation == "tanh" else np.maximum(value, 0.0)
    probabilities = _softmax(np.ravel(value))
    classes = tuple(map(str, artifact["classes"]))
    if len(classes) != len(probabilities):
        raise ValueError("property-router class/output mismatch")
    return {
        name: float(probability)
        for name, probability in zip(classes, probabilities)
    }


def validate_property_router(artifact: Mapping[str, Any]) -> None:
    body = dict(artifact)
    claimed = str(body.pop("artifact_sha256", ""))
    if stable_hash(body) != claimed:
        raise ValueError("target-native property router hash mismatch")
    if tuple(map(str, artifact.get("classes", ()))) != PROPERTY_CLASSES:
        raise ValueError("target-native property router has wrong classes")
    if artifact.get("training_authority") != "TARGET_ADAPTATION_EXPERT_ACTIONS_ONLY":
        raise ValueError("property router has invalid training authority")


def action_property(action: str) -> str:
    return PROPERTY_BY_VERB.get(action_verb(action), "NONE")


def property_label_from_actions(actions: Sequence[str]) -> str:
    labels = {
        action_property(action)
        for action in actions
        if action_property(action) != "NONE"
    }
    if len(labels) > 1:
        raise ValueError(f"expert trajectory has multiple unary properties: {labels}")
    return next(iter(labels), "NONE")


def parameterized_cycle_state(receipts: Sequence[str]) -> dict[str, bool]:
    bound = False
    property_achieved = False
    for receipt in receipts:
        if receipt not in RECEIPTS:
            raise ValueError(f"unknown target effect receipt: {receipt}")
        if receipt == "BIND_ROLE":
            bound = True
            property_achieved = False
        elif receipt == "MUTATE_REQUIRED_PROPERTY" and bound:
            property_achieved = True
        elif receipt == "RELATE_ROLE" and bound:
            bound = False
            property_achieved = False
    return {
        "goal_object_bound": bound,
        "required_property_achieved": property_achieved,
    }


def target_effect_receipt(
    *,
    action: str,
    grounding: Mapping[str, Any],
    required_property: str,
    minimum_role_binding: float,
) -> str:
    if float(grounding["binding"]) < minimum_role_binding:
        return "IGNORE"
    option = str(grounding["option"])
    if option == "ACQUIRE":
        return "BIND_ROLE"
    if (
        option == "TRANSFORM"
        and required_property != "NONE"
        and action_property(action) == required_property
    ):
        return "MUTATE_REQUIRED_PROPERTY"
    if option == "PLACE":
        return "RELATE_ROLE"
    return "IGNORE"


def _policy_score(
    action: str, row: Mapping[str, Any], history: Sequence[str]
) -> float:
    return float(row.get("policy", row["applicability"])) / (
        1.0 + history.count(action)
    )


def _realization_score(
    action: str, row: Mapping[str, Any], history: Sequence[str]
) -> float:
    return (
        float(row["applicability"])
        * (0.20 + 0.80 * float(row["completion"]))
        * (0.25 + 0.75 * float(row["binding"]))
        / (1.0 + history.count(action))
    )


def _permuted_property(property_name: str) -> str:
    permutation = {
        "NONE": "CLEAN",
        "CLEAN": "HEAT",
        "HEAT": "COOL",
        "COOL": "LIGHT",
        "LIGHT": "NONE",
    }
    return permutation[property_name]


def _requested_effect(
    state: Mapping[str, bool], required_property: str
) -> str:
    if not state["goal_object_bound"]:
        return "BIND"
    if required_property != "NONE" and not state["required_property_achieved"]:
        return "MUTATE"
    return "RELATE"


def choose_parameterized_action(
    *,
    condition: str,
    grounded: Mapping[str, Mapping[str, Any]],
    history: Sequence[str],
    effect_receipts: Sequence[str],
    source_ir: Mapping[str, Any],
    property_probabilities: Mapping[str, float],
    minimum_property_confidence: float,
    minimum_role_binding: float,
    minimum_realization_score: float,
    minimum_target_policy_ratio: float,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown parameterized Harness condition: {condition}")
    if not grounded:
        raise ValueError("parameterized Harness received no target-native actions")
    validate_parameterized_source_ir(source_ir)
    actions = tuple(grounded)
    fallback = max(
        actions,
        key=lambda action: (_policy_score(action, grounded[action], history), action),
    )
    fallback_effect = target_effect(str(grounded[fallback]["option"]))
    state = parameterized_cycle_state(effect_receipts)
    if condition == "target_only":
        return {
            "action": fallback,
            "fallback_action": fallback,
            "fallback_effect": fallback_effect,
            "target_realized_effect": fallback_effect,
            "source_selected_effect": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "diagnostic": "TARGET_ONLY",
            "target_cycle_state": state,
        }

    authentic_property = max(
        PROPERTY_CLASSES,
        key=lambda name: (float(property_probabilities.get(name, 0.0)), name),
    )
    property_confidence = float(property_probabilities.get(authentic_property, 0.0))
    required_property = (
        _permuted_property(authentic_property)
        if condition == "property_permuted_router"
        else authentic_property
    )
    requested_effect = _requested_effect(state, required_property)
    if condition == "edge_permuted_ir":
        requested_effect = {
            "BIND": "MUTATE",
            "MUTATE": "RELATE",
            "RELATE": "BIND",
        }[requested_effect]
    if property_confidence < minimum_property_confidence:
        return {
            "action": fallback,
            "fallback_action": fallback,
            "fallback_effect": fallback_effect,
            "target_realized_effect": fallback_effect,
            "source_selected_effect": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "diagnostic": "TARGET_PROPERTY_ROUTER_ABSTAINED",
            "target_cycle_state": state,
            "required_property": required_property,
            "authentic_required_property": authentic_property,
            "property_confidence": property_confidence,
        }

    candidates = []
    for action in actions:
        row = grounded[action]
        if float(row["binding"]) < minimum_role_binding:
            continue
        effect = target_effect(str(row["option"]))
        if effect != requested_effect:
            continue
        if effect == "MUTATE" and action_property(action) != required_property:
            continue
        candidates.append(action)
    if not candidates:
        return {
            "action": fallback,
            "fallback_action": fallback,
            "fallback_effect": fallback_effect,
            "target_realized_effect": fallback_effect,
            "source_selected_effect": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "diagnostic": "PARAMETERIZED_EFFECT_UNAVAILABLE_TARGET_ABSTENTION",
            "target_cycle_state": state,
            "requested_source_effect": requested_effect,
            "required_property": required_property,
            "authentic_required_property": authentic_property,
            "property_confidence": property_confidence,
        }

    selected = max(
        candidates,
        key=lambda action: (
            _policy_score(action, grounded[action], history),
            _realization_score(action, grounded[action], history),
            action,
        ),
    )
    selected_effect = target_effect(str(grounded[selected]["option"]))
    realization_score = _realization_score(selected, grounded[selected], history)
    selected_policy = _policy_score(selected, grounded[selected], history)
    fallback_policy = _policy_score(fallback, grounded[fallback], history)
    policy_ratio = selected_policy / max(fallback_policy, 1e-12)
    abstention = None
    if realization_score < minimum_realization_score:
        abstention = "TARGET_REALIZATION_SCORE_ABSTENTION"
    elif policy_ratio < minimum_target_policy_ratio:
        abstention = "TARGET_POLICY_RELATIVE_ABSTENTION"
    common = {
        "fallback_action": fallback,
        "fallback_effect": fallback_effect,
        "target_cycle_state": state,
        "requested_source_effect": requested_effect,
        "required_property": required_property,
        "authentic_required_property": authentic_property,
        "property_confidence": property_confidence,
        "best_realization_score": realization_score,
        "selected_target_policy_score": selected_policy,
        "fallback_target_policy_score": fallback_policy,
        "target_policy_ratio": policy_ratio,
    }
    if abstention:
        return common | {
            "action": fallback,
            "target_realized_effect": fallback_effect,
            "source_selected_effect": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "diagnostic": abstention,
        }
    return common | {
        "action": selected,
        "target_realized_effect": selected_effect,
        "source_selected_effect": selected_effect,
        "source_admitted": True,
        "changed_action": selected != fallback,
        "changed_effect": selected_effect != fallback_effect,
        "diagnostic": "PARAMETERIZED_SOURCE_EFFECT_TARGET_NEURAL_REALIZATION",
    }


__all__ = [
    "CONDITIONS",
    "PROPERTY_CLASSES",
    "action_property",
    "choose_parameterized_action",
    "parameterize_source_ir",
    "parameterized_cycle_state",
    "property_label_from_actions",
    "property_router_features",
    "property_router_probabilities",
    "target_effect_receipt",
    "validate_parameterized_source_ir",
    "validate_property_router",
]
