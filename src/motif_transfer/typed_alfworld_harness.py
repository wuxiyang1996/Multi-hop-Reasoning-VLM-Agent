"""Target-native realization of an action-free typed source-effect graph."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


CONDITIONS = (
    "target_only",
    "authentic_typed_ir",
    "edge_permuted_ir",
    "wrong_guard_ir",
)

CONCRETE_ACTION_RANKINGS = (
    "realization_first",
    "target_policy_within_effect",
)

TARGET_EFFECT_BY_OPTION = {
    "SEARCH": "POSITION",
    "ACQUIRE": "BIND",
    "TRANSFORM": "MUTATE",
    "PLACE": "RELATE",
    "VERIFY": "VERIFY",
}


def validate_typed_effect_ir(source_ir: Mapping[str, Any]) -> None:
    nodes = set(map(str, source_ir.get("nodes", ())))
    required = {"BIND", "MUTATE", "RELATE"}
    if not required.issubset(nodes):
        raise ValueError("typed source IR lacks required target-transfer nodes")
    edges = {
        (str(edge["from"]), str(edge["to"]), str(edge["guard"]))
        for edge in source_ir.get("edges", ())
    }
    if ("BIND", "MUTATE", "CARRIER_BOUND") not in edges:
        raise ValueError("typed source IR lacks guarded BIND->MUTATE edge")
    if ("BIND", "RELATE", "CARRIER_BOUND") not in edges:
        raise ValueError("typed source IR lacks guarded BIND->RELATE edge")
    prohibited = set(map(str, source_ir.get("prohibited_runtime_fields", ())))
    if not {"source_action_ordinal", "environment_id"}.issubset(prohibited):
        raise ValueError("typed source IR does not prohibit source-native runtime fields")
    if source_ir.get("target_grounding") != "TARGET_NATIVE_NEURAL_PROBES_ONLY":
        raise ValueError("typed source IR does not require target-native grounding")


def target_effect(option: str) -> str:
    return TARGET_EFFECT_BY_OPTION.get(str(option), "EXCLUDE")


def target_cycle_state(
    history: Sequence[str], grounded_history_options: Sequence[str] | None = None
) -> dict[str, bool]:
    """Track only target-native effect receipts from previously executed actions."""
    options = tuple(grounded_history_options or ())
    if len(options) != len(history):
        raise ValueError("history actions and target-native option receipts must align")
    carrier_bound = False
    mutated_since_bind = False
    for option in options:
        effect = target_effect(option)
        if effect == "BIND":
            carrier_bound = True
            mutated_since_bind = False
        elif effect == "MUTATE" and carrier_bound:
            mutated_since_bind = True
        elif effect == "RELATE" and carrier_bound:
            carrier_bound = False
            mutated_since_bind = False
    return {
        "carrier_bound": carrier_bound,
        "mutated_since_bind": mutated_since_bind,
    }


def _policy_score(
    action: str, row: Mapping[str, Any], history: Sequence[str]
) -> float:
    return float(row.get("policy", row["applicability"])) / (
        1.0 + history.count(action)
    )


def _realization_score(
    action: str, row: Mapping[str, Any], history: Sequence[str]
) -> float:
    effect = target_effect(str(row["option"]))
    binding = 1.0 if effect == "POSITION" else float(row["binding"])
    return (
        float(row["applicability"])
        * (0.20 + 0.80 * float(row["completion"]))
        * (0.25 + 0.75 * binding)
        / (1.0 + history.count(action))
    )


def _authentic_effects(state: Mapping[str, bool]) -> tuple[str, ...]:
    if not state["carrier_bound"]:
        return ("BIND",)
    if state["mutated_since_bind"]:
        return ("RELATE",)
    return ("MUTATE", "RELATE")


def _condition_effects(
    condition: str, state: Mapping[str, bool]
) -> tuple[str, ...]:
    if condition == "authentic_typed_ir":
        return _authentic_effects(state)
    if condition == "wrong_guard_ir":
        return _authentic_effects({
            "carrier_bound": not state["carrier_bound"],
            "mutated_since_bind": state["mutated_since_bind"],
        })
    if condition == "edge_permuted_ir":
        # Alpha-permute graph node identities BIND->MUTATE->RELATE->BIND while
        # leaving the target grounder and action budgets fixed.
        permutation = {"BIND": "MUTATE", "MUTATE": "RELATE", "RELATE": "BIND"}
        return tuple(permutation[effect] for effect in _authentic_effects(state))
    return ()


def choose_typed_action(
    *,
    condition: str,
    grounded: Mapping[str, Mapping[str, Any]],
    history: Sequence[str],
    grounded_history_options: Sequence[str],
    source_ir: Mapping[str, Any],
    minimum_realization_score: float = 0.0,
    concrete_action_ranking: str = "realization_first",
    minimum_target_policy_ratio: float = 0.0,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown typed target condition: {condition}")
    if not grounded:
        raise ValueError("typed target Harness received no native candidates")
    if concrete_action_ranking not in CONCRETE_ACTION_RANKINGS:
        raise ValueError(
            f"unknown concrete-action ranking: {concrete_action_ranking}"
        )
    if not 0.0 <= minimum_target_policy_ratio <= 1.0:
        raise ValueError("minimum target-policy ratio must be in [0, 1]")
    validate_typed_effect_ir(source_ir)
    actions = tuple(grounded)
    fallback = max(
        actions,
        key=lambda action: (_policy_score(action, grounded[action], history), action),
    )
    fallback_effect = target_effect(str(grounded[fallback]["option"]))
    state = target_cycle_state(history, grounded_history_options)
    if condition == "target_only":
        return {
            "action": fallback,
            "fallback_action": fallback,
            "fallback_effect": fallback_effect,
            "source_selected_effect": None,
            "target_realized_effect": fallback_effect,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "diagnostic": "TARGET_ONLY",
            "target_cycle_state": state,
        }

    requested_effects = _condition_effects(condition, state)
    candidates = [
        action
        for action in actions
        if target_effect(str(grounded[action]["option"])) in requested_effects
    ]
    if not candidates:
        position_candidates = [
            action
            for action in actions
            if target_effect(str(grounded[action]["option"])) == "POSITION"
        ]
        selected = max(
            position_candidates or list(actions),
            key=lambda action: (_policy_score(action, grounded[action], history), action),
        )
        selected_effect = target_effect(str(grounded[selected]["option"]))
        return {
            "action": selected,
            "fallback_action": fallback,
            "fallback_effect": fallback_effect,
            "source_selected_effect": None,
            "requested_source_effects": list(requested_effects),
            "target_realized_effect": selected_effect,
            "source_admitted": False,
            "changed_action": selected != fallback,
            "changed_effect": selected_effect != fallback_effect,
            "diagnostic": "TARGET_REALIZATION_UNAVAILABLE_REPLAN_POSITION",
            "target_cycle_state": state,
        }

    if concrete_action_ranking == "target_policy_within_effect":
        # The source graph selects only an effect.  Reuse the exact target
        # policy used by the null condition to realize a concrete action
        # inside that effect, so structural transfer is not confounded with
        # swapping in a different action-ranking objective.
        selected = max(
            candidates,
            key=lambda action: (
                _policy_score(action, grounded[action], history),
                _realization_score(action, grounded[action], history),
                action,
            ),
        )
    else:
        selected = max(
            candidates,
            key=lambda action: (
                _realization_score(action, grounded[action], history),
                _policy_score(action, grounded[action], history),
                action,
            ),
        )
    selected_score = _realization_score(selected, grounded[selected], history)
    selected_policy_score = _policy_score(selected, grounded[selected], history)
    fallback_policy_score = _policy_score(fallback, grounded[fallback], history)
    target_policy_ratio = selected_policy_score / max(fallback_policy_score, 1e-12)
    policy_diagnostics = (
        {
            "selected_target_policy_score": selected_policy_score,
            "fallback_target_policy_score": fallback_policy_score,
            "target_policy_ratio": target_policy_ratio,
        }
        if concrete_action_ranking == "target_policy_within_effect"
        or minimum_target_policy_ratio > 0.0
        else {}
    )
    selected_effect = target_effect(str(grounded[selected]["option"]))
    if selected_score < minimum_realization_score:
        return {
            "action": fallback,
            "fallback_action": fallback,
            "fallback_effect": fallback_effect,
            "source_selected_effect": None,
            "requested_source_effects": list(requested_effects),
            "target_realized_effect": fallback_effect,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "diagnostic": "TARGET_NEURAL_GROUNDER_ABSTAINED",
            "target_cycle_state": state,
            "best_realization_score": selected_score,
        } | policy_diagnostics
    if target_policy_ratio < minimum_target_policy_ratio:
        return {
            "action": fallback,
            "fallback_action": fallback,
            "fallback_effect": fallback_effect,
            "source_selected_effect": None,
            "requested_source_effects": list(requested_effects),
            "target_realized_effect": fallback_effect,
            "source_admitted": False,
            "changed_action": False,
            "changed_effect": False,
            "diagnostic": "TARGET_POLICY_RELATIVE_ABSTENTION",
            "target_cycle_state": state,
            "best_realization_score": selected_score,
            "selected_target_policy_score": selected_policy_score,
            "fallback_target_policy_score": fallback_policy_score,
            "target_policy_ratio": target_policy_ratio,
        }
    return {
        "action": selected,
        "fallback_action": fallback,
        "fallback_effect": fallback_effect,
        "source_selected_effect": selected_effect,
        "requested_source_effects": list(requested_effects),
        "target_realized_effect": selected_effect,
        "source_admitted": True,
        "changed_action": selected != fallback,
        "changed_effect": selected_effect != fallback_effect,
        "diagnostic": "SOURCE_EFFECT_REALIZED_BY_TARGET_NEURAL_GROUNDER",
        "target_cycle_state": state,
        "best_realization_score": selected_score,
    } | policy_diagnostics


__all__ = [
    "CONDITIONS",
    "CONCRETE_ACTION_RANKINGS",
    "TARGET_EFFECT_BY_OPTION",
    "choose_typed_action",
    "target_cycle_state",
    "target_effect",
    "validate_typed_effect_ir",
]
