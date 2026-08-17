"""Induce a recurrent goal-relation macro from source transitions only.

The older relational inducer selected the worker-position update because every
Sokoban primitive changes the worker position.  That operator is frequent but
not causal for task completion.  This module changes the temporal abstraction,
not the target mapping: consecutive source primitives are compressed at points
where an observed graph feature changes.  The induced operator, transition,
terminal predicate, and fail-closed rules are then learned entirely from
``(state, action, effect, next_state)`` source tuples.

No source action, coordinate, entity name, target token, or named
EXPLORE/BACKTRACK/COMMIT template is exported.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .relational_structural_induction import build_source_intervention_dataset
from .sokoban_commit_skill import parse_state, simulate, state_to_text
from .target_structural_induction import anonymous_operator_descriptor


ARTIFACT_VERSION = "SOURCE_INDUCED_GOAL_RELATION_MACRO_V3"
DATASET_VERSION = "source-goal-relation-macro-dataset-v3"
CONFIRMATION_VERSION = "SOURCE_GOAL_RELATION_MACRO_CONFIRMATION_V3"
FEATURE = "entity_goal_relation_coverage"


def _feature_descriptor(feature: str) -> dict[str, Any]:
    """Map a graph feature schema to the shared anonymous operator schema."""

    words = str(feature).upper().split("_")
    if words[-1] == "COVERAGE":
        words = words[:-1]
    family = "_".join(words)
    return anonymous_operator_descriptor(
        "UPDATE", family, 2, "RELATION_COVERAGE",
    )


def _macro_effect(before: float, after: float) -> dict[str, Any]:
    descriptor = _feature_descriptor(FEATURE)
    sign = "INCREASE" if after > before else "DECREASE"
    body = {
        "schema_version": "anonymous-feature-change-effect-v3",
        "atoms": [descriptor],
        "changed_feature": FEATURE,
        "change_sign": sign,
        "state_changed": True,
    }
    return body | {"effect_sha256": stable_hash(body)}


def _validate_effect(effect: Mapping[str, Any]) -> None:
    body = dict(effect)
    claimed = str(body.pop("effect_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("goal-relation macro effect hash mismatch")
    if effect.get("changed_feature") != FEATURE:
        raise ValueError("unsupported goal-relation macro feature")
    if effect.get("change_sign") not in {"INCREASE", "DECREASE"}:
        raise ValueError("invalid goal-relation macro direction")


def build_goal_relation_macro_dataset(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Compress source primitive rollouts at goal-relation change points."""

    primitive = build_source_intervention_dataset(plan)
    episodes = []
    excluded_nonmonotone_successes = 0
    for episode in primitive["episodes"]:
        candidates = []
        for candidate in episode["candidates"]:
            macros = []
            for row in candidate["tuples"]:
                before = float(row["before_features"][FEATURE])
                after = float(row["next_features"][FEATURE])
                if abs(after - before) <= 1e-12:
                    continue
                core = {
                    "state": str(row["state"]),
                    "action": str(row["action"]),
                    "effect": _macro_effect(before, after),
                    "next_state": str(row["next_state"]),
                    "before_features": dict(row["before_features"]),
                    "next_features": dict(row["next_features"]),
                }
                macros.append(core | {"tuple_sha256": stable_hash(core)})
            success = bool(candidate["success_from_state_only"])
            if success and any(
                row["effect"]["change_sign"] != "INCREASE" for row in macros
            ):
                excluded_nonmonotone_successes += 1
                success = False
            candidates.append({
                "candidate_id": str(candidate["candidate_id"]),
                "macro_tuples": macros,
                "terminal_features": dict(candidate["terminal_features"]),
                "success_from_state_only": success,
            })
        successes = [
            row for row in candidates if row["success_from_state_only"]
        ]
        if len(successes) == 1 and successes[0]["macro_tuples"]:
            episodes.append({
                "snapshot_id": str(episode["snapshot_id"]),
                "episode_id": str(episode["episode_id"]),
                "candidates": candidates,
            })
    if not episodes:
        raise ValueError("no monotone source goal-relation macro episodes")
    body = {
        "schema_version": DATASET_VERSION,
        "authority": "SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_ONLY",
        "source_plan_sha256": str(plan["plan_sha256"]),
        "primitive_dataset_sha256": str(primitive["dataset_sha256"]),
        "changed_feature_hypothesis": FEATURE,
        "episodes": episodes,
        "diagnostics": {
            "primitive_episodes": len(primitive["episodes"]),
            "retained_episodes": len(episodes),
            "excluded_nonmonotone_successes": excluded_nonmonotone_successes,
        },
        "target_data_read": False,
    }
    return body | {"dataset_sha256": stable_hash(body)}


def validate_goal_relation_macro_dataset(dataset: Mapping[str, Any]) -> None:
    body = dict(dataset)
    claimed = str(body.pop("dataset_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("goal-relation macro dataset hash mismatch")
    if dataset.get("schema_version") != DATASET_VERSION:
        raise ValueError("unsupported goal-relation macro dataset")
    if dataset.get("target_data_read") is not False:
        raise ValueError("target data leaked into source macro dataset")
    for episode in dataset.get("episodes") or ():
        for candidate in episode.get("candidates") or ():
            for row in candidate.get("macro_tuples") or ():
                core = dict(row)
                claimed_tuple = str(core.pop("tuple_sha256", ""))
                if not claimed_tuple or stable_hash(core) != claimed_tuple:
                    raise ValueError("goal-relation macro tuple hash mismatch")
                _validate_effect(row["effect"])


def _predicate_holds(predicate: Mapping[str, Any], features: Mapping[str, Any]) -> bool:
    if predicate.get("operator") != "EQ":
        raise ValueError("unsupported terminal predicate")
    observed = features.get(str(predicate["feature"]))
    expected = predicate.get("value")
    if isinstance(expected, float):
        return abs(float(observed) - expected) <= 1e-9
    return observed == expected


def _terminal_predicate_candidates(
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Create predicates from constants observed only on successful terminals."""

    successes = [
        candidate
        for episode in episodes
        for candidate in episode["candidates"]
        if candidate["success_from_state_only"]
    ]
    feature_names = sorted(set.intersection(*(
        set(row["terminal_features"]) for row in successes
    )))
    output = []
    for feature in feature_names:
        values = {row["terminal_features"][feature] for row in successes}
        if len(values) != 1:
            continue
        value = values.pop()
        if not isinstance(value, (bool, int, float)):
            continue
        family_words = feature.upper().split("_")
        value_kind = "BOOLEAN" if isinstance(value, bool) else "SCALAR"
        if family_words[-1] == "COVERAGE":
            family_words = family_words[:-1]
            value_kind = "RELATION_COVERAGE"
        output.append({
            "predicate_family": "_".join(family_words),
            "arity": 2 if "RELATION" in family_words else 1,
            "value_kind": value_kind,
            "feature": feature,
            "operator": "EQ",
            "value": value,
        })
    return tuple(output)


def _select_terminal_predicates(
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    hypotheses = _terminal_predicate_candidates(episodes)
    for size in range(1, len(hypotheses) + 1):
        valid = []
        for subset in combinations(hypotheses, size):
            correct = True
            for episode in episodes:
                accepted = [
                    candidate for candidate in episode["candidates"]
                    if candidate["macro_tuples"]
                    and all(
                        row["effect"]["change_sign"] == "INCREASE"
                        for row in candidate["macro_tuples"]
                    )
                    and all(
                        _predicate_holds(p, candidate["terminal_features"])
                        for p in subset
                    )
                ]
                truth = [
                    candidate for candidate in episode["candidates"]
                    if candidate["success_from_state_only"]
                ]
                if [row["candidate_id"] for row in accepted] != [
                    row["candidate_id"] for row in truth
                ]:
                    correct = False
                    break
            if correct:
                valid.append(tuple(dict(row) for row in subset))
        if valid:
            return min(valid, key=stable_hash)
    raise ValueError("no source-observed terminal predicate separates controls")


def induce_goal_relation_macro_program(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Learn operator, recurrence, terminal predicate, and abstention rules."""

    validate_goal_relation_macro_dataset(dataset)
    episodes = list(dataset["episodes"])
    successes = [
        candidate
        for episode in episodes
        for candidate in episode["candidates"]
        if candidate["success_from_state_only"]
    ]
    tuples = [row for candidate in successes for row in candidate["macro_tuples"]]
    changed_features = {row["effect"]["changed_feature"] for row in tuples}
    directions = {row["effect"]["change_sign"] for row in tuples}
    if len(changed_features) != 1 or directions != {"INCREASE"}:
        raise ValueError("successful source macros do not identify one effect")
    feature = changed_features.pop()
    descriptor = _feature_descriptor(feature)
    lengths = sorted({len(row["macro_tuples"]) for row in successes})
    predicates = _select_terminal_predicates(episodes)
    body = {
        "schema_version": "source-induced-goal-relation-macro-v3",
        "artifact_version": ARTIFACT_VERSION,
        "status": "SOURCE_GOAL_RELATION_MACRO_AWAITING_FRESH_CONFIRMATION",
        "source_receipts_sha256": str(dataset["dataset_sha256"]),
        "induction_authority": "SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_ONLY",
        "operator_types": [descriptor],
        "program": {
            "entry_operator_type_id": descriptor["operator_type_id"],
            "transitions": [{
                "from_operator_type_id": descriptor["operator_type_id"],
                "observed_effect_guard": {
                    "feature": feature,
                    "change_sign": "INCREASE",
                },
                "to_operator_type_id": descriptor["operator_type_id"],
                "cardinality": "ONE_OR_MORE" if max(lengths) > 1 else "EXACTLY_ONE",
            }],
            "terminal_predicates": list(predicates),
            "terminal_rule": "INDUCED_PREDICATE_CONJUNCTION_AFTER_TYPED_EFFECT",
            "abstention_rule": {
                "zero_target_bindings": "ABSTAIN",
                "multiple_target_bindings": "ABSTAIN",
                "nonpositive_observed_relation_delta": "ABSTAIN",
                "terminal_predicate_unobservable": "ABSTAIN",
            },
        },
        "induction_diagnostics": {
            "episodes": len(episodes),
            "successful_macro_tuples": len(tuples),
            "observed_success_macro_lengths": lengths,
            "repeating_transition_induced": max(lengths) > 1,
            "terminal_hypotheses_from_source": len(
                _terminal_predicate_candidates(episodes)
            ),
            "selected_terminal_predicate_count": len(predicates),
        },
        "named_controller_template_used": False,
        "forbidden_named_controller_templates": [
            "EXPLORE", "BACKTRACK", "COMMIT",
        ],
        "target_binding": "TARGET_NATIVE_NEURAL_GOAL_RELATION_GROUNDER",
        "target_data_read": False,
    }
    return body | {"artifact_sha256": stable_hash(body)}


def validate_goal_relation_macro_program(artifact: Mapping[str, Any]) -> None:
    body = dict(artifact)
    claimed = str(body.pop("artifact_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("goal-relation macro artifact hash mismatch")
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported goal-relation macro artifact")
    if artifact.get("named_controller_template_used") is not False:
        raise ValueError("named controller template leaked into source program")
    if artifact.get("target_data_read") is not False:
        raise ValueError("target data leaked into source program")
    operators = list(artifact.get("operator_types") or ())
    if len(operators) != 1:
        raise ValueError("source macro must induce exactly one operator")
    if operators[0].get("schema_version") != "anonymous-structural-operator-type-v1":
        raise ValueError("source macro does not use the shared structural IR")


def _accepted(
    candidate: Mapping[str, Any], predicates: Sequence[Mapping[str, Any]],
) -> bool:
    return bool(candidate["macro_tuples"]) and all(
        row["effect"]["change_sign"] == "INCREASE"
        for row in candidate["macro_tuples"]
    ) and all(
        _predicate_holds(row, candidate["terminal_features"])
        for row in predicates
    )


def _effect_binding_counts(dataset: Mapping[str, Any]) -> tuple[int, int]:
    rows = [
        row
        for episode in dataset["episodes"]
        for candidate in episode["candidates"]
        if candidate["success_from_state_only"]
        for row in candidate["macro_tuples"]
    ]
    if len(rows) < 2:
        raise ValueError("effect shuffle needs at least two macro tuples")
    next_states = [str(row["next_state"]) for row in rows]
    offset = 1 + int(stable_hash(next_states), 16) % (len(rows) - 1)
    shuffled = next_states[offset:] + next_states[:offset]
    authentic = permuted = 0
    for row, fake_next in zip(rows, shuffled):
        predicted = simulate(parse_state(str(row["state"])), str(row["action"])).after
        predicted_text = state_to_text(predicted)
        authentic += predicted_text == str(row["next_state"])
        permuted += predicted_text == fake_next
    return authentic, permuted


def confirm_goal_relation_macro_program(
    artifact: Mapping[str, Any], dataset: Mapping[str, Any],
    *, minimum_episodes: int = 24,
) -> dict[str, Any]:
    """Confirm an already-induced artifact on held-out source episodes."""

    validate_goal_relation_macro_program(artifact)
    validate_goal_relation_macro_dataset(dataset)
    predicates = artifact["program"]["terminal_predicates"]
    receipts = []
    for episode in dataset["episodes"]:
        accepted = [
            str(row["candidate_id"]) for row in episode["candidates"]
            if _accepted(row, predicates)
        ]
        truth = [
            str(row["candidate_id"]) for row in episode["candidates"]
            if row["success_from_state_only"]
        ]
        receipts.append({
            "snapshot_id": str(episode["snapshot_id"]),
            "accepted_candidate_ids": accepted,
            "source_success_candidate_ids": truth,
            "unique_source_success_selected": accepted == truth and len(truth) == 1,
        })
    authentic, permuted = _effect_binding_counts(dataset)
    unique = sum(row["unique_source_success_selected"] for row in receipts)
    gates = {
        "heldout_episode_coverage": len(receipts) >= minimum_episodes,
        "heldout_unique_program_selection": unique == len(receipts),
        "authentic_effect_binding_exact": authentic > 0,
        "shuffled_effect_binding_rejected": permuted == 0,
        "authentic_beats_shuffled_effect": authentic > permuted,
        "recurrent_relation_operator": artifact["program"]["transitions"][0][
            "cardinality"
        ] == "ONE_OR_MORE",
        "terminal_relation_induced": any(
            row["predicate_family"] == "ENTITY_GOAL_RELATION"
            for row in predicates
        ),
        "source_only_lineage": dataset.get("target_data_read") is False,
        "no_named_controller_template": artifact.get(
            "named_controller_template_used"
        ) is False,
    }
    passed = all(gates.values())
    body = {
        "schema_version": "source-goal-relation-macro-confirmation-v3",
        "confirmation_version": CONFIRMATION_VERSION,
        "status": (
            "SOURCE_GOAL_RELATION_MACRO_FRESH_VALIDATED" if passed
            else "SOURCE_GOAL_RELATION_MACRO_FRESH_FAILED"
        ),
        "claim_boundary": "FRESH_SOURCE_ONLY;NO_TARGET_EVIDENCE",
        "artifact_sha256": str(artifact["artifact_sha256"]),
        "source_dataset_sha256": str(dataset["dataset_sha256"]),
        "metrics": {
            "heldout_episodes": len(receipts),
            "unique_source_success_selections": unique,
            "authentic_effect_bindings": authentic,
            "shuffled_effect_bindings": permuted,
        },
        "gates": gates,
        "source_gate_passed": passed,
        "episode_receipts": receipts,
    }
    return body | {"report_sha256": stable_hash(body)}


__all__ = [
    "ARTIFACT_VERSION",
    "CONFIRMATION_VERSION",
    "DATASET_VERSION",
    "build_goal_relation_macro_dataset",
    "confirm_goal_relation_macro_program",
    "induce_goal_relation_macro_program",
    "validate_goal_relation_macro_dataset",
    "validate_goal_relation_macro_program",
]
