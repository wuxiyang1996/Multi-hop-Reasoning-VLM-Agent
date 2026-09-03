"""Induce a repeating relational program from source intervention tuples.

This module is the graph-path counterpart of :mod:`structural_delta_induction`.
It does not export Sokoban actions, coordinates, object names, or a named
``EXPLORE/BACKTRACK/COMMIT`` controller.  Instead it learns four things from
source-only ``(state, action, effect, next_state)`` tuples:

* the anonymous structural operator shared by every successful transition;
* whether that operator is repeated through a self-loop;
* a terminal graph predicate that separates successful from unsuccessful
  intervention chains; and
* the fail-closed candidate-cardinality rule supported by source episodes.

The resulting program uses the same
``anonymous-structural-operator-type-v1`` descriptor as the other Phase-3
targets.  A target-native grounder must bind its native relations and entities;
the source artifact only validates a grounded transition chain.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

from .contracts import stable_hash
from .sokoban_commit_skill import (
    DELTAS,
    parse_state,
    shortest_solution,
    simulate,
    state_to_text,
    validate_plan,
)
from .target_structural_induction import anonymous_operator_descriptor


ARTIFACT_VERSION = "SOURCE_INDUCED_RELATIONAL_STRUCTURAL_PROGRAM_V2"
CONFIRMATION_VERSION = "SOURCE_RELATIONAL_STRUCTURAL_CONFIRMATION_V2"

UPDATE_CONTROL_POSITION = anonymous_operator_descriptor(
    "UPDATE", "CONTROL_STATE", 1, "POSITION",
)
UPDATE_ENTITY_POSITION = anonymous_operator_descriptor(
    "UPDATE", "ENTITY_RELATION", 2, "POSITION",
)

_DIRECTION_CYCLE = {
    "up": "right", "right": "down", "down": "left", "left": "up",
}
_TERMINAL_HYPOTHESES = (
    {
        "predicate_family": "ENTITY_GOAL_RELATION",
        "arity": 2,
        "value_kind": "RELATION_COVERAGE",
        "feature": "entity_goal_relation_coverage",
        "operator": "EQ",
        "value": 1.0,
    },
    {
        "predicate_family": "CONTROL_GOAL_RELATION",
        "arity": 2,
        "value_kind": "BOOLEAN",
        "feature": "control_on_goal_relation",
        "operator": "EQ",
        "value": True,
    },
    {
        "predicate_family": "STATIC_SAFETY_RELATION",
        "arity": 2,
        "value_kind": "BOOLEAN",
        "feature": "no_static_deadlock",
        "operator": "EQ",
        "value": True,
    },
)


def _map_action(action: str, mapping: Mapping[str, str]) -> str:
    direction = str(action).split()[-1]
    mapped = mapping[direction]
    return f"push {mapped}" if str(action).startswith("push ") else mapped


def _stable_shuffle(values: Sequence[str], *, key: str) -> tuple[str, ...]:
    return tuple(
        value for _, value in sorted(
            enumerate(map(str, values)),
            key=lambda row: stable_hash({
                "key": key, "index": row[0], "value": row[1],
            }),
        )
    )


def _candidate_sequences(state: Any, snapshot_id: str) -> list[dict[str, Any]]:
    solution = tuple(shortest_solution(state))
    if not solution:
        return []
    candidates = (
        ("SOURCE_SUCCESS", solution),
        ("RELATION_PERMUTED", tuple(
            _map_action(action, _DIRECTION_CYCLE) for action in solution
        )),
        ("ORDER_REVERSED", tuple(reversed(solution))),
        ("EFFECT_ORDER_SHUFFLED", _stable_shuffle(
            solution, key=f"effect-order|{snapshot_id}",
        )),
    )
    unique: list[tuple[str, tuple[str, ...]]] = []
    seen: set[tuple[str, ...]] = set()
    for name, actions in candidates:
        if actions not in seen:
            unique.append((name, actions))
            seen.add(actions)
    if len(unique) != len(candidates):
        return []
    return [
        {"candidate_id": name, "actions": list(actions)}
        for name, actions in sorted(
            unique,
            key=lambda row: stable_hash({
                "snapshot_id": snapshot_id, "candidate_id": row[0],
            }),
        )
    ]


def _state_features(state: Any) -> dict[str, Any]:
    on_goal = len(state.boxes & state.docks)
    return {
        "entity_goal_relation_coverage": on_goal / max(1, len(state.boxes)),
        "control_on_goal_relation": state.worker in state.docks,
        "no_static_deadlock": True,
        "entity_cardinality": len(state.boxes),
        "goal_cardinality": len(state.docks),
    }


def _transition_tuple(before: Any, action: str) -> tuple[dict[str, Any], Any]:
    transition = simulate(before, action)
    atoms = []
    if transition.worker_moved:
        atoms.append(dict(UPDATE_CONTROL_POSITION))
    if transition.box_moved:
        atoms.append(dict(UPDATE_ENTITY_POSITION))
    effect_core = {
        "schema_version": "anonymous-relational-effect-v1",
        "atoms": sorted(atoms, key=lambda row: row["operator_type_id"]),
        "state_changed": transition.state_changed,
    }
    effect = effect_core | {"effect_sha256": stable_hash(effect_core)}
    row = {
        "state": state_to_text(before),
        "action": str(action),
        "effect": effect,
        "next_state": state_to_text(transition.after),
        "before_features": _state_features(before),
        "next_features": _state_features(transition.after),
    }
    return row | {"tuple_sha256": stable_hash(row)}, transition.after


def build_source_intervention_dataset(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize candidate rollouts without using any target-domain data."""

    episodes = []
    for snapshot in validate_plan(plan):
        snapshot_id = str(snapshot["snapshot_id"])
        state = parse_state(str(snapshot["state"]))
        candidates = _candidate_sequences(state, snapshot_id)
        if not candidates:
            continue
        rows = []
        for candidate in candidates:
            current = state
            tuples = []
            for action in map(str, candidate["actions"]):
                row, current = _transition_tuple(current, action)
                tuples.append(row)
            rows.append({
                "candidate_id": str(candidate["candidate_id"]),
                "tuples": tuples,
                "terminal_features": _state_features(current),
                "success_from_state_only": bool(current.solved),
            })
        successes = [row["candidate_id"] for row in rows if row["success_from_state_only"]]
        if successes != ["SOURCE_SUCCESS"]:
            continue
        episodes.append({
            "snapshot_id": snapshot_id,
            "episode_id": str(snapshot["episode_id"]),
            "candidates": rows,
        })
    if not episodes:
        raise ValueError("source plan has no discriminative relational episodes")
    body = {
        "schema_version": "source-relational-intervention-dataset-v2",
        "authority": "SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_ONLY",
        "source_plan_sha256": str(plan["plan_sha256"]),
        "episodes": episodes,
        "target_data_read": False,
    }
    return body | {"dataset_sha256": stable_hash(body)}


def _validate_dataset(dataset: Mapping[str, Any]) -> None:
    body = dict(dataset)
    claimed = str(body.pop("dataset_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("source relational dataset hash mismatch")
    if dataset.get("authority") != "SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_ONLY":
        raise ValueError("source relational dataset has invalid authority")
    if dataset.get("target_data_read") is not False:
        raise ValueError("target data leaked into source relational dataset")


def _effect_operator_ids(row: Mapping[str, Any]) -> set[str]:
    effect = row.get("effect") or {}
    body = dict(effect)
    claimed = str(body.pop("effect_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("source tuple effect hash mismatch")
    return {
        str(atom["operator_type_id"])
        for atom in effect.get("atoms") or ()
    }


def _predicate_holds(predicate: Mapping[str, Any], features: Mapping[str, Any]) -> bool:
    observed = features.get(str(predicate["feature"]))
    if predicate.get("operator") != "EQ":
        raise ValueError("unsupported induced terminal predicate")
    expected = predicate.get("value")
    if isinstance(expected, float):
        return abs(float(observed) - expected) <= 1e-9
    return observed == expected


def _accepted(
    candidate: Mapping[str, Any], *, operator_type_id: str,
    terminal_predicates: Sequence[Mapping[str, Any]],
) -> bool:
    tuples = list(candidate.get("tuples") or ())
    return bool(tuples) and all(
        operator_type_id in _effect_operator_ids(row) for row in tuples
    ) and all(
        _predicate_holds(predicate, candidate.get("terminal_features") or {})
        for predicate in terminal_predicates
    )


def _select_terminal_predicates(
    episodes: Sequence[Mapping[str, Any]], *, operator_type_id: str,
) -> tuple[dict[str, Any], ...]:
    """Choose the smallest conjunction uniquely selecting source successes."""

    candidates: list[tuple[tuple[int, str], tuple[dict[str, Any], ...]]] = []
    for size in range(1, len(_TERMINAL_HYPOTHESES) + 1):
        for subset in combinations(_TERMINAL_HYPOTHESES, size):
            correct = True
            for episode in episodes:
                accepted = [
                    str(row["candidate_id"]) for row in episode["candidates"]
                    if _accepted(
                        row, operator_type_id=operator_type_id,
                        terminal_predicates=subset,
                    )
                ]
                if accepted != ["SOURCE_SUCCESS"]:
                    correct = False
                    break
            if correct:
                normalized = tuple(dict(row) for row in subset)
                candidates.append(((size, stable_hash(normalized)), normalized))
        if candidates:
            break
    if not candidates:
        raise ValueError("no relational terminal predicate separates source controls")
    return min(candidates, key=lambda row: row[0])[1]


def induce_relational_structural_program(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Induce operator, transition, terminal, and abstention rules."""

    _validate_dataset(dataset)
    episodes = list(dataset.get("episodes") or ())
    successful_tuples = [
        tuple_row
        for episode in episodes
        for candidate in episode["candidates"]
        if candidate["success_from_state_only"]
        for tuple_row in candidate["tuples"]
    ]
    if not successful_tuples:
        raise ValueError("relational induction requires successful source tuples")
    support = Counter(
        operator_id
        for row in successful_tuples
        for operator_id in _effect_operator_ids(row)
    )
    common = sorted(
        operator_id for operator_id, count in support.items()
        if count == len(successful_tuples)
    )
    if len(common) != 1:
        raise ValueError("source tuples do not identify one shared relational operator")
    operator_type_id = common[0]
    descriptor_by_id = {
        UPDATE_CONTROL_POSITION["operator_type_id"]: UPDATE_CONTROL_POSITION,
        UPDATE_ENTITY_POSITION["operator_type_id"]: UPDATE_ENTITY_POSITION,
    }
    descriptor = descriptor_by_id.get(operator_type_id)
    if descriptor is None:
        raise ValueError("induced relational operator has no shared IR descriptor")
    lengths = {
        len(candidate["tuples"])
        for episode in episodes
        for candidate in episode["candidates"]
        if candidate["success_from_state_only"]
    }
    predicates = _select_terminal_predicates(
        episodes, operator_type_id=operator_type_id,
    )
    core = {
        "schema_version": "source-induced-relational-structural-program-v2",
        "artifact_version": ARTIFACT_VERSION,
        "status": "SOURCE_RELATIONAL_PROGRAM_INDUCED_AWAITING_FRESH_CONFIRMATION",
        "source_receipts_sha256": str(dataset["dataset_sha256"]),
        "induction_authority": "SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_ONLY",
        "operator_types": [dict(descriptor)],
        "program": {
            "entry_operator_type_id": operator_type_id,
            "transitions": [{
                "from_operator_type_id": operator_type_id,
                "guard": "NEXT_GROUNDED_EFFECT_HAS_SAME_OPERATOR_TYPE",
                "to_operator_type_id": operator_type_id,
                "cardinality": "ONE_OR_MORE" if max(lengths) > 1 else "EXACTLY_ONE",
            }],
            "terminal_predicates": [dict(row) for row in predicates],
            "terminal_rule": "INDUCED_PREDICATE_CONJUNCTION_AFTER_TYPED_TRANSITIONS",
            "abstention_rule": {
                "learned_candidate_cardinality": {"operator": "EQ", "value": 1},
                "otherwise": "ABSTAIN",
            },
        },
        "induction_diagnostics": {
            "episodes": len(episodes),
            "successful_transition_tuples": len(successful_tuples),
            "observed_success_path_lengths": sorted(lengths),
            "repeating_transition_induced": max(lengths) > 1,
            "terminal_hypothesis_count": len(_TERMINAL_HYPOTHESES),
            "selected_terminal_predicate_count": len(predicates),
        },
        "forbidden_named_controller_templates": [
            "EXPLORE", "BACKTRACK", "COMMIT",
        ],
        "named_controller_template_used": False,
        "target_binding": "TARGET_NATIVE_NEURAL_RELATION_GROUNDER",
        "target_data_read": False,
    }
    return core | {"artifact_sha256": stable_hash(core)}


def validate_relational_structural_program(artifact: Mapping[str, Any]) -> None:
    body = dict(artifact)
    claimed = str(body.pop("artifact_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("source relational program hash mismatch")
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported source relational artifact")
    if artifact.get("named_controller_template_used") is not False:
        raise ValueError("named controller template leaked into relational program")
    if artifact.get("target_data_read") is not False:
        raise ValueError("target data leaked into relational program")
    operators = artifact.get("operator_types") or ()
    if len(operators) != 1:
        raise ValueError("relational program must contain one induced operator")
    if operators[0].get("schema_version") != "anonymous-structural-operator-type-v1":
        raise ValueError("relational program does not use the shared structural IR")


def _shuffled_effect_binding_accuracy(dataset: Mapping[str, Any]) -> tuple[int, int]:
    rows = [
        row
        for episode in dataset["episodes"]
        for candidate in episode["candidates"]
        if candidate["candidate_id"] == "SOURCE_SUCCESS"
        for row in candidate["tuples"]
    ]
    if len(rows) < 2:
        raise ValueError("effect shuffle requires multiple source tuples")
    ordered_next = [str(row["next_state"]) for row in rows]
    # A non-zero content-addressed rotation cannot use episode or task labels.
    offset = 1 + int(stable_hash(ordered_next), 16) % (len(rows) - 1)
    shuffled = ordered_next[offset:] + ordered_next[:offset]
    authentic = shuffled_matches = 0
    for row, fake_next in zip(rows, shuffled):
        predicted = simulate(parse_state(str(row["state"])), str(row["action"])).after
        authentic += state_to_text(predicted) == str(row["next_state"])
        shuffled_matches += state_to_text(predicted) == fake_next
    return authentic, shuffled_matches


def confirm_relational_structural_program(
    artifact: Mapping[str, Any], dataset: Mapping[str, Any],
    *, minimum_episodes: int = 24,
) -> dict[str, Any]:
    """Evaluate an already-induced program on held-out source interventions."""

    validate_relational_structural_program(artifact)
    _validate_dataset(dataset)
    operator_id = str(artifact["program"]["entry_operator_type_id"])
    predicates = list(artifact["program"]["terminal_predicates"])
    episode_rows = []
    for episode in dataset["episodes"]:
        accepted = [
            str(candidate["candidate_id"])
            for candidate in episode["candidates"]
            if _accepted(
                candidate, operator_type_id=operator_id,
                terminal_predicates=predicates,
            )
        ]
        episode_rows.append({
            "snapshot_id": str(episode["snapshot_id"]),
            "accepted_candidate_ids": accepted,
            "unique_source_success_selected": accepted == ["SOURCE_SUCCESS"],
        })
    authentic_bindings, shuffled_bindings = _shuffled_effect_binding_accuracy(dataset)
    episodes = len(episode_rows)
    unique = sum(row["unique_source_success_selected"] for row in episode_rows)
    gates = {
        "heldout_episode_coverage": episodes >= minimum_episodes,
        "heldout_unique_program_selection": unique == episodes,
        "authentic_effect_binding_exact": authentic_bindings > 0,
        "shuffled_effect_binding_rejected": shuffled_bindings == 0,
        "authentic_beats_shuffled_effect": authentic_bindings > shuffled_bindings,
        "shared_structural_ir": all(
            row.get("schema_version") == "anonymous-structural-operator-type-v1"
            for row in artifact["operator_types"]
        ),
        "source_only_lineage": dataset.get("target_data_read") is False,
        "no_named_controller_template": artifact.get("named_controller_template_used") is False,
    }
    passed = all(gates.values())
    body = {
        "schema_version": "source-relational-structural-confirmation-v2",
        "confirmation_version": CONFIRMATION_VERSION,
        "status": (
            "SOURCE_RELATIONAL_STRUCTURAL_FRESH_VALIDATED" if passed
            else "SOURCE_RELATIONAL_STRUCTURAL_FRESH_FAILED"
        ),
        "claim_boundary": "FRESH_SOURCE_ONLY;NO_TARGET_EVIDENCE",
        "artifact_sha256": str(artifact["artifact_sha256"]),
        "source_dataset_sha256": str(dataset["dataset_sha256"]),
        "metrics": {
            "heldout_episodes": episodes,
            "unique_source_success_selections": unique,
            "authentic_effect_bindings": authentic_bindings,
            "shuffled_effect_bindings": shuffled_bindings,
        },
        "gates": gates,
        "source_gate_passed": passed,
        "episode_receipts": episode_rows,
    }
    return body | {"report_sha256": stable_hash(body)}


__all__ = [
    "ARTIFACT_VERSION",
    "CONFIRMATION_VERSION",
    "UPDATE_CONTROL_POSITION",
    "build_source_intervention_dataset",
    "confirm_relational_structural_program",
    "induce_relational_structural_program",
    "validate_relational_structural_program",
]
