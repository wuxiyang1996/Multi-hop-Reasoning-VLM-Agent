"""Induce the prerequisite of a source goal-relation update from interventions.

The relation macro identifies *what* must recur, but does not explain how a
state with no executable positive relation intervention becomes one with a
unique executable intervention.  This module learns that missing program edge
from source-only ``(state, action, effect, next_state)`` tuples.

The hypothesis class is deliberately structural.  It measures how many native
interventions would increase the already-induced terminal relation feature,
labels the observed zero-to-positive cardinality change, and learns which
anonymous source operators occur while that cardinality is zero.  No source
action, coordinate, entity name, or named EXPLORE/BACKTRACK/COMMIT controller
is exported.  A target-native neural grounder remains responsible for binding
the learned anonymous operator types to target actions.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .relational_structural_induction import build_source_intervention_dataset
from .sokoban_commit_skill import NATIVE_ACTIONS, parse_state, simulate
from .source_goal_relation_induction import (
    validate_goal_relation_macro_program,
)
from .target_structural_induction import anonymous_operator_descriptor


ARTIFACT_VERSION = "SOURCE_INDUCED_GOAL_ACQUISITION_PROGRAM_V1"
DATASET_VERSION = "source-goal-acquisition-dataset-v1"
CONFIRMATION_VERSION = "SOURCE_GOAL_ACQUISITION_CONFIRMATION_V1"

BINDING_CARDINALITY_FEATURE = "positive_relation_intervention_cardinality"
BINDING_OPERATOR = anonymous_operator_descriptor(
    "UPDATE", "POSITIVE_EFFECT_BINDING", 1, "CANDIDATE_CARDINALITY",
)


def _relation_contract(
    relation_artifact: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Read the relation feature/operator selected by the source inducer."""

    validate_goal_relation_macro_program(relation_artifact)
    predicates = list(relation_artifact["program"]["terminal_predicates"])
    numeric = [
        row for row in predicates
        if row.get("operator") == "EQ"
        and isinstance(row.get("value"), (int, float))
        and not isinstance(row.get("value"), bool)
    ]
    if len(numeric) != 1:
        raise ValueError("relation artifact does not identify one numeric feature")
    operators = list(relation_artifact["operator_types"])
    if len(operators) != 1:
        raise ValueError("relation artifact does not identify one operator")
    return str(numeric[0]["feature"]), dict(operators[0])


def _relation_coverage(state: Any, feature: str) -> float:
    """Ground the induced source relation feature without target vocabulary."""

    if feature != "entity_goal_relation_coverage":
        raise ValueError(f"unsupported source relation feature: {feature}")
    return len(state.boxes & state.docks) / max(1, len(state.boxes))


def _positive_binding_count(state: Any, feature: str) -> int:
    before = _relation_coverage(state, feature)
    return sum(
        _relation_coverage(simulate(state, action).after, feature) > before
        for action in NATIVE_ACTIONS
        if action != "no_op"
    )


def _checked_primitive_operator_ids(row: Mapping[str, Any]) -> tuple[str, ...]:
    effect = dict(row.get("effect") or {})
    claimed = str(effect.pop("effect_sha256", ""))
    if not claimed or stable_hash(effect) != claimed:
        raise ValueError("primitive source effect hash mismatch")
    return tuple(sorted(
        str(atom["operator_type_id"])
        for atom in effect.get("atoms") or ()
    ))


def _ordinary_descriptor(row: Mapping[str, Any]) -> dict[str, Any]:
    """Select the most relational observed primitive atom, without names."""

    atoms = list((row.get("effect") or {}).get("atoms") or ())
    if not atoms:
        raise ValueError("successful source transition has no observed operator")
    # Higher arity is the more specific graph update.  Stable hashing is only
    # a deterministic tie-break; source action tokens never participate.
    return dict(max(
        atoms,
        key=lambda atom: (int(atom.get("arity", 0)), stable_hash(atom)),
    ))


def _typed_transition(
    row: Mapping[str, Any], *, relation_feature: str,
    relation_operator: Mapping[str, Any],
) -> dict[str, Any]:
    before = parse_state(str(row["state"]))
    after = parse_state(str(row["next_state"]))
    before_relation = _relation_coverage(before, relation_feature)
    after_relation = _relation_coverage(after, relation_feature)
    before_binding = _positive_binding_count(before, relation_feature)
    after_binding = _positive_binding_count(after, relation_feature)
    primitive_ids = _checked_primitive_operator_ids(row)
    if after_relation > before_relation:
        descriptor = dict(relation_operator)
        effect_kind = "RELATION_FEATURE_INCREASE"
        changed_feature = relation_feature
        change_sign = "INCREASE"
    elif before_binding == 0 and after_binding > 0:
        descriptor = dict(BINDING_OPERATOR)
        effect_kind = "POSITIVE_BINDING_CARDINALITY_INCREASE"
        changed_feature = BINDING_CARDINALITY_FEATURE
        change_sign = "INCREASE"
    elif before_binding > 0 and after_binding == 0:
        descriptor = dict(BINDING_OPERATOR)
        effect_kind = "POSITIVE_BINDING_CARDINALITY_DECREASE"
        changed_feature = BINDING_CARDINALITY_FEATURE
        change_sign = "DECREASE"
    else:
        descriptor = _ordinary_descriptor(row)
        effect_kind = "OBSERVED_ANONYMOUS_STATE_UPDATE"
        changed_feature = None
        change_sign = None
    effect_body = {
        "schema_version": "anonymous-acquisition-effect-v1",
        "operator_type": descriptor,
        "effect_kind": effect_kind,
        "changed_feature": changed_feature,
        "change_sign": change_sign,
        "primitive_operator_type_ids": list(primitive_ids),
        "state_changed": bool((row.get("effect") or {}).get("state_changed")),
    }
    effect = effect_body | {"effect_sha256": stable_hash(effect_body)}
    core = {
        "state": str(row["state"]),
        "action": str(row["action"]),
        "effect": effect,
        "next_state": str(row["next_state"]),
        "before_features": {
            relation_feature: before_relation,
            BINDING_CARDINALITY_FEATURE: before_binding,
        },
        "next_features": {
            relation_feature: after_relation,
            BINDING_CARDINALITY_FEATURE: after_binding,
        },
    }
    return core | {"tuple_sha256": stable_hash(core)}


def build_goal_acquisition_dataset(
    plan: Mapping[str, Any], relation_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize successful source paths with intervention cardinalities."""

    relation_feature, relation_operator = _relation_contract(relation_artifact)
    primitive = build_source_intervention_dataset(plan)
    trajectories = []
    for episode in primitive["episodes"]:
        successes = [
            row for row in episode["candidates"]
            if row["success_from_state_only"]
        ]
        if len(successes) != 1:
            continue
        transitions = [
            _typed_transition(
                row,
                relation_feature=relation_feature,
                relation_operator=relation_operator,
            )
            for row in successes[0]["tuples"]
        ]
        if not transitions:
            continue
        trajectories.append({
            "snapshot_id": str(episode["snapshot_id"]),
            "episode_id": str(episode["episode_id"]),
            "candidate_id": str(successes[0]["candidate_id"]),
            "transitions": transitions,
        })
    if not trajectories:
        raise ValueError("no successful source acquisition trajectories")
    body = {
        "schema_version": DATASET_VERSION,
        "authority": "SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_ONLY",
        "source_plan_sha256": str(plan["plan_sha256"]),
        "primitive_dataset_sha256": str(primitive["dataset_sha256"]),
        "relation_artifact_sha256": str(relation_artifact["artifact_sha256"]),
        "relation_feature": relation_feature,
        "binding_cardinality_feature": BINDING_CARDINALITY_FEATURE,
        "trajectories": trajectories,
        "target_data_read": False,
    }
    return body | {"dataset_sha256": stable_hash(body)}


def validate_goal_acquisition_dataset(dataset: Mapping[str, Any]) -> None:
    body = dict(dataset)
    claimed = str(body.pop("dataset_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("source acquisition dataset hash mismatch")
    if dataset.get("schema_version") != DATASET_VERSION:
        raise ValueError("unsupported source acquisition dataset")
    if dataset.get("target_data_read") is not False:
        raise ValueError("target data leaked into source acquisition dataset")
    for trajectory in dataset.get("trajectories") or ():
        for row in trajectory.get("transitions") or ():
            core = dict(row)
            tuple_hash = str(core.pop("tuple_sha256", ""))
            if not tuple_hash or stable_hash(core) != tuple_hash:
                raise ValueError("source acquisition tuple hash mismatch")
            effect = dict(row.get("effect") or {})
            effect_hash = str(effect.pop("effect_sha256", ""))
            if not effect_hash or stable_hash(effect) != effect_hash:
                raise ValueError("source acquisition effect hash mismatch")


def _rows(dataset: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for trajectory in dataset["trajectories"]
        for row in trajectory["transitions"]
    ]


def _operator_id(row: Mapping[str, Any]) -> str:
    return str(row["effect"]["operator_type"]["operator_type_id"])


def induce_goal_acquisition_program(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Learn typed prerequisite operators, graph edges, and abstention rules."""

    validate_goal_acquisition_dataset(dataset)
    rows = _rows(dataset)
    binding_id = str(BINDING_OPERATOR["operator_type_id"])
    relation_rows = [
        row for row in rows
        if row["effect"]["effect_kind"] == "RELATION_FEATURE_INCREASE"
    ]
    onset_rows = [
        row for row in rows
        if row["effect"]["effect_kind"]
        == "POSITIVE_BINDING_CARDINALITY_INCREASE"
    ]
    if not relation_rows or not onset_rows:
        raise ValueError("source paths do not expose acquisition and relation effects")
    relation_ids = {_operator_id(row) for row in relation_rows}
    if len(relation_ids) != 1:
        raise ValueError("source paths do not identify one relation operator")
    relation_id = relation_ids.pop()
    relation_binding_counts = {
        int(row["before_features"][BINDING_CARDINALITY_FEATURE])
        for row in relation_rows
    }
    if len(relation_binding_counts) != 1:
        raise ValueError("relation update has ambiguous source binding cardinality")
    required_binding_count = relation_binding_counts.pop()
    if required_binding_count < 1:
        raise ValueError("relation update has no positive source binding")

    acquisition_rows = [
        row for row in rows
        if int(row["before_features"][BINDING_CARDINALITY_FEATURE]) == 0
        and row["effect"]["effect_kind"]
        == "OBSERVED_ANONYMOUS_STATE_UPDATE"
    ]
    descriptors = {
        _operator_id(row): dict(row["effect"]["operator_type"])
        for row in acquisition_rows
    }
    if not descriptors:
        raise ValueError("no source operator observed before positive binding")
    relation_descriptor = dict(relation_rows[0]["effect"]["operator_type"])

    edge_counts: Counter[tuple[str, str]] = Counter()
    for trajectory in dataset["trajectories"]:
        ids = [_operator_id(row) for row in trajectory["transitions"]]
        edge_counts.update(zip(ids, ids[1:]))
    transition_graph = [
        {
            "from_operator_type_id": left,
            "to_operator_type_id": right,
            "source_support": support,
        }
        for (left, right), support in sorted(edge_counts.items())
        if left in {*descriptors, binding_id, relation_id}
        and right in {*descriptors, binding_id, relation_id}
    ]
    onset_followed = onset_total = 0
    for trajectory in dataset["trajectories"]:
        transitions = list(trajectory["transitions"])
        for index, row in enumerate(transitions):
            if row["effect"]["effect_kind"] != (
                "POSITIVE_BINDING_CARDINALITY_INCREASE"
            ):
                continue
            onset_total += 1
            onset_followed += int(
                index + 1 < len(transitions)
                and transitions[index + 1]["effect"]["effect_kind"]
                == "RELATION_FEATURE_INCREASE"
            )
    if onset_followed != onset_total:
        raise ValueError("discovery binding onset does not deterministically precede relation")

    support = Counter(_operator_id(row) for row in acquisition_rows)
    operator_types = [
        descriptors[type_id] for type_id in sorted(descriptors)
    ] + [dict(BINDING_OPERATOR), relation_descriptor]
    body = {
        "schema_version": "source-induced-goal-acquisition-program-v1",
        "artifact_version": ARTIFACT_VERSION,
        "status": "SOURCE_GOAL_ACQUISITION_AWAITING_FRESH_CONFIRMATION",
        "source_receipts_sha256": str(dataset["dataset_sha256"]),
        "relation_artifact_sha256": str(dataset["relation_artifact_sha256"]),
        "induction_authority": "SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_ONLY",
        "operator_types": operator_types,
        "program": {
            "measured_precondition": {
                "feature": BINDING_CARDINALITY_FEATURE,
                "operator": "EQ",
                "value": 0,
            },
            "acquisition_operator_type_ids": sorted(descriptors),
            "binding_operator_type_id": binding_id,
            "relation_operator_type_id": relation_id,
            "relation_binding_cardinality": {
                "operator": "EQ",
                "value": required_binding_count,
            },
            "transition_graph": transition_graph,
            "binding_to_relation": {
                "from_operator_type_id": binding_id,
                "to_operator_type_id": relation_id,
                "guard": "OBSERVED_POSITIVE_BINDING_CARDINALITY",
                "discovery_support": onset_total,
                "discovery_precision": onset_followed / onset_total,
            },
            "abstention_rule": {
                "binding_cardinality_unobservable": "ABSTAIN",
                "binding_cardinality_above_induced_value": "ABSTAIN",
                "no_target_native_acquisition_binding": "ABSTAIN",
                "nonconforming_observed_effect": "ABSTAIN_AND_REMEASURE",
            },
        },
        "induction_diagnostics": {
            "trajectories": len(dataset["trajectories"]),
            "transitions": len(rows),
            "binding_onsets": onset_total,
            "relation_updates": len(relation_rows),
            "acquisition_operator_support": dict(sorted(support.items())),
            "learned_transition_edges": len(transition_graph),
        },
        "named_controller_template_used": False,
        "forbidden_named_controller_templates": [
            "EXPLORE", "BACKTRACK", "COMMIT",
        ],
        "target_binding": "TARGET_NATIVE_NEURAL_STRUCTURAL_GROUNDER",
        "target_data_read": False,
    }
    return body | {"artifact_sha256": stable_hash(body)}


def validate_goal_acquisition_program(artifact: Mapping[str, Any]) -> None:
    body = dict(artifact)
    claimed = str(body.pop("artifact_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("source acquisition artifact hash mismatch")
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported source acquisition artifact")
    if artifact.get("named_controller_template_used") is not False:
        raise ValueError("named controller template leaked into acquisition program")
    if artifact.get("target_data_read") is not False:
        raise ValueError("target data leaked into acquisition program")
    program = artifact.get("program") or {}
    known = {
        str(row["operator_type_id"])
        for row in artifact.get("operator_types") or ()
    }
    required = {
        *map(str, program.get("acquisition_operator_type_ids") or ()),
        str(program.get("binding_operator_type_id")),
        str(program.get("relation_operator_type_id")),
    }
    if not required <= known:
        raise ValueError("acquisition program references unknown operator types")


def _conforms(
    row: Mapping[str, Any], artifact: Mapping[str, Any], *,
    substitute_effect: Mapping[str, Any] | None = None,
) -> bool:
    effect = substitute_effect or row["effect"]
    operator_id = str(effect["operator_type"]["operator_type_id"])
    before = int(row["before_features"][BINDING_CARDINALITY_FEATURE])
    program = artifact["program"]
    if before == 0:
        return operator_id in {
            *map(str, program["acquisition_operator_type_ids"]),
            str(program["binding_operator_type_id"]),
        } and effect.get("effect_kind") != "POSITIVE_BINDING_CARDINALITY_DECREASE"
    required = int(program["relation_binding_cardinality"]["value"])
    return before == required and operator_id == str(
        program["relation_operator_type_id"]
    ) and effect.get("effect_kind") == "RELATION_FEATURE_INCREASE"


def _effect_binding_counts(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    if len(rows) < 2:
        raise ValueError("effect shuffle needs at least two source transitions")
    next_states = [str(row["next_state"]) for row in rows]
    offset = 1 + int(stable_hash(next_states), 16) % (len(rows) - 1)
    shuffled = next_states[offset:] + next_states[:offset]
    authentic = permuted = 0
    for row, fake_next in zip(rows, shuffled):
        predicted = simulate(
            parse_state(str(row["state"])), str(row["action"]),
        ).after
        authentic += predicted == parse_state(str(row["next_state"]))
        permuted += predicted == parse_state(fake_next)
    return authentic, permuted


def confirm_goal_acquisition_program(
    artifact: Mapping[str, Any], dataset: Mapping[str, Any], *,
    minimum_trajectories: int = 24,
    minimum_authentic_conformance: float = 0.95,
    minimum_authentic_minus_shuffled: float = 0.20,
    minimum_binding_to_relation_precision: float = 0.90,
) -> dict[str, Any]:
    """Confirm the induced program on held-out source transitions."""

    validate_goal_acquisition_program(artifact)
    validate_goal_acquisition_dataset(dataset)
    rows = _rows(dataset)
    effects = [row["effect"] for row in rows]
    offset = 1 + int(stable_hash(effects), 16) % (len(effects) - 1)
    shuffled = effects[offset:] + effects[:offset]
    authentic_count = sum(_conforms(row, artifact) for row in rows)
    shuffled_count = sum(
        _conforms(row, artifact, substitute_effect=effect)
        for row, effect in zip(rows, shuffled)
    )
    authentic_rate = authentic_count / len(rows)
    shuffled_rate = shuffled_count / len(rows)
    onset_total = onset_followed = 0
    receipts = []
    for trajectory in dataset["trajectories"]:
        transitions = list(trajectory["transitions"])
        trajectory_conforming = sum(_conforms(row, artifact) for row in transitions)
        for index, row in enumerate(transitions):
            if row["effect"]["effect_kind"] != (
                "POSITIVE_BINDING_CARDINALITY_INCREASE"
            ):
                continue
            onset_total += 1
            onset_followed += int(
                index + 1 < len(transitions)
                and transitions[index + 1]["effect"]["effect_kind"]
                == "RELATION_FEATURE_INCREASE"
            )
        receipts.append({
            "snapshot_id": str(trajectory["snapshot_id"]),
            "transitions": len(transitions),
            "conforming_transitions": trajectory_conforming,
        })
    precision = onset_followed / max(1, onset_total)
    authentic_bindings, shuffled_bindings = _effect_binding_counts(rows)
    gates = {
        "heldout_trajectory_coverage": (
            len(dataset["trajectories"]) >= minimum_trajectories
        ),
        "heldout_transition_conformance": (
            authentic_rate >= minimum_authentic_conformance
        ),
        "authentic_beats_shuffled_effect_conformance": (
            authentic_rate - shuffled_rate >= minimum_authentic_minus_shuffled
        ),
        "binding_onset_predicts_relation": (
            precision >= minimum_binding_to_relation_precision
        ),
        "authentic_effect_binding_exact": authentic_bindings == len(rows),
        "shuffled_effect_binding_rejected": shuffled_bindings == 0,
        "single_positive_binding_cardinality_induced": int(
            artifact["program"]["relation_binding_cardinality"]["value"]
        ) == 1,
        "source_only_lineage": dataset.get("target_data_read") is False,
        "no_named_controller_template": artifact.get(
            "named_controller_template_used"
        ) is False,
    }
    passed = all(gates.values())
    body = {
        "schema_version": "source-goal-acquisition-confirmation-v1",
        "confirmation_version": CONFIRMATION_VERSION,
        "status": (
            "SOURCE_GOAL_ACQUISITION_FRESH_VALIDATED" if passed
            else "SOURCE_GOAL_ACQUISITION_FRESH_FAILED"
        ),
        "claim_boundary": "FRESH_SOURCE_ONLY;NO_TARGET_EVIDENCE",
        "artifact_sha256": str(artifact["artifact_sha256"]),
        "source_dataset_sha256": str(dataset["dataset_sha256"]),
        "metrics": {
            "heldout_trajectories": len(dataset["trajectories"]),
            "heldout_transitions": len(rows),
            "authentic_conforming_transitions": authentic_count,
            "authentic_conformance_rate": authentic_rate,
            "shuffled_conforming_transitions": shuffled_count,
            "shuffled_conformance_rate": shuffled_rate,
            "binding_onsets": onset_total,
            "binding_to_relation_transitions": onset_followed,
            "binding_to_relation_precision": precision,
            "authentic_effect_bindings": authentic_bindings,
            "shuffled_effect_bindings": shuffled_bindings,
        },
        "gates": gates,
        "source_gate_passed": passed,
        "trajectory_receipts": receipts,
    }
    return body | {"report_sha256": stable_hash(body)}


__all__ = [
    "ARTIFACT_VERSION",
    "BINDING_CARDINALITY_FEATURE",
    "BINDING_OPERATOR",
    "CONFIRMATION_VERSION",
    "DATASET_VERSION",
    "build_goal_acquisition_dataset",
    "confirm_goal_acquisition_program",
    "induce_goal_acquisition_program",
    "validate_goal_acquisition_dataset",
    "validate_goal_acquisition_program",
]
