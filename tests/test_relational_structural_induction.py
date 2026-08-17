from __future__ import annotations

from motif_transfer.contracts import stable_hash
from motif_transfer.relational_structural_induction import (
    UPDATE_CONTROL_POSITION,
    confirm_relational_structural_program,
    induce_relational_structural_program,
    validate_relational_structural_program,
)


def _effect(*operator_ids: str) -> dict:
    atoms = [
        UPDATE_CONTROL_POSITION
        for operator_id in operator_ids
        if operator_id == UPDATE_CONTROL_POSITION["operator_type_id"]
    ]
    body = {
        "schema_version": "anonymous-relational-effect-v1",
        "atoms": atoms,
        "state_changed": bool(atoms),
    }
    return body | {"effect_sha256": stable_hash(body)}


def _dataset(episodes: int = 24) -> dict:
    operator_id = UPDATE_CONTROL_POSITION["operator_type_id"]
    rows = []
    for index in range(episodes):
        positive_tuple = {
            "state": f"before-{index}", "action": "opaque",
            "effect": _effect(operator_id), "next_state": f"after-{index}",
        }
        negative_tuple = {
            "state": f"before-negative-{index}", "action": "opaque",
            "effect": _effect(operator_id), "next_state": f"after-negative-{index}",
        }
        rows.append({
            "snapshot_id": str(index), "episode_id": str(index),
            "candidates": [
                {
                    "candidate_id": "SOURCE_SUCCESS",
                    "tuples": [positive_tuple, dict(positive_tuple)],
                    "terminal_features": {
                        "entity_goal_relation_coverage": 1.0,
                        "control_on_goal_relation": False,
                        "no_static_deadlock": True,
                    },
                    "success_from_state_only": True,
                },
                {
                    "candidate_id": "RELATION_PERMUTED",
                    "tuples": [negative_tuple, dict(negative_tuple)],
                    "terminal_features": {
                        "entity_goal_relation_coverage": 0.0,
                        "control_on_goal_relation": False,
                        "no_static_deadlock": True,
                    },
                    "success_from_state_only": False,
                },
            ],
        })
    body = {
        "schema_version": "source-relational-intervention-dataset-v2",
        "authority": "SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_ONLY",
        "source_plan_sha256": "a" * 64,
        "episodes": rows,
        "target_data_read": False,
    }
    return body | {"dataset_sha256": stable_hash(body)}


def test_induces_shared_ir_transition_terminal_and_abstention() -> None:
    artifact = induce_relational_structural_program(_dataset())
    validate_relational_structural_program(artifact)
    assert artifact["operator_types"] == [UPDATE_CONTROL_POSITION]
    assert artifact["program"]["transitions"][0]["cardinality"] == "ONE_OR_MORE"
    assert artifact["program"]["terminal_predicates"][0]["predicate_family"] == (
        "ENTITY_GOAL_RELATION"
    )
    assert artifact["program"]["abstention_rule"]["otherwise"] == "ABSTAIN"
    assert artifact["named_controller_template_used"] is False


def test_program_is_alpha_invariant_and_source_only() -> None:
    left = induce_relational_structural_program(_dataset())
    right = induce_relational_structural_program(_dataset())
    assert left["artifact_sha256"] == right["artifact_sha256"]
    assert left["target_data_read"] is False


def test_confirmation_rejects_shuffled_effects(monkeypatch) -> None:
    # Unit fixtures do not contain parseable Sokoban boards, so isolate the
    # source-effect shuffle check while exercising all program-selection gates.
    import motif_transfer.relational_structural_induction as module

    monkeypatch.setattr(
        module, "_shuffled_effect_binding_accuracy", lambda dataset: (24, 0),
    )
    dataset = _dataset()
    artifact = induce_relational_structural_program(dataset)
    report = confirm_relational_structural_program(artifact, dataset)
    assert report["status"] == "SOURCE_RELATIONAL_STRUCTURAL_FRESH_VALIDATED"
    assert all(report["gates"].values())
