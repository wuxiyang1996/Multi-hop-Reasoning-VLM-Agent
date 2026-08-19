from __future__ import annotations

from motif_transfer.contracts import stable_hash
import motif_transfer.source_goal_relation_causal_budget as causal
import motif_transfer.source_goal_relation_induction as base


def _effect(before: float, after: float) -> dict:
    return base._macro_effect(before, after)


def _tuple(index: int, before: float, after: float) -> dict:
    body = {
        "state": f"state-{index}-{before}",
        "action": "opaque-source-action",
        "effect": _effect(before, after),
        "next_state": f"state-{index}-{after}",
        "before_features": {
            "entity_goal_relation_coverage": before,
            "entity_cardinality": 2,
        },
        "next_features": {
            "entity_goal_relation_coverage": after,
            "entity_cardinality": 2,
        },
    }
    return body | {"tuple_sha256": stable_hash(body)}


def _dataset() -> dict:
    success = {
        "candidate_id": "SOURCE_SUCCESS",
        "macro_tuples": [_tuple(0, 0.0, 0.5), _tuple(1, 0.5, 1.0)],
        "terminal_features": {
            "entity_goal_relation_coverage": 1.0,
            "entity_cardinality": 2,
        },
        "success_from_state_only": True,
    }
    control = {
        "candidate_id": "ORDER_CONTROL",
        "macro_tuples": [_tuple(2, 0.0, 0.5)],
        "terminal_features": {
            "entity_goal_relation_coverage": 0.5,
            "entity_cardinality": 2,
        },
        "success_from_state_only": False,
    }
    body = {
        "schema_version": base.DATASET_VERSION,
        "authority": "SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_ONLY",
        "source_plan_sha256": "a" * 64,
        "primitive_dataset_sha256": "b" * 64,
        "changed_feature_hypothesis": base.FEATURE,
        "episodes": [{
            "snapshot_id": "snapshot-0",
            "episode_id": "episode-0",
            "candidates": [success, control],
        }],
        "diagnostics": {
            "primitive_episodes": 1,
            "retained_episodes": 1,
            "excluded_nonmonotone_successes": 0,
        },
        "target_data_read": False,
    }
    return body | {"dataset_sha256": stable_hash(body)}


def test_causal_projection_removes_unchanged_terminal_nuisance() -> None:
    dataset = _dataset()
    projected = causal.project_intervention_linked_terminals(dataset)

    assert causal.intervention_linked_features(dataset) == (
        "entity_goal_relation_coverage",
    )
    assert dataset["episodes"][0]["candidates"][0]["terminal_features"] == {
        "entity_goal_relation_coverage": 1.0,
        "entity_cardinality": 2,
    }
    assert projected["episodes"][0]["candidates"][0][
        "terminal_features"
    ] == {"entity_goal_relation_coverage": 1.0}
    assert projected["terminal_candidate_projection"][
        "named_terminal_feature_provided"
    ] is False
    base.validate_goal_relation_macro_dataset(projected)


def test_causal_inducer_selects_only_intervention_linked_terminal() -> None:
    artifact = causal.induce_causal_goal_relation_program(_dataset())
    causal.validate_causal_goal_relation_program(artifact)

    assert artifact["program"]["terminal_predicates"] == [{
        "predicate_family": "ENTITY_GOAL_RELATION",
        "arity": 2,
        "value_kind": "RELATION_COVERAGE",
        "feature": "entity_goal_relation_coverage",
        "operator": "EQ",
        "value": 1.0,
    }]
    assert artifact["terminal_candidate_authority"]["target_data_read"] is False


def test_causal_inducer_fails_closed_without_positive_effect() -> None:
    dataset = _dataset()
    dataset["episodes"][0]["candidates"][0]["macro_tuples"] = []
    body = dict(dataset)
    body.pop("dataset_sha256")
    dataset = body | {"dataset_sha256": stable_hash(body)}

    try:
        causal.induce_causal_goal_relation_program(dataset)
    except ValueError as error:
        assert "exactly one terminal feature" in str(error)
    else:
        raise AssertionError("ambiguous intervention-linked feature was accepted")
