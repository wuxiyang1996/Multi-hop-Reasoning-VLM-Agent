from __future__ import annotations

from motif_transfer.contracts import stable_hash
import motif_transfer.source_goal_relation_induction as module


def _effect(before: float, after: float) -> dict:
    return module._macro_effect(before, after)


def _tuple(index: int, before: float, after: float) -> dict:
    core = {
        "state": f"state-{index}-{before}",
        "action": "opaque-source-action",
        "effect": _effect(before, after),
        "next_state": f"state-{index}-{after}",
        "before_features": {
            "entity_goal_relation_coverage": before,
            "control_on_goal_relation": False,
            "no_static_deadlock": True,
            "entity_cardinality": 2,
            "goal_cardinality": 2,
        },
        "next_features": {
            "entity_goal_relation_coverage": after,
            "control_on_goal_relation": False,
            "no_static_deadlock": True,
            "entity_cardinality": 2,
            "goal_cardinality": 2,
        },
    }
    return core | {"tuple_sha256": stable_hash(core)}


def _dataset(episodes: int = 24) -> dict:
    rows = []
    for index in range(episodes):
        rows.append({
            "snapshot_id": str(index),
            "episode_id": str(index),
            "candidates": [
                {
                    "candidate_id": "SOURCE_SUCCESS",
                    "macro_tuples": [
                        _tuple(index * 10, 0.0, 0.5),
                        _tuple(index * 10 + 1, 0.5, 1.0),
                    ],
                    "terminal_features": {
                        "entity_goal_relation_coverage": 1.0,
                        "control_on_goal_relation": False,
                        "no_static_deadlock": True,
                        "entity_cardinality": 2,
                        "goal_cardinality": 2,
                    },
                    "success_from_state_only": True,
                },
                {
                    "candidate_id": "ORDER_CONTROL",
                    "macro_tuples": [_tuple(index * 10 + 2, 0.0, 0.5)],
                    "terminal_features": {
                        "entity_goal_relation_coverage": 0.5,
                        "control_on_goal_relation": False,
                        "no_static_deadlock": True,
                        "entity_cardinality": 2,
                        "goal_cardinality": 2,
                    },
                    "success_from_state_only": False,
                },
            ],
        })
    body = {
        "schema_version": module.DATASET_VERSION,
        "authority": "SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_ONLY",
        "source_plan_sha256": "a" * 64,
        "primitive_dataset_sha256": "b" * 64,
        "changed_feature_hypothesis": module.FEATURE,
        "episodes": rows,
        "diagnostics": {
            "primitive_episodes": episodes,
            "retained_episodes": episodes,
            "excluded_nonmonotone_successes": 0,
        },
        "target_data_read": False,
    }
    return body | {"dataset_sha256": stable_hash(body)}


def test_induces_relation_operator_recurrence_terminal_and_abstention() -> None:
    artifact = module.induce_goal_relation_macro_program(_dataset())
    module.validate_goal_relation_macro_program(artifact)
    operator = artifact["operator_types"][0]
    assert operator["predicate_family"] == "ENTITY_GOAL_RELATION"
    assert artifact["program"]["transitions"][0]["cardinality"] == "ONE_OR_MORE"
    assert artifact["program"]["terminal_predicates"] == [{
        "predicate_family": "ENTITY_GOAL_RELATION",
        "arity": 2,
        "value_kind": "RELATION_COVERAGE",
        "feature": "entity_goal_relation_coverage",
        "operator": "EQ",
        "value": 1.0,
    }]
    assert artifact["program"]["abstention_rule"][
        "nonpositive_observed_relation_delta"
    ] == "ABSTAIN"
    assert artifact["named_controller_template_used"] is False
    assert artifact["target_data_read"] is False


def test_confirmation_rejects_effect_shuffle(monkeypatch) -> None:
    dataset = _dataset()
    artifact = module.induce_goal_relation_macro_program(dataset)
    monkeypatch.setattr(module, "_effect_binding_counts", lambda _: (48, 0))
    report = module.confirm_goal_relation_macro_program(artifact, dataset)
    assert report["status"] == "SOURCE_GOAL_RELATION_MACRO_FRESH_VALIDATED"
    assert all(report["gates"].values())


def test_decreasing_success_effect_is_rejected() -> None:
    dataset = _dataset(1)
    dataset["episodes"][0]["candidates"][0]["macro_tuples"][0] = _tuple(
        100, 0.5, 0.0,
    )
    body = dict(dataset)
    body.pop("dataset_sha256")
    dataset = body | {"dataset_sha256": stable_hash(body)}
    try:
        module.induce_goal_relation_macro_program(dataset)
    except ValueError as error:
        assert "one effect" in str(error)
    else:
        raise AssertionError("decreasing successful source effect was accepted")
