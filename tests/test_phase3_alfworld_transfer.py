from motif_transfer.phase3_alfworld_transfer import (
    CONDITIONS,
    Phase3ALFWorldSelector,
    _candidate_view,
    effect_observation_horizon,
)


def _grounded():
    effects = {
        "EFFECT_BY_TRANSITION_1": 0.1,
        "EFFECT_BY_TRANSITION_4": 0.2,
        "EFFECT_BY_TRANSITION_8": 0.3,
        "EXECUTABLE_TRANSITION_PERSISTENCE": 0.4,
    }
    return {
        "go to shelf 1": {
            "action_sha256": "a",
            "target_policy_probability": 0.8,
            "typed_effect_probabilities": effects,
        },
        "look": {
            "action_sha256": "b",
            "target_policy_probability": 0.2,
            "typed_effect_probabilities": dict(effects) | {
                "EFFECT_BY_TRANSITION_4": 0.9,
            },
        },
    }


def test_source_free_arms_do_not_need_source_artifacts():
    grounded = _grounded()
    neural = Phase3ALFWorldSelector(
        condition="neural_only", source_artifacts=(),
    ).select(grounded=grounded, history=())
    generic = Phase3ALFWorldSelector(
        condition="generic_scaffold", source_artifacts=(),
    ).select(grounded=grounded, history=())
    ceiling = Phase3ALFWorldSelector(
        condition="target_native_ceiling", source_artifacts=(),
    ).select(grounded=grounded, history=(), expert_action="look")
    assert neural["action"] == "go to shelf 1"
    assert generic["action"] == "look"
    assert ceiling["action"] == "look"
    assert not neural["source_admitted"]


def test_condition_matrix_is_the_five_preregistered_arms():
    assert CONDITIONS == (
        "neural_only",
        "source_induced",
        "source_permuted",
        "generic_scaffold",
        "target_native_ceiling",
    )


def test_permuted_option_binding_treats_singleton_as_identity_control():
    effects, receipt = Phase3ALFWorldSelector._permuted_effects(
        artifact={"typed_effect_program": {}},
        ids=["only"],
        effects=[{"EFFECT_BY_TRANSITION_4": 0.9}],
    )
    assert effects == [{"EFFECT_BY_TRANSITION_4": 0.9}]
    assert receipt["status"] == "IDENTITY_CONTROL_SINGLETON_NOT_PERMUTABLE"


def test_option_effect_and_action_use_same_target_policy_realization():
    grounded = _grounded() | {
        "go to shelf 2": {
            "action_sha256": "c",
            "target_policy_probability": 0.1,
            "typed_effect_probabilities": {
                "EFFECT_BY_TRANSITION_1": 0.99,
                "EFFECT_BY_TRANSITION_4": 0.99,
                "EFFECT_BY_TRANSITION_8": 0.99,
                "EXECUTABLE_TRANSITION_PERSISTENCE": 0.99,
            },
        },
    }
    units, _, effects, _, realization = _candidate_view(
        grounded, (), "target_native_option",
    )
    search = units.index("SEARCH")
    assert realization["SEARCH"] == "go to shelf 1"
    assert effects[search]["EFFECT_BY_TRANSITION_4"] == 0.2


def test_typed_effect_horizons_preserve_source_measurement_type():
    assert effect_observation_horizon("EFFECT_BY_TRANSITION_1") == 1
    assert effect_observation_horizon("EFFECT_BY_TRANSITION_4") == 4
    assert effect_observation_horizon("EFFECT_BY_TRANSITION_8") == 8
    assert effect_observation_horizon("EXECUTABLE_TRANSITION_PERSISTENCE") == 8
