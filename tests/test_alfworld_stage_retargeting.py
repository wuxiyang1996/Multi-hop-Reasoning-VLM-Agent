from motif_transfer.alfworld_stage_retargeting import choose_action
from motif_transfer.contracts import stable_hash


def _artifact():
    body = {
        "artifact_version": "SOKOBAN_EFFECT_PROGRAM_V2",
        "program": {"rules": [
            {"select": "COMMIT"}, {"select": "POSITION"},
        ]},
    }
    return body | {"artifact_sha256": stable_hash(body)}


def _grounded(commit_completion=0.9):
    return {
        "go to cabinet 1": {
            "option": "SEARCH", "required_option": "ACQUIRE",
            "applicability": 0.9, "completion": 0.2, "policy": 0.55,
        },
        "take apple 1 from cabinet 1": {
            "option": "ACQUIRE", "required_option": "ACQUIRE",
            "applicability": 0.9, "completion": commit_completion, "policy": 0.8,
        },
        "put apple 1 in fridge 1": {
            "option": "PLACE", "required_option": "ACQUIRE",
            "applicability": 0.7, "completion": 0.3, "policy": 0.2,
        },
    }


def test_authentic_selects_advance_but_target_policy_ranks_native_action():
    result = choose_action(
        condition="authentic_source_skill",
        grounded=_grounded(),
        history=(),
        source_artifact=_artifact(),
        effect_threshold=0.7,
    )
    assert result["source_selected_option"] == "COMMIT"
    assert result["action"] == "take apple 1 from cabinet 1"
    assert result["changed_option"] is False


def test_authentic_vetoes_advance_when_effect_is_not_grounded():
    result = choose_action(
        condition="authentic_source_skill",
        grounded=_grounded(commit_completion=0.1),
        history=(),
        source_artifact=_artifact(),
        effect_threshold=0.7,
    )
    assert result["source_selected_option"] == "POSITION"
    assert result["action"] == "go to cabinet 1"
    assert result["changed_option"] is True


def test_null_and_oracle_share_target_neural_realizer():
    null = choose_action(
        condition="null_skill_same_harness",
        grounded=_grounded(), history=(), source_artifact=_artifact(),
        effect_threshold=0.7,
    )
    oracle = choose_action(
        condition="target_oracle_skill",
        grounded=_grounded(), history=(), source_artifact=_artifact(),
        effect_threshold=0.7,
    )
    assert null["action"] == oracle["action"] == "take apple 1 from cabinet 1"
