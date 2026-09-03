from __future__ import annotations

import numpy as np

import pytest

from scripts.run_webshop_transfer_qualification_v5 import (
    _candidate_actions,
    _canonicalize_session_text,
    _decision_candidates,
    _select_action,
    _source_for_condition,
)


def _source() -> dict:
    return {
        "artifact_sha256": "source",
        "cluster_count": 3,
        "effect_scaler": {"mean": [0] * 7, "scale": [1] * 7},
        "cluster_centers": [
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
        ],
        "value_model": {
            "coefficients": [
                [0] * 10,
                [0] * 10,
                [0] * 10,
                [0, 0, 0, 0, 1, 2, 0, 0, 0, 0],
            ],
            "intercept": [0, 0, 0, 0],
        },
    }


def test_target_only_preserves_decision_rank_zero() -> None:
    selected, receipt = _select_action(
        condition="target_only",
        candidates=("a", "b"),
        effects=np.zeros((2, 7)),
        authentic_source=_source(),
        other_source=_source(),
        context=(0, 1, 0),
        previous_option=None,
    )
    assert selected == 0
    assert not receipt["source_admitted"]


def test_selective_other_game_uses_other_source_for_selection_and_state_update() -> None:
    authentic = {"artifact_sha256": "authentic"}
    other = {"artifact_sha256": "other"}
    assert _source_for_condition("selective_other_game_source", authentic, other) is other
    assert _source_for_condition("other_game_source", authentic, other) is other
    assert _source_for_condition("selective_authentic_source", authentic, other) is authentic


def test_session_canonicalization_preserves_common_request_identity() -> None:
    assert _canonicalize_session_text(
        "http://x/run-a_fixed_28/item/run-a_fixed_28",
        "run-a_fixed_28",
        "fixed_28",
    ) == "http://x/fixed_28/item/fixed_28"


def test_authentic_source_can_change_target_option() -> None:
    effects = np.asarray([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
    ])
    selected, receipt = _select_action(
        condition="authentic_game_source",
        candidates=("a", "b"),
        effects=effects,
        authentic_source=_source(),
        other_source=_source(),
        context=(0, 1, 0),
        previous_option=None,
    )
    assert selected == 1
    assert receipt["source_admitted"]


def test_candidate_parser_rejects_json_scalar() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        _candidate_actions("-1e308", axtree="[12] button 'x'", maximum=5)


def test_selective_source_only_overrides_predicted_repeat() -> None:
    source = _source()
    source["effect_feature_names"] = [
        "state_changed", "line_change_fraction", "character_length_delta_tanh",
        "available_action_count_delta_tanh", "action_repeated",
        "normalized_immediate_reward_tanh", "terminated",
    ]
    effects = np.asarray([
        [0, 0, 0, 0, 0.9, 0, 0],
        [1, 0, 0, 0, 0.1, 0, 0],
    ])
    selected, receipt = _select_action(
        condition="selective_authentic_source",
        candidates=("repeat", "recover"),
        effects=effects,
        authentic_source=source,
        other_source=source,
        context=(0, 1, 0),
        previous_option=None,
    )
    assert selected == 1
    assert receipt["selective_gate"]["gate_open"]

    effects[0, 4] = 0.1
    selected, receipt = _select_action(
        condition="selective_authentic_source",
        candidates=("progress", "alternative"),
        effects=effects,
        authentic_source=source,
        other_source=source,
        context=(0, 1, 0),
        previous_option=None,
    )
    assert selected == 0
    assert not receipt["selective_gate"]["gate_open"]


def test_minimum_repeat_is_source_free_matched_baseline() -> None:
    source = _source()
    source["effect_feature_names"] = [
        "state_changed", "line_change_fraction", "character_length_delta_tanh",
        "available_action_count_delta_tanh", "action_repeated",
        "normalized_immediate_reward_tanh", "terminated",
    ]
    effects = np.asarray([
        [0, 0, 0, 0, 0.9, 0, 0],
        [0, 0, 0, 0, 0.3, 0, 0],
        [0, 0, 0, 0, 0.1, 0, 0],
    ])
    selected, receipt = _select_action(
        condition="selective_minimum_repeat",
        candidates=("repeat", "better", "best"),
        effects=effects,
        authentic_source=source,
        other_source=source,
        context=(0, 1, 0),
        previous_option=None,
    )
    assert selected == 2
    assert receipt["neural_only"]
    assert receipt["selective_gate"]["gate_open"]


def test_safe_source_requires_exact_stall_and_preserves_constraint_action() -> None:
    source = _source()
    source["effect_feature_names"] = [
        "state_changed", "line_change_fraction", "character_length_delta_tanh",
        "available_action_count_delta_tanh", "action_repeated",
        "normalized_immediate_reward_tanh", "terminated",
    ]
    effects = np.asarray([
        [0, 0, 0, 0, 0.9, 0, 0],
        [1, 0, 0, 0, 0.1, 0, 0],
    ])
    navigation_rows = [
        {"is_commit": False, "is_constraint": False, "is_noop": False, "is_navigation": True},
        {"is_commit": False, "is_constraint": False, "is_noop": False, "is_navigation": True},
    ]
    selected, receipt = _select_action(
        condition="selective_safe_authentic_source",
        candidates=("repeat", "recover"),
        effects=effects,
        authentic_source=source,
        other_source=source,
        context=(0, 1, 0),
        previous_option=None,
        candidate_semantics_rows=navigation_rows,
        observed_stall=True,
    )
    assert selected == 1
    assert receipt["applicability_gate"]["gate_open"]

    constraint_rows = [
        {"is_commit": False, "is_constraint": True, "is_noop": False, "is_navigation": False},
        navigation_rows[1],
    ]
    selected, receipt = _select_action(
        condition="selective_safe_authentic_source",
        candidates=("repeat", "recover"),
        effects=effects,
        authentic_source=source,
        other_source=source,
        context=(0, 1, 0),
        previous_option=None,
        candidate_semantics_rows=constraint_rows,
        observed_stall=True,
    )
    assert selected == 0
    assert receipt["applicability_gate"]["closed_reason"] == "preserve_target_constraint_action"


def test_decision_schema_retries_are_receipted() -> None:
    class Backend:
        def __init__(self) -> None:
            self.outputs = iter(("None", '-1e308', '{"candidates":[{"action":"click(\'12\')"}]}'))
            self.last_usage = {}

        def complete(self, role: str, system: str, payload: dict) -> str:
            del role, system, payload
            return next(self.outputs)

    attempts: list[dict] = []
    candidates, _, attempts = _decision_candidates(
        backend=Backend(),
        system="system",
        payload={},
        axtree="[12] button 'go'",
        maximum=5,
        schema_retries=2,
        attempts_out=attempts,
    )
    assert candidates == ("click('12')",)
    assert [attempt["completion_diagnostic"]["json_type"] for attempt in attempts] == [
        "invalid_json", "float", "dict",
    ]
