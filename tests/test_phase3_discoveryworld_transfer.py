import json
from pathlib import Path

from dataclasses import replace
import pytest

from motif_transfer.discoveryworld_sokoban_transfer import (
    DiscoveryWorldGroundedCandidate,
    DiscoveryWorldTargetBinding,
)
from motif_transfer.discoveryworld_env import DiscoveryWorldObservation
from motif_transfer.phase3_discoveryworld_transfer import (
    PHASE3_TARGET_GROUNDER_SYSTEM_PROMPT,
    SOURCE_INDUCED,
    SOURCE_PERMUTED,
    call_phase3_binder,
    Phase3GroundedCandidate,
    Phase3DiscoveryWorldSelector,
    Phase3DiscoveryWorldPortfolioSelector,
    call_phase3_grounder,
    canonical_position_candidates,
    phase3_candidate_set_complete,
    outcome_blind_target_native_facts,
    phase3_position_action_catalog,
    phase3_target_binding_catalog,
    validate_phase3_target_binding_semantics,
)


def candidate(name, role="POSITION", info=0.5, effect=0.5, prereq=0.5):
    return DiscoveryWorldGroundedCandidate(
        action={"action": "MOVE_DIRECTION", "arg1": name},
        target_role=role,
        prerequisite_probability=prereq,
        positive_effect_probability=effect,
        information_gain_probability=info,
        expected_effect="observable target change",
        evidence=(), reason="test",
    )


def typed_candidate(name, persistence, role="POSITION"):
    base = candidate(name, role=role)
    return Phase3GroundedCandidate(
        **base.__dict__,
        typed_effect_probabilities={
            "EFFECT_BY_TRANSITION_1": 0.2,
            "EFFECT_BY_TRANSITION_4": 0.4,
            "EFFECT_BY_TRANSITION_8": 0.6,
            "EXECUTABLE_TRANSITION_PERSISTENCE": persistence,
        },
    )


def test_phase3_prompt_requires_nontrivial_multiplicity():
    assert "exactly five" in PHASE3_TARGET_GROUNDER_SYSTEM_PROMPT
    assert "four distinct reversible POSITION" in PHASE3_TARGET_GROUNDER_SYSTEM_PROMPT
    assert "EFFECT_BY_TRANSITION_1" in PHASE3_TARGET_GROUNDER_SYSTEM_PROMPT
    assert "EXECUTABLE_TRANSITION_PERSISTENCE" in PHASE3_TARGET_GROUNDER_SYSTEM_PROMPT


def test_candidate_completeness_is_exact_four_plus_one():
    positions = [candidate(name) for name in ("north", "east", "south", "west")]
    commit = candidate("commit", role="COMMIT")
    assert phase3_candidate_set_complete((*positions, commit))
    assert not phase3_candidate_set_complete((*positions[:3], commit))
    assert not phase3_candidate_set_complete((*positions, commit, replace(commit, reason="x")))


def test_target_candidate_order_is_semantic_and_model_order_invariant():
    rows = (
        candidate("north", info=0.2),
        candidate("east", info=0.9),
        candidate("south", info=0.4),
        candidate("west", info=0.7),
    )
    expected = tuple(row.action["arg1"] for row in canonical_position_candidates(rows))
    reversed_order = tuple(
        row.action["arg1"] for row in canonical_position_candidates(tuple(reversed(rows)))
    )
    assert reversed_order == expected
    assert set(expected) == {"north", "east", "south", "west"}


def _grounder_observation():
    return DiscoveryWorldObservation(
        scenario="Proteomics", difficulty="Easy", seed=0, episode_step=20,
        ui={
            "taskProgress": [{
                "description": "drop flag west of statue",
                "completed": True, "completedSuccessfully": True,
                "score": 1, "maxScore": 1,
            }],
            "agentLocation": {
                "x": 15, "y": 13, "directions_you_can_move": ["south"],
            },
            "inventoryObjects": [{"uuid": 7, "name": "flag"}],
            "accessibleEnvironmentObjects": [{"uuid": 9, "name": "statue"}],
            "nearbyObjects": {"objects": {"east": [
                {"uuid": 9, "name": "statue", "distance": 1},
            ]}},
        },
        known_actions={
            "DROP": {"args": ["arg1"]},
            "MOVE_DIRECTION": {"args": ["arg1"]},
            "ROTATE_DIRECTION": {"args": ["arg1"]},
        },
        teleport_locations={}, last_action_result=None, vision=None,
        in_dialog=False, terminal=False, official_success=False,
    )


def _grounder_response(position_actions):
    candidates = [{
        **action,
        "target_role": "POSITION",
        "prerequisite_probability": 1.0,
        "positive_effect_probability": 0.2,
        "information_gain_probability": 0.5,
        "typed_effect_probabilities": {
            "EFFECT_BY_TRANSITION_1": 0.2,
            "EFFECT_BY_TRANSITION_4": 0.4,
            "EFFECT_BY_TRANSITION_8": 0.6,
            "EXECUTABLE_TRANSITION_PERSISTENCE": 0.8,
        },
        "expected_effect": "reversible positioning observation",
        "evidence": [],
        "reason": "valid reversible intervention",
    } for action in position_actions]
    candidates.append({
        "action": "DROP", "arg1": 7, "target_role": "COMMIT",
        "prerequisite_probability": 0.2,
        "positive_effect_probability": 0.9,
        "information_gain_probability": 0.0,
        "typed_effect_probabilities": {
            "EFFECT_BY_TRANSITION_1": 0.9,
            "EFFECT_BY_TRANSITION_4": 0.9,
            "EFFECT_BY_TRANSITION_8": 0.9,
            "EXECUTABLE_TRANSITION_PERSISTENCE": 0.1,
        },
        "expected_effect": "drop the bound flag",
        "evidence": [], "reason": "bound final action",
    })
    return json.dumps({
        "memory": "", "running_hypotheses": [], "candidates": candidates,
    })


def test_initial_multiplicity_retry_reports_native_parse_rejections():
    invalid = _grounder_response([
        {"action": "MOVE_DIRECTION", "arg1": "north"},
        {"action": "ROTATE_DIRECTION", "arg1": "east"},
        {"action": "ROTATE_DIRECTION", "arg1": "south"},
        {"action": "ROTATE_DIRECTION", "arg1": "west"},
    ])
    valid = _grounder_response([
        {"action": "MOVE_DIRECTION", "arg1": "south"},
        {"action": "ROTATE_DIRECTION", "arg1": "east"},
        {"action": "ROTATE_DIRECTION", "arg1": "south"},
        {"action": "ROTATE_DIRECTION", "arg1": "west"},
    ])

    class Backend:
        def __init__(self):
            self.responses = [invalid, valid]
            self.payloads = []
            self.last_usage = {}

        def complete(self, role, system_prompt, payload):
            self.payloads.append(payload)
            return self.responses.pop(0)

    backend = Backend()
    observation = _grounder_observation()
    binding = DiscoveryWorldTargetBinding(
        target_uuid=9, target_name="statue",
        commit_subject_relation_to_target="west",
        target_relation_from_agent="east", target_distance=1,
        commit_action={"action": "DROP", "arg1": 7}, confidence=1.0,
        hypothesis_used="task", reason="test",
    )
    _, candidates, _, audit = call_phase3_grounder(
        backend, observation, memory="", hypotheses=(), recent=[],
        target_binding=binding, attempts=2,
    )
    repair = backend.payloads[1]["previous_response_rejected"]
    assert "MOVE_DIRECTION arg1 is not currently movable" in repair
    assert "valid_positions=3" in repair
    assert phase3_candidate_set_complete(candidates)
    assert [row["accepted"] for row in audit] == [False, True]
    assert all(row["formal_outcome_fields_visible"] is False for row in audit)
    assert all(
        payload["formal_outcome_fields_visible"] is False
        and "completed" not in json.dumps(payload["target_native_facts"])
        and "score" not in json.dumps(payload["target_native_facts"])
        for payload in backend.payloads
    )


def test_phase3_binder_and_facts_strip_formal_outcome_fields():
    observation = _grounder_observation()
    facts = outcome_blind_target_native_facts(observation)
    serialized = json.dumps(facts)
    assert "completed" not in serialized
    assert "score" not in serialized

    class Backend:
        last_usage = {}

        def __init__(self):
            self.payload = None

        def complete(self, role, system_prompt, payload):
            self.payload = payload
            return json.dumps({
                "target_uuid": 9, "target_name": "statue",
                "commit_subject_relation_to_target": "west",
                "target_distance": 1,
                "commit_action": {"action": "DROP", "arg1": 7},
                "confidence": 1.0, "hypothesis_used": "task",
                "reason": "bind exact supplied target",
            })

    backend = Backend()
    binding, _, audit = call_phase3_binder(
        backend, observation, memory="", hypotheses=(), attempts=1,
    )
    assert binding.target_uuid == 9
    assert audit[0]["formal_outcome_fields_visible"] is False
    assert backend.payload["formal_outcome_fields_visible"] is False
    required_type, catalog = phase3_target_binding_catalog(observation)
    assert required_type == "statue"
    assert catalog == ({"uuid": 9, "name": "statue"},)
    assert backend.payload["phase3_target_binding_catalog"] == list(catalog)
    payload_text = json.dumps(backend.payload["target_native_facts"])
    assert "completed" not in payload_text
    assert "score" not in payload_text

    wrong_type = replace(
        binding, target_uuid=10, target_name="prismatic beast",
    )
    with pytest.raises(ValueError, match="requires a statue target"):
        validate_phase3_target_binding_semantics(wrong_type, observation)


def test_phase3_catalog_compiles_only_currently_valid_native_actions():
    observation = _grounder_observation()
    binding = DiscoveryWorldTargetBinding(
        target_uuid=9, target_name="statue",
        commit_subject_relation_to_target="west",
        target_relation_from_agent="east", target_distance=1,
        commit_action={"action": "DROP", "arg1": 7}, confidence=1.0,
        hypothesis_used="task", reason="test",
    )
    catalog = phase3_position_action_catalog(observation, binding)
    assert {"action": "MOVE_DIRECTION", "arg1": "south"} in catalog
    assert {"action": "MOVE_DIRECTION", "arg1": "north"} not in catalog
    assert sum(row["action"] == "ROTATE_DIRECTION" for row in catalog) == 4


def test_typed_source_runtime_rebinds_successor_candidate_set():
    repo = Path(__file__).resolve().parents[1]
    artifact = json.loads((
        repo / "configs/phase3_source_induction_v3/frozen_reserve/"
        "programs/tetris.json"
    ).read_text())
    selector = Phase3DiscoveryWorldSelector(
        authentic_artifact=artifact, permuted_artifact=artifact,
    )
    observation = _grounder_observation()
    binding = DiscoveryWorldTargetBinding(
        target_uuid=9, target_name="statue",
        commit_subject_relation_to_target="west",
        target_relation_from_agent="north", target_distance=1,
        commit_action={"action": "DROP", "arg1": 7}, confidence=1.0,
        hypothesis_used="task", reason="test",
    )
    initial = tuple(
        typed_candidate(name, persistence)
        for name, persistence in zip(
            ("north", "east", "south", "west"), (0.1, 0.9, 0.3, 0.2)
        )
    ) + (typed_candidate("commit", 0.1, role="COMMIT"),)
    selected, first = selector.select(
        SOURCE_INDUCED, initial, observation, target_binding=binding,
    )
    assert selected.action["arg1"] == "east"
    assert first.source_admitted is True

    successor = tuple(
        typed_candidate(name, persistence)
        for name, persistence in zip(
            ("novel-a", "novel-b", "novel-c", "novel-d"),
            (0.2, 0.4, 0.95, 0.1),
        )
    ) + (typed_candidate("commit-2", 0.1, role="COMMIT"),)
    selected, second = selector.select(
        SOURCE_INDUCED, successor, observation, target_binding=binding,
    )
    assert selected.action["arg1"] == "novel-c"
    assert second.source_admitted is True
    assert second.selection_reason == "SOURCE_INDUCED_ANONYMOUS_TRIAL_DELTA"


def test_portfolio_selector_uses_program_content_and_permuted_binding_control():
    repo = Path(__file__).resolve().parents[1]
    program_dir = (
        repo / "configs/phase3_source_induction_v3/frozen_reserve/programs"
    )
    selector = Phase3DiscoveryWorldPortfolioSelector(source_artifacts=[
        json.loads(path.read_text()) for path in sorted(program_dir.glob("*.json"))
    ])
    observation = _grounder_observation()
    binding = DiscoveryWorldTargetBinding(
        target_uuid=9, target_name="statue",
        commit_subject_relation_to_target="west",
        target_relation_from_agent="north", target_distance=1,
        commit_action={"action": "DROP", "arg1": 7}, confidence=1.0,
        hypothesis_used="task", reason="test",
    )
    effect_rows = (
        ("north", 0.95, 0.4, 0.4),
        ("east", 0.10, 0.9, 0.5),
        ("south", 0.20, 0.3, 0.6),
        ("west", 0.30, 0.2, 0.7),
    )
    positions = []
    for name, short, medium, persistence in effect_rows:
        base = candidate(name)
        positions.append(Phase3GroundedCandidate(
            **base.__dict__,
            typed_effect_probabilities={
                "EFFECT_BY_TRANSITION_1": 0.2,
                "EFFECT_BY_TRANSITION_4": short,
                "EFFECT_BY_TRANSITION_8": medium,
                "EXECUTABLE_TRANSITION_PERSISTENCE": persistence,
            },
        ))
    commit = typed_candidate("commit", 0.1, role="COMMIT")
    candidates = (*positions, commit)
    authentic, authentic_receipt = selector.select(
        SOURCE_INDUCED, candidates, observation, target_binding=binding,
    )
    control, control_receipt = selector.select(
        SOURCE_PERMUTED, candidates, observation, target_binding=binding,
    )
    assert authentic.action["arg1"] == "north"
    assert control.action != authentic.action
    assert authentic_receipt.portfolio_receipt_sha256
    assert (
        control_receipt.portfolio_receipt_sha256
        == authentic_receipt.portfolio_receipt_sha256
    )
    assert control_receipt.effect_binding_control_receipt_sha256
    assert authentic_receipt.validate()
    assert control_receipt.validate()


def test_source_trial_must_be_regrounded_in_current_candidate_set():
    repo = Path(__file__).resolve().parents[1]
    artifact = json.loads((
        repo / "configs/phase3_source_induction_v1/frozen_confirmation/"
        "programs/candy_crush.json"
    ).read_text())
    selector = Phase3DiscoveryWorldSelector(
        authentic_artifact=artifact, permuted_artifact=artifact,
    )
    observation = _grounder_observation()
    binding = DiscoveryWorldTargetBinding(
        target_uuid=9, target_name="statue",
        commit_subject_relation_to_target="west",
        target_relation_from_agent="north", target_distance=1,
        commit_action={"action": "DROP", "arg1": 7}, confidence=1.0,
        hypothesis_used="task", reason="test",
    )
    initial = tuple(candidate(name) for name in ("north", "east", "south", "west"))
    initial += (candidate("unwitnessed", role="COMMIT"),)
    selected, first = selector.select(
        SOURCE_INDUCED, initial, observation, target_binding=binding,
    )
    assert first.selection_reason == "SOURCE_INDUCED_ANONYMOUS_TRIAL_DELTA"
    assert selected in initial

    # None of the remaining source-ordered fork actions is present in this
    # state-specific target grounding, so the source program must fail closed.
    current = (
        candidate("novel", info=0.9),
        candidate("still-unwitnessed", role="COMMIT", effect=0.1),
    )
    selected, second = selector.select(
        SOURCE_INDUCED, current, observation, target_binding=binding,
    )
    assert selected in current
    assert second.source_admitted is False
    assert second.selection_reason == (
        "SOURCE_TRIAL_NOT_REGROUNDED_IN_CURRENT_STATE_"
        "TO_MATCHED_SOURCE_FREE_GROUNDER"
    )
