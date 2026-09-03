from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from motif_transfer.search_automaton_transfer_v16 import (
    AttemptLedger,
    OUTCOME_NONTERMINAL_EFFECT,
    OUTCOME_REFUTED,
    OUTCOME_TERMINAL_VERIFIED,
    SourceSearchAutomaton,
    bind_native_action,
    ground_target_event,
)
from motif_transfer.sokoban_search_automaton_v16 import (
    BACKTRACK,
    COMMIT,
    EXPLORE,
    REFUTED,
    UNBOUND,
    VERIFIED,
)


REPO = Path(__file__).resolve().parents[1]


def _artifact() -> dict:
    return json.loads(
        (REPO / "runs/sokoban_search_automaton_v16/artifact.json").read_text()
    )


@pytest.mark.parametrize(
    ("domain", "flags", "event_name", "abstract_action", "native_action"),
    [
        ("webshop", (False, False, True), VERIFIED, COMMIT, "click[Buy Now]"),
        ("alfworld", (True, False, False), UNBOUND, EXPLORE, "go to fridge 1"),
        (
            "discoveryworld",
            (False, True, False),
            REFUTED,
            BACKTRACK,
            {"action": "TELEPORT_TO_OBJECT", "arg1": 17},
        ),
        ("tirbench", (True, False, False), UNBOUND, EXPLORE, "zoom_region:C2"),
    ],
)
def test_one_source_policy_routes_four_target_native_action_spaces(
    domain: str,
    flags: tuple[bool, bool, bool],
    event_name: str,
    abstract_action: str,
    native_action,
) -> None:
    source = SourceSearchAutomaton(_artifact())
    event = ground_target_event(
        domain=domain,
        episode_id=f"{domain}-episode",
        decision_index=0,
        untried_candidate_available=flags[0],
        active_candidate_refuted=flags[1],
        terminal_commit_verified=flags[2],
        evidence_kind=f"{domain}_native_predicate",
        evidence_payload={"target_state_digest": "f" * 64},
        grounding_confidence=0.9,
    )
    assert event is not None and event.event == event_name
    binding = bind_native_action(
        event,
        abstract_action=abstract_action,
        native_action_id=f"{domain}-native-action",
        native_action=native_action,
        grounding_confidence=0.8,
    )
    decision = source.route(event, {abstract_action: binding})
    assert decision.admitted
    assert decision.source_action == abstract_action
    assert decision.native_action == native_action
    assert decision.validate()


def test_source_artifact_and_target_binding_fail_closed() -> None:
    tampered = copy.deepcopy(_artifact())
    tampered["learned_policy"][UNBOUND] = COMMIT
    with pytest.raises(ValueError, match="self-hash"):
        SourceSearchAutomaton(tampered)

    source = SourceSearchAutomaton(_artifact())
    event = ground_target_event(
        domain="alfworld",
        episode_id="episode",
        decision_index=0,
        untried_candidate_available=True,
        active_candidate_refuted=False,
        terminal_commit_verified=False,
        evidence_kind="target_native_candidates",
        evidence_payload={"candidate_count": 2},
        grounding_confidence=0.4,
    )
    assert event is not None
    decision = source.route(event, {})
    assert not decision.admitted
    assert decision.native_action is None
    assert decision.reason == "ABSTAIN_LOW_EVENT_GROUNDING_CONFIDENCE"


def test_source_native_fields_cannot_cross_target_boundary() -> None:
    with pytest.raises(ValueError, match="source-native field"):
        ground_target_event(
            domain="webshop",
            episode_id="episode",
            decision_index=0,
            untried_candidate_available=True,
            active_candidate_refuted=False,
            terminal_commit_verified=False,
            evidence_kind="bad",
            evidence_payload={"sokoban_coordinate": [1, 2]},
            grounding_confidence=1.0,
        )


def test_attempt_ledger_rejects_repeats_and_separates_terminal_verification() -> None:
    ledger = AttemptLedger()
    ledger.begin_scope("state-a")
    assert ledger.unbound_event(["c1", "c2"]) == UNBOUND
    assert ledger.next_untried(["c1", "c2"]) == "c1"
    assert ledger.observe("c1", OUTCOME_REFUTED) == REFUTED
    assert ledger.next_untried(["c1", "c2"]) == "c2"
    assert ledger.observe("c2", OUTCOME_NONTERMINAL_EFFECT) is None
    assert ledger.unbound_event(["c1", "c2"]) is None

    ledger.begin_scope("state-b")
    assert ledger.next_untried(["c3"]) == "c3"
    assert ledger.observe("c3", OUTCOME_TERMINAL_VERIFIED) == VERIFIED
    assert ledger.refuted == set()


def test_conflicting_target_events_are_rejected() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ground_target_event(
            domain="tirbench",
            episode_id="episode",
            decision_index=0,
            untried_candidate_available=True,
            active_candidate_refuted=True,
            terminal_commit_verified=False,
            evidence_kind="conflict",
            evidence_payload={},
            grounding_confidence=1.0,
        )
