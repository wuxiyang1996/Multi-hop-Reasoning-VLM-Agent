from motif_transfer.contracts import stable_hash
from motif_transfer.phase1_target_relineage import (
    relineage_decision,
    walk_routed_decisions,
)
from motif_transfer.search_automaton_transfer_v16 import (
    NativeBinding,
    SourceSearchAutomaton,
    TargetEvent,
)
from motif_transfer.sokoban_search_automaton_v16 import (
    BACKTRACK,
    COMMIT,
    EXPLORE,
    REFUTED,
    UNBOUND,
    VERIFIED,
)


POLICY = {UNBOUND: EXPLORE, REFUTED: BACKTRACK, VERIFIED: COMMIT}


def _artifact(lineage: str):
    body = {
        "schema_version": "sokoban-search-automaton-artifact-v16",
        "status": "SOURCE_SEARCH_AUTOMATON_FROZEN",
        "target_authorized": True,
        "learned_policy": POLICY,
        "source_lineage": lineage,
    }
    return body | {"artifact_sha256": stable_hash(body)}


def test_relineage_changes_only_source_and_receipt_hashes() -> None:
    old = SourceSearchAutomaton(_artifact("old"))
    new = SourceSearchAutomaton(_artifact("six-games"))
    event = TargetEvent(
        domain="webshop",
        episode_id="task-1",
        decision_index=0,
        event=UNBOUND,
        evidence_kind="target-native-state",
        evidence_payload={"candidate_count": 3},
        grounding_confidence=1.0,
    )
    binding = NativeBinding(
        abstract_action=EXPLORE,
        native_action_id="candidate-1",
        native_action="target_native_action",
        grounding_confidence=1.0,
        target_evidence_sha256=event.evidence_sha256,
    )
    historical = old.route(event, {EXPLORE: binding})

    relined = relineage_decision(
        historical.__dict__, old_source=old, new_source=new
    )

    assert relined["source_artifact_sha256"] == new.artifact_sha256
    assert relined["receipt_sha256"] != historical.receipt_sha256
    for key, value in historical.__dict__.items():
        if key not in {"source_artifact_sha256", "receipt_sha256"}:
            assert relined[key] == value


def test_recursive_decision_walk_finds_nested_receipts() -> None:
    marker = {key: key for key in (
        "domain", "episode_id", "decision_index", "target_event",
        "source_action", "native_action_id", "native_action", "admitted",
        "reason", "source_artifact_sha256", "target_evidence_sha256",
        "receipt_sha256",
    )}

    found = list(walk_routed_decisions({"episodes": [[marker]]}))

    assert found == [marker]
