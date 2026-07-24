import hashlib
import json

from motif_transfer.phase1_assets import audit_evidence_batch


def dump_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_skill_candidates_are_not_misclassified_as_action_proposals(tmp_path):
    events = [
        {
            "episode_id": "e",
            "sequence": 0,
            "event_sha256": "h0",
            "previous_event_sha256": None,
            "kind": "AGENT_PROPOSAL_SET",
            "payload": {"skill_candidates": [{"skill_id": "s"}]},
        },
        {
            "episode_id": "e",
            "sequence": 1,
            "event_sha256": "h1",
            "previous_event_sha256": "h0",
            "kind": "AGENT_DECISION",
            "payload": {"decision_origin": "AGENT", "decision_type": "EXECUTE"},
        },
    ]
    episodes = [{"episode_id": "e", "game": "game", "steps": 1}]
    dump_jsonl(tmp_path / "events.jsonl", events)
    dump_jsonl(tmp_path / "episodes.jsonl", episodes)
    manifest = {
        "metadata": {"game": "game"},
        "files": {
            name: {"sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}
            for name in ("events.jsonl", "episodes.jsonl")
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    audit = audit_evidence_batch(tmp_path)
    assert audit.skill_candidate_sets == 1
    assert audit.action_proposal_attempts == 0
    assert audit.action_proposal_sets == 0
    assert "NO_VALID_ACTION_PROPOSAL_SETS" in audit.gaps
    assert audit.content_hashes_valid and audit.event_chains_valid


def test_invalid_agent_protocol_attempts_are_visible_but_not_admitted(tmp_path):
    events = [
        {
            "episode_id": "e",
            "sequence": 0,
            "event_sha256": "h0",
            "previous_event_sha256": None,
            "kind": "AGENT_ACTION_PROPOSAL_SET",
            "payload": {"step": 0, "schema_valid": False, "proposal_set": None},
        },
        {
            "episode_id": "e",
            "sequence": 1,
            "event_sha256": "h1",
            "previous_event_sha256": "h0",
            "kind": "AGENT_POST_TRANSITION_VERDICT",
            "payload": {"step": 0, "schema_valid": False, "verdict": None},
        },
    ]
    episodes = [{"episode_id": "e", "game": "game", "steps": 1, "total_reward": 0.0}]
    dump_jsonl(tmp_path / "events.jsonl", events)
    dump_jsonl(tmp_path / "episodes.jsonl", episodes)
    manifest = {
        "metadata": {"game": "game"},
        "files": {
            name: {"sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}
            for name in ("events.jsonl", "episodes.jsonl")
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    audit = audit_evidence_batch(tmp_path)
    assert audit.action_proposal_attempts == 1
    assert audit.action_proposal_sets == 0
    assert audit.post_transition_verdict_attempts == 1
    assert audit.post_transition_verdicts == 0
    assert not audit.motif_ready


def test_policy_and_skill_conditioning_receipts_are_counted(tmp_path):
    events = [
        {
            "episode_id": "e", "sequence": 0, "event_sha256": "h0",
            "previous_event_sha256": None, "kind": "AGENT_ACTION_PROPOSAL_SET",
            "payload": {
                "step": 0, "schema_valid": True,
                "proposal_set": {"proposals": [{"proposal_id": "p"}]},
                "policy_adapter_requested": "action_taking",
                "policy_adapter_used": "action_taking",
                "conditioning_skill_sha256": "skill-hash",
            },
        },
        {
            "episode_id": "e", "sequence": 1, "event_sha256": "h1",
            "previous_event_sha256": "h0", "kind": "AGENT_POST_TRANSITION_VERDICT",
            "payload": {"step": 0, "schema_valid": True, "verdict": {}},
        },
    ]
    episodes = [{"episode_id": "e", "game": "game", "steps": 1, "total_reward": 0.0}]
    replay = [{"receipt_id": "r"}]
    for name, rows in (
        ("events.jsonl", events), ("episodes.jsonl", episodes),
        ("replay_receipts.jsonl", replay),
    ):
        dump_jsonl(tmp_path / name, rows)
    manifest = {
        "metadata": {"game": "game"},
        "files": {
            name: {"sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}
            for name in ("events.jsonl", "episodes.jsonl", "replay_receipts.jsonl")
        },
        "replay_forks": {"file": "replay_receipts.jsonl"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    audit = audit_evidence_batch(tmp_path)
    assert audit.action_adapter_grounded_proposal_sets == 1
    assert audit.skill_conditioned_proposal_sets == 1
    assert "ACTION_PROPOSALS_NOT_GROUNDED_IN_CHECKPOINT_POLICY" not in audit.gaps
