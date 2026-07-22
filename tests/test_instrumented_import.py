import hashlib
import json

from motif_transfer.instrumented_import import import_instrumented_batch, import_native_source_batch


def dump(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def event(kind, step, payload):
    return {"episode_id": "e", "kind": kind, "payload": {"step": step, **payload}}


def test_complete_external_cycle_import(tmp_path):
    events = [
        event("OBSERVATION", 0, {"observable_state": "before", "structured_state": {}}),
        event(
            "AGENT_ACTION_PROPOSAL_SET",
            0,
            {
                "schema_valid": True,
                "proposal_set_sha256": "set",
                "proposal_set": {
                    "decision": "EXECUTE",
                    "selected_proposal_id": "p0",
                    "proposals": [
                        {
                            "proposal_id": "p0",
                            "action_number": 1,
                            "predicted_observable_delta": "change",
                            "rationale": "untrusted",
                        }
                    ],
                },
            },
        ),
        event("NATIVE_ADMISSIBILITY", 0, {"native_actions": ["go"]}),
        event("AGENT_DECISION", 0, {"decision_origin": "AGENT"}),
        event("ENVIRONMENT_STEP", 0, {"executed_action": "go", "reward": 1, "terminated": True}),
        event("OBSERVATION", 1, {"observable_state": "after", "structured_state": {}}),
        event(
            "AGENT_POST_TRANSITION_VERDICT",
            0,
            {
                "schema_valid": True,
                "verdict": {
                    "verdict": "SUPPORTED",
                    "decision": "TERMINATE",
                    "evidence_claim": "observed",
                },
            },
        ),
    ]
    dump(tmp_path / "events.jsonl", events)
    dump(tmp_path / "episodes.jsonl", [{"episode_id": "e", "game": "g", "steps": 1, "total_reward": 1}])
    dump(tmp_path / "replay_receipts.jsonl", [])
    imported = import_instrumented_batch(tmp_path)
    assert len(imported) == 1
    assert len(imported[0].records) == 1
    assert imported[0].records[0].validate()


def test_native_source_policy_import_does_not_fabricate_proposals(tmp_path):
    events = [
        event("OBSERVATION", 0, {"observable_state": "before", "structured_state": {}}),
        event("NATIVE_ADMISSIBILITY", 0, {"native_actions": ["go", "wait"]}),
        event("AGENT_PROPOSAL_SET", 0, {
            "selected_skill_id": "skill-1",
            "selected_skill_sha256": "skill-hash",
            "skill_candidates": [{"skill_id": "skill-1"}],
        }),
        event("AGENT_RESPONSE", 0, {
            "adapter": "action_taking", "raw_response_sha256": "response-hash",
        }),
        event("PARSED_DECISION", 0, {"reasoning": "source policy reasoning"}),
        event("AGENT_DECISION", 0, {
            "decision_origin": "AGENT", "executed_action": "go",
        }),
        event("ENVIRONMENT_STEP", 0, {
            "executed_action": "go", "reward": 1, "terminated": True,
        }),
        event("OBSERVATION", 1, {"observable_state": "after", "structured_state": {}}),
    ]
    dump(tmp_path / "events.jsonl", events)
    dump(tmp_path / "episodes.jsonl", [{
        "episode_id": "e", "game": "g", "steps": 1, "total_reward": 1,
    }])
    dump(tmp_path / "replay_receipts.jsonl", [])
    imported = import_native_source_batch(tmp_path)
    assert len(imported) == 1
    assert len(imported[0].records) == 1
    record = imported[0].records[0]
    assert record.selected_skill_id == "skill-1"
    assert record.policy_adapter == "action_taking"
    assert record.action == "go"
    assert record.validate()


def test_native_source_policy_preserves_postprocessor_action_origin(tmp_path):
    events = [
        event("OBSERVATION", 0, {"observable_state": "before", "structured_state": {}}),
        event("NATIVE_ADMISSIBILITY", 0, {"native_actions": ["go", "wait"]}),
        event("AGENT_PROPOSAL_SET", 0, {
            "selected_skill_id": "skill-1", "selected_skill_sha256": "skill-hash",
        }),
        event("AGENT_RESPONSE", 0, {
            "adapter": "action_taking", "raw_response_sha256": "response-hash",
        }),
        event("PARSED_DECISION", 0, {"reasoning": "proposed wait"}),
        event("AGENT_DECISION", 0, {
            "decision_origin": "POLICY_POSTPROCESSOR", "executed_action": "go",
        }),
        event("ENVIRONMENT_STEP", 0, {
            "executed_action": "go", "reward": 0, "terminated": True,
        }),
        event("OBSERVATION", 1, {"observable_state": "after", "structured_state": {}}),
    ]
    dump(tmp_path / "events.jsonl", events)
    dump(tmp_path / "episodes.jsonl", [{
        "episode_id": "e", "game": "g", "steps": 1, "total_reward": 0,
    }])
    dump(tmp_path / "replay_receipts.jsonl", [])
    imported = import_native_source_batch(tmp_path)
    assert imported[0].gaps == ()
    assert imported[0].records[0].action_origin == "POLICY_POSTPROCESSOR"
    assert imported[0].records[0].validate()
