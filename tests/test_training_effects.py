from __future__ import annotations

import hashlib
import json

from motif_transfer.training_effects import analyze_training_effects


def _hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_matched_effect_analysis_is_fail_closed(tmp_path):
    treatments = ("B", "G_MINUS_S", "G_PLUS_S", "G_PLUS_RANDOM")
    records, replays = [], []
    for seed in range(6):
        for offset in range(2):
            step = 4 + offset
            for treatment in treatments:
                prompt = "masked" if treatment in {"B", "G_MINUS_S"} else treatment
                action = "A" if treatment in {"B", "G_MINUS_S"} else "B"
                if treatment == "G_PLUS_RANDOM":
                    action = "C"
                response = f"ACTION: {action}"
                replay = {
                    "intervention_id": f"e{seed}.matched_step_{step}.{treatment}",
                    "seed": seed, "prefix_actions": ["A"] * step,
                    "expected_fork_state_sha256": _hash([seed, step]),
                    "replayed_fork_state_sha256": _hash([seed, step]),
                    "alternative_action": action,
                    "admissible_actions_sha256": _hash(["A", "B", "C"]),
                    "alternative_next_state_sha256": _hash([seed, step, action]),
                    "status": "INTERVENTION_OBSERVED", "failure_codes": [],
                }
                receipt_id = _hash(replay)
                replays.append(replay | {
                    "receipt_sha256": receipt_id,
                    "alternative_reward": 1.0 if treatment == "G_PLUS_S" else 0.0,
                })
                records.append({
                    "episode_id": f"e{seed}", "episode_seed": seed, "step": step,
                    "treatment": treatment,
                    "sampling_order": "AUTHENTIC_FIRST_SHADOW_AFTER_V1",
                    "requested_adapter": None if treatment == "B" else "action_taking",
                    "used_adapter": "base" if treatment == "B" else "action_taking",
                    "source_skill_id": "COMMIT/ATTACK",
                    "context_skill_id": (
                        "COMMIT/ATTACK" if treatment == "G_PLUS_S" else
                        "OTHER" if treatment == "G_PLUS_RANDOM" else None
                    ),
                    "prefix_actions": ["A"] * step,
                    "before_observable_sha256": _hash([seed, step]),
                    "native_actions": ["A", "B", "C"],
                    "native_actions_sha256": _hash(["A", "B", "C"]),
                    "prompt": prompt, "prompt_sha256": _hash(prompt),
                    "raw_response": response, "raw_response_sha256": _hash(response),
                    "parsed_action": action, "parser_fallback": False,
                    "replay_receipt_sha256": receipt_id,
                    "replay_status": "INTERVENTION_OBSERVED",
                    "after_observable_sha256": _hash([seed, step, action]),
                })
    records_path = tmp_path / "matched_policy_records.jsonl"
    replay_path = tmp_path / "matched_policy_replays.jsonl"
    records_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    replay_path.write_text("".join(json.dumps(row) + "\n" for row in replays))
    manifest = {
        "matched_policy_treatments": {
            "source_skill_id": "COMMIT/ATTACK", "snapshot_count": 12,
            "records_file": records_path.name, "records_sha256": _file_hash(records_path),
            "replays_file": replay_path.name, "replays_sha256": _file_hash(replay_path),
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    report = analyze_training_effects(tmp_path)
    assert report["gates"] == {
        "SOURCE_AUTHORITY_ORDER_SAFE": True,
        "SOURCE_CAUSAL_SUPPORTED": True,
        "SOURCE_GRAPH_SUPPORTED": True,
        "SOURCE_VALUE_SUPPORTED": True,
        "PHASE7_PASS": True,
    }
