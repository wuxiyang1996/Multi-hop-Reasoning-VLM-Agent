from __future__ import annotations

import json

from motif_transfer.multihorizon_replay import stable_hash
from motif_transfer.source_multihorizon import (
    build_multihorizon_plan,
    prompt_parts,
    render_continuation_prompt,
    validate_plan,
)


def _prompt(skill: str = "") -> str:
    prefix = "SYSTEM"
    if skill:
        prefix += f"\n--- Active Skill: {skill} ---\n  Strategy: frozen"
    return (
        f"{prefix}\n\nGame state:\n\nOLD_STATE\n\n"
        "Subgoal: [EXECUTE] test\n"
        + (f"Active skill: {skill}\n" if skill else "")
        + "Recent actions and rewards:\n"
        "  LEFT -> reward 0.0\n"
        "  RIGHT -> reward 1.0\n\n"
        "Available actions (pick ONE by number):\n"
        "  1. LEFT\n  2. RIGHT\n\n"
        "Brief REASONING (1 sentence max) then ACTION: <number>."
    )


def _record(episode_id, seed, treatment, skill=""):
    prompt = _prompt(skill)
    response = "REASONING: test\nACTION: 2"
    adapter = None if treatment == "B" else "action_taking"
    return {
        "episode_id": episode_id,
        "episode_seed": seed,
        "step": 1,
        "source_skill_id": "COMMIT/TEST",
        "treatment": treatment,
        "context_skill_id": skill or None,
        "prefix_actions": ["LEFT"],
        "before_observable_sha256": stable_hash("fork"),
        "native_actions": ["LEFT", "RIGHT"],
        "native_actions_sha256": stable_hash(["LEFT", "RIGHT"]),
        "prompt": prompt,
        "prompt_sha256": stable_hash(prompt),
        "raw_response": response,
        "raw_response_sha256": stable_hash(response),
        "parsed_action": "RIGHT",
        "parser_fallback": False,
        "requested_adapter": adapter,
        "used_adapter": adapter,
        "replay_status": "INTERVENTION_OBSERVED",
    }


def test_prompt_renderer_keeps_frozen_context_and_updates_live_state():
    source = _prompt("Commit/Test")
    parts = prompt_parts(source)
    assert parts["subgoal"] == "[EXECUTE] test"
    assert parts["static_context"] == "Active skill: Commit/Test"
    rendered = render_continuation_prompt(
        source,
        state_markup="NEW_STATE",
        native_actions=("UP", "DOWN"),
        branch_history=(("UP", 2.0),),
    )
    assert "OLD_STATE" not in rendered
    assert "NEW_STATE" in rendered
    assert "Active skill: Commit/Test" in rendered
    assert "  1. UP\n  2. DOWN" in rendered
    assert "UP -> reward 2.0" in rendered


def test_build_plan_selects_blind_splits_without_reading_outcomes(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    episodes = [
        {"episode_id": "episode-a"},
        {"episode_id": "episode-b"},
        {"episode_id": "episode-c"},
    ]
    (evidence / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8"
    )
    records = []
    for episode_id, seed in (("episode-b", 11), ("episode-c", 12)):
        for treatment, skill in (
            ("B", ""),
            ("G_MINUS_S", ""),
            ("G_PLUS_S", "Commit/Test"),
            ("G_PLUS_RANDOM", "Other/Test"),
        ):
            records.append(_record(episode_id, seed, treatment, skill))
    (evidence / "matched_policy_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    (evidence / "manifest.json").write_text(json.dumps({
        "metadata": {"game": "gymv_test", "max_steps": 20},
    }), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"frozen": True}), encoding="utf-8")

    plan = build_multihorizon_plan(
        evidence,
        config_path=config,
        maximum_per_split=1,
    )
    validate_plan(plan)
    assert plan["selected_counts"] == {"qualification": 1, "held_out": 1}
    assert {row["episode_id"] for row in plan["snapshots"]} == {
        "episode-b", "episode-c",
    }
    assert plan["selection_authority"].startswith("LINEAGE_ONLY")
