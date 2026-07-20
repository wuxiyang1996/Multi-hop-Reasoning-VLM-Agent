from __future__ import annotations

import json
from pathlib import Path

from scripts.build_source_only_game_sft import build


def test_builder_uses_only_exact_executed_source_labels(tmp_path: Path) -> None:
    source = tmp_path / "source" / "game_a"
    source.mkdir(parents=True)
    episode = {
        "game_name": "game_a",
        "episode_id": "e1",
        "task": "collect blocks",
        "experiences": [
            {
                "idx": 0,
                "summary_state": "at start",
                "action": "right",
                "available_actions": ["left", "right"],
                "skills": {"skill_id": "NAVIGATE"},
                "skill_candidates": ["COLLECT", "NAVIGATE"],
            },
            {
                "idx": 1,
                "action": "invented",
                "available_actions": ["left", "right"],
                "skills": {"skill_id": "NAVIGATE"},
                "skill_candidates": ["NAVIGATE"],
            },
        ],
    }
    (source / "episode_000.json").write_text(json.dumps(episode))
    output = tmp_path / "out"
    manifest = build(tmp_path / "source", output, per_game=10, seed=1)
    assert manifest["n_selected"] == 1
    action = json.loads((output / "game_a/action_taking.jsonl").read_text())
    skill = json.loads((output / "game_a/skill_selection.jsonl").read_text())
    assert action["completion"] == "ACTION: 2"
    assert skill["completion"] == "SKILL: 2"
    assert action["source_only"] is True
    assert manifest["target_examples"] == 0
