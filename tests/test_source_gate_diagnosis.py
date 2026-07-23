from __future__ import annotations

import json

from motif_transfer.source_gate_diagnosis import (
    paired_return_diagnostic,
    summarize_source_evidence,
)


def _write_evidence(root, seed, rewards, skill="S", total=None):
    root.mkdir(parents=True)
    episode_id = f"e{seed}"
    events = [{
        "episode_id": episode_id,
        "kind": "RESET",
        "payload": {"requested_seed": seed},
    }]
    for step, reward in enumerate(rewards):
        events.extend([
            {
                "episode_id": episode_id,
                "kind": "AGENT_PROPOSAL_SET",
                "payload": {"step": step, "selected_skill_id": skill},
            },
            {
                "episode_id": episode_id,
                "kind": "ENVIRONMENT_STEP",
                "payload": {"step": step, "reward": reward},
            },
        ])
    (root / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    (root / "episodes.jsonl").write_text(json.dumps({
        "episode_id": episode_id,
        "total_reward": sum(rewards) if total is None else total,
    }) + "\n")


def test_source_summary_measures_reward_support_without_skill_semantics(tmp_path):
    root = tmp_path / "run" / "evidence"
    _write_evidence(root, 7, [0.0, 0.0, 5.0, 0.0])
    summary = summarize_source_evidence(root)
    assert summary["positive_reward_density"] == 0.25
    assert summary["skills"] == [{
        "skill_id": "S",
        "selected_steps": 4,
        "continuous_edges": 3,
        "positive_reward_support": {"h1": 1, "h2": 2, "h4": 3, "h8": 3},
    }]


def test_paired_returns_are_labeled_noncausal(tmp_path):
    authentic = tmp_path / "auth" / "evidence"
    skill_off = tmp_path / "off" / "evidence"
    _write_evidence(authentic, 7, [2.0], total=2.0)
    _write_evidence(skill_off, 7, [1.0], total=1.0)
    report = paired_return_diagnostic(authentic, skill_off)
    assert report["paired_differences"] == [1.0]
    assert "does not" in report["claim_limit"]
