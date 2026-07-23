from __future__ import annotations

from motif_transfer.discovery_selection import select_discovery_candidate


def _episode(episode_id, seed, skill, rewards):
    rows = [{
        "episode_id": episode_id,
        "kind": "RESET",
        "payload": {"requested_seed": seed},
        "event_sha256": f"reset-{episode_id}",
    }]
    for step, reward in enumerate(rewards):
        rows.extend([
            {
                "episode_id": episode_id,
                "kind": "AGENT_PROPOSAL_SET",
                "payload": {"step": step, "selected_skill_id": skill},
                "event_sha256": f"proposal-{episode_id}-{step}",
            },
            {
                "episode_id": episode_id,
                "kind": "ENVIRONMENT_STEP",
                "payload": {"step": step, "reward": reward},
                "event_sha256": f"transition-{episode_id}-{step}",
            },
        ])
    return rows


def test_selection_reads_only_discovery_content():
    # Seeds 0 and 3 are discovery. Seeds 1/4 and 2/5 are qualification/heldout.
    discovery = _episode("d0", 0, "A", [0, 1, 0]) + _episode(
        "d3", 3, "A", [0, 0, 1]
    )
    hidden_a = (
        _episode("q1", 1, "B", [100, 100])
        + _episode("h2", 2, "B", [100, 100])
        + _episode("q4", 4, "B", [100, 100])
        + _episode("h5", 5, "B", [100, 100])
    )
    hidden_b = (
        _episode("q1", 1, "C", [0, 0])
        + _episode("h2", 2, "C", [0, 0])
        + _episode("q4", 4, "C", [0, 0])
        + _episode("h5", 5, "C", [0, 0])
    )
    first = select_discovery_candidate(discovery + hidden_a, events_sha256="same")
    second = select_discovery_candidate(discovery + hidden_b, events_sha256="same")
    assert first["selected_skill_id"] == second["selected_skill_id"] == "A"
    assert first["candidates"] == second["candidates"]
    assert first["content_scope"] == "DISCOVERY_EVENTS_ONLY"


def test_selection_fails_closed_without_reward_supported_recurrence():
    rows = (
        _episode("d0", 0, "A", [0, 0])
        + _episode("q1", 1, "A", [1])
        + _episode("h2", 2, "A", [1])
        + _episode("d3", 3, "A", [0, 0])
    )
    report = select_discovery_candidate(rows, events_sha256="x")
    assert report["status"] == "NO_ELIGIBLE_CANDIDATE"
    assert report["selected_skill_id"] is None
