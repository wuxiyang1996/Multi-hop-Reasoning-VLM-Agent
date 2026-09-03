import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_workspace_freezes_all_five_named_worktrees_and_roles() -> None:
    config = json.loads((
        REPO / "configs/reproducible_workspace_v1.json"
    ).read_text(encoding="utf-8"))
    components = {row["directory"]: row for row in config["components"]}
    assert set(components) == {
        "Multi-hop-Reasoning-VLM-Agent-source-fresh-v1",
        "Multi-hop-Reasoning-VLM-Agent-two-agent-clean",
        "Multi-hop-Reasoning-VLM-Agent-github-main",
        "Multi-hop-Reasoning-VLM-Agent-experiment-clean",
        "Multi-hop-Reasoning-VLM-Agent",
    }
    assert {
        name for name, row in components.items()
        if row["required_for_v3_substitution"]
    } == {
        "Multi-hop-Reasoning-VLM-Agent-two-agent-clean",
        "Multi-hop-Reasoning-VLM-Agent-github-main",
    }


def test_workspace_does_not_mislabel_frozen_cohort_as_full_outcome() -> None:
    config = json.loads((
        REPO / "configs/reproducible_workspace_v1.json"
    ).read_text(encoding="utf-8"))
    boundary = config["result_boundary"]
    assert boundary["full_official_six_benchmark_outcome_reproduced"] is False
    assert boundary["full_protocol_sizes"] == {
        "webshop": 500,
        "alfworld": 134,
        "discoveryworld": 120,
        "tirbench": 1215,
        "clevrer": 76368,
        "agqa2": 669207,
    }
