from __future__ import annotations

import json

from motif_transfer.contracts import stable_hash
from motif_transfer.direct_prospective_matrix_v1 import SOURCE_GAMES
from motif_transfer.phase2_discoveryworld_utility_v1 import CONDITIONS
from motif_transfer.phase2_discoveryworld_utility_v2 import (
    SCHEMA, STATUS, build_report, file_sha256, make_cell, sign_p,
)


def test_selective_report_keeps_abstention_and_validates_effect(tmp_path) -> None:
    assert sign_p(9, 1) == 0.021484375
    tasks = []
    for index in range(36):
        episode = tmp_path / f"episode-{index}.json"
        episode.write_text("{}")
        fork = tmp_path / f"fork-{index}.json"
        fork.write_text("{}")
        tasks.append({
            "task_id": f"task-{index}", "source_game": SOURCE_GAMES[index % 6],
            "source_artifact_sha256": f"source-{index % 6}",
            "applicable": index != 35,
            "target_episode": episode.name,
            "target_episode_file_sha256": file_sha256(episode),
            "fork_config": fork.name if index != 35 else None,
            "fork_config_file_sha256": file_sha256(fork) if index != 35 else None,
            "abstention_rule": None if index != 35 else "INHERIT_RECORDED_TARGET_ONLY_OUTCOME_FOR_ALL_ARMS",
        })
    manifest_body = {
        "schema_version": SCHEMA, "status": STATUS,
        "claim_boundary": "unit test", "matched_outcomes_visible_at_freeze": False,
        "eligibility_read_target_outcome": False,
        "primary_endpoint": {"maximum_p": .05, "maximum_negative_rate": .25},
        "tasks": tasks, "runtime_file_sha256": {},
    }
    manifest = manifest_body | {"manifest_sha256": stable_hash(manifest_body)}
    cells = []
    for index, task in enumerate(tasks):
        if not task["applicable"]:
            cells.append(make_cell(
                manifest_sha256=manifest["manifest_sha256"], task=task,
                outcomes={condition: False for condition in CONDITIONS},
                recovery_steps={condition: 0 for condition in CONDITIONS}, routes=[],
                matched_result_file_sha256=None, all_matched_forks=True,
                all_selection_receipts_valid=True, runtime_error=None,
            ))
            continue
        raw = index == 9 or index >= 10
        authentic = index < 9 or index >= 10
        route_body = {
            "admitted": True, "source_artifact_sha256": task["source_artifact_sha256"],
            "source_action": "EXPLORE_UNTRIED",
        }
        route = route_body | {"receipt_sha256": stable_hash(route_body)}
        cells.append(make_cell(
            manifest_sha256=manifest["manifest_sha256"], task=task,
            outcomes={
                CONDITIONS[0]: raw, CONDITIONS[1]: authentic,
                CONDITIONS[2]: False, CONDITIONS[3]: False, CONDITIONS[4]: False,
            },
            recovery_steps={condition: 1 for condition in CONDITIONS}, routes=[route],
            matched_result_file_sha256="x", all_matched_forks=True,
            all_selection_receipts_valid=True, runtime_error=None,
        ))
    report = build_report(manifest, cells, repo=tmp_path)
    assert report["status"] == "PHASE2_DISCOVERYWORLD_SELECTIVE_CAUSAL_UTILITY_VALIDATED"
    assert report["eligible_matched_tasks"] == 35
    assert report["fail_closed_abstentions"] == 1
    assert report["authentic_vs_raw"]["wins"] == 9
    assert report["authentic_vs_raw"]["losses"] == 1
    assert all(report["gates"].values())
