from __future__ import annotations

import json

from motif_transfer.contracts import stable_hash
from motif_transfer.direct_prospective_matrix_v1 import SOURCE_GAMES
from motif_transfer.phase2_discoveryworld_utility_v1 import (
    AUTHENTIC,
    CONDITIONS,
    RAW,
    SCHEMA,
    STATUS,
    build_report,
    exact_two_sided_sign_test,
    file_sha256,
)


def test_exact_sign_test() -> None:
    assert exact_two_sided_sign_test(0, 0) == 1.0
    assert exact_two_sided_sign_test(6, 0) == 0.03125
    assert exact_two_sided_sign_test(9, 1) == 0.021484375


def test_powered_report_validates_matched_causal_effect(tmp_path) -> None:
    tasks = []
    sources = {}
    for game in SOURCE_GAMES:
        source_path = tmp_path / f"{game}.json"
        artifact_sha = stable_hash({"game": game})
        source_path.write_text(json.dumps({
            "artifact_sha256": artifact_sha,
            "source_lineage": {"game": game},
        }))
        sources[game] = (source_path, artifact_sha)
    for index in range(36):
        game = SOURCE_GAMES[index % 6]
        path, artifact_sha = sources[game]
        tasks.append({
            "task_id": f"proteomics.easy.seed{45 + index}",
            "scenario": "Proteomics",
            "difficulty": "Easy",
            "seed": 45 + index,
            "source_game": game,
            "source_artifact": path.name,
            "source_artifact_sha256": artifact_sha,
            "source_artifact_file_sha256": file_sha256(path),
            "selected_target_previously_executed": False,
        })
    manifest_body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_boundary": "unit test",
        "selection_read_target_outcome": False,
        "historical_target_outcome_reuse_allowed": False,
        "primary_endpoint": {
            "maximum_exact_two_sided_sign_p": 0.05,
            "maximum_discordant_negative_transfer_rate": 0.25,
        },
        "conditions": list(CONDITIONS),
        "tasks": tasks,
        "runtime_file_sha256": {},
    }
    manifest = manifest_body | {"manifest_sha256": stable_hash(manifest_body)}
    cells = []
    # Nine strict wins, one strict loss, and 26 ties: p=.021484375.
    for index, task in enumerate(tasks):
        raw = index == 9 or index >= 10
        authentic = index < 9 or index >= 10
        outcomes = {
            CONDITIONS[0]: raw,
            CONDITIONS[1]: authentic,
            CONDITIONS[2]: False,
            CONDITIONS[3]: False,
            CONDITIONS[4]: False,
        }
        route_body = {
            "source_artifact_sha256": task["source_artifact_sha256"],
            "source_action": "EXPLORE_UNTRIED",
            "admitted": True,
        }
        route = route_body | {"receipt_sha256": stable_hash(route_body)}
        cell_body = {
            "schema_version": "phase2-discoveryworld-causal-utility-cell-v1",
            "task_id": task["task_id"],
            "source_game": task["source_game"],
            "source_artifact_sha256": task["source_artifact_sha256"],
            "matched_result_file_sha256": "x",
            "outcomes": outcomes,
            "recovery_steps": {condition: 1 for condition in CONDITIONS},
            "all_matched_forks": True,
            "all_selection_receipts_valid": True,
            "mechanism_complete": True,
            "policy_runtime_saw_oracle_scorecard": False,
            "authentic_source_routes": [route],
            "runtime_error": None,
        }
        cells.append(cell_body | {"cell_sha256": stable_hash(cell_body)})
    report = build_report(manifest, cells, repo=tmp_path)
    assert report["status"] == "PHASE2_DISCOVERYWORLD_CAUSAL_UTILITY_VALIDATED"
    assert report["authentic_vs_raw"]["wins"] == 9
    assert report["authentic_vs_raw"]["losses"] == 1
    assert report["authentic_vs_raw"]["exact_two_sided_sign_test_p"] == 0.021484375
    assert report["condition_successes"][AUTHENTIC] == 35
    assert report["condition_successes"][RAW] == 27
    assert all(report["gates"].values())
