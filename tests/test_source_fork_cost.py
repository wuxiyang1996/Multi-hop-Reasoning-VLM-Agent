from __future__ import annotations

import json
from pathlib import Path

from motif_transfer.contracts import stable_hash
from motif_transfer.relational_structural_induction import (
    build_source_intervention_dataset,
)
from motif_transfer.sokoban_commit_skill import PLAN_VERSION
from motif_transfer.source_fork_cost import reconstruct_source_fork_cost


REPO = Path(__file__).resolve().parents[1]


def _plan() -> dict:
    source = json.loads((
        REPO / "runs/sokoban_effect_program_v2/fresh_confirmation_plan.json"
    ).read_text(encoding="utf-8"))
    snapshot = source["snapshots"][0]
    body = {
        "plan_version": PLAN_VERSION,
        "split_counts": {str(snapshot["split"]): 1},
        "snapshots": [snapshot],
    }
    return body | {"plan_sha256": stable_hash(body)}


def test_complete_source_fork_cost_reconstructs_exact_dataset() -> None:
    plan = _plan()
    primitive = build_source_intervention_dataset(plan)
    episode = primitive["episodes"][0]
    result = reconstruct_source_fork_cost(
        plan,
        expected_primitive_dataset_sha256=primitive["dataset_sha256"],
        selected_episode_keys=[(
            episode["snapshot_id"], episode["episode_id"],
        )],
    )
    assert result["reconstruction_exact_hash_match"] is True
    assert result["source_snapshot_episodes"] == 1
    assert result["candidate_forks"] == 4
    assert result["successful_candidate_forks"] == 1
    assert result["failed_candidate_forks"] == 3
    success_steps = result["successful_path_primitive_transitions"]
    assert success_steps > 1
    assert result["all_candidate_primitive_transitions"] == 4 * success_steps
    assert result["failed_path_primitive_transitions"] == 3 * success_steps


def test_complete_source_fork_cost_rejects_wrong_lineage_hash() -> None:
    plan = _plan()
    primitive = build_source_intervention_dataset(plan)
    episode = primitive["episodes"][0]
    try:
        reconstruct_source_fork_cost(
            plan,
            expected_primitive_dataset_sha256="wrong",
            selected_episode_keys=[(
                episode["snapshot_id"], episode["episode_id"],
            )],
        )
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("wrong primitive lineage hash was accepted")
