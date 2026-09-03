"""Reconstruct complete primitive cost from a hash-bound source plan."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .relational_structural_induction import build_source_intervention_dataset


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["snapshot_id"]), str(row["episode_id"])


def reconstruct_source_fork_cost(
    plan: Mapping[str, Any], *, expected_primitive_dataset_sha256: str,
    selected_episode_keys: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Replay the deterministic source collector and count every fork step.

    ``build_source_intervention_dataset`` executes all candidate action
    sequences through the source simulator.  Matching its dataset hash to the
    hash embedded in the downstream compact receipts proves this is the exact
    pre-compression dataset rather than an approximate reconstruction.
    """

    primitive = build_source_intervention_dataset(plan)
    observed_hash = str(primitive["dataset_sha256"])
    if observed_hash != str(expected_primitive_dataset_sha256):
        raise ValueError("reconstructed source primitive dataset hash mismatch")
    selected = set(selected_episode_keys)
    episodes = [
        row for row in primitive["episodes"] if _key(row) in selected
    ]
    if {_key(row) for row in episodes} != selected:
        raise ValueError("selected source episode was absent after reconstruction")
    candidates = [
        candidate
        for episode in episodes
        for candidate in episode["candidates"]
    ]
    successful = [
        candidate for candidate in candidates
        if candidate["success_from_state_only"]
    ]
    unsuccessful = [
        candidate for candidate in candidates
        if not candidate["success_from_state_only"]
    ]
    per_episode = []
    for episode in episodes:
        rows = list(episode["candidates"])
        per_episode.append({
            "snapshot_id": str(episode["snapshot_id"]),
            "episode_id": str(episode["episode_id"]),
            "candidate_forks": len(rows),
            "all_candidate_primitive_transitions": sum(
                len(row["tuples"]) for row in rows
            ),
            "successful_path_primitive_transitions": sum(
                len(row["tuples"])
                for row in rows if row["success_from_state_only"]
            ),
            "failed_path_primitive_transitions": sum(
                len(row["tuples"])
                for row in rows if not row["success_from_state_only"]
            ),
        })
    return {
        "reconstruction_exact_hash_match": True,
        "primitive_dataset_sha256": observed_hash,
        "source_snapshot_episodes": len(episodes),
        "candidate_fork_resets": len(candidates),
        "candidate_forks": len(candidates),
        "successful_candidate_forks": len(successful),
        "failed_candidate_forks": len(unsuccessful),
        "all_candidate_primitive_transitions": sum(
            len(row["tuples"]) for row in candidates
        ),
        "successful_path_primitive_transitions": sum(
            len(row["tuples"]) for row in successful
        ),
        "failed_path_primitive_transitions": sum(
            len(row["tuples"]) for row in unsuccessful
        ),
        "per_episode": sorted(
            per_episode,
            key=lambda row: (row["snapshot_id"], row["episode_id"]),
        ),
    }


__all__ = ["reconstruct_source_fork_cost"]
