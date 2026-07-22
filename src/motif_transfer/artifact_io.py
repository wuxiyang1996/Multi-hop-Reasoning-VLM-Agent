from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    Lifecycle,
    MotifCandidate,
    MotifEdge,
    MotifNode,
    SourceStepSignature,
    stable_hash,
)


def load_json(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not one JSON object: {path}")
    return value


def load_first_source_motif(path: str | Path, *, status: Lifecycle) -> MotifCandidate:
    artifact = load_json(path)
    for episode in artifact.get("episodes", []):
        candidates = episode.get("candidates") or []
        if not candidates:
            continue
        raw = candidates[0]
        nodes = []
        for node in raw["nodes"]:
            signatures = tuple(
                SourceStepSignature(
                    bool(row["skill_conditioned"]),
                    str(row["action_origin"]),
                    str(row["reward_sign"]),
                    bool(row["terminal"]),
                    row.get("skill_class_ordinal"),
                )
                for row in node.get("decision_signatures", [])
            )
            nodes.append(MotifNode(
                str(node["node_id"]),
                tuple(str(value) for value in node["transition_receipt_ids"]),
                signatures,
            ))
        edges = tuple(MotifEdge(
            str(edge["source"]),
            str(edge["target"]),
            tuple(str(value) for value in edge["replay_receipt_ids"]),
            str(edge.get("untrusted_claim", "")),
        ) for edge in raw["edges"])
        return MotifCandidate(
            str(raw["motif_id"]),
            tuple(str(value) for value in raw["source_lineage"]),
            tuple(nodes),
            edges,
            status,
            str(raw.get("untrusted_description", "")),
        )
    raise ValueError(f"no source motif candidate in {path}")


def adaptation_example_view(path: str | Path) -> Mapping[str, Any]:
    artifact = load_json(path)
    if not artifact.get("official_success"):
        raise ValueError("one-shot adaptation example is not officially successful")
    actions = artifact.get("actions") or []
    return {
        "artifact_sha256": stable_hash(artifact),
        "demo_id": artifact.get("demo_id"),
        "target_domain": artifact.get("target_domain"),
        "task_family": artifact.get("task_family"),
        "official_success": True,
        "transitions": [
            {
                "index": row.get("transition_index", index),
                "action": row.get("action"),
                "before_native_actions": row.get("before_admissible_actions", []),
                "after_native_actions": row.get("after_admissible_actions", []),
                "reward": row.get("reward"),
                "official_success_after": row.get("official_success_after"),
                "state_sha256": row.get("state_sha256"),
                "next_state_sha256": row.get("next_state_sha256"),
            }
            for index, row in enumerate(actions)
        ],
    }
