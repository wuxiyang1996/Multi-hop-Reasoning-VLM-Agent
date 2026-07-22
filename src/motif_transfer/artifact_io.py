from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    BindingHypothesis,
    Lifecycle,
    MotifCandidate,
    MotifEdge,
    MotifNode,
    SourceStepSignature,
    stable_hash,
)
from .binding import (
    AttributedBinding,
    BindingArtifactStatus,
    BindingAttribution,
    FrozenBindingArtifact,
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


def write_frozen_binding_artifact(path: str | Path, artifact: FrozenBindingArtifact) -> None:
    if not artifact.validate():
        raise ValueError("refusing to write an invalid frozen binding artifact")
    payload = artifact.unsigned_payload()
    payload["artifact_hash"] = artifact.artifact_hash
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_frozen_binding_artifact(path: str | Path) -> FrozenBindingArtifact:
    raw = load_json(path)
    bindings = []
    for item in raw.get("bindings", []):
        hypothesis = item["hypothesis"]
        bindings.append(AttributedBinding(
            BindingHypothesis(
                binding_id=str(hypothesis["binding_id"]),
                motif_id=str(hypothesis["motif_id"]),
                target_claim=str(hypothesis["target_claim"]),
                testable_prediction=str(hypothesis["testable_prediction"]),
                adaptation_receipt_ids=tuple(hypothesis["adaptation_receipt_ids"]),
                verifier_id=str(hypothesis["verifier_id"]),
                status=Lifecycle(str(hypothesis["status"])),
                node_alignment=tuple(
                    (int(node), tuple(int(index) for index in indices))
                    for node, indices in hypothesis["node_alignment"]
                ),
                edge_alignment=tuple(
                    (int(edge), tuple(int(index) for index in boundary))
                    for edge, boundary in hypothesis["edge_alignment"]
                ),
                invariance_signature=str(hypothesis["invariance_signature"]),
            ),
            BindingAttribution(str(item["attribution"])),
        ))
    artifact = FrozenBindingArtifact(
        schema_version=int(raw["schema_version"]),
        motif_id=str(raw["motif_id"]),
        adaptation_example_sha256=str(raw["adaptation_example_sha256"]),
        induction_repetitions=int(raw["induction_repetitions"]),
        raw_signature_sets=tuple(tuple(row) for row in raw["raw_signature_sets"]),
        alpha_signature_sets=tuple(tuple(row) for row in raw["alpha_signature_sets"]),
        bindings=tuple(bindings),
        status=BindingArtifactStatus(str(raw["status"])),
        backend_identity_sha256=str(raw["backend_identity_sha256"]),
        call_receipt_hashes=tuple(raw["call_receipt_hashes"]),
        artifact_hash=str(raw["artifact_hash"]),
    )
    if not artifact.validate():
        raise ValueError(f"frozen binding artifact hash mismatch: {path}")
    return artifact
