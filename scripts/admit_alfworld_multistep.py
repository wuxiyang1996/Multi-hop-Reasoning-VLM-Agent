#!/usr/bin/env python3
"""Freeze every evidence-qualified v3 multi-step Agent binding candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.multistep_binding import (  # noqa: E402
    FrozenMultiStepArtifactStore,
    MultiStepTargetAdmission,
    multistep_candidate_from_dict,
)
from harness.skill_admission import target_demo_receipt_from_dict  # noqa: E402


def _hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()

    demo = target_demo_receipt_from_dict(json.loads(args.demo.read_text(encoding="utf-8")))
    proposal_set = json.loads(args.proposals.read_text(encoding="utf-8"))
    claimed_proposal_hash = str(proposal_set.get("artifact_sha256") or "")
    unsigned_proposals = dict(proposal_set)
    unsigned_proposals.pop("artifact_sha256", None)
    if not claimed_proposal_hash or _hash(unsigned_proposals) != claimed_proposal_hash:
        raise SystemExit("v3 proposal artifact hash mismatch")
    if int(proposal_set.get("schema_version", 0)) != 2:
        raise SystemExit("unsupported v3 proposal schema")
    if proposal_set.get("candidate_source") != "independent_untrusted_agents":
        raise SystemExit("v3 proposals must come from independent untrusted Agents")
    source_treatment = str(proposal_set.get("source_treatment") or "")
    source_control_receipt = dict(proposal_set.get("source_control_receipt") or {})
    source_control_unsigned = dict(source_control_receipt)
    source_control_hash = str(source_control_unsigned.pop("receipt_sha256", ""))
    if (
        source_treatment not in {"empty", "correct", "wrong", "renamed"}
        or source_control_receipt.get("treatment") != source_treatment
        or not source_control_hash
        or _hash(source_control_unsigned) != source_control_hash
        or proposal_set.get("source_control_applied_before_binding_generation") is not True
    ):
        raise SystemExit("invalid or late binding source control receipt")
    candidates = [
        multistep_candidate_from_dict(item)
        for item in proposal_set.get("candidates") or ()
    ]
    proposal_receipts = proposal_set.get("proposal_receipts") or ()
    known_proposal_hashes = []
    for item in proposal_receipts:
        if set(item) != {"receipt_sha256", "receipt_payload"}:
            raise SystemExit("binding proposal receipt has wrong keys")
        receipt_hash = str(item["receipt_sha256"])
        if _hash(item["receipt_payload"]) != receipt_hash:
            raise SystemExit("binding proposal receipt hash mismatch")
        known_proposal_hashes.append(receipt_hash)
    known_source_hypotheses = {
        str(item["hypothesis_hash"]): [
            str(node["node_id"])
            for node in item["hypothesis"].get("nodes") or ()
        ]
        for item in proposal_set.get("qualified_source_hypotheses") or ()
    }
    known_source_node_conditioning = {}
    for graph in proposal_set.get("source_graphs") or ():
        graph_hash = str(graph["source_hypothesis_hash"])
        edges = list(graph.get("edges") or ())
        known_source_node_conditioning[graph_hash] = {
            str(node["node_id"]): {
                "observed_transitions": list(node.get("observed_transitions") or ()),
                "incident_edges": [
                    edge for edge in edges
                    if str(node["node_id"]) in (
                        str(edge["source_node_id"]), str(edge["target_node_id"])
                    )
                ],
            }
            for node in graph.get("nodes") or ()
        }
    artifact = MultiStepTargetAdmission().admit(
        candidates=candidates,
        demo=demo,
        known_proposal_receipt_hashes=known_proposal_hashes,
        known_source_hypothesis_nodes=known_source_hypotheses,
        known_source_node_conditioning=known_source_node_conditioning,
        source_treatment=source_treatment,
        source_control_receipt_sha256=source_control_hash,
    )
    path = FrozenMultiStepArtifactStore(args.output_root).freeze(artifact)
    manifest = {
        "schema_version": 3,
        "artifact_hash": artifact.artifact_hash,
        "artifact_path": path.name,
        "demo_hash": demo.content_hash(),
        "n_proposed": len(candidates),
        "n_qualified": len(artifact.candidates),
        "n_rejected": len(artifact.rejected_candidates),
        "candidate_retention": "content_hash_dedup_then_set_union_no_rank_no_vote",
        "runtime_rule": (
            "actor_selects_once_from_exact_intersection_of_all_candidate_supported_"
            "native_action_sets"
        ),
        "target_gradient_updates": 0,
        "semantic_alignment_claimed": False,
        "source_treatment": source_treatment,
        "source_control_receipt_sha256": source_control_hash,
        "source_control_applied_before_binding_generation": True,
        "gaps": list(proposal_set.get("gaps") or ()),
    }
    output = args.manifest_output or args.output_root / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
