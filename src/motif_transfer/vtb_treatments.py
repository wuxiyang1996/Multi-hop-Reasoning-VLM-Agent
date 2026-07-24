from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .contracts import stable_hash
from .transfer_matrix import REQUIRED_TARGET_CONDITIONS


def _qualified_candidate(bundle: Mapping[str, Any]) -> dict[str, Any]:
    candidate = deepcopy(dict(bundle.get("candidate") or {}))
    qualification = dict(bundle.get("qualification") or {})
    if qualification.get("lifecycle") != "SOURCE_SUPPORTED":
        raise ValueError("VTB source treatment requires SOURCE_SUPPORTED qualification")
    if qualification.get("candidate_sha256") != stable_hash(candidate):
        raise ValueError("source candidate does not match its qualification receipt")
    if not candidate.get("motif_id") or len(candidate.get("nodes") or []) < 2:
        raise ValueError("qualified source candidate has no non-trivial graph")
    if not candidate.get("edges"):
        raise ValueError("qualified source candidate has no control edge")
    return candidate


def _receipt_ids(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    values = []
    for node in candidate.get("nodes") or []:
        values.extend(str(row) for row in node.get("transition_receipt_ids") or [])
    for edge in candidate.get("edges") or []:
        values.extend(str(row) for row in edge.get("replay_receipt_ids") or [])
    result = tuple(dict.fromkeys(values))
    if not result:
        raise ValueError("qualified source candidate carries no source receipts")
    return result


def _alpha_renamed(candidate: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(candidate))
    node_ids = [str(row["node_id"]) for row in result["nodes"]]
    node_map = {node_id: f"NODE_{index}" for index, node_id in enumerate(node_ids)}
    for node in result["nodes"]:
        node["node_id"] = node_map[str(node["node_id"])]
    for edge in result["edges"]:
        edge["source"] = node_map[str(edge["source"])]
        edge["target"] = node_map[str(edge["target"])]
        edge["untrusted_claim"] = ""
    result["source_lineage"] = [
        f"LINEAGE_{index}" for index, _ in enumerate(result.get("source_lineage") or [])
    ]
    result["motif_id"] = "ALPHA_RENAMED_MOTIF"
    result["untrusted_description"] = ""
    return result


def _shuffled(candidate: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(candidate))
    node_ids = [str(row["node_id"]) for row in result["nodes"]]
    rotated = node_ids[1:] + node_ids[:1]
    mapping = dict(zip(node_ids, rotated))
    for edge in result["edges"]:
        # Preserve edge count and source evidence slots while breaking which
        # observed control state each target is attached to.
        edge["target"] = mapping[str(edge["target"])]
        edge["untrusted_claim"] = ""
    result["motif_id"] = "SHUFFLED_TOPOLOGY_CONTROL"
    result["untrusted_description"] = ""
    if stable_hash(result) == stable_hash(candidate):
        raise ValueError("shuffled control did not alter the candidate")
    return result


def _generic(candidate: Mapping[str, Any]) -> dict[str, Any]:
    receipt_slot = 0
    nodes = []
    for node_index, node in enumerate(candidate.get("nodes") or []):
        slots = []
        for _ in node.get("transition_receipt_ids") or []:
            slots.append(f"RECEIPT_SLOT_{receipt_slot}")
            receipt_slot += 1
        nodes.append({
            "node_id": f"NODE_SLOT_{node_index}",
            "transition_receipt_ids": slots,
            "decision_signatures": ["SIGNATURE_SLOT"] * len(node.get("decision_signatures") or []),
        })
    node_map = {
        str(node["node_id"]): f"NODE_SLOT_{index}"
        for index, node in enumerate(candidate.get("nodes") or [])
    }
    edges = []
    for edge in candidate.get("edges") or []:
        replay_slots = []
        for _ in edge.get("replay_receipt_ids") or []:
            replay_slots.append(f"RECEIPT_SLOT_{receipt_slot}")
            receipt_slot += 1
        edges.append({
            "source": node_map[str(edge["source"])],
            "target": node_map[str(edge["target"])],
            "replay_receipt_ids": replay_slots,
            "untrusted_claim": "",
        })
    return {
        "motif_id": "MATCHED_GENERIC_CONTROL",
        "source_lineage": ["LINEAGE_SLOT"] * len(candidate.get("source_lineage") or []),
        "nodes": nodes,
        "edges": edges,
        "untrusted_description": "",
    }


def compile_vtb_treatments(
    authentic_bundle: Mapping[str, Any], other_game_bundle: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    authentic = _qualified_candidate(authentic_bundle)
    other = _qualified_candidate(other_game_bundle)
    if other.get("motif_id") == authentic.get("motif_id"):
        raise ValueError("other-game control must be a different qualified motif")
    authentic_receipts = _receipt_ids(authentic)
    other_receipts = _receipt_ids(other)
    payloads = {
        "generic_reasoning": (_generic(authentic), ()),
        "authentic_game_source": (authentic, authentic_receipts),
        "renamed_game_source": (_alpha_renamed(authentic), authentic_receipts),
        "shuffled_game_source": (_shuffled(authentic), authentic_receipts),
        "other_game_source": (other, other_receipts),
    }
    result = {}
    for condition, (payload, receipt_ids) in payloads.items():
        result[condition] = {
            "schema_version": 1,
            "condition": condition,
            "source_lifecycle": "CONTROL" if condition == "generic_reasoning" else "SOURCE_SUPPORTED",
            "source_receipt_ids": list(receipt_ids),
            "payload": payload,
            "payload_sha256": stable_hash(payload),
        }
    if set(result) != set(REQUIRED_TARGET_CONDITIONS) - {"target_only"}:
        raise AssertionError("compiler did not produce the frozen five treatment artifacts")
    return result
