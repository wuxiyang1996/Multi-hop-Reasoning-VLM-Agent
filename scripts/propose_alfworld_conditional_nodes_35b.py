#!/usr/bin/env python3
"""Ask matched untrusted Agents for multi-example conditional node programs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.binding_source_controls import apply_binding_source_control  # noqa: E402
from harness.conditional_node_program import (  # noqa: E402
    ConditionalProgramProposal, ExampleSegmentation, ProposedSegment, SegmentKind,
)
from harness.frozen_transfer_policy import StrictOpenAIClient  # noqa: E402
from harness.skill_admission import target_demo_receipt_from_dict  # noqa: E402
from scripts.propose_alfworld_multistep_bindings_35b import _source_graphs  # noqa: E402


_ROLES = ("proposer_a", "proposer_b", "skeptic")
_JSON_ONLY = re.compile(r"\A\s*\{.*\}\s*\Z", re.DOTALL)


def _hash(value) -> str:
    raw = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _demo_trace(demo):
    return {
        "demo_id": demo.demo_id,
        "transitions": [{
            "target_transition_id": f"target_t{step.transition_index}",
            "operator": step.operator, "argument_types": dict(step.argument_types),
            "action": step.action, "state_sha256": step.state_sha256,
            "next_state_sha256": step.next_state_sha256,
            "official_success_after": step.official_success_after,
        } for step in demo.actions],
    }


def _prompt(role, graphs, demos, repair_context=None, condition="source"):
    role_instruction = {
        "proposer_a": "Find one receipt-compatible reusable node-local program.",
        "proposer_b": "Independently find a different compatible segmentation if one exists.",
        "skeptic": "Try to expose non-transferable steps using explicit TARGET_NATIVE_GAP segments.",
    }[role]
    condition_text = (
        "The graph contains source transition evidence."
        if condition == "source" else
        "This is a target-only matched control. Graph nodes are opaque capacity slots "
        "and contain no source transition evidence; do not infer source semantics."
    )
    return (
        f"You are {role}, an untrusted proposal Agent. {role_instruction} "
        f"{condition_text} "
        "You may propose structure but cannot verify semantics. Select exactly one listed "
        "source_hypothesis_hash. For EACH target demo, return an ordered segmentation that "
        "uses every target transition exactly once. Every source node must occur exactly once "
        "and in listed source-node order. A SOURCE_NODE segment has that exact source_node_id. "
        "Any target transition that is contextual scaffolding or cannot share a source-node "
        "local program across all demos must be retained in a TARGET_NATIVE_GAP segment with "
        "source_node_id=null; never drop it. For each source node, its ordered operator and "
        "argument-type sequence (mechanically determined from the referenced transitions) must "
        "be exactly identical across all demos. Gaps may differ across demos. Do not provide "
        "semantics, predicates, confidence, rationale, action names, operators, or types. "
        "If no exact segmentation exists, abstain.\n"
        "REQUIRED_SOURCE_NODE_REGISTRY="
        f"{json.dumps({g['source_hypothesis_hash']: [n['node_id'] for n in g['nodes']] for g in graphs}, sort_keys=True)}\n"
        f"SOURCE_GRAPHS={json.dumps(graphs, sort_keys=True)}\n"
        f"TARGET_DEMOS={json.dumps([_demo_trace(d) for d in demos], sort_keys=True)}\n"
        + (
            "PRIOR_REPAIR_CONTEXT=" + json.dumps(repair_context, sort_keys=True) + "\n"
            "The prior proposals were not admitted. Use only the deterministic Harness "
            "failure codes above to repair the segmentation, or abstain. Do not copy a "
            "proposal whose failure remains unresolved. The prior source_hypothesis_hash "
            "is immutable during repair; do not switch source graphs.\n"
            if repair_context is not None else ""
        )
        +
        "Return exactly one JSON object with keys source_hypothesis_hash,examples,abstain. "
        "Each example has exactly demo_id,segments. Each segment has exactly segment_id,kind,"
        "source_node_id,target_transition_ids. kind is SOURCE_NODE or TARGET_NATIVE_GAP. "
        "No markdown."
    )


def _parse(raw, *, graphs, demos):
    if _JSON_ONLY.fullmatch(raw) is None:
        raise ValueError("NOT_EXACT_JSON_OBJECT")
    payload = json.loads(raw)
    if set(payload) != {"source_hypothesis_hash", "examples", "abstain"}:
        raise ValueError("WRONG_TOP_LEVEL_KEYS")
    if not isinstance(payload["abstain"], bool):
        raise ValueError("ABSTAIN_NOT_BOOLEAN")
    if payload["abstain"]:
        if payload["source_hypothesis_hash"] is not None or payload["examples"]:
            raise ValueError("MALFORMED_ABSTENTION")
        return None, True
    graph_by_hash = {graph["source_hypothesis_hash"]: graph for graph in graphs}
    source_hash = str(payload["source_hypothesis_hash"])
    if source_hash not in graph_by_hash:
        raise ValueError("UNKNOWN_SOURCE_HYPOTHESIS")
    demos_by_id = {demo.demo_id: demo for demo in demos}
    if not isinstance(payload["examples"], list):
        raise ValueError("EXAMPLES_NOT_LIST")
    examples = []
    for row in payload["examples"]:
        if set(row) != {"demo_id", "segments"} or row["demo_id"] not in demos_by_id:
            raise ValueError("INVALID_DEMO_ROW")
        demo = demos_by_id[row["demo_id"]]
        aliases = {f"target_t{i}": i for i in range(len(demo.actions))}
        segments = []
        for segment in row["segments"]:
            if set(segment) != {
                "segment_id", "kind", "source_node_id", "target_transition_ids",
            }:
                raise ValueError("INVALID_SEGMENT_KEYS")
            try:
                indices = tuple(aliases[str(item)] for item in segment["target_transition_ids"])
            except KeyError as exc:
                raise ValueError("UNKNOWN_TARGET_TRANSITION") from exc
            segments.append(ProposedSegment(
                segment_id=str(segment["segment_id"]),
                kind=SegmentKind(str(segment["kind"])),
                source_node_id=(
                    str(segment["source_node_id"])
                    if segment["source_node_id"] is not None else None
                ),
                target_transition_indices=indices,
            ))
        examples.append(ExampleSegmentation(str(row["demo_id"]), tuple(segments)))
    return (source_hash, tuple(examples)), False


def _api_key(endpoint, env_name):
    key = os.environ.get(env_name, "").strip()
    if not key and "openrouter.ai" in endpoint.lower():
        try:
            from API_func import open_router_api_key
            key = str(open_router_api_key or "").strip()
        except Exception:
            key = ""
    if not key and "openrouter.ai" in endpoint.lower():
        key_file = REPO_ROOT.parent / "keys.py"
        if key_file.is_file():
            key = str(runpy.run_path(str(key_file)).get("OPENROUTER_API_KEY") or "").strip()
    return key


def _edge_evidence_gate(graphs):
    eligible, excluded = [], []
    for graph in graphs:
        node_ids = [str(node["node_id"]) for node in graph["nodes"]]
        edges = {
            (str(edge["source_node_id"]), str(edge["target_node_id"])): edge
            for edge in graph["edges"]
        }
        failures = []
        for left, right in zip(node_ids, node_ids[1:]):
            edge = edges.get((left, right))
            receipts = list((edge or {}).get("intervention_receipt_sha256s") or ())
            if edge is None:
                failures.append(f"MISSING_REGISTERED_EDGE:{left}->{right}")
            elif not receipts or any(
                len(str(item)) != 64
                or any(char not in "0123456789abcdef" for char in str(item))
                for item in receipts
            ):
                failures.append(f"MISSING_VALID_INTERVENTION_RECEIPTS:{left}->{right}")
        if failures:
            excluded.append({
                "source_hypothesis_hash": graph["source_hypothesis_hash"],
                "failure_codes": failures,
            })
        else:
            eligible.append(graph)
    if not eligible:
        raise ValueError("NO_SOURCE_GRAPH_PASSES_EDGE_EVIDENCE_GATE")
    return eligible, excluded


def _repair_proposal_view(candidate):
    return {
        "proposal_id": candidate["proposal_id"],
        "source_hypothesis_hash": candidate["source_hypothesis_hash"],
        "examples": [{
            "demo_id": example["demo_id"],
            "segments": [{
                "segment_id": segment["segment_id"], "kind": segment["kind"],
                "source_node_id": segment["source_node_id"],
                "target_transition_ids": [
                    f"target_t{index}"
                    for index in segment["target_transition_indices"]
                ],
            } for segment in example["segments"]],
        } for example in candidate["examples"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-hypotheses", type=Path, required=True)
    parser.add_argument("--source-treatment", choices=("correct", "renamed"), default="correct")
    parser.add_argument("--source-control-seed", type=int, default=1729)
    parser.add_argument("--fixed-source-hypothesis-hash")
    parser.add_argument(
        "--role", action="append", choices=_ROLES,
        help="Registered proposal role; repeat to request multiple roles (default: all).",
    )
    parser.add_argument("--demo", type=Path, action="append", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-proposals", type=Path)
    parser.add_argument("--repair-admission", type=Path)
    args = parser.parse_args()
    if len(args.demo) < 2:
        parser.error("conditional proposal requires at least two --demo values")
    if bool(args.repair_proposals) != bool(args.repair_admission):
        parser.error("repair requires both --repair-proposals and --repair-admission")
    payload = json.loads(args.source_hypotheses.read_text(encoding="utf-8"))
    raw_graphs = _source_graphs(payload)
    graphs, control_receipt = apply_binding_source_control(
        raw_graphs, treatment=args.source_treatment, seed=args.source_control_seed,
    )
    graphs, excluded_graphs = _edge_evidence_gate(graphs)
    if args.fixed_source_hypothesis_hash:
        graphs = [
            graph for graph in graphs
            if graph["source_hypothesis_hash"] == args.fixed_source_hypothesis_hash
        ]
        if len(graphs) != 1:
            raise SystemExit(
                "fixed source hypothesis is absent or failed the edge-evidence gate"
            )
    demos = tuple(target_demo_receipt_from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    ) for path in args.demo)
    for demo in demos:
        demo.validate_for_admission()
    repair_context = None
    if args.repair_proposals is not None:
        prior = json.loads(args.repair_proposals.read_text(encoding="utf-8"))
        admission = json.loads(args.repair_admission.read_text(encoding="utf-8"))
        rejected_by_id = {
            row["proposal_id"]: row
            for row in admission.get("rejected_candidates") or ()
        }
        repair_context = [{
            "proposal": _repair_proposal_view(candidate),
            "harness_rejection_receipt": rejected_by_id.get(
                candidate["proposal_id"], {
                    "failure_codes": [
                        "PRIOR_PROPOSAL_NOT_PRESENT_IN_ADMISSION_RECEIPT"
                    ],
                },
            ),
        } for candidate in prior.get("candidates") or ()]
    key = _api_key(args.endpoint, args.api_key_env)
    if "openrouter.ai" in args.endpoint.lower() and not key:
        raise SystemExit("OpenRouter API key unavailable")
    client = StrictOpenAIClient(args.endpoint, timeout_s=180, api_key=key or "EMPTY")
    rows, candidates = [], []
    try:
        registered_roles = tuple(args.role or _ROLES)
        for role_index, role in enumerate(registered_roles):
            role_repair = None
            role_graphs = graphs
            if repair_context is not None:
                role_repair = [repair_context[role_index % len(repair_context)]]
                fixed_hash = role_repair[0]["proposal"]["source_hypothesis_hash"]
                role_graphs = [
                    graph for graph in graphs
                    if graph["source_hypothesis_hash"] == fixed_hash
                ]
                if not role_graphs:
                    raise ValueError("REPAIR_SOURCE_GRAPH_FAILED_EDGE_EVIDENCE_GATE")
            prompt = _prompt(role, role_graphs, demos, repair_context=role_repair)
            reply, usage, error, abstained, parsed = "", {}, None, False, None
            try:
                reply, usage = client.complete(model=args.model, prompt=prompt, max_tokens=1400)
                parsed, abstained = _parse(reply, graphs=role_graphs, demos=demos)
            except Exception as exc:
                error = f"{type(exc).__name__}:{exc}"
            receipt_payload = {
                "role": role, "model": args.model, "prompt_sha256": _hash(prompt),
                "raw_reply": reply, "usage": dict(usage),
                "source_treatment": args.source_treatment,
            }
            receipt_hash = _hash(receipt_payload)
            rows.append({
                "receipt_sha256": receipt_hash, "receipt_payload": receipt_payload,
                "error": error, "abstained": abstained,
            })
            if parsed is not None:
                source_hash, examples = parsed
                candidate = ConditionalProgramProposal(
                    proposal_id="conditional-" + _hash({
                        "role": role, "receipt": receipt_hash, "examples": examples,
                    })[:20],
                    proposal_source=f"{args.model}:{role}",
                    proposal_receipt_sha256=receipt_hash,
                    source_hypothesis_hash=source_hash, examples=examples,
                )
                candidates.append(_jsonable(candidate))
    finally:
        client.close()
    output = {
        "schema_version": 1, "candidate_source": "independent_untrusted_agents",
        "source_treatment": args.source_treatment,
        "source_control_receipt": control_receipt.to_dict(),
        "source_graph_edge_evidence_gate": {
            "rule": "all_consecutive_node_edges_registered_with_valid_intervention_receipts",
            "n_eligible": len(graphs), "excluded": excluded_graphs,
        },
        "source_graphs": graphs,
        "demo_ids": [demo.demo_id for demo in demos],
        "demo_hashes": [demo.content_hash() for demo in demos],
        "model": args.model, "rows": rows, "candidates": candidates,
        "registered_roles": list(registered_roles),
        "fixed_source_hypothesis_hash": args.fixed_source_hypothesis_hash,
        "n_candidates": len(candidates),
        "n_abstain": sum(row["abstained"] for row in rows),
        "n_invalid": sum(row["error"] is not None for row in rows),
        "semantic_alignment_claimed": False,
        "repair_context_sha256": (
            _hash(repair_context) if repair_context is not None else None
        ),
    }
    output["artifact_sha256"] = _hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: output[key] for key in (
        "n_candidates", "n_abstain", "n_invalid", "artifact_sha256",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
