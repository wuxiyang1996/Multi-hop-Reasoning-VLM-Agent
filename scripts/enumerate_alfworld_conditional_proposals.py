#!/usr/bin/env python3
"""Enumerate one conditional proposal call per evidence-qualified graph/control slot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.binding_source_controls import apply_binding_source_control  # noqa: E402
from harness.conditional_node_program import ConditionalProgramProposal  # noqa: E402
from harness.frozen_transfer_policy import StrictOpenAIClient  # noqa: E402
from harness.skill_admission import target_demo_receipt_from_dict  # noqa: E402
from scripts.propose_alfworld_conditional_nodes_35b import (  # noqa: E402
    _api_key, _edge_evidence_gate, _hash, _jsonable, _parse, _prompt,
)
from scripts.propose_alfworld_multistep_bindings_35b import _source_graphs  # noqa: E402


def _target_only_skeletons(graphs):
    skeletons = []
    for graph_index, graph in enumerate(graphs):
        node_count = len(graph["nodes"])
        graph_hash = _hash({
            "condition": "target_only", "slot": graph_index,
            "matched_source_graph_hash": graph["source_hypothesis_hash"],
            "node_count": node_count,
        })
        node_ids = [f"opaque_node_{index}" for index in range(node_count)]
        skeleton = {
            "source_hypothesis_hash": graph_hash,
            "source_program_hash": None,
            "nodes": [{
                "node_id": node_id, "observed_transitions": [],
            } for node_id in node_ids],
            "edges": [{
                "source_node_id": left, "target_node_id": right,
                "kind": "OPAQUE_ORDER", "agent_claim": {},
                "intervention_receipt_sha256s": [],
                "status": "TARGET_ONLY_CONTROL",
            } for left, right in zip(node_ids, node_ids[1:])],
        }
        source_chars = len(json.dumps(graph, sort_keys=True))
        skeleton_chars = len(json.dumps(skeleton, sort_keys=True))
        padding_chars = 3 * max(0, source_chars - skeleton_chars)
        skeleton["opaque_context_padding"] = (
            "pad " * ((padding_chars + 3) // 4)
        )[:padding_chars]
        skeletons.append(skeleton)
    return skeletons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=(
        "correct", "renamed", "randomized", "receipt_null", "target_only",
    ), required=True)
    parser.add_argument("--source-hypotheses", type=Path, required=True)
    parser.add_argument("--source-control-seed", type=int, default=1729)
    parser.add_argument("--demo", type=Path, action="append", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="qwen/qwen3-max")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--role", choices=(
        "proposer_a", "proposer_b", "skeptic",
    ), default="proposer_a")
    parser.add_argument("--max-graphs", type=int)
    parser.add_argument(
        "--graph-index", action="append", type=int,
        help="Original eligible-graph slot to run; repeat for endpoint-only retries.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.demo) < 2:
        parser.error("enumeration requires at least two adaptation examples")
    source_payload = json.loads(args.source_hypotheses.read_text(encoding="utf-8"))
    raw_graphs, initially_excluded = _edge_evidence_gate(
        _source_graphs(source_payload, require_agent_reasoning_receipts=True)
    )
    if args.condition == "target_only":
        graphs = _target_only_skeletons(raw_graphs)
        source_treatment = "empty"
        control_receipt = None
    else:
        graphs, control_receipt = apply_binding_source_control(
            raw_graphs, treatment=args.condition, seed=args.source_control_seed,
        )
        source_treatment = args.condition
    indexed_graphs = list(enumerate(graphs))
    if args.graph_index:
        requested = list(args.graph_index)
        if len(requested) != len(set(requested)) or any(
            index < 0 or index >= len(indexed_graphs) for index in requested
        ):
            parser.error("--graph-index values must be unique registered slots")
        indexed_graphs = [indexed_graphs[index] for index in requested]
    if args.max_graphs is not None:
        if args.max_graphs < 1:
            parser.error("--max-graphs must be positive")
        indexed_graphs = indexed_graphs[:args.max_graphs]
    selected_graphs = [graph for _, graph in indexed_graphs]
    demos = tuple(target_demo_receipt_from_dict(json.loads(
        path.read_text(encoding="utf-8")
    )) for path in args.demo)
    for demo in demos:
        demo.validate_for_admission()
    key = _api_key(args.endpoint, args.api_key_env)
    if "openrouter.ai" in args.endpoint.lower() and not key:
        raise SystemExit("OpenRouter API key unavailable")
    client = StrictOpenAIClient(args.endpoint, timeout_s=180, api_key=key or "EMPTY")
    rows, candidates = [], []
    try:
        for graph_index, graph in indexed_graphs:
            prompt_condition = "target_only" if args.condition == "target_only" else "source"
            prompt = _prompt(
                args.role, [graph], demos, condition=prompt_condition,
            )
            reply, usage, error, abstained, parsed = "", {}, None, False, None
            try:
                proposal_seed = args.source_control_seed + graph_index
                reply, usage = client.complete(
                    model=args.model, prompt=prompt, max_tokens=1400,
                    seed=proposal_seed,
                )
                parsed, abstained = _parse(reply, graphs=[graph], demos=demos)
            except Exception as exc:
                error = f"{type(exc).__name__}:{exc}"
            receipt_payload = {
                "condition": args.condition, "graph_index": graph_index,
                "source_hypothesis_hash": graph["source_hypothesis_hash"],
                "role": args.role, "model": args.model,
                "proposal_seed": args.source_control_seed + graph_index,
                "prompt_sha256": _hash(prompt), "raw_reply": reply,
                "usage": dict(usage),
            }
            receipt_hash = _hash(receipt_payload)
            rows.append({
                "receipt_sha256": receipt_hash,
                "receipt_payload": receipt_payload,
                "error": error, "abstained": abstained,
            })
            if parsed is not None:
                source_hash, examples = parsed
                candidate = ConditionalProgramProposal(
                    proposal_id="enumerated-" + _hash({
                        "condition": args.condition, "graph_index": graph_index,
                        "receipt": receipt_hash, "examples": examples,
                    })[:20],
                    proposal_source=f"{args.model}:{args.role}:graph{graph_index}",
                    proposal_receipt_sha256=receipt_hash,
                    source_hypothesis_hash=source_hash, examples=examples,
                )
                candidates.append(_jsonable(candidate))
    finally:
        client.close()
    output = {
        "schema_version": 1,
        "candidate_source": "complete_graph_enumeration_untrusted_agent",
        "condition": args.condition, "source_treatment": source_treatment,
        "source_control_receipt": (
            control_receipt.to_dict() if control_receipt is not None else None
        ),
        "source_graph_edge_evidence_gate": {
            "rule": "all_consecutive_edges_have_intervention_receipts_before_controls",
            "n_eligible": len(raw_graphs), "excluded": initially_excluded,
        },
        "source_graphs": selected_graphs,
        "demo_ids": [demo.demo_id for demo in demos],
        "demo_hashes": [demo.content_hash() for demo in demos],
        "model": args.model, "role": args.role,
        "enumeration_complete": (
            args.max_graphs is None and not args.graph_index
        ),
        "total_eligible_graph_count": len(graphs),
        "selected_graph_indices": [index for index, _ in indexed_graphs],
        "registered_graph_count": len(selected_graphs), "rows": rows,
        "candidates": candidates, "n_candidates": len(candidates),
        "n_abstain": sum(bool(row["abstained"]) for row in rows),
        "n_invalid": sum(row["error"] is not None for row in rows),
        "semantic_alignment_claimed": False,
    }
    output["artifact_sha256"] = _hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    total_cost = sum(float(
        row["receipt_payload"]["usage"].get("cost") or 0.0
    ) for row in rows)
    total_prompt_tokens = sum(int(
        row["receipt_payload"]["usage"].get("prompt_tokens") or 0
    ) for row in rows)
    print(json.dumps({
        "condition": args.condition, "registered_graph_count": len(selected_graphs),
        "n_candidates": len(candidates), "n_invalid": output["n_invalid"],
        "n_abstain": output["n_abstain"], "reported_cost": total_cost,
        "total_prompt_tokens": total_prompt_tokens,
        "artifact_sha256": output["artifact_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
