#!/usr/bin/env python3
"""Generate matched source-informed or target-only v3 binding candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.frozen_transfer_policy import StrictOpenAIClient  # noqa: E402
from harness.binding_source_controls import apply_binding_source_control  # noqa: E402
from harness.skill_admission import target_demo_receipt_from_dict  # noqa: E402


_ROLES = ("proposer_a", "proposer_b", "skeptic")
_JSON_ONLY = re.compile(r"\A\s*\{.*\}\s*\Z", re.DOTALL)


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _target_trace(demo) -> list[dict[str, Any]]:
    return [{
        "target_transition_id": f"target_t{item.transition_index}",
        "operator": item.operator,
        "argument_types": dict(item.argument_types),
        "arguments": dict(item.arguments),
        "action": item.action,
        "state_sha256": item.state_sha256,
        "next_state_sha256": item.next_state_sha256,
        "official_success_after": item.official_success_after,
    } for item in demo.actions]


def _verify_source_artifact(source_payload: dict[str, Any]) -> None:
    """Reject corrupted or legacy-shaped source evidence instead of using an empty set."""
    claimed = str(source_payload.get("artifact_sha256") or "")
    unsigned = dict(source_payload)
    unsigned.pop("artifact_sha256", None)
    if not claimed or _hash(unsigned) != claimed:
        raise ValueError("SOURCE_ARTIFACT_HASH_MISMATCH")
    if source_payload.get("candidate_source") != "independent_untrusted_agents":
        raise ValueError("SOURCE_ARTIFACT_NOT_AGENT_GENERATED")
    if source_payload.get("semantic_scoring") is not False:
        raise ValueError("SOURCE_ARTIFACT_SEMANTIC_SCORING_NOT_DISABLED")
    if source_payload.get("ranking") is not False or source_payload.get("voting") is not False:
        raise ValueError("SOURCE_ARTIFACT_RANKING_OR_VOTING_ENABLED")
    if not source_payload.get("full_observed_path_partition_required"):
        raise ValueError("SOURCE_ARTIFACT_FULL_PARTITION_NOT_REQUIRED")


def _source_graphs(
    source_payload: dict[str, Any], *, require_agent_reasoning_receipts: bool = False,
) -> list[dict[str, Any]]:
    """Expose exact source receipts and untrusted control claims to binding Agents."""
    _verify_source_artifact(source_payload)
    graphs = []
    for program_row in source_payload.get("programs") or ():
        program = program_row.get("program") or {}
        reasoning_trace = program_row.get("source_reasoning_trace") or {}
        if reasoning_trace:
            unsigned_trace = dict(reasoning_trace)
            claimed_trace_hash = str(unsigned_trace.pop("trace_sha256", ""))
            if not claimed_trace_hash or _hash(unsigned_trace) != claimed_trace_hash:
                raise ValueError("SOURCE_REASONING_TRACE_HASH_MISMATCH")
            reasoning_by_transition = {}
            for step in reasoning_trace.get("steps") or ():
                transition_id = str(step.get("transition_id") or "")
                claim = str(step.get("agent_reasoning_claim") or "")
                claim_hash = str(step.get("agent_response_sha256") or "")
                if (
                    not transition_id or transition_id in reasoning_by_transition
                    or not claim or _hash(claim) != claim_hash
                    or step.get("claim_status") != "UNTRUSTED_AGENT_CLAIM"
                ):
                    raise ValueError("INVALID_SOURCE_REASONING_STEP_RECEIPT")
                reasoning_by_transition[transition_id] = {
                    "agent_reasoning_claim": claim,
                    "agent_response_sha256": claim_hash,
                    "claim_status": "UNTRUSTED_AGENT_CLAIM",
                    **({
                        "action_proposal_receipt": dict(step["action_proposal_receipt"]),
                        "action_proposal_event_sha256": str(
                            step["action_proposal_event_sha256"]
                        ),
                        "post_transition_verdict_receipt": dict(
                            step["post_transition_verdict_receipt"]
                        ),
                        "post_transition_verdict_event_sha256": str(
                            step["post_transition_verdict_event_sha256"]
                        ),
                    } if "action_proposal_receipt" in step else {}),
                }
        else:
            reasoning_by_transition = {}
        if require_agent_reasoning_receipts and not reasoning_by_transition:
            raise ValueError("SOURCE_REASONING_RECEIPTS_REQUIRED")
        transitions = {
            str(row["transition_id"]): {
                "transition_id": str(row["transition_id"]),
                "action": str(row["action"]),
                "reward": float(row["reward"]),
                "done": bool(row["done"]),
                "state_sha256": str(row["state_sha256"]),
                "next_state_sha256": str(row["next_state_sha256"]),
                **reasoning_by_transition.get(str(row["transition_id"]), {}),
            }
            for row in program.get("transitions") or ()
        }
        if require_agent_reasoning_receipts and set(reasoning_by_transition) != set(transitions):
            raise ValueError("SOURCE_REASONING_RECEIPT_COVERAGE_MISMATCH")
        qualified = program_row.get("qualified_hypotheses") or ()
        if int(program_row.get("n_qualified", -1)) != len(qualified):
            raise ValueError("SOURCE_ARTIFACT_QUALIFIED_COUNT_MISMATCH")
        for row in qualified:
            checks = row.get("checks") or {}
            if not checks or not all(value is True for value in checks.values()):
                raise ValueError("SOURCE_ARTIFACT_CONTAINS_UNQUALIFIED_HYPOTHESIS")
            hypothesis = row.get("hypothesis") or {}
            nodes = []
            flattened_ids = []
            for node in hypothesis.get("nodes") or ():
                ids = [str(item) for item in node.get("transition_ids") or ()]
                if not ids or any(item not in transitions for item in ids):
                    raise ValueError("SOURCE_ARTIFACT_UNKNOWN_TRANSITION")
                flattened_ids.extend(ids)
                nodes.append({
                    "node_id": str(node["node_id"]),
                    "observed_transitions": [transitions[item] for item in ids],
                })
            expected_ids = [str(row["transition_id"]) for row in program.get("transitions") or ()]
            if flattened_ids != expected_ids:
                raise ValueError("SOURCE_ARTIFACT_NOT_FULL_ORDERED_PARTITION")
            graphs.append({
                "source_hypothesis_hash": str(row["hypothesis_hash"]),
                "source_program_hash": str(program.get("program_hash") or ""),
                "source_reasoning_trace_sha256": (
                    str(reasoning_trace.get("trace_sha256")) if reasoning_trace else None
                ),
                "nodes": nodes,
                "edges": [{
                    "source_node_id": str(edge["source_node_id"]),
                    "target_node_id": str(edge["target_node_id"]),
                    "kind": str(edge["kind"]),
                    "agent_claim": dict(edge.get("agent_claim") or {}),
                    "intervention_receipt_sha256s": list(
                        edge.get("intervention_receipt_sha256s") or ()
                    ),
                    "status": "AGENT_HYPOTHESIS",
                } for edge in hypothesis.get("edges") or ()],
            })
    if not graphs:
        raise ValueError("SOURCE_ARTIFACT_HAS_NO_QUALIFIED_HYPOTHESES")
    return graphs


def _prompt(*, condition: str, role: str, demo, graphs) -> str:
    role_text = {
        "proposer_a": "Propose one executable linear binding candidate.",
        "proposer_b": "Independently seek a different evidence-covered binding.",
        "skeptic": "Expose ambiguity with a competing binding, or abstain.",
    }[role]
    if condition == "source":
        evidence = (
            "SOURCE_GRAPH_CANDIDATES=" + json.dumps(graphs, sort_keys=True) + "\n"
            "Select one listed source_hypothesis_hash. Return every node of that graph "
            "exactly once and preserve its listed node order. Source edge prose is an "
            "untrusted Agent hypothesis; transition action/reward/done fields are observed "
            "receipts. Decide whether one graph can structure the target trace, or abstain."
        )
    else:
        evidence = (
            "SOURCE_GRAPH_CANDIDATES=[]\n"
            "This is the matched target-only condition. source_hypothesis_hash must be null. "
            "Propose your own opaque node IDs from the target demo only."
        )
    return (
        f"You are {role}, an untrusted binding proposer. {role_text} "
        "You do not verify, rank, vote, or claim semantic equivalence. A single target demo "
        "supports observed linear order only, never branches/loops/guards. Choose at least two "
        "target transitions. Across all nodes, use every listed target_transition_id exactly "
        "once and preserve the fixed target-demo order. The only legal "
        f"target_transition_ids are {[f'target_t{i}' for i in range(len(demo.actions))]}. "
        "Source evidence has no target_tN IDs. Every node needs a non-empty "
        "target_transition_ids span, or abstain. Do not output operators, "
        "argument types, confidence, or rationale; the Harness derives native signatures from "
        "the immutable demo.\n"
        f"CONDITION={condition}\n{evidence}\n"
        f"FIXED_TARGET_DEMO={json.dumps(_target_trace(demo), sort_keys=True)}\n"
        "Return exactly one JSON object with keys source_hypothesis_hash,nodes,abstain. "
        "Each nodes item has exactly node_id,target_transition_ids. No markdown."
    )


def _parse(raw: str, *, condition: str, graphs, demo):
    if _JSON_ONLY.fullmatch(raw) is None:
        raise ValueError("NOT_EXACT_JSON_OBJECT")
    payload = json.loads(raw)
    if set(payload) != {"source_hypothesis_hash", "nodes", "abstain"}:
        raise ValueError("WRONG_TOP_LEVEL_KEYS")
    if not isinstance(payload["abstain"], bool):
        raise ValueError("ABSTAIN_NOT_BOOLEAN")
    if payload["abstain"]:
        if payload["nodes"] or payload["source_hypothesis_hash"] is not None:
            raise ValueError("MALFORMED_ABSTENTION")
        return None, True
    if not isinstance(payload["nodes"], list):
        raise ValueError("NODES_NOT_LIST")
    parsed_nodes = []
    target_aliases = {
        f"target_t{item.transition_index}": item.transition_index for item in demo.actions
    }
    for row in payload["nodes"]:
        if set(row) != {"node_id", "target_transition_ids"}:
            raise ValueError("WRONG_NODE_KEYS")
        try:
            indices = [target_aliases[str(item)] for item in row["target_transition_ids"]]
        except KeyError as exc:
            raise ValueError("UNKNOWN_TARGET_TRANSITION_ID") from exc
        if not indices:
            raise ValueError("TARGET_TRANSITION_OUT_OF_RANGE_OR_EMPTY")
        parsed_nodes.append((str(row["node_id"]), indices))
    source_hash = payload["source_hypothesis_hash"]
    if condition == "source":
        by_hash = {item["source_hypothesis_hash"]: item for item in graphs}
        if source_hash not in by_hash:
            raise ValueError("HALLUCINATED_SOURCE_HYPOTHESIS")
        expected = [item["node_id"] for item in by_hash[source_hash]["nodes"]]
        if [item[0] for item in parsed_nodes] != expected:
            raise ValueError("SOURCE_NODE_SEQUENCE_MISMATCH")
    elif source_hash is not None:
        raise ValueError("TARGET_ONLY_LEAKED_SOURCE_IDENTITY")
    return {"source_hypothesis_hash": source_hash, "nodes": parsed_nodes}, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("source", "target_only"), required=True)
    parser.add_argument("--source-hypotheses", type=Path)
    parser.add_argument(
        "--source-treatment", choices=("empty", "correct", "wrong", "renamed"),
        help="Source exposure applied before any binding Agent call.",
    )
    parser.add_argument(
        "--control-source-hypotheses", type=Path,
        help="Separately frozen cross-game source artifact required by treatment=wrong.",
    )
    parser.add_argument("--source-control-seed", type=int, default=1729)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse-proposal-receipts", type=Path,
        help="Revalidate stored raw replies with zero new API calls.",
    )
    args = parser.parse_args()
    source_treatment = args.source_treatment or (
        "correct" if args.condition == "source" else "empty"
    )
    if (
        args.condition == "source" and source_treatment in {"correct", "renamed"}
        and not args.source_hypotheses
    ):
        parser.error("correct/renamed source treatment requires --source-hypotheses")
    if args.condition == "target_only" and source_treatment != "empty":
        parser.error("target_only condition requires source treatment=empty")
    if args.condition == "source" and source_treatment == "empty":
        parser.error("source condition cannot use source treatment=empty")
    if source_treatment == "wrong" and args.control_source_hypotheses is None:
        parser.error("source treatment=wrong requires --control-source-hypotheses")

    source_payload = (
        json.loads(args.source_hypotheses.read_text(encoding="utf-8"))
        if args.source_hypotheses else {}
    )
    if source_treatment == "wrong":
        control_payload = json.loads(
            args.control_source_hypotheses.read_text(encoding="utf-8")
        )
        raw_graphs = _source_graphs(control_payload)
    else:
        raw_graphs = _source_graphs(source_payload) if args.condition == "source" else []
    graphs, source_control_receipt = apply_binding_source_control(
        raw_graphs, treatment=source_treatment, seed=args.source_control_seed,
    )
    demo = target_demo_receipt_from_dict(json.loads(args.demo.read_text(encoding="utf-8")))
    demo.validate_for_admission()
    key = os.environ.get(args.api_key_env, "").strip()
    if (
        not key and args.reuse_proposal_receipts is None
        and "openrouter.ai" in args.endpoint.lower()
    ):
        try:
            from API_func import open_router_api_key
            key = str(open_router_api_key or "").strip()
        except Exception:
            key = ""
    if (
        not key and args.reuse_proposal_receipts is None
        and "openrouter.ai" in args.endpoint.lower()
    ):
        key_file = REPO_ROOT.parent / "keys.py"
        if key_file.is_file():
            key = str(
                runpy.run_path(str(key_file)).get("OPENROUTER_API_KEY") or ""
            ).strip()
    if (
        args.reuse_proposal_receipts is None
        and "openrouter.ai" in args.endpoint.lower() and not key
    ):
        raise SystemExit("OpenRouter API key unavailable")

    client = (
        StrictOpenAIClient(args.endpoint, timeout_s=180, api_key=key or "EMPTY")
        if args.reuse_proposal_receipts is None else None
    )
    reused = {}
    if args.reuse_proposal_receipts is not None:
        prior = json.loads(args.reuse_proposal_receipts.read_text(encoding="utf-8"))
        if prior.get("condition") != args.condition:
            raise SystemExit("reused binding proposal condition mismatch")
        if prior.get("source_treatment") != source_treatment:
            raise SystemExit("reused binding proposal source treatment mismatch")
        reused = {
            str(row["receipt_payload"]["role"]): row["receipt_payload"]
            for row in prior.get("rows") or ()
        }
    rows = []
    candidates = []
    try:
        for role in _ROLES:
            prompt = _prompt(condition=args.condition, role=role, demo=demo, graphs=graphs)
            reply, usage, error, endpoint_failure, abstained = "", {}, None, False, False
            parsed = None
            try:
                if reused:
                    stored = reused[role]
                    if stored["model"] != args.model or stored["prompt_sha256"] != _hash(prompt):
                        raise ValueError("REUSED_PROPOSAL_IDENTITY_MISMATCH")
                    reply, usage = str(stored["raw_reply"]), dict(stored.get("usage") or {})
                else:
                    reply, usage = client.complete(model=args.model, prompt=prompt, max_tokens=600)
                parsed, abstained = _parse(
                    reply, condition=args.condition, graphs=graphs, demo=demo,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}:{exc}"
                endpoint_failure = type(exc).__module__.startswith("httpx")
            receipt_payload = {
                "condition": args.condition, "source_treatment": source_treatment,
                "role": role, "model": args.model,
                "prompt_sha256": _hash(prompt), "raw_reply": reply, "usage": dict(usage),
            }
            receipt_hash = _hash(receipt_payload)
            rows.append({
                "receipt_sha256": receipt_hash,
                "receipt_payload": receipt_payload,
                "error": error,
                "endpoint_failure": endpoint_failure,
                "abstained": abstained,
            })
            if parsed is not None:
                nodes = []
                selected_graph = next((
                    graph for graph in graphs
                    if graph["source_hypothesis_hash"] == parsed["source_hypothesis_hash"]
                ), None)
                source_nodes = {
                    node["node_id"]: node for node in (selected_graph or {}).get("nodes", [])
                }
                for node_id, indices in parsed["nodes"]:
                    target_steps = []
                    for index in indices:
                        action = demo.actions[index]
                        target_steps.append({
                            "target_transition_index": index,
                            "target_operator": action.operator,
                            "argument_types": dict(action.argument_types),
                        })
                    nodes.append({
                        "node_id": node_id,
                        "target_steps": target_steps,
                        "source_conditioning": ({
                            "observed_transitions": list(
                                source_nodes[node_id]["observed_transitions"]
                            ),
                            "incident_edges": [
                                edge for edge in selected_graph["edges"]
                                if node_id in (
                                    edge["source_node_id"], edge["target_node_id"]
                                )
                            ],
                        } if selected_graph is not None else {}),
                    })
                identity = {
                    "condition": args.condition, "role": role,
                    "source_hypothesis_hash": parsed["source_hypothesis_hash"],
                    "nodes": nodes, "proposal_receipt_sha256": receipt_hash,
                }
                candidates.append({
                    "candidate_id": "v3-" + _hash(identity)[:20],
                    "origin": (
                        "SOURCE_HYPOTHESIS" if args.condition == "source"
                        else "TARGET_NATIVE_SAME_DEMO"
                    ),
                    "proposal_source": f"{args.model}:{role}",
                    "proposal_receipt_sha256": receipt_hash,
                    "source_hypothesis_hash": parsed["source_hypothesis_hash"],
                    "nodes": nodes,
                })
    finally:
        if client is not None:
            client.close()

    output = {
        "schema_version": 2,
        "candidate_source": "independent_untrusted_agents",
        "condition": args.condition,
        "source_treatment": source_treatment,
        "source_control_receipt": source_control_receipt.to_dict(),
        "source_control_applied_before_binding_generation": True,
        "model": args.model,
        "reused_proposal_receipts": args.reuse_proposal_receipts is not None,
        "api_calls_made_this_run": 0 if args.reuse_proposal_receipts is not None else len(rows),
        "matched_design": {
            "roles": list(_ROLES), "model": args.model, "max_completion_tokens": 600,
            "fixed_demo_hash": demo.content_hash(), "n_calls": len(_ROLES),
            "source_exposure_stage": "before_binding_generation",
        },
        "qualified_source_hypotheses": (
            [{
                "hypothesis_hash": graph["source_hypothesis_hash"],
                "hypothesis": {
                    "nodes": [{"node_id": node["node_id"]} for node in graph["nodes"]],
                },
            } for graph in graphs]
            if args.condition == "source" else []
        ),
        "source_graphs": graphs if args.condition == "source" else [],
        "proposal_receipts": [{
            "receipt_sha256": item["receipt_sha256"],
            "receipt_payload": item["receipt_payload"],
        } for item in rows],
        "rows": rows,
        "candidates": candidates,
        "n_candidates": len(candidates),
        "n_abstain": sum(bool(item["abstained"]) for item in rows),
        "n_invalid": sum(item["error"] is not None and not item["endpoint_failure"] for item in rows),
        "n_endpoint_failures": sum(bool(item["endpoint_failure"]) for item in rows),
        "gaps": [
            "source_control_claims_remain_agent_hypotheses",
            "single_target_demo_proves_linear_coverage_only",
        ],
    }
    output["artifact_sha256"] = _hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in (
        "condition", "n_candidates", "n_abstain", "n_invalid", "n_endpoint_failures",
    )}, indent=2))
    return 1 if output["n_endpoint_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
