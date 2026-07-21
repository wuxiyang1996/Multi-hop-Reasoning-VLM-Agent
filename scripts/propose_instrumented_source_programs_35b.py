#!/usr/bin/env python3
"""Propose full-path source control programs from instrumented evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.frozen_transfer_policy import StrictOpenAIClient  # noqa: E402
from harness.instrumented_source_evidence import load_instrumented_source_batch  # noqa: E402
from skill_agents.control_hypotheses import (  # noqa: E402
    AgentControlHypothesis,
    ControlHypothesisValidator,
    HypothesisEdge,
    HypothesisNode,
    union_qualified_hypotheses,
)
from skill_agents.evidence_query import EvidenceQuery  # noqa: E402
from skill_bank.trace_program_ir import ControlClaimKind  # noqa: E402


_JSON_ONLY = re.compile(r"\A\s*\{.*\}\s*\Z", re.DOTALL)
_ROLES = ("proposer_a", "proposer_b", "skeptic")


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _prompt(evidence, response, role: str):
    program = evidence.program
    transition_aliases = {
        f"t{index}": item.transition_id for index, item in enumerate(program.transitions)
    }
    visible_transitions = []
    for index, row in enumerate(response.transitions):
        native = row["native_evidence"]
        visible_transitions.append({
            "transition_id": f"t{index}",
            "step_index": row["step_index"],
            "state": native["state"],
            "available_actions": native["available_actions"],
            "raw_agent_response": native["raw_agent_response"],
            "executed_action": row["action"],
            "next_state": native["next_state"],
            "reward": row["reward"],
            "done": row["done"],
        })
    receipt_aliases = {
        f"r{index}": row["receipt_sha256"]
        for index, row in enumerate(evidence.intervention_receipts)
    }
    visible_receipts = [{
        "receipt_id": f"r{index}",
        "prefix_actions": row["prefix_actions"],
        "expected_fork_state_sha256": row["expected_fork_state_sha256"],
        "alternative_action": row["alternative_action"],
        "alternative_next_state_sha256": row["alternative_next_state_sha256"],
        "status": row["status"],
    } for index, row in enumerate(evidence.intervention_receipts)]
    role_instruction = {
        "proposer_a": "Produce one compact full-path multi-node control decomposition.",
        "proposer_b": "Independently produce a structurally different full-path decomposition.",
        "skeptic": (
            "Expose a competing decomposition if evidence permits; otherwise abstain. "
            "Do not manufacture uncertainty claims."
        ),
    }[role]
    prompt = (
        f"You are {role}, an untrusted source-program proposer. {role_instruction} "
        "Use every transition exactly once, in listed order, partitioned into at least "
        "two non-empty contiguous nodes. Choose node boundaries and edges yourself. "
        "Do not translate to another domain or use skill-name analogies. Observed paths "
        "do not prove causality. Intervention receipts prove only the exact replayed "
        "fork they contain; citing one never automatically proves your prose claim. "
        "Use only listed tN/rN IDs. If you cannot satisfy the full partition, abstain.\n"
        f"OPAQUE_PROGRAM_ID={program.program_id}\n"
        f"PROGRAM_HASH={program.content_hash()}\n"
        f"EVIDENCE_RESPONSE_HASH={response.response_sha256}\n"
        f"ALLOWED_EDGE_KINDS={json.dumps([item.value for item in ControlClaimKind])}\n"
        f"OBSERVED_TRANSITIONS={json.dumps(visible_transitions, ensure_ascii=False)}\n"
        f"OBSERVED_INTERVENTIONS={json.dumps(visible_receipts, ensure_ascii=False)}\n"
        "Return exactly one JSON object with keys hypothesis_id,nodes,edges,abstain. "
        "Each node has exactly node_id,transition_ids. Each edge has exactly edge_id,"
        "source_node_id,target_node_id,kind,agent_claim,intervention_receipt_ids. "
        "agent_claim must be a JSON object."
    )
    return prompt, transition_aliases, receipt_aliases


def _parse(raw: str, *, evidence, response, role: str, transition_aliases, receipt_aliases):
    if _JSON_ONLY.fullmatch(raw) is None:
        raise ValueError("NOT_EXACT_JSON_OBJECT")
    payload = json.loads(raw)
    if set(payload) != {"hypothesis_id", "nodes", "edges", "abstain"}:
        raise ValueError("WRONG_TOP_LEVEL_KEYS")
    nodes = []
    for row in payload["nodes"]:
        if set(row) != {"node_id", "transition_ids"}:
            raise ValueError("WRONG_NODE_KEYS")
        nodes.append(HypothesisNode(
            str(row["node_id"]),
            tuple(transition_aliases[str(item)] for item in row["transition_ids"]),
        ))
    edges = []
    expected_edge_keys = {
        "edge_id", "source_node_id", "target_node_id", "kind", "agent_claim",
        "intervention_receipt_ids",
    }
    for row in payload["edges"]:
        if set(row) != expected_edge_keys or not isinstance(row["agent_claim"], dict):
            raise ValueError("WRONG_EDGE_SCHEMA")
        edges.append(HypothesisEdge(
            str(row["edge_id"]), str(row["source_node_id"]),
            str(row["target_node_id"]), ControlClaimKind(str(row["kind"])),
            dict(row["agent_claim"]),
            tuple(receipt_aliases[str(item)] for item in row["intervention_receipt_ids"]),
        ))
    hypothesis = AgentControlHypothesis(
        hypothesis_id=str(payload["hypothesis_id"]),
        program_id=evidence.program.program_id,
        program_hash=evidence.program.content_hash(),
        proposal_source=role,
        evidence_response_hashes=[response.response_sha256],
        nodes=nodes,
        edges=edges,
        abstain=bool(payload["abstain"]),
    )
    return ControlHypothesisValidator().validate(
        hypothesis,
        program=evidence.program,
        evidence_responses=[response],
        intervention_receipts=evidence.intervention_receipts,
        require_full_partition=True,
        require_multinode=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://openrouter.ai/api")
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--max-programs", type=int, default=None,
        help="Use the first N episode IDs in lexical order; never select by reward/content.",
    )
    parser.add_argument(
        "--reuse-proposal-receipts", type=Path, default=None,
        help="Revalidate stored raw replies without making API calls.",
    )
    args = parser.parse_args()
    key = os.environ.get(args.api_key_env, "").strip()
    if not key and args.reuse_proposal_receipts is None:
        try:
            from API_func import open_router_api_key
            key = str(open_router_api_key or "").strip()
        except Exception:
            key = ""
    if not key and args.reuse_proposal_receipts is None:
        raise SystemExit("OpenRouter API key unavailable")
    evidence_rows = load_instrumented_source_batch(args.source_batch)
    if args.max_programs is not None:
        if args.max_programs < 1:
            raise SystemExit("--max-programs must be positive")
        evidence_rows = tuple(sorted(
            evidence_rows, key=lambda item: item.program.episode_id,
        )[:args.max_programs])
    prepared = []
    for evidence in evidence_rows:
        response = evidence.evidence_session.query(EvidenceQuery(
            query_id="full-instrumented-observed-episode",
            program_id=evidence.program.program_id,
            program_hash=evidence.program.content_hash(),
            transition_ids=[item.transition_id for item in evidence.program.transitions],
        ))
        for role in _ROLES:
            prompt, transition_aliases, receipt_aliases = _prompt(
                evidence, response, role,
            )
            prepared.append((evidence, response, role, prompt, transition_aliases, receipt_aliases))
    client = (
        StrictOpenAIClient(args.endpoint, timeout_s=180, api_key=key)
        if args.reuse_proposal_receipts is None else None
    )
    reused = {}
    if args.reuse_proposal_receipts is not None:
        previous = json.loads(args.reuse_proposal_receipts.read_text(encoding="utf-8"))
        reused = {
            (str(row["episode_id"]), str(row["role"])): row
            for row in previous.get("proposal_receipts", ())
        }

    def _call(item):
        evidence, response, role, prompt, transition_aliases, receipt_aliases = item
        raw = ""; usage = {}; error = None; candidate = None; endpoint_failure = False
        try:
            if reused:
                stored = reused[(evidence.program.episode_id, role)]
                if stored["prompt_sha256"] != _hash(prompt):
                    raise ValueError("REUSED_PROMPT_HASH_MISMATCH")
                raw, usage = str(stored["raw_reply"]), dict(stored.get("usage") or {})
            else:
                raw, usage = client.complete(model=args.model, prompt=prompt, max_tokens=1400)
            candidate = _parse(
                raw, evidence=evidence, response=response, role=role,
                transition_aliases=transition_aliases, receipt_aliases=receipt_aliases,
            )
            if candidate.status != "AGENT_HYPOTHESIS":
                error = ",".join(candidate.failure_codes)
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            endpoint_failure = type(exc).__module__.startswith("httpx")
        receipt_payload = {
            "episode_id": evidence.program.episode_id,
            "program_id": evidence.program.program_id,
            "program_hash": evidence.program.content_hash(),
            "role": role,
            "model": args.model,
            "prompt_sha256": _hash(prompt),
            "raw_reply": raw,
            "usage": dict(usage),
        }
        return {
            **receipt_payload,
            "proposal_receipt_sha256": _hash(receipt_payload),
            "candidate": candidate,
            "error": error,
            "endpoint_failure": endpoint_failure,
        }

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
            rows = list(pool.map(_call, prepared))
    finally:
        if client is not None:
            client.close()
    output_programs = []
    for evidence in evidence_rows:
        matching = [row for row in rows if row["episode_id"] == evidence.program.episode_id]
        union = union_qualified_hypotheses([
            row["candidate"] for row in matching
            if row["candidate"] is not None
        ])
        output_programs.append({
            "episode_id": evidence.program.episode_id,
            "program": evidence.program.to_dict(),
            "n_agent_calls": len(matching),
            "n_qualified": len(union),
            "qualified_hypotheses": [{
                "hypothesis_hash": item.hypothesis_hash,
                "checks": dict(item.checks),
                "hypothesis": {
                    "hypothesis_id": item.hypothesis.hypothesis_id,
                    "proposal_source": item.hypothesis.proposal_source,
                    "nodes": [{
                        "node_id": node.node_id,
                        "transition_ids": list(node.transition_ids),
                    } for node in item.hypothesis.nodes],
                    "edges": [{
                        "edge_id": edge.edge_id,
                        "source_node_id": edge.source_node_id,
                        "target_node_id": edge.target_node_id,
                        "kind": edge.kind.value,
                        "agent_claim": dict(edge.agent_claim),
                        "intervention_receipt_sha256s": list(
                            edge.intervention_receipt_sha256s
                        ),
                    } for edge in item.hypothesis.edges],
                },
            } for item in union],
        })
    output = {
        "schema_version": 1,
        "candidate_source": "independent_untrusted_agents",
        "source_batch": str(args.source_batch),
        "model": args.model,
        "roles": list(_ROLES),
        "full_observed_path_partition_required": True,
        "semantic_scoring": False,
        "ranking": False,
        "voting": False,
        "reused_proposal_receipts": args.reuse_proposal_receipts is not None,
        "api_calls_made_this_run": (
            0 if args.reuse_proposal_receipts is not None else len(rows)
        ),
        "program_selection": {
            "rule": "lexicographically_first_episode_ids_no_reward_or_content_selection",
            "max_programs": args.max_programs,
            "selected_episode_ids": [
                item.program.episode_id for item in evidence_rows
            ],
        },
        "programs": output_programs,
        "proposal_receipts": [{
            key: value for key, value in row.items() if key != "candidate"
        } for row in rows],
        "claim_limit": (
            "All accepted structures remain Agent hypotheses. Intervention citations "
            "prove exact fork identity only, not semantic control claims."
        ),
    }
    output["artifact_sha256"] = _hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        output, indent=2, sort_keys=True, ensure_ascii=False,
    ) + "\n", encoding="utf-8")
    historical_cost = sum(
        float((row["usage"] or {}).get("cost", 0) or 0) for row in rows
    )
    print(json.dumps({
        "programs": len(output_programs),
        "agent_calls": len(rows),
        "api_calls_made_this_run": output["api_calls_made_this_run"],
        "qualified": sum(row["n_qualified"] for row in output_programs),
        "endpoint_failures": sum(bool(row["endpoint_failure"]) for row in rows),
        "errors": sum(row["error"] is not None for row in rows),
        "reported_cost_this_run": (
            0.0 if args.reuse_proposal_receipts is not None else historical_cost
        ),
        "reused_historical_reported_cost": (
            historical_cost if args.reuse_proposal_receipts is not None else 0.0
        ),
    }, indent=2, sort_keys=True))
    return 1 if any(row["endpoint_failure"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
