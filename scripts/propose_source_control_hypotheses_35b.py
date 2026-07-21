#!/usr/bin/env python3
"""Ask independent Agents for evidence-referenced source control hypotheses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.frozen_transfer_policy import StrictOpenAIClient  # noqa: E402
from skill_agents.control_hypotheses import (  # noqa: E402
    AgentControlHypothesis,
    ControlHypothesisValidator,
    HypothesisEdge,
    HypothesisNode,
    union_qualified_hypotheses,
)
from skill_agents.evidence_query import (  # noqa: E402
    ContentAddressedEvidenceSession,
    EvidenceQuery,
)
from skill_bank.trace_program_ir import ControlClaimKind  # noqa: E402
from skill_bank.trace_program_validator import compile_observed_episode  # noqa: E402


_JSON_ONLY = re.compile(r"\A\s*\{.*\}\s*\Z", re.DOTALL)
_ROLES = ("proposer_a", "proposer_b", "skeptic")


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _prompt(program, response, role: str) -> tuple[str, Mapping[str, str]]:
    aliases = {
        f"t{index}": item.transition_id for index, item in enumerate(program.transitions)
    }
    visible = []
    for index, row in enumerate(response.transitions):
        evidence = row["native_evidence"]
        visible.append({
            "transition_id": f"t{index}",
            "step_index": row["step_index"],
            "state": evidence["state"],
            "available_actions": evidence["available_actions"],
            "executed_action": row["action"],
            "next_state": evidence["next_state"],
            "reward": row["reward"],
            "done": row["done"],
        })
    role_text = {
        "proposer_a": "Propose one compact multi-node control hypothesis supported by the observed path.",
        "proposer_b": "Independently propose a non-identical plausible control hypothesis.",
        "skeptic": "Propose a competing structure exposing ambiguity, or abstain.",
    }[role]
    kinds = [item.value for item in ControlClaimKind]
    prompt = (
        f"You are {role}, an untrusted hypothesis proposer. {role_text} "
        "The trace is observational only. Never claim causality, semantic transfer, "
        "complete branches, or verified guards. Use only listed transition IDs. "
        "Nodes must contain non-empty contiguous transition spans. Edges are untrusted "
        "claims for later interventions. If evidence is insufficient, abstain=true and "
        "return empty nodes/edges. Do not use game/skill names.\n"
        f"OPAQUE_PROGRAM_ID={program.program_id}\n"
        f"PROGRAM_HASH={program.content_hash()}\n"
        f"EVIDENCE_RESPONSE_HASH={response.response_sha256}\n"
        f"ALLOWED_EDGE_KINDS={json.dumps(kinds)}\n"
        f"OBSERVED_NATIVE_TRACE={json.dumps(visible, ensure_ascii=False, sort_keys=True)}\n"
        "Return exactly one JSON object with keys hypothesis_id,nodes,edges,abstain. "
        "Each node has exactly node_id,transition_ids. Each edge has exactly edge_id,"
        "source_node_id,target_node_id,kind,agent_claim. agent_claim must be a JSON object."
    )
    return prompt, aliases


def _parse(raw: str, *, program, response, role: str, aliases: Mapping[str, str]):
    if _JSON_ONLY.fullmatch(raw) is None:
        raise ValueError("NOT_EXACT_JSON_OBJECT")
    payload = json.loads(raw)
    if set(payload) != {"hypothesis_id", "nodes", "edges", "abstain"}:
        raise ValueError("WRONG_TOP_LEVEL_KEYS")
    nodes = []
    for row in payload["nodes"]:
        if set(row) != {"node_id", "transition_ids"}:
            raise ValueError("WRONG_NODE_KEYS")
        ids = [aliases[item] for item in row["transition_ids"]]
        nodes.append(HypothesisNode(str(row["node_id"]), ids))
    edges = []
    for row in payload["edges"]:
        if set(row) not in ({
            "edge_id", "source_node_id", "target_node_id", "kind", "agent_claim",
        }, {
            "edge_id", "source_node_id", "target_node_id", "kind", "agent_claim",
            "intervention_receipt_ids",
        }):
            raise ValueError("WRONG_EDGE_KEYS")
        if not isinstance(row["agent_claim"], dict):
            raise ValueError("AGENT_CLAIM_NOT_OBJECT")
        edges.append(HypothesisEdge(
            str(row["edge_id"]), str(row["source_node_id"]),
            str(row["target_node_id"]), ControlClaimKind(str(row["kind"])),
            dict(row["agent_claim"]),
            tuple(str(item) for item in row.get("intervention_receipt_ids") or ()),
        ))
    return AgentControlHypothesis(
        hypothesis_id=str(payload["hypothesis_id"]),
        program_id=program.program_id,
        program_hash=program.content_hash(),
        proposal_source=role,
        evidence_response_hashes=[response.response_sha256],
        nodes=nodes, edges=edges, abstain=bool(payload["abstain"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-episode", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    key = os.environ.get(args.api_key_env, "").strip()
    if not key and "openrouter.ai" in args.endpoint.lower():
        try:
            from API_func import open_router_api_key
            key = str(open_router_api_key or "").strip()
        except Exception:
            key = ""
    if "openrouter.ai" in args.endpoint.lower() and not key:
        raise SystemExit("OpenRouter API key unavailable")

    program = compile_observed_episode(args.source_episode)
    session = ContentAddressedEvidenceSession.from_source_episode(program, args.source_episode)
    response = session.query(EvidenceQuery(
        "full-observed-episode", program.program_id, program.content_hash(),
        [item.transition_id for item in program.transitions],
    ))
    client = StrictOpenAIClient(args.endpoint, timeout_s=180, api_key=key or "EMPTY")
    rows = []
    qualified = []
    try:
        for role in _ROLES:
            prompt, aliases = _prompt(program, response, role)
            reply = ""
            usage = {}
            error = None
            endpoint_failure = False
            candidate = None
            abstained = False
            try:
                reply, usage = client.complete(model=args.model, prompt=prompt, max_tokens=1200)
                hypothesis = _parse(
                    reply, program=program, response=response, role=role, aliases=aliases,
                )
                if hypothesis.abstain:
                    abstained = True
                else:
                    candidate = ControlHypothesisValidator().validate(
                        hypothesis, program=program, evidence_responses=[response],
                    )
                    if candidate.status == "AGENT_HYPOTHESIS":
                        qualified.append(candidate)
                    else:
                        error = ",".join(candidate.failure_codes)
            except Exception as exc:
                error = f"{type(exc).__name__}:{exc}"
                endpoint_failure = type(exc).__module__.startswith("httpx")
            receipt_payload = {
                "role": role, "model": args.model, "prompt_sha256": _hash(prompt),
                "raw_reply": reply, "usage": dict(usage),
            }
            rows.append({
                **receipt_payload,
                "receipt_sha256": _hash(receipt_payload),
                "qualified_hypothesis_hash": candidate.hypothesis_hash if candidate and candidate.status == "AGENT_HYPOTHESIS" else None,
                "error": error,
                "endpoint_failure": endpoint_failure,
                "abstained": abstained,
            })
    finally:
        client.close()
    union = union_qualified_hypotheses(qualified)
    output = {
        "schema_version": 1,
        "candidate_source": "independent_untrusted_agents",
        "status_semantics": "AGENT_HYPOTHESIS_NOT_CONTROL_PROOF",
        "program_id": program.program_id,
        "program_hash": program.content_hash(),
        "source_file_sha256": program.source_file_sha256,
        "evidence_response_sha256": response.response_sha256,
        "roles": list(_ROLES),
        "n_agent_calls": len(rows),
        "n_qualified": len(union),
        "n_abstain": sum(bool(row["abstained"]) for row in rows),
        "n_endpoint_failures": sum(bool(row["endpoint_failure"]) for row in rows),
        "qualified_source_hypotheses": [
            {"hypothesis_hash": item.hypothesis_hash, "hypothesis": {
                "hypothesis_id": item.hypothesis.hypothesis_id,
                "proposal_source": item.hypothesis.proposal_source,
                "nodes": [{"node_id": n.node_id, "transition_ids": list(n.transition_ids)} for n in item.hypothesis.nodes],
                "edges": [{"edge_id": e.edge_id, "source_node_id": e.source_node_id, "target_node_id": e.target_node_id, "kind": e.kind.value, "agent_claim": dict(e.agent_claim)} for e in item.hypothesis.edges],
            }} for item in union
        ],
        "proposal_receipts": rows,
        "claim_limit": "No intervention or transferable reasoning claim is established.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in (
        "program_id", "n_agent_calls", "n_qualified", "n_abstain", "n_endpoint_failures",
    )}, indent=2))
    return 1 if output["n_endpoint_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
