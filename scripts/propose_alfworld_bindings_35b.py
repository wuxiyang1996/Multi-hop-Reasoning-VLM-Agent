#!/usr/bin/env python3
"""Ask independent agents for candidates; never ask them for a verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.frozen_transfer_policy import StrictOpenAIClient  # noqa: E402
from harness.skill_admission import target_demo_receipt_from_dict  # noqa: E402
from skill_bank.program_ir import canonical_program_from_dict  # noqa: E402


_JSON_ONLY = re.compile(r"\A\s*\{.*\}\s*\Z", re.DOTALL)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _program_ref(program: Any) -> str:
    return "program-" + program.content_hash()[:16]


def _step_aliases(program: Any) -> Dict[str, str]:
    return {f"step-{index}": item.step_id for index, item in enumerate(program.steps)}


def _prompt(program: Any, demo: Any, role: str) -> str:
    schemas = {
        item.operator: dict(item.argument_types)
        for item in demo.actions
    }
    allowed = [{"operator": name, "argument_types": slots} for name, slots in schemas.items()]
    action_counts = Counter(item.action for item in program.evidence)
    source = {
        # Human-authored game/skill/operator labels are deliberately hidden.
        # Agents receive an opaque reference plus executable source receipts.
        "program_id": _program_ref(program),
        "program_hash": program.content_hash(),
        "source_step_ids": list(_step_aliases(program)),
        "observed_action_counts": action_counts.most_common(20),
        "evidence_count": len(program.evidence),
    }
    target_trace = [
        {
            "transition_index": item.transition_index,
            "operator": item.operator,
            "argument_types": dict(item.argument_types),
            "arguments": dict(item.arguments),
            "state_sha256": item.state_sha256,
            "next_state_sha256": item.next_state_sha256,
        }
        for item in demo.actions
    ]
    role_instruction = {
        "proposer_a": "Propose the strongest executable structural instantiation.",
        "proposer_b": "Independently search for a different plausible instantiation.",
        "skeptic": (
            "Search for a competing instantiation that would expose ambiguity. "
            "Use ABSTAIN when no evidence-referenced competitor exists."
        ),
    }[role]
    return (
        f"You are {role}, an untrusted hypothesis generator. {role_instruction} "
        "Propose at most one source-step to target-demo action binding. You do not "
        "verify, score, vote on, or admit it. Semantic names and rationale are never "
        "evidence. If the supplied receipts are insufficient, set operator to ABSTAIN.\n"
        f"SOURCE_PROGRAM={json.dumps(source, sort_keys=True)}\n"
        f"FIXED_TARGET_DEMO={json.dumps(target_trace, sort_keys=True)}\n"
        f"ALLOWED_SCHEMAS={json.dumps(allowed, sort_keys=True)}\n"
        "Return exactly one JSON object with keys: source_program_id, "
        "source_step_id, operator, argument_types, rationale. No markdown."
    )


def _validate(
    raw: str,
    program: Any,
    allowed_schemas: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[Dict[str, Any] | None, str | None]:
    if _JSON_ONLY.fullmatch(raw) is None:
        return None, "NOT_EXACT_JSON_OBJECT"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None, "INVALID_JSON"
    if not isinstance(payload, dict):
        return None, "JSON_NOT_OBJECT"
    expected_keys = {
        "source_program_id", "source_step_id", "operator", "argument_types", "rationale",
    }
    if set(payload) != expected_keys:
        return None, "WRONG_JSON_KEYS"
    if payload["source_program_id"] != _program_ref(program):
        return None, "HALLUCINATED_PROGRAM_ID"
    aliases = _step_aliases(program)
    if payload["source_step_id"] not in aliases:
        return None, "HALLUCINATED_SOURCE_STEP"
    operator = str(payload["operator"])
    if operator == "ABSTAIN":
        if payload["argument_types"] not in ({}, None):
            return None, "ABSTAIN_WITH_ARGUMENTS"
        normalized = dict(payload)
        normalized["source_program_id"] = program.program_id
        normalized["source_step_id"] = aliases[str(payload["source_step_id"])]
        return normalized, None
    if allowed_schemas is None:
        return None, "MISSING_TARGET_NATIVE_SCHEMA_SET"
    schema = allowed_schemas.get(operator)
    if schema is None:
        return None, "HALLUCINATED_OPERATOR"
    if dict(payload["argument_types"] or {}) != dict(schema):
        return None, "ARGUMENT_SCHEMA_MISMATCH"
    normalized = dict(payload)
    normalized["source_program_id"] = program.program_id
    normalized["source_step_id"] = aliases[str(payload["source_step_id"])]
    return normalized, None


def _request_proposal(
    client: StrictOpenAIClient, *, model: str, prompt: str, program: Any,
    allowed_schemas: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[str, Mapping[str, Any], Dict[str, Any] | None, str | None, bool]:
    """Keep endpoint failures separate from model schema failures."""
    try:
        reply, usage = client.complete(model=model, prompt=prompt, max_tokens=256)
    except Exception as exc:
        return (
            "",
            {},
            None,
            f"ENDPOINT_FAILURE:{type(exc).__name__}:{exc}",
            True,
        )
    proposal, error = _validate(reply, program, allowed_schemas)
    return reply, usage, proposal, error, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument(
        "--programs", type=Path,
        default=REPO_ROOT / "artifacts/source_evidence_index/source_programs.jsonl",
    )
    parser.add_argument(
        "--demo", type=Path, required=True,
        help="the one fixed successful target demonstration",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = StrictOpenAIClient(args.endpoint, timeout_s=180.0)
    programs = [
        canonical_program_from_dict(json.loads(line))
        for line in args.programs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    demo = target_demo_receipt_from_dict(json.loads(args.demo.read_text(encoding="utf-8")))
    demo.validate_for_admission()
    allowed_schemas = {item.operator: dict(item.argument_types) for item in demo.actions}
    rows: List[Dict[str, Any]] = []
    bindings: List[Dict[str, Any]] = []
    roles = ("proposer_a", "proposer_b", "skeptic")
    try:
        for index, program in enumerate(programs):
            identity = (program.source_games[0], program.name)
            for role in roles:
                prompt = _prompt(program, demo, role)
                reply, usage, proposal, error, endpoint_failure = _request_proposal(
                    client, model=args.model, prompt=prompt, program=program,
                    allowed_schemas=allowed_schemas,
                )
                proposed_operator = proposal.get("operator") if proposal else None
                row = {
                    "program_id": program.program_id,
                    "program_hash": program.content_hash(),
                    "source_game": identity[0],
                    "source_skill_name": identity[1],
                    "agent_role": role,
                    "proposal": proposal,
                    "valid_closed_schema_proposal": proposal is not None,
                    "proposal_error": error,
                    "endpoint_failure": endpoint_failure,
                    "raw_reply": reply[:2000],
                    "usage": dict(usage),
                    "admission_effect": "CANDIDATE_ONLY_HARNESS_DECIDES",
                }
                rows.append(row)
                if proposal is not None and proposed_operator != "ABSTAIN":
                    candidate_key = json.dumps({
                        "program": program.program_id,
                        "step": proposal["source_step_id"],
                        "operator": proposed_operator,
                        "arguments": proposal["argument_types"],
                        "role": role,
                        "model": args.model,
                    }, sort_keys=True, separators=(",", ":"))
                    bindings.append({
                        "candidate_id": "agent-" + hashlib.sha256(
                            candidate_key.encode("utf-8")
                        ).hexdigest()[:20],
                        "source_game": identity[0],
                        "source_skill_name": identity[1],
                        "source_step_id": proposal["source_step_id"],
                        "target_operator": proposed_operator,
                        "argument_types": proposal["argument_types"],
                        "proposal_source": f"{args.model}:{role}",
                    })
                print(
                    f"[35B proposal] {index + 1}/{len(programs)} {program.program_id} "
                    f"role={role} operator={proposed_operator} error={error}",
                    flush=True,
                )
    finally:
        client.close()
    summary = {
        "schema_version": 1,
        "role": "untrusted_agent_candidate_set",
        "candidate_source": "independent_untrusted_agents",
        "target_domain": demo.target_domain,
        "task_family": demo.task_family,
        "demo_id": demo.demo_id,
        "demo_hash": demo.content_hash(),
        "model": args.model,
        "programs_sha256": _sha256(args.programs),
        "demo_sha256": _sha256(args.demo),
        "n_programs": len(programs),
        "n_agent_calls": len(rows),
        "n_valid_closed_schema": sum(row["valid_closed_schema_proposal"] for row in rows),
        "n_abstain": sum(
            bool(row.get("proposal") and row["proposal"].get("operator") == "ABSTAIN")
            for row in rows
        ),
        "n_invalid_or_hallucinated": sum(
            not row["valid_closed_schema_proposal"] and not row["endpoint_failure"]
            for row in rows
        ),
        "n_endpoint_failures": sum(row["endpoint_failure"] for row in rows),
        "n_candidates": len(bindings),
        "admission_artifacts_modified": False,
        "bindings": bindings,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({key: summary[key] for key in (
        "n_programs", "n_valid_closed_schema", "n_abstain",
        "n_agent_calls", "n_candidates", "n_invalid_or_hallucinated", "n_endpoint_failures",
    )}, indent=2))
    return 1 if summary["n_endpoint_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
