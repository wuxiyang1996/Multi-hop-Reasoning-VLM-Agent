#!/usr/bin/env python3
"""Ask 35B for closed-schema binding proposals, never admission verdicts."""

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
from harness.skill_admission import TARGET_OPERATOR_SCHEMAS  # noqa: E402
from skill_bank.program_ir import canonical_program_from_dict  # noqa: E402


_JSON_ONLY = re.compile(r"\A\s*\{.*\}\s*\Z", re.DOTALL)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt(program: Any) -> str:
    schemas = TARGET_OPERATOR_SCHEMAS["alfworld"]
    allowed = [{"operator": name, "argument_types": slots} for name, slots in schemas.items()]
    action_counts = Counter(item.action for item in program.evidence)
    source = {
        "program_id": program.program_id,
        "program_hash": program.content_hash(),
        "name": program.name,
        "source_games": program.source_games,
        "source_skill_ids": program.source_skill_ids,
        "source_step_ids": [item.step_id for item in program.steps],
        "observed_action_counts": action_counts.most_common(20),
        "evidence_count": len(program.evidence),
    }
    return (
        "You are an untrusted hypothesis generator. Propose at most one mapping "
        "from the source program to a closed ALFWorld operator schema. You do not "
        "verify or admit it. If evidence is insufficient, set operator to ABSTAIN.\n"
        f"SOURCE_PROGRAM={json.dumps(source, sort_keys=True)}\n"
        f"ALLOWED_SCHEMAS={json.dumps(allowed, sort_keys=True)}\n"
        "Return exactly one JSON object with keys: source_program_id, "
        "source_step_id, operator, argument_types, rationale. No markdown."
    )


def _validate(raw: str, program: Any) -> tuple[Dict[str, Any] | None, str | None]:
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
    if payload["source_program_id"] != program.program_id:
        return None, "HALLUCINATED_PROGRAM_ID"
    if payload["source_step_id"] not in {item.step_id for item in program.steps}:
        return None, "HALLUCINATED_SOURCE_STEP"
    operator = str(payload["operator"])
    if operator == "ABSTAIN":
        if payload["argument_types"] not in ({}, None):
            return None, "ABSTAIN_WITH_ARGUMENTS"
        return payload, None
    schema = TARGET_OPERATOR_SCHEMAS["alfworld"].get(operator)
    if schema is None:
        return None, "HALLUCINATED_OPERATOR"
    if dict(payload["argument_types"] or {}) != dict(schema):
        return None, "ARGUMENT_SCHEMA_MISMATCH"
    return payload, None


def _request_proposal(
    client: StrictOpenAIClient, *, model: str, prompt: str, program: Any,
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
    proposal, error = _validate(reply, program)
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
        "--frozen-bindings", type=Path,
        default=REPO_ROOT / "configs/alfworld_one_shot_bindings.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = StrictOpenAIClient(args.endpoint, timeout_s=180.0)
    programs = [
        canonical_program_from_dict(json.loads(line))
        for line in args.programs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frozen = json.loads(args.frozen_bindings.read_text(encoding="utf-8"))
    frozen_identity = {
        (str(item["source_game"]), str(item["source_skill_name"])): str(item["target_operator"])
        for item in frozen.get("bindings", [])
    }
    rows: List[Dict[str, Any]] = []
    try:
        for index, program in enumerate(programs):
            prompt = _prompt(program)
            reply = ""
            error = None
            proposal = None
            usage: Mapping[str, Any] = {}
            reply, usage, proposal, error, endpoint_failure = _request_proposal(
                client, model=args.model, prompt=prompt, program=program,
            )
            identity = (program.source_games[0], program.name)
            frozen_operator = frozen_identity.get(identity)
            proposed_operator = proposal.get("operator") if proposal else None
            rows.append({
                "program_id": program.program_id,
                "program_hash": program.content_hash(),
                "source_game": identity[0],
                "source_skill_name": identity[1],
                "proposal": proposal,
                "valid_closed_schema_proposal": proposal is not None,
                "proposal_error": error,
                "endpoint_failure": endpoint_failure,
                "raw_reply": reply[:2000],
                "usage": dict(usage),
                "frozen_operator_for_audit_only": frozen_operator,
                "matches_frozen_proposal": bool(
                    frozen_operator is not None and proposed_operator == frozen_operator
                ),
                "admission_effect": "NONE_FROZEN_ARTIFACTS_IMMUTABLE",
            })
            print(
                f"[35B proposal] {index + 1}/{len(programs)} {program.program_id} "
                f"operator={proposed_operator} error={error}",
                flush=True,
            )
    finally:
        client.close()
    summary = {
        "schema_version": 1,
        "role": "untrusted_proposal_only",
        "model": args.model,
        "programs_sha256": _sha256(args.programs),
        "frozen_bindings_sha256": _sha256(args.frozen_bindings),
        "n_programs": len(rows),
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
        "n_frozen_candidates": len(frozen_identity),
        "n_matches_frozen": sum(row["matches_frozen_proposal"] for row in rows),
        "admission_artifacts_modified": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({key: summary[key] for key in (
        "n_programs", "n_valid_closed_schema", "n_abstain",
        "n_invalid_or_hallucinated", "n_endpoint_failures", "n_matches_frozen",
    )}, indent=2))
    return 1 if summary["n_endpoint_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
