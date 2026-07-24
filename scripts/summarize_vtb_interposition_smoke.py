#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the matched VTB interposition mechanism smoke.")
    parser.add_argument("--target-only", type=Path, required=True)
    parser.add_argument("--generic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = _read(args.target_only)
    generic = _read(args.generic)
    if target["sample_id"] != generic["sample_id"]:
        raise SystemExit("UNMATCHED: sample ID differs")
    identity_checks = {
        "initial_image_sha256_equal": target["image_sha256"] == generic["image_sha256"],
        "official_commit_equal": target["official_commit"] == generic["official_commit"],
        "tool_contract_equal": target["tool_contract_sha256"] == generic["tool_contract_sha256"],
        "initial_dynamic_tool_schema_equal": (
            target["dynamic_tool_schema_sha256_by_round"][0]
            == generic["dynamic_tool_schema_sha256_by_round"][0]
        ),
        "decision_model_equal": target["decision_model"] == generic["decision_model"],
        "budget_equal": target["max_tool_rounds"] == generic["max_tool_rounds"],
        "generic_first_decision_request_cache_hit": bool(
            generic.get("decision_usage") and generic["decision_usage"][0].get("cache_hit")
        ),
        "generic_first_reviewed_proposal_equals_target_first": bool(
            generic.get("reviews")
            and generic["reviews"][0].get("reviewed_proposal_ids")
            and target.get("tool_trace")
            and generic["reviews"][0]["reviewed_proposal_ids"][0]
            == target["tool_trace"][0]["proposal"]["proposal_id"]
        ),
        "generic_harness_payloads_exact_6000_tokens": bool(
            generic.get("harness_input_token_counts")
            and all(
                row.get("o200k_base_tokens") == 6000
                for row in generic["harness_input_token_counts"]
            )
        ),
    }
    matched = all(identity_checks.values())
    target_successes = sum(bool(row["receipt"]["success"]) for row in target["tool_trace"])
    generic_successes = sum(bool(row["receipt"]["success"]) for row in generic["tool_trace"])
    if not matched:
        status = "UNMATCHED_INVALID"
    elif target.get("final_answer_present") or generic.get("final_answer_present"):
        status = "MECHANISM_SMOKE_WITH_FINAL_ANSWER"
    elif generic.get("replan_count", 0) > 0 and generic_successes > target_successes:
        status = "GENERIC_INTERPOSITION_IMPROVED_TOOL_EXECUTION_WITHOUT_TASK_SUCCESS"
    elif generic.get("replan_count", 0) > 0 and generic_successes < target_successes:
        status = "GENERIC_INTERPOSITION_HARMED_TOOL_EXECUTION_WITHOUT_TASK_SUCCESS"
    else:
        status = "NO_OBSERVED_GENERIC_INTERPOSITION_EFFECT"
    payload = {
        "schema_version": 1,
        "status": status,
        "sample_id": target["sample_id"],
        "identity_checks": identity_checks,
        "target_only": {
            "tool_calls": target["tool_call_count"],
            "successful_tool_calls": target_successes,
            "final_answer_present": target["final_answer_present"],
            "termination_reason": target["termination_reason"],
        },
        "generic_reasoning": {
            "tool_calls": generic["tool_call_count"],
            "successful_tool_calls": generic_successes,
            "reviews": len(generic["reviews"]),
            "verifications": len(generic["verifications"]),
            "replans": generic["replan_count"],
            "final_answer_present": generic["final_answer_present"],
            "termination_reason": generic["termination_reason"],
        },
        "claim_limit": (
            "This adaptation-only, capability-degraded, three-round run validates interposition "
            "and common-randomness mechanics. It is not source-motif transfer evidence and has no "
            "official task score."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
