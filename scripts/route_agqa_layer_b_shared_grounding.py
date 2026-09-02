#!/usr/bin/env python3
"""Outcome-blind routing between frozen Layer-B grounding candidates.

The router is part of the shared target-native grounder.  It never reads game
capabilities, source-induced plans, programs, or answers.  A candidate is
considered complete only when the unrestricted target-native executor can
execute the operator-free semantic parse.  Ties use the declared candidate
order, not a target outcome.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_executor import execute_layer_b_semantics
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import _grounding, _semantic


GENERIC_VM_OPERATORS = (
    "AND", "ARGMAX", "CHOOSE", "COMPARE", "EXISTS", "FILTER_EQ",
    "FIRST", "INTERVAL_OF", "LAST", "PROJECT", "TEMPORAL_SELECT",
    "UNIQUE", "XOR",
)


def _generic_status(raw_row: dict, compact_semantics: str) -> str:
    result = execute_layer_b_semantics(
        compact_semantics=compact_semantics,
        grounding=_grounding(raw_row["grounding_receipt"]),
        semantic=_semantic(raw_row["semantic_receipt"]),
        authorized_operators=GENERIC_VM_OPERATORS,
        authorized_compositions=None,
        ambiguity_policy="EAGER",
    )
    return result.receipt.status


def choose_candidate(statuses: tuple[str, ...]) -> int:
    """Choose the first generically executable candidate; fixed-order fallback."""
    for index, status in enumerate(statuses):
        if status == "COMMITTED":
            return index
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--shared-action-frame-budget", type=int, default=320)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("routed grounding output is immutable")
    if len(args.candidate) < 2:
        raise ValueError("routing requires at least two frozen candidates")

    cohort = json.loads(args.cohort.read_text())
    runtime = json.loads(args.semantic_runtime.read_text())
    candidates = [json.loads(path.read_text()) for path in args.candidate]
    if runtime["status"] != "SEMANTIC_RUNTIME_FROZEN_BEFORE_VIDEO_OR_OUTCOME":
        raise ValueError("semantic runtime is not frozen")
    if any(candidate["status"] != "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES"
           for candidate in candidates):
        raise ValueError("candidate grounding was not frozen before outcomes")
    if any(candidate["cohort_sha256"] != cohort["cohort_sha256"]
           for candidate in candidates):
        raise ValueError("candidate/cohort mismatch")
    forbidden = ("answer_read", "official_scene_graph_read", "functional_program_read", "source_controller_read")
    if any(candidate.get(key) for candidate in candidates for key in forbidden):
        raise ValueError("candidate crossed an authority boundary")

    compact = {str(row["task_id"]): str(row["predicted_semantics"])
               for row in runtime["rows"]}
    by_candidate = [
        {str(row["task_id"]): row for row in candidate["rows"]}
        for candidate in candidates
    ]
    wanted = [str(row["task_id"]) for row in cohort["rows"]]
    if any(set(rows) != set(wanted) for rows in by_candidate):
        raise ValueError("candidate task set mismatch")

    rows = []
    route_counts = [0] * len(candidates)
    for task_id in wanted:
        candidate_rows = [values[task_id] for values in by_candidate]
        statuses = tuple(_generic_status(row, compact[task_id]) for row in candidate_rows)
        selected = choose_candidate(statuses)
        route_counts[selected] += 1
        row = dict(candidate_rows[selected])
        row["shared_grounder_route"] = {
            "candidate_index": selected,
            "generic_execution_statuses": list(statuses),
            "rule": "FIRST_GENERIC_COMMIT_ELSE_FIRST_CANDIDATE",
            "answer_read": False,
            "source_controller_read": False,
        }
        rows.append(row)

    # Both composite candidates reuse one identical frozen SlowFast pass.  Do
    # not double-count it in the actual frame-presentation budget.
    frame_budget = sum(int(candidate["frame_budget"]) for candidate in candidates)
    action_hashes = {candidate.get("action_probe_sha256") for candidate in candidates}
    if len(action_hashes) == 1 and None not in action_hashes:
        frame_budget -= args.shared_action_frame_budget * (len(candidates) - 1)
    backend = stable_hash({
        "router": "FIRST_GENERIC_COMMIT_ELSE_FIRST_CANDIDATE_V1",
        "candidate_report_sha256s": [candidate["report_sha256"] for candidate in candidates],
        "generic_vm_operators": GENERIC_VM_OPERATORS,
        "total_frame_presentation_budget": frame_budget,
        "shared_action_frame_budget": args.shared_action_frame_budget,
    })
    body = {
        "schema_version": "agqa-layer-b-shared-grounder-router-v1",
        "status": "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES",
        "cohort_sha256": cohort["cohort_sha256"],
        "semantic_runtime_sha256": runtime["runtime_sha256"],
        "grounder_backend_sha256": backend,
        "frame_budget": frame_budget,
        "candidate_report_sha256s": [candidate["report_sha256"] for candidate in candidates],
        "candidate_paths": [str(path) for path in args.candidate],
        "route_counts": route_counts,
        "rows": rows,
        "all_harness_arms_share_exact_receipts": True,
        "answer_read": False,
        "official_scene_graph_read": False,
        "functional_program_read": False,
        "source_controller_read": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "rows": len(rows), "route_counts": route_counts,
        "frame_budget": frame_budget, "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
