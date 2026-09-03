#!/usr/bin/env python3
"""Open a fresh strict-boundary AGQA reserve exactly once."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_harness import ARMS
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import _gold_rows, _matches, _mcnemar
from scripts.evaluate_agqa_query_grounder_v2_fresh_formal import _formal_gates


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--preoutcome", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("strict-boundary formal evaluation is immutable")
    protocol = json.loads(args.protocol.read_text())
    manifest = json.loads(args.manifest.read_text())
    cohort = json.loads(args.cohort.read_text())
    preoutcome = json.loads(args.preoutcome.read_text())
    if preoutcome.get("status") != (
        "ALL_FIVE_ARM_STRICT_BOUNDARY_DECISIONS_FROZEN_BEFORE_FORMAL_OUTCOMES"
    ) or not all(preoutcome.get("invariants", {}).values()):
        raise ValueError("five-arm decisions were not frozen before outcomes")
    if preoutcome["protocol_file_sha256"] != _sha256(args.protocol):
        raise ValueError("formal protocol changed after decisions froze")
    if preoutcome["manifest_file_sha256"] != _sha256(args.manifest):
        raise ValueError("formal manifest changed after decisions froze")
    if preoutcome["cohort_file_sha256"] != _sha256(args.cohort):
        raise ValueError("formal cohort changed after decisions froze")
    if manifest.get("status") != (
        "AGQA_QUERY_GROUNDER_V2_STRICT_BOUNDARY_FRESH_FORMAL_FROZEN"
    ):
        raise ValueError("strict-boundary formal reserve is not frozen")
    wanted = {str(row["task_id"]) for row in cohort["rows"]}
    frozen_rows = {str(row["task_id"]): row for row in preoutcome["rows"]}
    if set(frozen_rows) != wanted:
        raise ValueError("preoutcome decisions do not exactly cover the cohort")

    # The first target-outcome read occurs here, after every arm is frozen.
    evaluator = _gold_rows(args.archive, args.entry, wanted)
    public_by_task = {str(row["task_id"]): row for row in cohort["rows"]}
    rows = []
    for task_id in [str(row["task_id"]) for row in cohort["rows"]]:
        frozen = frozen_rows[task_id]
        gold = str(evaluator[task_id]["answer"])
        correct = {arm: _matches(frozen["predictions"][arm], gold) for arm in ARMS}
        rows.append({
            **frozen,
            "answer_type": str(public_by_task[task_id].get("answer_type") or "unknown"),
            "gold_answer_evaluator_only": gold,
            "correct": correct,
            "official_answer_first_read_after_all_five_arm_decisions_froze": True,
        })
    correct = {arm: [row["correct"][arm] for row in rows] for arm in ARMS}
    n = len(rows)
    summaries = {arm: {
        "correct": sum(correct[arm]), "total": n,
        "accuracy": sum(correct[arm]) / n,
        "symbolic_commits": sum(row["commits"][arm] for row in rows),
    } for arm in ARMS}
    comparisons = {
        baseline: _mcnemar(correct["source_induced"], correct[baseline])
        for baseline in ("neural_only", "generic_scaffold", "source_permuted")
    }
    gates = _formal_gates(summaries, comparisons, rows, protocol["formal_gates"])
    failure_taxonomy = {
        "symbolic_recovery": sum(
            row["correct"]["source_induced"] and not row["correct"]["neural_only"]
            for row in rows
        ),
        "negative_transfer": sum(
            row["correct"]["neural_only"] and not row["correct"]["source_induced"]
            for row in rows
        ),
        "committed_both_correct": sum(
            row["commits"]["source_induced"] and row["correct"]["source_induced"]
            and row["correct"]["neural_only"] for row in rows
        ),
        "committed_shared_failure": sum(
            row["commits"]["source_induced"] and not row["correct"]["source_induced"]
            and not row["correct"]["neural_only"] for row in rows
        ),
        "abstained_fallback_correct": sum(
            not row["commits"]["source_induced"] and row["correct"]["neural_only"]
            for row in rows
        ),
        "abstained_generic_headroom": sum(
            not row["commits"]["source_induced"] and not row["correct"]["neural_only"]
            and row["correct"]["generic_scaffold"] for row in rows
        ),
        "abstained_shared_failure": sum(
            not row["commits"]["source_induced"] and not row["correct"]["neural_only"]
            and not row["correct"]["generic_scaffold"] for row in rows
        ),
    }
    ablations: dict[str, dict] = {}
    for key in ("root_predicate", "requested_role", "answer_type"):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row)
        ablations[key] = {
            value: {
                "tasks": len(group),
                "neural_correct": sum(row["correct"]["neural_only"] for row in group),
                "source_correct": sum(row["correct"]["source_induced"] for row in group),
                "source_commits": sum(row["commits"]["source_induced"] for row in group),
                "wins": sum(row["correct"]["source_induced"] and not row["correct"]["neural_only"] for row in group),
                "losses": sum(row["correct"]["neural_only"] and not row["correct"]["source_induced"] for row in group),
            } for value, group in sorted(groups.items())
        }
    secondary = {
        "source_overall_accuracy_strictly_above_55_percent": (
            summaries["source_induced"]["accuracy"]
            > float(protocol["secondary_target"]["overall_source_accuracy_strictly_above"])
        ),
        "is_formal_pass_gate": False,
    }
    body = {
        "schema_version": "agqa-query-grounder-v2-strict-boundary-formal-evaluation-v1",
        "status": (
            "AGQA_QUERY_GROUNDER_V2_STRICT_BOUNDARY_FRESH_FORMAL_TRANSFER_VALIDATED"
            if all(gates.values()) else
            "AGQA_QUERY_GROUNDER_V2_STRICT_BOUNDARY_FRESH_FORMAL_GATES_FAILED"
        ),
        "claim_scope": "CONTROLLED_BALANCED_TRAIN_QUERY_OBJECT_NOT_OFFICIAL_TEST_OR_SOTA",
        "protocol_file_sha256": _sha256(args.protocol),
        "manifest_file_sha256": _sha256(args.manifest),
        "cohort_file_sha256": _sha256(args.cohort),
        "preoutcome_file_sha256": _sha256(args.preoutcome),
        "preoutcome_receipt_sha256": preoutcome["receipt_sha256"],
        "rows": rows, "summaries": summaries, "comparisons": comparisons,
        "gates": gates, "secondary_target": secondary,
        "failure_taxonomy": failure_taxonomy, "ablations": ablations,
        "cost": {"provider_calls": 0, "provider_cost_usd": 0.0,
                 "local_gpu_job_seconds_added_posthoc_from_scheduler_receipts": True},
        "all_five_arms_share_raw_frames_grounding_parser_executor_and_fallback": True,
        "only_symbolic_harness_varies": True,
        "official_scene_graph_or_functional_program_used_at_runtime": False,
        "target_outcomes_opened_only_here": True, "official_test_claim": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "summaries": summaries,
        "comparisons": comparisons, "gates": gates,
        "secondary_target": secondary, "failure_taxonomy": failure_taxonomy,
        "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
