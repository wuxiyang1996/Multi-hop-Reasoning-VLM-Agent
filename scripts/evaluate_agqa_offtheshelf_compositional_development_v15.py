#!/usr/bin/env python3
"""Open V15 development answers once after five-arm decisions are frozen."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from motif_transfer.agqa_layer_b_harness import ARMS
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import _gold_rows, _matches, _mcnemar


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--preoutcome", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-transfer-fraction", type=float, default=0.05)
    parser.add_argument("--formal-protocol", type=Path)
    parser.add_argument("--formal-manifest", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("V15 development evaluation is immutable")

    cohort = json.loads(args.cohort.read_text())
    pre = json.loads(args.preoutcome.read_text())
    protocol = json.loads(args.formal_protocol.read_text()) if args.formal_protocol else None
    manifest = json.loads(args.formal_manifest.read_text()) if args.formal_manifest else None
    formal = protocol is not None or manifest is not None
    if protocol is None and manifest is not None or protocol is not None and manifest is None:
        raise ValueError("formal protocol and manifest must be supplied together")
    expected_pre_statuses = (
        {"V17_FORMAL_FIVE_ARM_DECISIONS_FROZEN_BEFORE_OUTCOMES"}
        if formal else {
            "V15_COMPOSITIONAL_DEVELOPMENT_DECISIONS_FROZEN",
            "V18_COMPOSITIONAL_DEVELOPMENT_DECISIONS_FROZEN",
        }
    )
    if pre.get("status") not in expected_pre_statuses or not all(pre.get("gates", {}).values()):
        raise ValueError("five-arm decisions did not pass outcome-blind gates")
    if formal:
        if protocol.get("status") != "QWEN32_COMPOSITIONAL_FORMAL_PROTOCOL_FROZEN_AFTER_DEVELOPMENT":
            raise ValueError("formal protocol is invalid")
        if manifest.get("status") != "AGQA_QWEN32_COMPOSITIONAL_FRESH_FORMAL_V17_FROZEN":
            raise ValueError("formal manifest is invalid")
        if manifest.get("protocol_file_sha256") != _sha256(args.formal_protocol):
            raise ValueError("formal manifest/protocol mismatch")
        if pre.get("formal_protocol_file_sha256") != _sha256(args.formal_protocol):
            raise ValueError("formal decisions do not bind protocol")
        if pre.get("formal_manifest_file_sha256") != _sha256(args.formal_manifest):
            raise ValueError("formal decisions do not bind manifest")
        if _sha256(Path(__file__)) != protocol["components"]["evaluator_sha256"]:
            raise ValueError("formal evaluator implementation changed")
        if args.negative_transfer_fraction != float(protocol["formal_gates"]["negative_transfer_fraction_at_most"]):
            raise ValueError("formal negative-transfer threshold differs from protocol")
    if pre.get("cohort_file_sha256") != _sha256(args.cohort):
        raise ValueError("V15 cohort changed after decisions froze")
    if pre.get("cohort_sha256") != cohort.get("cohort_sha256"):
        raise ValueError("V15 cohort content identity changed")

    wanted = [str(row["task_id"]) for row in cohort["rows"]]
    frozen = {str(row["task_id"]): row for row in pre["rows"]}
    if set(frozen) != set(wanted):
        raise ValueError("preoutcome decisions do not exactly cover V15")

    # The first task-answer read for this development cohort occurs here.
    evaluator = _gold_rows(args.archive, args.entry, set(wanted))
    rows = []
    for task_id in wanted:
        row = frozen[task_id]
        gold = str(evaluator[task_id]["answer"])
        correct = {arm: _matches(row["predictions"][arm], gold) for arm in ARMS}
        rows.append({
            **row, "gold_answer_evaluator_only": gold, "correct": correct,
            "task_answer_first_read_after_all_five_arm_decisions_froze": True,
        })
    n = len(rows)
    correct = {arm: [row["correct"][arm] for row in rows] for arm in ARMS}
    summaries = {arm: {
        "correct": sum(correct[arm]), "total": n,
        "accuracy": sum(correct[arm]) / n,
        "symbolic_commits": sum(row["commits"][arm] for row in rows),
    } for arm in ARMS}
    comparisons = {
        baseline: _mcnemar(correct["source_induced"], correct[baseline])
        for baseline in ("neural_only", "generic_scaffold", "source_permuted")
    }
    max_losses = math.floor(args.negative_transfer_fraction * n)
    gates = {
        "source_beats_neural": summaries["source_induced"]["correct"] > summaries["neural_only"]["correct"],
        "source_vs_neural_significant": comparisons["neural_only"]["exact_two_sided_p"] < 0.05,
        "source_negative_transfer_bounded": comparisons["neural_only"]["losses"] <= max_losses,
        "source_beats_matched_permuted": summaries["source_induced"]["correct"] > summaries["source_permuted"]["correct"],
        "source_vs_matched_permuted_significant": comparisons["source_permuted"]["exact_two_sided_p"] < 0.05,
        "target_written_isomorphic_equivalence": all(
            row["predictions"]["source_induced"] == row["predictions"]["target_written_isomorphic"]
            for row in rows
        ),
        "preoutcome_gates_passed": all(pre["gates"].values()),
    }
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["semantic_root"]].append(row)
    root_breakdown = {
        root: {
            "tasks": len(group),
            "neural_correct": sum(row["correct"]["neural_only"] for row in group),
            "source_correct": sum(row["correct"]["source_induced"] for row in group),
            "generic_correct": sum(row["correct"]["generic_scaffold"] for row in group),
            "source_commits": sum(row["commits"]["source_induced"] for row in group),
            "wins": sum(row["correct"]["source_induced"] and not row["correct"]["neural_only"] for row in group),
            "losses": sum(row["correct"]["neural_only"] and not row["correct"]["source_induced"] for row in group),
        } for root, group in sorted(groups.items())
    }
    failure_taxonomy = {
        "symbolic_recovery": comparisons["neural_only"]["wins"],
        "negative_transfer": comparisons["neural_only"]["losses"],
        "committed_shared_failure": sum(
            row["commits"]["source_induced"] and not row["correct"]["source_induced"]
            and not row["correct"]["neural_only"] for row in rows
        ),
        "abstained_fallback_correct": sum(
            not row["commits"]["source_induced"] and row["correct"]["neural_only"] for row in rows
        ),
        "abstained_shared_failure": sum(
            not row["commits"]["source_induced"] and not row["correct"]["neural_only"] for row in rows
        ),
    }
    development_status_prefix = (
        "V18_COMPOSITIONAL_DEVELOPMENT_TRANSFER_SIGNAL"
        if pre.get("status", "").startswith("V18_") else
        "V15_COMPOSITIONAL_DEVELOPMENT_TRANSFER_SIGNAL"
    )
    body = {
        "schema_version": "agqa-offtheshelf-compositional-development-evaluation-v15",
        "status": (
            "AGQA_QWEN32_COMPOSITIONAL_FRESH_FORMAL_TRANSFER_VALIDATED"
            if formal and all(gates.values()) else
            "AGQA_QWEN32_COMPOSITIONAL_FRESH_FORMAL_GATES_FAILED"
            if formal else
            f"{development_status_prefix}_PASSED"
            if all(gates.values()) else f"{development_status_prefix}_FAILED"
        ),
        "claim_scope": (
            "FRESH_VIDEO_AND_TASK_DISJOINT_BALANCED_TRAIN_COMPOSITIONAL_TRANSFER"
            if formal else "CONSUMED_VIDEO_DEVELOPMENT_QUALIFICATION_NOT_TRANSFER_EVIDENCE"
        ),
        "cohort_file_sha256": _sha256(args.cohort),
        "preoutcome_file_sha256": _sha256(args.preoutcome),
        "preoutcome_receipt_sha256": pre["receipt_sha256"],
        "rows": rows, "summaries": summaries, "comparisons": comparisons,
        "negative_transfer_max_losses": max_losses, "gates": gates,
        "secondary_target": {
            "source_overall_accuracy_strictly_above_55_percent": summaries["source_induced"]["accuracy"] > 0.55,
            "is_primary_gate": False,
        },
        "root_breakdown": root_breakdown, "failure_taxonomy": failure_taxonomy,
        "all_five_arms_share_raw_frames_grounding_parser_executor_and_fallback": True,
        "only_symbolic_harness_varies": True,
        "official_scene_graph_or_functional_program_used_at_runtime": False,
        "formal_protocol_file_sha256": _sha256(args.formal_protocol) if formal else None,
        "formal_manifest_file_sha256": _sha256(args.formal_manifest) if formal else None,
        "development_only": not formal, "target_outcomes_opened_only_here": True,
        "executor_version": pre.get("executor_version", "v2"),
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "summaries": summaries,
        "comparisons": comparisons, "gates": gates,
        "secondary_target": body["secondary_target"],
        "root_breakdown": root_breakdown, "failure_taxonomy": failure_taxonomy,
        "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
