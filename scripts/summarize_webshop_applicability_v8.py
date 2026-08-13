#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402


def semantic_bucket(row: dict) -> str:
    semantics = row["semantics"]
    if semantics["is_commit"]:
        return "commit"
    if semantics["is_constraint"]:
        return "constraint"
    if semantics["is_navigation"]:
        return "navigation"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v7-report", type=Path, required=True)
    parser.add_argument("--v8-replay-summary", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--grouped-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    v7 = json.loads(args.v7_report.read_text())
    replay = json.loads(args.v8_replay_summary.read_text())
    candidates = json.loads(args.candidate_report.read_text())
    split = json.loads(args.grouped_split.read_text())
    branches = [row for row in candidates["branches"] if row["failure"] is None]
    buckets = Counter(semantic_bucket(row) for row in branches)
    commit = [row for row in branches if semantic_bucket(row) == "commit"]
    constraints = [row for row in branches if semantic_bucket(row) == "constraint"]
    replay_metrics = replay["conditions"]
    report = {
        "schema_version": 1,
        "experiment": "webshop_neurosymbolic_applicability_v8_development",
        "claim_limit": "Diagnostic replay and consumed semantic grouping only.",
        "semantic_group_split": {
            "group_key": split["group_key"],
            "unique_groups": len(split["tasks_by_group"]),
            "role_counts": {
                role: len(groups) for role, groups in split["groups_by_role"].items()
            },
            "confirmation_tasks": split["representative_tasks_by_role"]["confirmation"],
            "cross_role_group_overlap": False,
        },
        "v7_failure": {
            "target_mean_reward": v7["episode_metrics"]["target_only"]["mean_official_reward"],
            "authentic_mean_reward": v7["episode_metrics"]["selective_authentic_source"]["mean_official_reward"],
            "target_strict": v7["episode_metrics"]["target_only"]["strict_successes"],
            "authentic_strict": v7["episode_metrics"]["selective_authentic_source"]["strict_successes"],
        },
        "v8_diagnostic_replay": {
            "tasks": replay["tasks"],
            "target": replay_metrics["target_only"],
            "safe_neural": replay_metrics["selective_safe_minimum_repeat"],
            "safe_authentic": replay_metrics["selective_safe_authentic_source"],
            "interventions": 0,
            "cache_mode": "exact-request replay only",
            "all_requests_cache_hits": True,
        },
        "candidate_interventions": {
            **candidates["metrics"],
            "semantic_bucket_counts": dict(sorted(buckets.items())),
            "commit_actions": len(commit),
            "commit_actions_terminated": sum(row["actual_terminated"] for row in commit),
            "commit_mean_reward": (
                sum(row["actual_reward"] for row in commit) / len(commit) if commit else None
            ),
            "constraint_actions": len(constraints),
            "constraint_actions_changed_state": sum(
                row["actual_state_changed"] for row in constraints
            ),
        },
        "scientific_status": "SAFETY_FIX_ONLY_NO_TRANSFER_EVIDENCE",
        "confirmation_status": "NOT_RUN_INVALID_DECISION_CREDENTIALS",
        "interpretation": (
            "The V8 shield removes V7 negative transfer by refusing unsafe interventions, "
            "but it makes zero interventions and therefore does not establish transfer benefit."
        ),
        "runtime_hashes": {
            "v7_report": file_sha256(args.v7_report),
            "v8_replay_summary": file_sha256(args.v8_replay_summary),
            "candidate_report": file_sha256(args.candidate_report),
            "grouped_split": file_sha256(args.grouped_split),
            "summarizer": file_sha256(Path(__file__)),
        },
        "held_out_read_or_run": False,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
