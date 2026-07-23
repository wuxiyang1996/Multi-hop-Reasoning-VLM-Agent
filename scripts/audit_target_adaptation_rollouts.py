#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import TransitionReceipt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed audit for frozen target adaptation rollouts"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--cell", default="alfworld_valid_unseen")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    expected = list(manifest["cells"][args.cell]["splits"]["adaptation"])
    rows = []
    failures = []
    for offset, task_id in enumerate(expected):
        path = args.input_dir / f"task_{offset}.json"
        if not path.is_file():
            failures.append(f"MISSING_TASK_{offset}")
            continue
        row = json.loads(path.read_text())
        row_failures = []
        if row.get("task_id") != task_id:
            row_failures.append("TASK_ID_MISMATCH")
        if row.get("collection_split") != "adaptation":
            row_failures.append("SPLIT_MISMATCH")
        if row.get("condition") != "BASE_DECISION_TARGET_ONLY":
            row_failures.append("CONDITION_MISMATCH")
        if row.get("harness_used") or row.get("source_motif_used"):
            row_failures.append("ASSISTANCE_PRESENT_IN_TARGET_ONLY")
        error = row.get("error")
        decision_failure = (
            isinstance(error, str)
            and error.startswith("ValueError:decision model ")
        )
        if error is not None and not decision_failure:
            row_failures.append("INFRASTRUCTURE_ROLLOUT_ERROR")
        receipts = [
            TransitionReceipt(**receipt)
            for receipt in row.get("transition_receipts", [])
        ]
        if not receipts or not all(receipt.validate() for receipt in receipts):
            row_failures.append("INVALID_OR_EMPTY_TRANSITION_RECEIPTS")
        metrics = row.get("metrics") or {}
        if metrics.get("steps") != len(receipts):
            row_failures.append("METRIC_STEP_COUNT_MISMATCH")
        failures.extend(f"TASK_{offset}_{failure}" for failure in row_failures)
        rows.append({
            "task_offset": offset,
            "task_id": task_id,
            "accepted": not row_failures,
            "failure_codes": row_failures,
            "steps": len(receipts),
            "official_success": metrics.get("official_success"),
            "official_score": metrics.get("official_score"),
            "repeated_actions": metrics.get("repeated_actions"),
            "no_observable_progress": metrics.get("no_observable_progress"),
            "decision_invalid_output": decision_failure,
        })
    payload = {
        "schema_version": 1,
        "authority": "MECHANICAL_TARGET_ADAPTATION_RECEIPT_AUDIT",
        "cell": args.cell,
        "expected_tasks": len(expected),
        "observed_tasks": len(rows),
        "all_accepted": not failures and len(rows) == len(expected),
        "failure_codes": sorted(set(failures)),
        "summary": {
            "success_counts": dict(sorted(Counter(
                str(row["official_success"]) for row in rows
            ).items())),
            "total_steps": sum(row["steps"] for row in rows),
            "mean_steps": (
                sum(row["steps"] for row in rows) / len(rows) if rows else None
            ),
        },
        "rows": rows,
        "claim_boundary": (
            "Adaptation receipts may train or propose target-native motifs. "
            "They are not held-out transfer results."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "all_accepted": payload["all_accepted"],
        **payload["summary"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    if not payload["all_accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
