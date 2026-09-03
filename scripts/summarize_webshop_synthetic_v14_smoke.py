#!/usr/bin/env python3
"""Summarize only complete paired groups from the V14 development smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.webshop_constraint_coverage_v14 import (  # noqa: E402
    audit_receipt_commits,
)
from motif_transfer.webshop_sokoban_effect_transfer import CONDITIONS  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_cost(path: Path) -> tuple[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", {})
    return len(entries), sum(
        float((row.get("usage") or {}).get("cost") or 0.0)
        for row in entries.values()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO / "runs/webshop_synthetic_unique_v14_development_smoke_v1",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "configs/webshop_synthetic_unique_v14_frozen.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "docs/results/webshop_synthetic_v14_development_smoke_v1.json",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    intended = [row["task_id"] for row in manifest["roles"]["development"][:4]]
    complete = [
        task_id for task_id in intended
        if all((args.run_dir / f"{task_id}.{condition}.json").exists()
               for condition in CONDITIONS)
    ]
    receipts: list[dict[str, Any]] = []
    for task_id in complete:
        receipts.extend(
            json.loads((args.run_dir / f"{task_id}.{condition}.json").read_text(
                encoding="utf-8"
            ))
            for condition in CONDITIONS
        )
    condition_metrics = {}
    for condition in CONDITIONS:
        rows = [row for row in receipts if row["condition"] == condition]
        condition_metrics[condition] = {
            "tasks": len(rows),
            "strict_successes": sum(row["strict_success"] for row in rows),
            "rewards": [row["official_reward"] for row in rows],
            "source_decisions": sum(row["source_decision_count"] for row in rows),
            "changed_from_target_rank_zero": sum(
                row["changed_from_target_rank_zero_count"] for row in rows
            ),
            "failures": sum(row["failure"] is not None for row in rows),
        }
    cache_calls = 0
    provider_cost = 0.0
    cache_hashes = {}
    for task_id in complete:
        path = args.run_dir / "attempts" / task_id / "attempt_0/decision_cache.json"
        calls, cost = _cache_cost(path)
        cache_calls += calls
        provider_cost += cost
        cache_hashes[task_id] = file_sha256(path)
    authentic = [
        row for row in receipts
        if row["condition"] == "authentic_sokoban_effect_plus_target"
    ]
    commit_audits = {
        row["task_id"]: audit_receipt_commits(row) for row in authentic
    }
    artifact = {
        "schema_version": 1,
        "status": "WEBSHOP_V14_DEVELOPMENT_PREFLIGHT_FAILED_ZERO_SOURCE_APPLICABILITY",
        "role": "DEVELOPMENT_DIAGNOSIS_ONLY",
        "formal_reserve_opened": False,
        "intended_smoke_tasks": intended,
        "complete_paired_tasks": complete,
        "incomplete_tasks_excluded": sorted(set(intended) - set(complete)),
        "stop_rule": (
            "Stopped after two complete six-condition groups both produced zero authentic "
            "source decisions; did not spend on the remaining smoke tasks."
        ),
        "conditions": condition_metrics,
        "authentic_total_source_decisions": sum(
            row["source_decision_count"] for row in authentic
        ),
        "coverage_counterfactual_audit": commit_audits,
        "diagnosis": [
            "The synthetic server and BrowserGym transitions work end to end.",
            "The frozen V13 target MLP never opened authentic source authority on either complete new goal.",
            "On webshop.1, repeated direct radio clicks were no-ops and target rank zero committed with both required options unverified, yielding reward 5/7 rather than strict success.",
            "The next repair is target-native label-action completion plus set coverage, tested against coverage-only; changing the source threshold is not justified.",
        ],
        "next_gate": (
            "Require nonzero action contrast and authentic improvement over both target-only "
            "and target-native coverage-only on development before opening formal reserve."
        ),
        "operational": {
            "cached_provider_calls": cache_calls,
            "estimated_provider_cost_usd": provider_cost,
            "complete_receipts": len(receipts),
            "final_failures": sum(row["failure"] is not None for row in receipts),
        },
        "runtime_hashes": {
            "manifest": file_sha256(args.manifest),
            "decision_caches": cache_hashes,
            "summarizer": file_sha256(Path(__file__)),
        },
    }
    artifact["artifact_sha256"] = stable_hash(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
