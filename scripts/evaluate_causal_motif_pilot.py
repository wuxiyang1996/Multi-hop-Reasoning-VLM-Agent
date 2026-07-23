#!/usr/bin/env python3
"""Evaluate matched causal-motif target runs using official outcomes only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.causal_reasoning_motif import (  # noqa: E402
    MatchedEnvironmentOutcome,
    evaluate_matched_environment_contrasts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    outcomes = []
    ignored = []
    for path in args.result:
        payload = json.loads(path.read_text(encoding="utf-8"))
        treatment = str(payload.get("treatment") or "")
        if treatment not in {
            "authentic", "target_only", "generic_protocol",
            "shuffled_topology", "other_source",
        }:
            ignored.append({"path": str(path), "treatment": treatment})
            continue
        for row in payload.get("rows") or ():
            identity = row.get("matched_identity") or {}
            outcomes.append(MatchedEnvironmentOutcome(
                comparison_id=str(identity.get("comparison_id") or ""),
                treatment=treatment,
                initial_state_sha256=str(identity.get("initial_state_sha256") or ""),
                prefix_sha256=str(identity.get("prefix_sha256") or ""),
                policy_identity_sha256=str(identity.get("policy_identity_sha256") or ""),
                budget_sha256=str(identity.get("budget_sha256") or ""),
                official_success=bool(row.get("success")),
                official_score=float(row.get("cumulative_reward") or 0.0),
                valid_execution=(
                    row.get("error") is None
                    and all(
                        trace.get("policy_execution_identity", {}).get(
                            "execution_matches_policy", False,
                        )
                        for trace in row.get("traces") or ()
                    )
                ),
            ))
    report = evaluate_matched_environment_contrasts(
        outcomes, claim="target_incremental_value",
    )
    output = {
        "schema_version": 1,
        "report": asdict(report),
        "ignored_results": ignored,
        "gpt_verdict_used_as_outcome": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(output["report"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
