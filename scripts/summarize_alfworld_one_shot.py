#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


CONDITIONS = ("target_only", "generic_protocol", "authentic", "shuffled_topology", "other_source")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for condition in CONDITIONS:
        path = args.run_dir / f"{condition}.json"
        artifact = json.loads(path.read_text(encoding="utf-8"))
        matching = [row for row in artifact["rows"] if row["condition"] == condition]
        if len(matching) != 1:
            raise ValueError(f"{path} does not contain exactly one {condition} row")
        rows[condition] = matching[0]
    initial_hashes = {row["initial_state_hash"] for row in rows.values()}
    if None in initial_hashes or len(initial_hashes) != 1:
        raise ValueError(f"conditions are not initial-state matched: {initial_hashes}")
    if any(row["error"] is not None for row in rows.values()):
        raise ValueError("at least one condition has an error")
    metrics = {condition: row["metrics"] for condition, row in rows.items()}
    authentic_step_delta = metrics["authentic"]["steps"] - metrics["target_only"]["steps"]
    all_success = all(row["official_success"] for row in metrics.values())
    verdict = "INCONCLUSIVE_NO_ATTRIBUTABLE_SUCCESS_SEPARATION"
    if all_success and authentic_step_delta > 0:
        verdict = "NO_POSITIVE_VALUE_WITH_NEGATIVE_EFFICIENCY_SIGNAL"
    summary = {
        "schema_version": 1,
        "initial_state_match": True,
        "initial_state_hash": next(iter(initial_hashes)),
        "conditions_complete": True,
        "source_qualification_at_run": "GENERIC_ONLY_NOT_SOURCE_SUPPORTED",
        "one_shot_online_execution_reached": True,
        "metrics": metrics,
        "authentic_minus_target_only_steps": authentic_step_delta,
        "verdict": verdict,
        "claim_limit": (
            "The one-shot Harness mechanism executed online in ALFWorld. One matched episode cannot "
            "establish transfer, and authentic source content did not outperform controls."
        ),
        "actions": {condition: row["actions"] for condition, row in rows.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
