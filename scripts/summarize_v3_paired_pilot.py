#!/usr/bin/env python3
"""Summarize a paired source-vs-target-only v3 development pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _initial_hash(row):
    return next(
        event["payload"]["observation_sha256"]
        for event in row["reasoning_event_log"]["events"]
        if event["kind"] == "OBSERVATION"
    )


def _metrics(payload):
    rows = payload["rows"]
    actor_rows = [
        actor for row in rows for trace in row["traces"] for actor in trace["actor_rows"]
    ]
    usages = [
        actor.get("usage") or {} for actor in actor_rows
    ]
    reasons = {}
    for row in rows:
        key = row["abstain_reason"] or "NONE"
        reasons[key] = reasons.get(key, 0) + 1
    return {
        "n": len(rows),
        "n_success": sum(bool(row["success"]) for row in rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
        "mean_steps": mean(row["steps"] for row in rows),
        "n_abstain": sum(row["abstain_reason"] is not None for row in rows),
        "n_error": sum(row["error"] is not None for row in rows),
        "actor_calls": len(usages),
        "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in usages),
        "completion_tokens": sum(item.get("completion_tokens", 0) for item in usages),
        "openrouter_reported_cost": sum(item.get("cost", 0.0) for item in usages),
        "actor_calls_with_source_conditioning": sum(
            int(actor.get("n_source_conditioning", 0)) > 0 for actor in actor_rows
        ),
        "source_conditioning_entries_shown": sum(
            int(actor.get("n_source_conditioning", 0)) for actor in actor_rows
        ),
        "abstain_reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target-only", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    target = json.loads(args.target_only.read_text(encoding="utf-8"))
    source_rows, target_rows = source["rows"], target["rows"]
    if len(source_rows) != len(target_rows):
        raise SystemExit("paired pilot row count mismatch")
    paired = [
        _initial_hash(left) == _initial_hash(right)
        for left, right in zip(source_rows, target_rows)
    ]
    source_metrics, target_metrics = _metrics(source), _metrics(target)
    advantage = source_metrics["success_rate"] - target_metrics["success_rate"]
    source_treatment_active = (
        source_metrics["actor_calls"] > 0
        and source_metrics["actor_calls_with_source_conditioning"]
        == source_metrics["actor_calls"]
        and target_metrics["actor_calls_with_source_conditioning"] == 0
    )
    paired_outcomes = [
        bool(left["success"]) == bool(right["success"])
        for left, right in zip(source_rows, target_rows)
    ]
    source_only_successes = sum(
        bool(left["success"]) and not bool(right["success"])
        for left, right in zip(source_rows, target_rows)
    )
    target_only_successes = sum(
        bool(right["success"]) and not bool(left["success"])
        for left, right in zip(source_rows, target_rows)
    )
    report = {
        "schema_version": 2,
        "report_role": "development_pilot_not_held_out_result",
        "paired_initial_observations": sum(paired),
        "n_pairs": len(paired),
        "all_initial_observations_paired": all(paired),
        "paired_outcomes_equal": sum(paired_outcomes),
        "source_only_success_pairs": source_only_successes,
        "target_only_success_pairs": target_only_successes,
        "source_treatment_active": source_treatment_active,
        "source": source_metrics,
        "target_only": target_metrics,
        "source_success_rate_minus_target_only": advantage,
        "development_conclusion": (
            "SOURCE_ADVANTAGE_OBSERVED" if advantage > 0
            else "NO_SOURCE_ADVANTAGE_IN_DEV_PILOT"
        ),
        "authorizes_large_scale_2x4": False,
        "large_scale_blocker": (
            "Source receipts reached every source Actor call but produced no positive "
            "source-vs-target-only development signal. Do not scale 2x4; next test a "
            "different source domain/program family or improve Agent-side use of receipts "
            "without adding semantic heuristics."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
