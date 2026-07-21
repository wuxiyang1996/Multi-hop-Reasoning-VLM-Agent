#!/usr/bin/env python3
"""Audit a paired B/N/H/R online-transfer development matrix."""

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
    usages = []
    actor_calls = rebind_calls = 0
    for row in rows:
        for trace in row["traces"]:
            for actor in trace.get("actor_rows") or ():
                actor_calls += 1
                usages.append(actor.get("usage") or {})
            if trace.get("rebind_agent"):
                rebind_calls += 1
                usages.append(trace["rebind_agent"].get("usage") or {})
    successes = [bool(row["success"]) for row in rows]
    return {
        "n": len(rows),
        "n_success": sum(successes),
        "success_rate": sum(successes) / len(rows),
        "mean_steps": mean(row["steps"] for row in rows),
        "n_abstain": sum(row["abstain_reason"] is not None for row in rows),
        "n_error": sum(row["error"] is not None for row in rows),
        "actor_calls": actor_calls,
        "rebind_agent_calls": rebind_calls,
        "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in usages),
        "completion_tokens": sum(item.get("completion_tokens", 0) for item in usages),
        "total_tokens": sum(item.get("total_tokens", 0) for item in usages),
        "reported_cost": sum(item.get("cost", 0.0) for item in usages),
        "success_vector": [int(value) for value in successes],
    }


def _validate_protocol(label, payload):
    expected = {
        "B": ("target_only", False, "none"),
        "N": ("source", False, "none"),
        "H": ("source", True, "none"),
        "R": ("source", True, "rotate"),
    }[label]
    actual = (
        payload.get("condition"),
        bool(payload.get("online_source_control", False)),
        str(payload.get("source_conditioning_control", "none")),
    )
    if actual != expected:
        raise ValueError(f"{label} protocol mismatch: {actual} != {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-only", type=Path, required=True)
    parser.add_argument("--naive-source", type=Path, required=True)
    parser.add_argument("--online-harness", type=Path, required=True)
    parser.add_argument("--randomized-conditioning", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "B": args.target_only,
        "N": args.naive_source,
        "H": args.online_harness,
        "R": args.randomized_conditioning,
    }
    payloads = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    for key, payload in payloads.items():
        _validate_protocol(key, payload)
    counts = {len(payload["rows"]) for payload in payloads.values()}
    if len(counts) != 1 or not counts or next(iter(counts)) == 0:
        raise ValueError("matrix row counts are empty or unmatched")
    n = next(iter(counts))
    paired = [
        len({_initial_hash(payloads[key]["rows"][index]) for key in payloads}) == 1
        for index in range(n)
    ]
    if not all(paired):
        raise ValueError("matrix initial observation hashes are not paired")
    model_set = {payload.get("model") for payload in payloads.values()}
    max_steps_set = {payload.get("max_steps") for payload in payloads.values()}
    if len(model_set) != 1 or len(max_steps_set) != 1:
        raise ValueError("matrix model/max-steps protocol mismatch")
    metrics = {key: _metrics(payload) for key, payload in payloads.items()}
    b, n_metric, h, r = (metrics[key] for key in ("B", "N", "H", "R"))
    mitigation_rate = None
    if b["success_rate"] > n_metric["success_rate"]:
        mitigation_rate = (
            (h["success_rate"] - n_metric["success_rate"])
            / (b["success_rate"] - n_metric["success_rate"])
        )
    report = {
        "schema_version": 1,
        "report_role": "development_matrix_not_held_out_causal_result",
        "conditions": {
            "B": "target_only",
            "N": "naive_source",
            "H": "verified_source_plus_online_harness",
            "R": "rotated_conditioning_plus_online_harness",
        },
        "randomized_control_scope": (
            "R rotates untrusted source-conditioning payloads across frozen candidate identities; "
            "it does not randomize admitted target program topology."
        ),
        "n_pairs": n,
        "all_initial_observations_paired": all(paired),
        "model": next(iter(model_set)),
        "max_steps": next(iter(max_steps_set)),
        "metrics": metrics,
        "success_rate_deltas": {
            "N_minus_B": n_metric["success_rate"] - b["success_rate"],
            "H_minus_B": h["success_rate"] - b["success_rate"],
            "H_minus_N": h["success_rate"] - n_metric["success_rate"],
            "H_minus_R": h["success_rate"] - r["success_rate"],
        },
        "mitigation_rate": mitigation_rate,
        "success_negative_transfer_observed": n_metric["success_rate"] < b["success_rate"],
        "source_cost_overhead_without_success_gain": {
            key: (
                metrics[key]["reported_cost"] > b["reported_cost"]
                and metrics[key]["success_rate"] <= b["success_rate"]
            ) for key in ("N", "H", "R")
        },
        "actual_token_consumption_matched": len({
            metrics[key]["total_tokens"] for key in metrics
        }) == 1,
        "development_conclusion": (
            "POSITIVE_ONLINE_SOURCE_SIGNAL"
            if h["success_rate"] > max(b["success_rate"], r["success_rate"])
            else "NO_POSITIVE_ONLINE_SOURCE_SIGNAL"
        ),
        "authorizes_large_scale_2x4": False,
        "claim_limit": (
            "Four development pairs have no inferential power. Unequal actual prompt-token "
            "consumption prevents attributing differences solely to source content."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
