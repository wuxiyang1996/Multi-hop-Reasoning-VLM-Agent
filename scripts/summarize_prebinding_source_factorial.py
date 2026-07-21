#!/usr/bin/env python3
"""Audit a paired pre-binding E/S/W/R × Harness development pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _metrics(payload):
    usages = []
    calls = {"action": 0, "contract": 0, "rebind": 0}
    for row in payload["rows"]:
        for trace in row["traces"]:
            for actor in trace.get("actor_rows") or ():
                calls["action"] += 1
                usages.append(actor.get("usage") or {})
            if trace.get("contract_agent"):
                calls["contract"] += 1
                usages.append(trace["contract_agent"].get("usage") or {})
            if trace.get("rebind_agent"):
                calls["rebind"] += 1
                usages.append(trace["rebind_agent"].get("usage") or {})
    successes = [bool(row["success"]) for row in payload["rows"]]
    return {
        "n": len(successes),
        "n_success": sum(successes),
        "success_rate": sum(successes) / len(successes),
        "success_vector": [int(item) for item in successes],
        "mean_steps": mean(row["steps"] for row in payload["rows"]),
        "n_error": sum(row["error"] is not None for row in payload["rows"]),
        "n_abstain": sum(row["abstain_reason"] is not None for row in payload["rows"]),
        "calls": calls,
        "prompt_tokens": sum(row.get("prompt_tokens", 0) for row in usages),
        "completion_tokens": sum(row.get("completion_tokens", 0) for row in usages),
        "total_tokens": sum(row.get("total_tokens", 0) for row in usages),
        "reported_cost": sum(row.get("cost", 0.0) for row in usages),
    }


def _validate(label, payload):
    treatment, harness = label.rsplit("_", 1)
    expected_condition = "target_only" if treatment == "empty" else "source"
    if payload.get("condition") != expected_condition:
        raise ValueError(f"{label}: condition mismatch")
    if payload.get("source_treatment") != treatment:
        raise ValueError(f"{label}: source treatment mismatch")
    online = bool(payload.get("online_source_control"))
    shadow = bool(payload.get("shadow_source_control"))
    expected_modes = (
        (False, False) if treatment == "empty"
        else (True, False) if harness == "on"
        else (False, True)
    )
    if (online, shadow) != expected_modes:
        raise ValueError(f"{label}: Harness/shadow state mismatch")
    if payload.get("source_control_applied_before_binding_generation") is not True:
        raise ValueError(f"{label}: source control was not applied before binding")
    if not payload.get("source_control_receipt_sha256"):
        raise ValueError(f"{label}: missing source control receipt")
    if not payload.get("rows"):
        raise ValueError(f"{label}: empty result")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empty-off", type=Path, required=True)
    for treatment in ("correct", "wrong", "renamed"):
        parser.add_argument(f"--{treatment}-off", type=Path, required=True)
        parser.add_argument(f"--{treatment}-on", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "empty_off": args.empty_off,
        "correct_off": args.correct_off,
        "correct_on": args.correct_on,
        "wrong_off": args.wrong_off,
        "wrong_on": args.wrong_on,
        "renamed_off": args.renamed_off,
        "renamed_on": args.renamed_on,
    }
    payloads = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in paths.items()
    }
    for label, payload in payloads.items():
        _validate(label, payload)
    counts = {len(payload["rows"]) for payload in payloads.values()}
    if len(counts) != 1:
        raise ValueError("unmatched row counts")
    n = next(iter(counts))
    for index in range(n):
        identities = {
            payload["rows"][index]["target_instance_identity"]["identity_sha256"]
            for payload in payloads.values()
        }
        if len(identities) != 1:
            raise ValueError(f"target instance mismatch at pair {index}")
    protocol_keys = ("model", "config_sha256", "max_steps", "registered_call_caps")
    for key in protocol_keys:
        if len({json.dumps(payload.get(key), sort_keys=True) for payload in payloads.values()}) != 1:
            raise ValueError(f"unmatched protocol field: {key}")
    metrics = {label: _metrics(payload) for label, payload in payloads.items()}
    source_signal = {
        "correct_on_minus_empty_off": (
            metrics["correct_on"]["success_rate"]
            - metrics["empty_off"]["success_rate"]
        ),
        "correct_on_minus_wrong_on": (
            metrics["correct_on"]["success_rate"]
            - metrics["wrong_on"]["success_rate"]
        ),
        "correct_on_minus_renamed_on": (
            metrics["correct_on"]["success_rate"]
            - metrics["renamed_on"]["success_rate"]
        ),
    }
    harness_effect = {
        treatment: (
            metrics[f"{treatment}_on"]["success_rate"]
            - metrics[f"{treatment}_off"]["success_rate"]
        )
        for treatment in ("correct", "wrong", "renamed")
    }
    realized_calls_matched = len({
        json.dumps(row["calls"], sort_keys=True) for row in metrics.values()
    }) == 1
    action_sensitivity = {}
    for label in ("correct_off", "correct_on", "wrong_off", "wrong_on", "renamed_off", "renamed_on"):
        exact = [
            payloads[label]["rows"][index]["actions"]
            == payloads["empty_off"]["rows"][index]["actions"]
            for index in range(n)
        ]
        action_sensitivity[label] = {
            "exact_action_sequence_equal_to_empty_count": sum(exact),
            "exact_action_sequence_differs_from_empty_count": n - sum(exact),
        }
    report = {
        "schema_version": 1,
        "report_role": "development_source_dependence_factorial_not_causal_result",
        "n_pairs": n,
        "all_target_instance_identities_paired": True,
        "source_control_stage": "before_binding_generation",
        "paper_condition_labels": {
            "correct": "designated_source_not_semantically_prevalidated",
            "wrong": "cross_game_source_control_not_semantically_prevalidated",
            "renamed": "content_independent_identity_renaming_control",
        },
        "metrics": metrics,
        "source_signal_deltas": source_signal,
        "harness_effect_deltas": harness_effect,
        "native_action_sensitivity_to_source_treatment": action_sensitivity,
        "registered_caps_matched": True,
        "realized_calls_matched": realized_calls_matched,
        "realized_tokens_matched": len({
            row["total_tokens"] for row in metrics.values()
        }) == 1,
        "positive_source_ordering_observed": all(value > 0 for value in source_signal.values()),
        "authorizes_large_scale_2x4": False,
        "claim_limit": (
            "This audit reports paired development contrasts only. A preregistered held-out "
            "threshold and uncertainty interval are still required; unequal realized calls "
            "must be reported and may prevent a source-causal interpretation. The protocol "
            "enums correct/wrong denote designated/cross-game treatments, not known semantic "
            "correctness."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
