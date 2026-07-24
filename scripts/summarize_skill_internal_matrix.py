#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact audit of a full source matrix")
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_bytes = args.report.read_bytes()
    report = json.loads(raw_bytes)
    by_condition = defaultdict(Counter)
    error_types = Counter()
    usage = Counter()
    eligible = []
    model_calls = 0
    for skill in report["skills"]:
        runs = [(row["condition"], row) for row in skill["conditions"]]
        if "skill_off_control" in skill:
            runs.append(("skill_off", skill["skill_off_control"]))
        for label, run in runs:
            model_calls += 1
            by_condition[label]["calls"] += 1
            by_condition[label]["candidates"] += len(run.get("candidates", []))
            by_condition[label]["backbone_eligible"] += sum(
                audit.get("backbone_eligible", False) for audit in run.get("audits", [])
            )
            if run.get("model_error"):
                by_condition[label]["model_errors"] += 1
                error_types[run["model_error"].split(":", 1)[-1]] += 1
            for key, value in run.get("agent_call", {}).get("usage", {}).items():
                if isinstance(value, int):
                    usage[key] += value
            for index, audit in enumerate(run.get("audits", [])):
                if not audit.get("backbone_eligible"):
                    continue
                candidate = run["candidates"][index]

                def topology_fit(rows):
                    return [
                        {
                            "accepted": item.get("audit", {}).get("accepted"),
                            "node_coverage": item.get("audit", {}).get("node_coverage"),
                            "edge_recurrence": item.get("audit", {}).get("edge_recurrence"),
                            "error": item.get("error"),
                        }
                        for item in rows
                    ]

                eligible.append({
                    "game": skill["game"],
                    "skill_id": skill["skill_id"],
                    "condition": label,
                    "graph_id": candidate["graph_id"],
                    "nodes": len(candidate["nodes"]),
                    "edges": len(candidate["edges"]),
                    "qualification_topology_fit": topology_fit(run.get("qualification", [])),
                    "held_out_topology_fit": topology_fit(run.get("held_out", [])),
                })
    compact = {
        "schema_version": "SKILL_INTERNAL_MATRIX_V1_COMPACT_AUDIT",
        "source_report_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "model_identity": report.get("model_identity"),
        "totals": {
            "skills": len(report["skills"]),
            "model_calls": model_calls,
            "model_errors": sum(error_types.values()),
            "backbone_eligible_all_conditions": len(eligible),
            "backbone_eligible_authentic": sum(
                row["condition"] == "authentic" for row in eligible
            ),
            "usage": dict(usage),
        },
        "by_condition": {
            key: dict(value) for key, value in sorted(by_condition.items())
        },
        "error_types": dict(error_types),
        "eligible_candidates": eligible,
        "interpretation": [
            "No authentic condition produced a backbone-eligible candidate.",
            "All eligible candidates came from masked, receipt-only, shuffled, or skill-off controls.",
            "Qualification and held-out alignment are topology-fit diagnostics, not blind prediction accuracy.",
            "No candidate is SOURCE_SUPPORTED and none should be transferred to a target domain.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(compact["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
