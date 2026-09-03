#!/usr/bin/env python3
"""Apply the frozen V23 independent-candidate pilot advancement gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_natural_video_v19_formal import paired_metrics, sha256  # noqa: E402


FIELDS = {
    "primary": "primary_correct",
    "generic_direct": "generic_direct_correct",
    "joint_typed_proof": "joint_typed_proof_correct",
    "independent_executor": "independent_executor_correct",
    "source_authentic": "source_authentic_correct",
    "shuffled_binding_control": "shuffled_binding_control_correct",
    "shuffled_topology_control": "shuffled_topology_control_correct",
}


def _metrics(
    rows: Sequence[Mapping[str, Any]], indices: Sequence[int], baseline: str,
) -> dict[str, Any]:
    reference = [bool(rows[i][FIELDS[baseline]]) for i in indices]
    return {
        name: paired_metrics(
            [bool(rows[i][field]) for i in indices], reference,
        )
        for name, field in FIELDS.items()
    }


def analyze(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    if len(rows) != 28:
        raise ValueError("V23 pilot requires one row from each of 28 video clusters")
    identities = {(str(row["benchmark"]), str(row["sample_id"])) for row in rows}
    clusters = {(str(row["benchmark"]), str(row["video_id"])) for row in rows}
    if len(identities) != 28 or len(clusters) != 28:
        raise ValueError("V23 pilot identities/clusters are not one-to-one")
    if any(not bool(row.get("each_neural_call_saw_exactly_one_candidate")) for row in rows):
        raise ValueError("a V23 neural call was not candidate-isolated")
    if any(bool(row.get("runtime_saw_gold_or_official_structure", True)) for row in rows):
        raise ValueError("a V23 runtime branch saw forbidden supervision")
    pooled_generic = _metrics(rows, list(range(len(rows))), "generic_direct")
    pooled_primary = _metrics(rows, list(range(len(rows))), "primary")
    by_benchmark = {
        benchmark: _metrics(
            rows,
            [i for i, row in enumerate(rows) if str(row["benchmark"]) == benchmark],
            "generic_direct",
        )
        for benchmark in ("star", "nextqa")
    }
    gates = config["pilot_advancement_gates"]
    source = pooled_generic["source_authentic"]
    gate_results = {
        "minimum_source_authentic_net_wins_vs_generic_direct": (
            source["net_wins"]
            >= int(gates["minimum_source_authentic_net_wins_vs_generic_direct"])
        ),
        "source_authentic_not_below_generic_each_benchmark": all(
            metrics["source_authentic"]["correct"]
            >= metrics["source_authentic"]["baseline_correct"]
            for metrics in by_benchmark.values()
        ),
        "source_authentic_strictly_above_binding_control": (
            pooled_generic["source_authentic"]["correct"]
            > pooled_generic["shuffled_binding_control"]["correct"]
        ),
        "source_authentic_strictly_above_topology_control": (
            pooled_generic["source_authentic"]["correct"]
            > pooled_generic["shuffled_topology_control"]["correct"]
        ),
        "independent_executor_not_below_generic_direct": (
            pooled_generic["independent_executor"]["correct"]
            >= pooled_generic["generic_direct"]["correct"]
        ),
    }
    return {
        "schema_version": 23,
        "status": "ADVANCE_TO_EXPANDED_DEVELOPMENT" if all(gate_results.values()) else "DO_NOT_ADVANCE",
        "claim_boundary": config["claim_boundary"],
        "rows": len(rows),
        "video_clusters": len(clusters),
        "metrics_vs_generic_direct": pooled_generic,
        "metrics_vs_primary": pooled_primary,
        "by_benchmark_vs_generic_direct": by_benchmark,
        "advancement_gates": gate_results,
        "all_advancement_gates_passed": all(gate_results.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--receipts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = json.loads(args.receipts.read_text(encoding="utf-8"))
    report = analyze(rows, config)
    report["artifacts"] = {
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "receipts": str(args.receipts.resolve()),
        "receipts_sha256": sha256(args.receipts),
        "analyzer": str(Path(__file__).resolve()),
        "analyzer_sha256": sha256(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "source_vs_generic": report["metrics_vs_generic_direct"]["source_authentic"],
        "executor_vs_generic": report["metrics_vs_generic_direct"]["independent_executor"],
        "failed_gates": [
            name for name, passed in report["advancement_gates"].items() if not passed
        ],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
