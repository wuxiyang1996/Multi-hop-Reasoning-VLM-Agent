#!/usr/bin/env python3
"""Apply frozen question- and video-cluster gates to V24 fresh STAR transfer."""

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

from analyze_natural_video_v19_formal import cluster_metrics, paired_metrics, sha256  # noqa: E402


FIELDS = {
    "primary": "primary_correct",
    "matched_generic_direct": "generic_direct_correct",
    "raw_typed_proof": "typed_proof_correct",
    "source_authentic": "source_authentic_correct",
    "shuffled_binding_control": "shuffled_binding_correct",
    "shuffled_topology_control": "shuffled_topology_correct",
}


def analyze(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    expected = int(config["benchmark"]["samples"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} V24 rows, got {len(rows)}")
    identities = {str(row["sample_id"]) for row in rows}
    clusters = {str(row["video_id"]) for row in rows}
    if len(identities) != expected or len(clusters) != int(config["benchmark"]["video_clusters"]):
        raise ValueError("V24 identity/cluster count mismatch")
    if any(str(row["family"]) != "Interaction" for row in rows):
        raise ValueError("V24 contains a non-Interaction row")
    if any(bool(row.get("runtime_saw_gold_or_official_structure", True)) for row in rows):
        raise ValueError("a V24 runtime branch saw forbidden supervision")
    if any(not bool(row.get("generic_and_proof_panels_identical")) for row in rows):
        raise ValueError("V24 matched neural controls did not see identical panels")

    vectors = {
        name: [bool(row[field]) for row in rows] for name, field in FIELDS.items()
    }
    source = vectors["source_authentic"]
    metrics = {
        name: {
            "vs_primary": paired_metrics(values, vectors["primary"]),
            "vs_matched_generic_direct": paired_metrics(
                values, vectors["matched_generic_direct"]
            ),
        }
        for name, values in vectors.items()
    }
    bootstrap = config["cluster_bootstrap"]
    source_vs_generic_clusters = cluster_metrics(
        rows,
        source,
        vectors["matched_generic_direct"],
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]),
    )
    source_vs_controls = {
        name: {
            "paired": paired_metrics(source, vectors[name]),
            "cluster": cluster_metrics(
                rows,
                source,
                vectors[name],
                resamples=int(bootstrap["resamples"]),
                seed=int(bootstrap["seed"]) + offset,
            ),
        }
        for offset, name in enumerate(
            ("raw_typed_proof", "shuffled_binding_control", "shuffled_topology_control"),
            start=1,
        )
    }
    gates = config["formal_gates"]
    source_primary = metrics["source_authentic"]["vs_primary"]
    source_generic = metrics["source_authentic"]["vs_matched_generic_direct"]
    gate_results = {
        "minimum_source_net_wins_vs_primary": (
            source_primary["net_wins"] >= int(gates["minimum_source_net_wins_vs_primary"])
        ),
        "minimum_source_net_wins_vs_matched_generic_direct": (
            source_generic["net_wins"]
            >= int(gates["minimum_source_net_wins_vs_matched_generic_direct"])
        ),
        "maximum_source_vs_generic_question_exact_two_sided_p": (
            source_generic["exact_two_sided_p"]
            <= float(gates["maximum_source_vs_generic_question_exact_two_sided_p"])
        ),
        "source_vs_generic_video_cluster_bootstrap_lower_bound_must_exceed": (
            source_vs_generic_clusters["stratified_cluster_bootstrap"]["lower_95"]
            > float(gates["source_vs_generic_video_cluster_bootstrap_lower_bound_must_exceed"])
        ),
        "minimum_positive_minus_negative_video_clusters_vs_generic": (
            source_vs_generic_clusters["positive"] - source_vs_generic_clusters["negative"]
            >= int(gates["minimum_positive_minus_negative_video_clusters_vs_generic"])
        ),
        "source_strictly_above_raw_typed_proof": (
            sum(source) > sum(vectors["raw_typed_proof"])
        ),
        "source_strictly_above_binding_control": (
            sum(source) > sum(vectors["shuffled_binding_control"])
        ),
        "source_strictly_above_topology_control": (
            sum(source) > sum(vectors["shuffled_topology_control"])
        ),
    }
    return {
        "schema_version": 24,
        "status": "PASS" if all(gate_results.values()) else "FAIL",
        "claim": (
            "adaptation-based fresh-cluster STAR Interaction transfer validated"
            if all(gate_results.values())
            else "adaptation-based fresh-cluster STAR Interaction transfer not validated"
        ),
        "claim_boundary": config["claim_boundary"],
        "rows": len(rows),
        "video_clusters": len(clusters),
        "condition_metrics": metrics,
        "source_vs_generic_video_cluster_inference": source_vs_generic_clusters,
        "source_vs_controls": source_vs_controls,
        "formal_gate_results": gate_results,
        "all_formal_gates_passed": all(gate_results.values()),
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
        "claim": report["claim"],
        "source_vs_primary": report["condition_metrics"]["source_authentic"]["vs_primary"],
        "source_vs_generic": report["condition_metrics"]["source_authentic"][
            "vs_matched_generic_direct"
        ],
        "source_vs_generic_cluster_bootstrap": report[
            "source_vs_generic_video_cluster_inference"
        ]["stratified_cluster_bootstrap"],
        "failed_gates": [
            name for name, passed in report["formal_gate_results"].items() if not passed
        ],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
