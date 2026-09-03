#!/usr/bin/env python3
"""Apply frozen same-Qwen V25 development advancement gates."""

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
    "same_model_direct": "direct_correct",
    "raw_typed_proof": "typed_proof_correct",
    "source_authentic": "source_authentic_correct",
    "shuffled_binding_control": "shuffled_binding_correct",
    "shuffled_topology_control": "shuffled_topology_correct",
}


def analyze(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    expected = int(config["expected_rows"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} V25 rows, got {len(rows)}")
    identities = {str(row["sample_id"]) for row in rows}
    clusters = {str(row["video_id"]) for row in rows}
    if len(identities) != expected or len(clusters) != 64:
        raise ValueError("V25 identities/video clusters are incomplete")
    if any(bool(row.get("runtime_saw_gold_or_official_structure", True)) for row in rows):
        raise ValueError("a V25 runtime branch saw forbidden supervision")
    if any(not bool(row.get("direct_and_proof_panels_identical")) for row in rows):
        raise ValueError("V25 direct/proof panels differ")
    vectors = {
        name: [bool(row[field]) for row in rows] for name, field in FIELDS.items()
    }
    direct = vectors["same_model_direct"]
    metrics_vs_direct = {
        name: paired_metrics(values, direct) for name, values in vectors.items()
    }
    source = vectors["source_authentic"]
    cluster_config = config["cluster_bootstrap"]
    clustered = cluster_metrics(
        rows,
        source,
        direct,
        resamples=int(cluster_config["resamples"]),
        seed=int(cluster_config["seed"]),
    )
    source_vs_controls = {
        name: paired_metrics(source, vectors[name])
        for name in (
            "raw_typed_proof", "shuffled_binding_control", "shuffled_topology_control",
        )
    }
    gates = config["development_advancement_gates"]
    source_direct = metrics_vs_direct["source_authentic"]
    gate_results = {
        "minimum_source_net_wins_vs_same_model_direct": (
            source_direct["net_wins"]
            >= int(gates["minimum_source_net_wins_vs_same_model_direct"])
        ),
        "maximum_source_vs_direct_question_exact_two_sided_p": (
            source_direct["exact_two_sided_p"]
            <= float(gates["maximum_source_vs_direct_question_exact_two_sided_p"])
        ),
        "source_vs_direct_video_cluster_bootstrap_lower_bound_must_exceed": (
            clustered["stratified_cluster_bootstrap"]["lower_95"]
            > float(gates["source_vs_direct_video_cluster_bootstrap_lower_bound_must_exceed"])
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
        "schema_version": 25,
        "status": "QUALIFIED_FOR_FRESH_CONFIRMATION" if all(gate_results.values()) else "NOT_QUALIFIED",
        "claim_boundary": config["claim_boundary"],
        "rows": len(rows),
        "video_clusters": len(clusters),
        "metrics_vs_same_model_direct": metrics_vs_direct,
        "source_vs_controls": source_vs_controls,
        "source_vs_direct_video_cluster_inference": clustered,
        "advancement_gate_results": gate_results,
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
        "source_vs_direct": report["metrics_vs_same_model_direct"]["source_authentic"],
        "cluster_bootstrap": report["source_vs_direct_video_cluster_inference"][
            "stratified_cluster_bootstrap"
        ],
        "failed_gates": [
            name for name, passed in report["advancement_gate_results"].items() if not passed
        ],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
