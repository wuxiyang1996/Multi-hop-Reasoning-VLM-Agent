#!/usr/bin/env python3
"""Analyze matched wrapper-grounded STAR Interaction development branches."""

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

from analyze_natural_video_v19_formal import (  # noqa: E402
    cluster_metrics,
    paired_metrics,
    sha256,
)


FIELDS = {
    "uniform_v25_direct": "uniform_v25_direct_correct",
    "uniform_v25_source": "uniform_v25_source_correct",
    "active_generic": "active_generic_correct",
    "active_raw_typed_proof": "active_typed_proof_correct",
    "active_source": "active_source_correct",
    "active_binding_control": "active_binding_control_correct",
    "active_topology_control": "active_topology_control_correct",
}


def analyze(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    expected = int(config["expected_rows"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} V26 rows, got {len(rows)}")
    identities = {str(row["sample_id"]) for row in rows}
    clusters = {str(row["video_id"]) for row in rows}
    if len(identities) != expected or len(clusters) != int(config["expected_clusters"]):
        raise ValueError("V26 identities/video clusters are incomplete")
    if any(bool(row.get("runtime_saw_gold_or_official_structure", True)) for row in rows):
        raise ValueError("a V26 runtime branch saw forbidden supervision")
    if any(not bool(row.get("active_direct_and_proof_panels_identical")) for row in rows):
        raise ValueError("V26 active direct/proof panels differ")
    expected_protocol = str(config["grounding_contract"]["protocol"])
    for row in rows:
        receipt = row.get("transition_grounding_receipt") or {}
        if receipt.get("protocol") != expected_protocol:
            raise ValueError("V26 transition-grounding protocol drift")
        if int(receipt.get("pair_count", -1)) != int(config["media"]["transition_pair_count"]):
            raise ValueError("V26 transition pair-count drift")
        if receipt.get("tool_sequence", [None])[0] != "detect_scene_changes":
            raise ValueError("V26 did not execute the required grounding tool first")
        if len(receipt.get("comparisons") or ()) != int(config["media"]["transition_pair_count"]):
            raise ValueError("V26 compare_frames receipt count drift")

    vectors = {
        name: [bool(row[field]) for row in rows] for name, field in FIELDS.items()
    }
    active_generic = vectors["active_generic"]
    active_source = vectors["active_source"]
    uniform_direct = vectors["uniform_v25_direct"]
    uniform_source = vectors["uniform_v25_source"]
    active_vs_generic = {
        name: paired_metrics(values, active_generic)
        for name, values in vectors.items()
    }
    evidence_view_effect = paired_metrics(active_generic, uniform_direct)
    source_protocol_effect_vs_v25 = paired_metrics(active_source, uniform_source)
    source_vs_controls = {
        name: paired_metrics(active_source, vectors[name])
        for name in (
            "active_raw_typed_proof",
            "active_binding_control",
            "active_topology_control",
        )
    }
    cluster_config = config["cluster_bootstrap"]
    clustered = cluster_metrics(
        rows,
        active_source,
        active_generic,
        resamples=int(cluster_config["resamples"]),
        seed=int(cluster_config["seed"]),
    )
    gates = config["development_advancement_gates"]
    source_direct = active_vs_generic["active_source"]
    gate_results = {
        "active_source_minimum_net_wins_vs_active_generic": (
            source_direct["net_wins"]
            >= int(gates["active_source_minimum_net_wins_vs_active_generic"])
        ),
        "active_source_maximum_exact_two_sided_p": (
            source_direct["exact_two_sided_p"]
            <= float(gates["active_source_maximum_exact_two_sided_p"])
        ),
        "active_source_cluster_bootstrap_lower_must_exceed": (
            clustered["stratified_cluster_bootstrap"]["lower_95"]
            > float(gates["active_source_cluster_bootstrap_lower_must_exceed"])
        ),
        "active_source_strictly_above_raw_typed_proof": (
            sum(active_source) > sum(vectors["active_raw_typed_proof"])
        ),
        "active_source_strictly_above_binding_control": (
            sum(active_source) > sum(vectors["active_binding_control"])
        ),
        "active_source_strictly_above_topology_control": (
            sum(active_source) > sum(vectors["active_topology_control"])
        ),
        "active_generic_not_below_uniform_direct": (
            evidence_view_effect["net_wins"]
            >= int(gates["active_generic_minimum_net_wins_vs_uniform_direct"])
        ),
    }
    return {
        "schema_version": 26,
        "status": (
            "QUALIFIED_FOR_FRESH_ACTIVE_GROUNDING_CONFIRMATION"
            if all(gate_results.values())
            else "NOT_QUALIFIED"
        ),
        "claim_boundary": config["claim_boundary"],
        "rows": len(rows),
        "video_clusters": len(clusters),
        "metrics_vs_matched_active_generic": active_vs_generic,
        "active_evidence_view_vs_uniform_v25_direct": evidence_view_effect,
        "active_source_vs_uniform_v25_source": source_protocol_effect_vs_v25,
        "active_source_vs_controls": source_vs_controls,
        "active_source_vs_generic_cluster_inference": clustered,
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
        "active_source_vs_generic": report[
            "metrics_vs_matched_active_generic"
        ]["active_source"],
        "active_view_vs_uniform": report["active_evidence_view_vs_uniform_v25_direct"],
        "cluster_bootstrap": report["active_source_vs_generic_cluster_inference"][
            "stratified_cluster_bootstrap"
        ],
        "failed_gates": [
            name
            for name, passed in report["advancement_gate_results"].items()
            if not passed
        ],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
