#!/usr/bin/env python3
"""Analyze the matched uniform/active x direct/source STAR factorial."""

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


FIELDS = (
    "uniform_direct_correct",
    "uniform_typed_proof_correct",
    "uniform_source_correct",
    "uniform_binding_control_correct",
    "uniform_topology_control_correct",
    "active_direct_correct",
    "active_typed_proof_correct",
    "active_source_correct",
    "active_binding_control_correct",
    "active_topology_control_correct",
)


def analyze(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    expected = int(config["expected_rows"])
    clusters = {str(row["video_id"]) for row in rows}
    if len(rows) != expected or len({str(row["sample_id"]) for row in rows}) != expected:
        raise ValueError("V27 identities are incomplete")
    if len(clusters) != int(config["expected_clusters"]):
        raise ValueError("V27 video clusters are incomplete")
    if any(bool(row.get("runtime_saw_gold_or_official_structure", True)) for row in rows):
        raise ValueError("a V27 runtime branch saw forbidden supervision")
    expected_pairs = int(config["media"]["transition_pair_count"])
    expected_protocol = str(config["grounding_contract"]["protocol"])
    for row in rows:
        if not bool(row.get("within_view_direct_and_proof_panels_identical")):
            raise ValueError("V27 within-view panels differ")
        if not bool(row.get("uniform_and_active_use_same_proxy_frames")):
            raise ValueError("V27 views use different proxy frames")
        receipt = row.get("transition_grounding_receipt") or {}
        if receipt.get("protocol") != expected_protocol:
            raise ValueError("V27 grounding protocol drift")
        if len(receipt.get("comparisons") or ()) != expected_pairs:
            raise ValueError("V27 compare_frames receipt count drift")

    v = {field: [bool(row[field]) for row in rows] for field in FIELDS}
    uniform_direct = v["uniform_direct_correct"]
    uniform_source = v["uniform_source_correct"]
    active_direct = v["active_direct_correct"]
    active_source = v["active_source_correct"]
    effects = {
        "uniform_source_minus_uniform_direct": paired_metrics(
            uniform_source, uniform_direct,
        ),
        "active_source_minus_active_direct": paired_metrics(
            active_source, active_direct,
        ),
        "active_direct_minus_uniform_direct": paired_metrics(
            active_direct, uniform_direct,
        ),
        "active_source_minus_uniform_source": paired_metrics(
            active_source, uniform_source,
        ),
    }
    active_source_vs_controls = {
        "raw_typed_proof": paired_metrics(
            active_source, v["active_typed_proof_correct"],
        ),
        "binding_control": paired_metrics(
            active_source, v["active_binding_control_correct"],
        ),
        "topology_control": paired_metrics(
            active_source, v["active_topology_control_correct"],
        ),
    }
    cluster_config = config["cluster_bootstrap"]
    cluster = cluster_metrics(
        rows,
        active_source,
        active_direct,
        resamples=int(cluster_config["resamples"]),
        seed=int(cluster_config["seed"]),
    )
    uniform_net = effects["uniform_source_minus_uniform_direct"]["net_wins"]
    active_net = effects["active_source_minus_active_direct"]["net_wins"]
    interaction_net_wins = active_net - uniform_net
    gates = config["development_advancement_gates"]
    active_effect = effects["active_source_minus_active_direct"]
    gate_results = {
        "active_source_minimum_net_wins_vs_active_direct": (
            active_effect["net_wins"]
            >= int(gates["active_source_minimum_net_wins_vs_active_direct"])
        ),
        "active_source_maximum_exact_two_sided_p": (
            active_effect["exact_two_sided_p"]
            <= float(gates["active_source_maximum_exact_two_sided_p"])
        ),
        "active_source_cluster_bootstrap_lower_must_exceed": (
            cluster["stratified_cluster_bootstrap"]["lower_95"]
            > float(gates["active_source_cluster_bootstrap_lower_must_exceed"])
        ),
        "active_source_strictly_above_raw_typed_proof": (
            sum(active_source) > sum(v["active_typed_proof_correct"])
        ),
        "active_source_strictly_above_binding_control": (
            sum(active_source) > sum(v["active_binding_control_correct"])
        ),
        "active_source_strictly_above_topology_control": (
            sum(active_source) > sum(v["active_topology_control_correct"])
        ),
        "minimum_active_by_source_interaction_net_wins": (
            interaction_net_wins
            >= int(gates["minimum_active_by_source_interaction_net_wins"])
        ),
        "active_direct_not_below_uniform_direct": (
            effects["active_direct_minus_uniform_direct"]["net_wins"]
            >= int(gates["active_direct_minimum_net_wins_vs_uniform_direct"])
        ),
    }
    return {
        "schema_version": 27,
        "status": (
            "QUALIFIED_FOR_FRESH_ACTIVE_GROUNDING_CONFIRMATION"
            if all(gate_results.values())
            else "NOT_QUALIFIED"
        ),
        "claim_boundary": config["claim_boundary"],
        "rows": len(rows),
        "video_clusters": len(clusters),
        "condition_correct": {
            field.removesuffix("_correct"): sum(values)
            for field, values in v.items()
        },
        "paired_effects": effects,
        "active_by_source_interaction": {
            "uniform_source_net_wins": uniform_net,
            "active_source_net_wins": active_net,
            "difference_in_net_wins": interaction_net_wins,
            "accuracy_difference_in_differences": (
                (sum(active_source) - sum(active_direct))
                - (sum(uniform_source) - sum(uniform_direct))
            ) / len(rows),
        },
        "active_source_vs_controls": active_source_vs_controls,
        "active_source_vs_active_direct_cluster_inference": cluster,
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
        "condition_correct": report["condition_correct"],
        "paired_effects": report["paired_effects"],
        "interaction": report["active_by_source_interaction"],
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
