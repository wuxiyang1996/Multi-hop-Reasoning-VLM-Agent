#!/usr/bin/env python3
"""Analyze the pre-registered V37 matched-model video transfer."""

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

import analyze_natural_video_v19_formal as stats  # noqa: E402
import collect_natural_video_matched_cate_v37 as collector  # noqa: E402
from motif_transfer.natural_video_matched_cate import validate_v36_artifact  # noqa: E402


def sha256(path: Path) -> str:
    return collector.sha256(path)


def _vectors(
    rows: Sequence[Mapping[str, Any]], conditions: Sequence[str],
) -> dict[str, list[bool]]:
    return {
        name: [bool(row["conditions"][name]["correct"]) for row in rows]
        for name in conditions
    }


def _subset(
    vectors: Mapping[str, Sequence[bool]], indices: Sequence[int],
) -> dict[str, Any]:
    direct = [vectors["matched_direct"][index] for index in indices]
    return {
        name: stats.paired_metrics([values[index] for index in indices], direct)
        for name, values in vectors.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--receipts", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("status") != "FROZEN_BEFORE_V37_MATCHED_FORMAL_OUTCOMES":
        raise ValueError("V37 formal config is not prospectively sealed")
    if sha256(Path(__file__).resolve()) != config["frozen_lineage"]["analyzer_sha256"]:
        raise ValueError("V37 analyzer lineage mismatch")
    manifest = json.loads(Path(config["formal_manifest"]).read_text())
    expected = [
        (benchmark, str(row["sample_id"]))
        for benchmark in ("star", "nextqa") for row in manifest["benchmarks"][benchmark]
    ]
    rows = json.loads(args.receipts.read_text())
    identities = [(str(row["benchmark"]), str(row["sample_id"])) for row in rows]
    if identities != expected:
        raise ValueError("V37 receipts do not exactly match the frozen manifest")
    if any(bool(row.get("runtime_saw_gold_or_official_structure", True)) for row in rows):
        raise ValueError("a V37 runtime branch saw forbidden supervision")
    if any(
        not bool(row.get("same_model_direct_and_proof"))
        or not bool(row.get("same_frames_direct_and_proof"))
        for row in rows
    ):
        raise ValueError("V37 direct/proof matching invariant failed")
    expected_contract = collector.content_hash({
        "config_sha256": sha256(args.config),
        "manifest_sha256": sha256(Path(config["formal_manifest"])),
        "artifact_sha256": sha256(Path(config["cate_artifact"])),
        "collector_sha256": sha256(REPO / "scripts/collect_natural_video_matched_cate_v37.py"),
        "ordered_pairs": expected,
    })
    if {str(row.get("collection_contract_sha256", "")) for row in rows} != {expected_contract}:
        raise ValueError("V37 collection contract mismatch")
    artifact = json.loads(Path(config["cate_artifact"]).read_text())
    validate_v36_artifact(artifact)
    replay = collector._finalize([dict(row) for row in rows], config, artifact)
    for stored, rebuilt in zip(rows, replay):
        if stored["conditions"] != rebuilt["conditions"] or stored["cate"] != rebuilt["cate"]:
            raise ValueError(f"V37 CATE replay mismatch: {stored['sample_id']}")
        if stored["runtime_binding_target"] != rebuilt["runtime_binding_target"]:
            raise ValueError(f"V37 binding replay mismatch: {stored['sample_id']}")
    index = {(str(row["benchmark"]), str(row["sample_id"])): row for row in rows}
    binding_audit = []
    for row in rows:
        target = row["runtime_binding_target"]
        bound = index[(str(target["benchmark"]), str(target["sample_id"]))]
        binding_audit.append({
            "same_sample": str(row["sample_id"]) == str(bound["sample_id"]),
            "same_video": str(row["video_id"]) == str(bound["video_id"]),
            "same_benchmark": str(row["benchmark"]) == str(bound["benchmark"]),
        })
    if any(value["same_sample"] or value["same_video"] or not value["same_benchmark"]
           for value in binding_audit):
        raise ValueError("V37 runtime binding is not an exact cross-video derangement")

    conditions = list(config["controls"]["conditions"])
    if any(set(row["conditions"]) != set(conditions) for row in rows):
        raise ValueError("V37 formal condition set drift")
    vectors = _vectors(rows, conditions)
    direct = vectors["matched_direct"]
    pooled = {
        name: stats.paired_metrics(values, direct) for name, values in vectors.items()
    }
    by_benchmark = {
        benchmark: _subset(
            vectors, [i for i, row in enumerate(rows) if row["benchmark"] == benchmark],
        )
        for benchmark in ("star", "nextqa")
    }
    by_family = {
        f"{benchmark}:{family}": _subset(
            vectors, [
                i for i, row in enumerate(rows)
                if row["benchmark"] == benchmark and row["family"] == family
            ],
        )
        for benchmark, family in sorted({(row["benchmark"], row["family"]) for row in rows})
    }
    cluster_cfg = config["cluster_inference"]
    source = vectors["source_proof_cate"]
    source_clusters = stats.cluster_metrics(
        rows, source, direct,
        resamples=int(cluster_cfg["bootstrap_resamples"]),
        seed=int(cluster_cfg["bootstrap_seed"]),
    )
    controls = [name for name in conditions if name != "source_proof_cate"]
    source_vs_controls = {
        name: {
            "paired": stats.paired_metrics(source, vectors[name]),
            "cluster": stats.cluster_metrics(
                rows, source, vectors[name],
                resamples=int(cluster_cfg["bootstrap_resamples"]),
                seed=int(cluster_cfg["bootstrap_seed"]) + offset,
            ),
        }
        for offset, name in enumerate(controls, start=1)
    }
    gate = config["formal_gates"]
    authentic = pooled["source_proof_cate"]
    gates = {
        "minimum_source_net_wins_pooled": authentic["net_wins"] >= int(gate["minimum_source_net_wins_pooled"]),
        "minimum_source_net_wins_each_benchmark": all(
            by_benchmark[value]["source_proof_cate"]["net_wins"]
            >= int(gate["minimum_source_net_wins_each_benchmark"])
            for value in ("star", "nextqa")
        ),
        "maximum_question_exact_p": authentic["exact_two_sided_p"] <= float(gate["maximum_question_exact_p"]),
        "positive_cluster_bootstrap_lower_bound": (
            source_clusters["stratified_cluster_bootstrap"]["lower_95"]
            > float(gate["cluster_bootstrap_lower_bound_must_exceed"])
        ),
        "minimum_positive_minus_negative_clusters_pooled": (
            source_clusters["positive_minus_negative"]
            >= int(gate["minimum_positive_minus_negative_clusters_pooled"])
        ),
        "minimum_positive_minus_negative_clusters_each_benchmark": all(
            value["positive_minus_negative"]
            >= int(gate["minimum_positive_minus_negative_clusters_each_benchmark"])
            for value in source_clusters["by_benchmark"].values()
        ),
        "source_strictly_above_matched_direct": authentic["correct"] > pooled["matched_direct"]["correct"],
        "source_strictly_above_raw_typed_proof": authentic["correct"] > pooled["raw_typed_proof"]["correct"],
        "source_strictly_above_base_only": authentic["correct"] > pooled["base_only_cate"]["correct"],
        "source_strictly_above_permuted_uplift": authentic["correct"] > pooled["permuted_uplift_cate"]["correct"],
        "source_strictly_above_binding_training": authentic["correct"] > pooled["binding_training_cate"]["correct"],
        "source_strictly_above_runtime_binding": authentic["correct"] > pooled["runtime_cross_video_binding"]["correct"],
        "source_strictly_above_inverted_contract": authentic["correct"] > pooled["inverted_source_contract"]["correct"],
        "source_strictly_above_same_rate_marginal": authentic["correct"] > pooled["same_rate_marginal"]["correct"],
    }
    passed = all(gates.values())
    report = {
        "schema_version": 37,
        "status": (
            "SOKOBAN_TO_MATCHED_STAR_NEXTQA_NEUROSYMBOLIC_TRANSFER_FORMAL_VALIDATED"
            if passed else "SOKOBAN_TO_MATCHED_STAR_NEXTQA_NEUROSYMBOLIC_TRANSFER_FORMAL_FAILED"
        ),
        "rows": len(rows),
        "video_clusters": len({(row["benchmark"], row["video_id"]) for row in rows}),
        "zero_sample_overlap_with_v36_adaptation": True,
        "zero_video_overlap_with_v36_adaptation": True,
        "same_model_and_frames_direct_vs_proof": True,
        "condition_metrics_vs_matched_direct": pooled,
        "by_benchmark_vs_matched_direct": by_benchmark,
        "by_family_vs_matched_direct": by_family,
        "source_video_cluster_inference": source_clusters,
        "source_vs_controls": source_vs_controls,
        "binding_audit": {
            "rows": len(binding_audit),
            "same_sample": sum(value["same_sample"] for value in binding_audit),
            "same_video": sum(value["same_video"] for value in binding_audit),
            "same_benchmark": sum(value["same_benchmark"] for value in binding_audit),
        },
        "formal_gates": gates,
        "all_formal_gates_passed": passed,
        "claim_boundary": config["claim_boundary"],
        "artifacts": {
            "config": str(args.config.resolve()), "config_sha256": sha256(args.config),
            "manifest": str(Path(config["formal_manifest"]).resolve()),
            "manifest_sha256": sha256(Path(config["formal_manifest"])),
            "cate_artifact": str(Path(config["cate_artifact"]).resolve()),
            "cate_artifact_sha256": sha256(Path(config["cate_artifact"])),
            "receipts": str(args.receipts.resolve()), "receipts_sha256": sha256(args.receipts),
            "analyzer": str(Path(__file__).resolve()),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "metrics": pooled,
        "formal_gates": gates, "report": str(args.report.resolve()),
        "report_sha256": sha256(args.report),
    }, indent=2))


if __name__ == "__main__":
    main()
