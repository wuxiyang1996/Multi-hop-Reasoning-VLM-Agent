#!/usr/bin/env python3
"""Analyze the prospectively frozen V35 game-to-natural-video transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_natural_video_v19_formal as v19_analysis  # noqa: E402
import collect_natural_video_proof_cate_v35 as collector  # noqa: E402
from motif_transfer.natural_video_proof_cate import (  # noqa: E402
    compile_v19_features,
    validate_v34_artifact,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _condition_vectors(
    rows: Sequence[Mapping[str, Any]], conditions: Sequence[str],
) -> dict[str, list[bool]]:
    return {
        name: [bool(row["conditions"][name]["correct"]) for row in rows]
        for name in conditions
    }


def _subset_metrics(
    rows: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, Sequence[bool]],
    indices: Sequence[int],
) -> dict[str, Any]:
    baseline = [vectors["primary"][index] for index in indices]
    return {
        name: v19_analysis.paired_metrics(
            [values[index] for index in indices], baseline,
        )
        for name, values in vectors.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--receipts", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_V35_PROSPECTIVE_FORMAL_OUTCOMES":
        raise ValueError("V35 config was not prospectively frozen")
    if sha256(Path(__file__).resolve()) != config["frozen_lineage"].get("analyzer_sha256"):
        raise ValueError("V35 analyzer lineage mismatch")
    rows = json.loads(args.receipts.read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["formal_manifest"]).read_text())
    expected = [
        (benchmark, str(row["sample_id"]))
        for benchmark in ("star", "nextqa")
        for row in manifest["benchmarks"][benchmark]
    ]
    identities = [(str(row["benchmark"]), str(row["sample_id"])) for row in rows]
    if identities != expected or len(identities) != 126:
        raise ValueError("V35 formal receipts do not exactly match frozen order")
    if any(bool(row.get("runtime_saw_gold_or_official_structure", True)) for row in rows):
        raise ValueError("V35 runtime branch saw forbidden target supervision")
    contracts = {str(row.get("collection_contract_sha256", "")) for row in rows}
    expected_contract = collector.content_hash({
        "config_sha256": sha256(args.config),
        "manifest_sha256": sha256(Path(config["formal_manifest"])),
        "artifact_sha256": sha256(Path(config["cate_artifact"])),
        "collector_sha256": sha256(REPO / "scripts/collect_natural_video_proof_cate_v35.py"),
        "ordered_pairs": expected,
    })
    if contracts != {expected_contract}:
        raise ValueError("V35 collection contract is missing or inconsistent")
    for row in rows:
        compile_v19_features(row)
    artifact = json.loads(Path(config["cate_artifact"]).read_text())
    validate_v34_artifact(artifact)
    recomputed = collector._finalize([dict(row) for row in rows], config, artifact)
    for stored, replay in zip(rows, recomputed):
        if stored["conditions"] != replay["conditions"]:
            raise ValueError(f"V35 condition replay mismatch: {stored['sample_id']}")
        if stored["binding_rotation_target"] != replay["binding_rotation_target"]:
            raise ValueError(f"V35 binding replay mismatch: {stored['sample_id']}")
        if stored["cate"] != replay["cate"]:
            raise ValueError(f"V35 neural CATE replay mismatch: {stored['sample_id']}")
    by_identity = {
        (str(row["benchmark"]), str(row["sample_id"])): row for row in rows
    }
    binding_audit = []
    for row in rows:
        target = row["binding_rotation_target"]
        bound = by_identity[(str(target["benchmark"]), str(target["sample_id"]))]
        binding_audit.append({
            "same_sample": str(row["sample_id"]) == str(bound["sample_id"]),
            "same_video": str(row["video_id"]) == str(bound["video_id"]),
            "same_benchmark": str(row["benchmark"]) == str(bound["benchmark"]),
            "same_family": str(row["family"]) == str(bound["family"]),
        })
    if any(value["same_sample"] or value["same_video"] for value in binding_audit):
        raise ValueError("V35 binding rotation failed sample/video derangement")
    if any(not value["same_benchmark"] or not value["same_family"] for value in binding_audit):
        raise ValueError("V35 binding rotation crossed task family")

    conditions = list(config["controls"]["conditions"])
    if any(set(row["conditions"]) != set(conditions) for row in rows):
        raise ValueError("V35 formal condition set drift")
    vectors = _condition_vectors(rows, conditions)
    primary = vectors["primary"]
    pooled = {
        name: v19_analysis.paired_metrics(values, primary)
        for name, values in vectors.items()
    }
    by_benchmark = {
        benchmark: _subset_metrics(
            rows, vectors,
            [i for i, row in enumerate(rows) if str(row["benchmark"]) == benchmark],
        )
        for benchmark in ("star", "nextqa")
    }
    by_family = {
        f"{benchmark}:{family}": _subset_metrics(
            rows, vectors,
            [
                i for i, row in enumerate(rows)
                if str(row["benchmark"]) == benchmark and str(row["family"]) == family
            ],
        )
        for benchmark, family in sorted({
            (str(row["benchmark"]), str(row["family"])) for row in rows
        })
    }
    cluster_config = config["cluster_inference"]
    source = vectors["source_proof_cate"]
    source_clusters = v19_analysis.cluster_metrics(
        rows, source, primary,
        resamples=int(cluster_config["bootstrap_resamples"]),
        seed=int(cluster_config["bootstrap_seed"]),
    )
    controls = [name for name in conditions if name != "source_proof_cate"]
    source_vs_controls = {
        name: {
            "paired": v19_analysis.paired_metrics(source, vectors[name]),
            "cluster": v19_analysis.cluster_metrics(
                rows, source, vectors[name],
                resamples=int(cluster_config["bootstrap_resamples"]),
                seed=int(cluster_config["bootstrap_seed"]) + offset,
            ),
        }
        for offset, name in enumerate(controls, start=1)
    }
    formal = config["formal_gates"]
    authentic = pooled["source_proof_cate"]
    gates = {
        "minimum_source_net_wins_pooled": (
            authentic["net_wins"] >= int(formal["minimum_source_net_wins_pooled"])
        ),
        "minimum_source_net_wins_each_benchmark": all(
            by_benchmark[value]["source_proof_cate"]["net_wins"]
            >= int(formal["minimum_source_net_wins_each_benchmark"])
            for value in ("star", "nextqa")
        ),
        "maximum_question_level_exact_two_sided_p": (
            authentic["exact_two_sided_p"]
            <= float(formal["maximum_question_level_exact_two_sided_p"])
        ),
        "positive_cluster_bootstrap_lower_bound": (
            source_clusters["stratified_cluster_bootstrap"]["lower_95"]
            > float(formal["cluster_bootstrap_lower_bound_must_exceed"])
        ),
        "minimum_positive_minus_negative_video_clusters_pooled": (
            source_clusters["positive_minus_negative"]
            >= int(formal["minimum_positive_minus_negative_video_clusters_pooled"])
        ),
        "minimum_positive_minus_negative_video_clusters_each_benchmark": all(
            value["positive_minus_negative"]
            >= int(formal["minimum_positive_minus_negative_video_clusters_each_benchmark"])
            for value in source_clusters["by_benchmark"].values()
        ),
        "source_strictly_above_primary": (
            authentic["correct"] > pooled["primary"]["correct"]
        ),
        "source_strictly_above_always_proof": (
            authentic["correct"] > pooled["always_proof"]["correct"]
        ),
        "source_strictly_above_base_only": (
            authentic["correct"] > pooled["base_only_cate"]["correct"]
        ),
        "source_strictly_above_permuted_uplift": (
            authentic["correct"] > pooled["permuted_uplift_cate"]["correct"]
        ),
        "source_strictly_above_shuffled_proof_training": (
            authentic["correct"] > pooled["shuffled_proof_training_cate"]["correct"]
        ),
        "source_strictly_above_binding_rotation": (
            authentic["correct"] > pooled["binding_rotation_cate"]["correct"]
        ),
        "source_strictly_above_inverted_contract": (
            authentic["correct"] > pooled["inverted_source_contract"]["correct"]
        ),
        "source_strictly_above_same_rate_marginal": (
            authentic["correct"] > pooled["same_rate_marginal"]["correct"]
        ),
    }
    passed = all(gates.values())
    report = {
        "schema_version": 35,
        "status": (
            "SOKOBAN_TO_STAR_NEXTQA_NEUROSYMBOLIC_TRANSFER_FORMAL_VALIDATED"
            if passed else "SOKOBAN_TO_STAR_NEXTQA_NEUROSYMBOLIC_TRANSFER_FORMAL_FAILED"
        ),
        "rows": len(rows),
        "video_clusters": len({(row["benchmark"], row["video_id"]) for row in rows}),
        "zero_sample_overlap_with_adaptation": True,
        "zero_video_overlap_with_adaptation": True,
        "condition_metrics_vs_primary": pooled,
        "by_benchmark_vs_primary": by_benchmark,
        "by_family_vs_primary": by_family,
        "source_video_cluster_inference": source_clusters,
        "source_vs_controls": source_vs_controls,
        "binding_audit": {
            "rows": len(binding_audit),
            "same_sample": sum(value["same_sample"] for value in binding_audit),
            "same_video": sum(value["same_video"] for value in binding_audit),
            "same_benchmark": sum(value["same_benchmark"] for value in binding_audit),
            "same_family": sum(value["same_family"] for value in binding_audit),
        },
        "formal_gates": gates,
        "all_formal_gates_passed": passed,
        "claim_boundary": config["claim_boundary"],
        "artifacts": {
            "config": str(args.config.resolve()),
            "config_sha256": sha256(args.config),
            "manifest": str(Path(config["formal_manifest"]).resolve()),
            "manifest_sha256": sha256(Path(config["formal_manifest"])),
            "cate_artifact": str(Path(config["cate_artifact"]).resolve()),
            "cate_artifact_sha256": sha256(Path(config["cate_artifact"])),
            "formal_receipts": str(args.receipts.resolve()),
            "formal_receipts_sha256": sha256(args.receipts),
            "analyzer": str(Path(__file__).resolve()),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "metrics": pooled,
        "formal_gates": gates,
        "report": str(args.report.resolve()),
        "report_sha256": sha256(args.report),
    }, indent=2))


if __name__ == "__main__":
    main()
