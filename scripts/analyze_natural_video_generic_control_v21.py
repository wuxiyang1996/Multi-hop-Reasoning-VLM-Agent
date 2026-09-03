#!/usr/bin/env python3
"""Apply the frozen V21 matched prompt-ablation qualification gates."""

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


def _subset(
    rows: Sequence[Mapping[str, Any]], indices: Sequence[int],
) -> dict[str, Any]:
    typed = [bool(rows[i]["typed_proof_correct"]) for i in indices]
    generic = [bool(rows[i]["generic_direct_correct"]) for i in indices]
    primary = [bool(rows[i]["primary_correct"]) for i in indices]
    return {
        "typed_vs_generic": paired_metrics(typed, generic),
        "typed_vs_primary": paired_metrics(typed, primary),
        "generic_vs_primary": paired_metrics(generic, primary),
    }


def analyze(
    rows: Sequence[Mapping[str, Any]],
    v19_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    expected = int(config["expected_rows"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} V21 control rows, got {len(rows)}")
    identities = [(str(row["benchmark"]), str(row["sample_id"])) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate V21 identities")
    if any(bool(row.get("runtime_saw_gold_or_official_structure", True)) for row in rows):
        raise ValueError("a generic runtime branch saw forbidden supervision")
    if any(list(row["proof_panel_sha256"]) == [] for row in rows):
        raise ValueError("matched proof-panel hashes are missing")
    contracts = {str(row.get("collection_contract_sha256", "")) for row in rows}
    if len(contracts) != 1 or "" in contracts:
        raise ValueError("generic-control collection contract mismatch")
    v19_by_identity = {
        (str(row["benchmark"]), str(row["sample_id"])): row for row in v19_rows
    }
    if set(v19_by_identity) != set(identities):
        raise ValueError("V21 generic and V19 authentic identities do not align")
    if str(protocol["authentic_v19_field"]) != "unrestricted_correct":
        raise ValueError("unexpected V21 authentic-policy field")

    all_indices = list(range(len(rows)))
    pooled = _subset(rows, all_indices)
    benchmarks = {
        benchmark: _subset(
            rows,
            [i for i, row in enumerate(rows) if str(row["benchmark"]) == benchmark],
        )
        for benchmark in ("star", "nextqa")
    }
    families = sorted({(str(row["benchmark"]), str(row["family"])) for row in rows})
    by_family = {
        f"{benchmark}:{family}": _subset(
            rows,
            [
                i for i, row in enumerate(rows)
                if str(row["benchmark"]) == benchmark and str(row["family"]) == family
            ],
        )
        for benchmark, family in families
    }
    typed = [bool(row["typed_proof_correct"]) for row in rows]
    generic = [bool(row["generic_direct_correct"]) for row in rows]
    bootstrap = config["cluster_bootstrap"]
    clustered = cluster_metrics(
        rows,
        typed,
        generic,
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]),
    )
    authentic = [
        bool(v19_by_identity[identity]["unrestricted_correct"])
        for identity in identities
    ]
    generic = [bool(row["generic_direct_correct"]) for row in rows]
    primary = [bool(row["primary_correct"]) for row in rows]
    raw_proof = [bool(row["typed_proof_correct"]) for row in rows]
    authentic_pooled = {
        "source_guard_vs_generic": paired_metrics(authentic, generic),
        "source_guard_vs_primary": paired_metrics(authentic, primary),
        "source_guard_vs_raw_proof": paired_metrics(authentic, raw_proof),
    }
    authentic_by_benchmark = {}
    for benchmark in ("star", "nextqa"):
        indices = [
            i for i, row in enumerate(rows) if str(row["benchmark"]) == benchmark
        ]
        authentic_by_benchmark[benchmark] = {
            "source_guard_vs_generic": paired_metrics(
                [authentic[i] for i in indices], [generic[i] for i in indices]
            ),
            "source_guard_vs_primary": paired_metrics(
                [authentic[i] for i in indices], [primary[i] for i in indices]
            ),
        }
    authentic_clustered = cluster_metrics(
        rows,
        authentic,
        generic,
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]) + 1,
    )
    gates = config["development_qualification_gates"]
    results = {
        "typed_proof_strictly_above_generic_direct_pooled": (
            pooled["typed_vs_generic"]["correct"]
            > pooled["typed_vs_generic"]["baseline_correct"]
        ),
        "typed_proof_not_below_generic_direct_each_benchmark": all(
            metrics["typed_vs_generic"]["correct"]
            >= metrics["typed_vs_generic"]["baseline_correct"]
            for metrics in benchmarks.values()
        ),
        "typed_proof_vs_generic_exact_two_sided_p_maximum": (
            pooled["typed_vs_generic"]["exact_two_sided_p"]
            <= float(gates["typed_proof_vs_generic_exact_two_sided_p_maximum"])
        ),
        "typed_proof_minus_generic_stratified_video_cluster_bootstrap_lower_bound_must_exceed": (
            clustered["stratified_cluster_bootstrap"]["lower_95"]
            > float(gates[
                "typed_proof_minus_generic_stratified_video_cluster_bootstrap_lower_bound_must_exceed"
            ])
        ),
    }
    source_gates = protocol["additional_source_authentic_qualification_gates"]
    authentic_results = {
        "source_guard_strictly_above_generic_direct_pooled": (
            authentic_pooled["source_guard_vs_generic"]["correct"]
            > authentic_pooled["source_guard_vs_generic"]["baseline_correct"]
        ),
        "source_guard_not_below_generic_direct_each_benchmark": all(
            metrics["source_guard_vs_generic"]["correct"]
            >= metrics["source_guard_vs_generic"]["baseline_correct"]
            for metrics in authentic_by_benchmark.values()
        ),
        "source_guard_vs_generic_exact_two_sided_p_maximum": (
            authentic_pooled["source_guard_vs_generic"]["exact_two_sided_p"]
            <= float(source_gates["source_guard_vs_generic_exact_two_sided_p_maximum"])
        ),
        "source_guard_minus_generic_stratified_video_cluster_bootstrap_lower_bound_must_exceed": (
            authentic_clustered["stratified_cluster_bootstrap"]["lower_95"]
            > float(source_gates[
                "source_guard_minus_generic_stratified_video_cluster_bootstrap_lower_bound_must_exceed"
            ])
        ),
    }
    all_passed = all(results.values()) and all(authentic_results.values())
    return {
        "schema_version": 21,
        "status": "QUALIFIED_FOR_RESERVE" if all_passed else "NOT_QUALIFIED",
        "claim_boundary": config["claim_boundary"],
        "rows": len(rows),
        "video_clusters": len({
            (str(row["benchmark"]), str(row["video_id"])) for row in rows
        }),
        "pooled": pooled,
        "by_benchmark": benchmarks,
        "by_family": by_family,
        "typed_minus_generic_cluster_inference": clustered,
        "source_authentic_policy": protocol["authentic_policy"],
        "source_authentic_pooled": authentic_pooled,
        "source_authentic_by_benchmark": authentic_by_benchmark,
        "source_guard_minus_generic_cluster_inference": authentic_clustered,
        "typed_prompt_qualification_gates": results,
        "source_authentic_qualification_gates": authentic_results,
        "all_qualification_gates_passed": all_passed,
        "collection_contract_sha256": next(iter(contracts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v19-receipts", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rows = json.loads(args.receipts.read_text(encoding="utf-8"))
    v19_rows = json.loads(args.v19_receipts.read_text(encoding="utf-8"))
    if sha256(args.config) != str(protocol["generic_control_config_sha256"]):
        raise ValueError("V21 analysis protocol references a different control config")
    if sha256(args.v19_receipts) != str(protocol["input_v19_receipts_sha256"]):
        raise ValueError("V21 analysis protocol references different V19 receipts")
    report = analyze(rows, v19_rows, config, protocol)
    report["artifacts"] = {
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "receipts": str(args.receipts.resolve()),
        "receipts_sha256": sha256(args.receipts),
        "v19_receipts": str(args.v19_receipts.resolve()),
        "v19_receipts_sha256": sha256(args.v19_receipts),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol),
        "analyzer": str(Path(__file__).resolve()),
        "analyzer_sha256": sha256(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "typed_vs_generic": report["pooled"]["typed_vs_generic"],
        "cluster_bootstrap": report["typed_minus_generic_cluster_inference"][
            "stratified_cluster_bootstrap"
        ],
        "source_guard_vs_generic": report["source_authentic_pooled"][
            "source_guard_vs_generic"
        ],
        "source_guard_cluster_bootstrap": report[
            "source_guard_minus_generic_cluster_inference"
        ]["stratified_cluster_bootstrap"],
        "failed_gates": [
            name for name, passed in {
                **report["typed_prompt_qualification_gates"],
                **report["source_authentic_qualification_gates"],
            }.items() if not passed
        ],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
