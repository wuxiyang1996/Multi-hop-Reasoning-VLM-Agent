#!/usr/bin/env python3
"""Analyze the prospectively frozen V19 natural-video transfer receipts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from motif_transfer.sokoban_video_recovery import exact_binomial_two_sided  # noqa: E402


CONDITION_FIELDS = {
    "primary": "primary_correct",
    "always_cross_model_proof": "proof_correct",
    "authentic_source_compatible_guard": "authentic_correct",
    "unrestricted_typed_proof_guard": "unrestricted_correct",
    "inverted_source_applicability": "inverted_applicability_correct",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _answer(row: Mapping[str, Any], branch: str) -> str:
    if branch == "primary":
        return str(row["primary"]["answer"])
    if branch == "proof":
        return str(row["proof"]["answer"])
    raise ValueError(f"unknown answer branch: {branch}")


def build_fixed_control_masks(
    rows: Sequence[Mapping[str, Any]], *, shuffled_seed: int, marginal_seed: int,
) -> dict[str, list[bool]]:
    """Build outcome-blind masks while preserving the frozen recovery rates."""

    shuffled = [False] * len(rows)
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        cells[(str(row["benchmark"]), str(row["family"]))].append(index)
    shuffled_rng = random.Random(shuffled_seed)
    for cell in sorted(cells):
        indices = sorted(cells[cell], key=lambda i: str(rows[i]["sample_id"]))
        values = [bool(rows[i]["authentic_recover"]) for i in indices]
        shuffled_rng.shuffle(values)
        for index, value in zip(indices, values):
            shuffled[index] = value

    marginal = [False] * len(rows)
    marginal_rng = random.Random(marginal_seed)
    for benchmark in sorted({str(row["benchmark"]) for row in rows}):
        compatible = [
            i for i, row in enumerate(rows)
            if str(row["benchmark"]) == benchmark and bool(row["source_compatible"])
        ]
        compatible.sort(key=lambda i: str(rows[i]["sample_id"]))
        recovery_count = sum(
            bool(row["authentic_recover"])
            for row in rows if str(row["benchmark"]) == benchmark
        )
        chosen = set(marginal_rng.sample(compatible, recovery_count))
        for index in chosen:
            marginal[index] = True

    return {
        "within_benchmark_family_shuffled_recovery_mask": shuffled,
        "same_rate_marginal_recovery_mask": marginal,
    }


def correct_from_mask(
    rows: Sequence[Mapping[str, Any]], mask: Sequence[bool],
) -> list[bool]:
    if len(rows) != len(mask):
        raise ValueError("control mask length mismatch")
    return [
        (_answer(row, "proof") if recover else _answer(row, "primary"))
        == str(row["gold_answer"])
        for row, recover in zip(rows, mask)
    ]


def paired_metrics(candidate: Sequence[bool], baseline: Sequence[bool]) -> dict[str, Any]:
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("paired vectors must be nonempty and aligned")
    wins = sum(bool(left) and not bool(right) for left, right in zip(candidate, baseline))
    losses = sum(not bool(left) and bool(right) for left, right in zip(candidate, baseline))
    correct = sum(map(bool, candidate))
    baseline_correct = sum(map(bool, baseline))
    return {
        "n": len(candidate),
        "correct": correct,
        "accuracy": correct / len(candidate),
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / len(candidate),
        "wins": wins,
        "losses": losses,
        "ties": len(candidate) - wins - losses,
        "net_wins": wins - losses,
        "accuracy_delta": (correct - baseline_correct) / len(candidate),
        "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
    }


def percentile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid percentile input")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def cluster_metrics(
    rows: Sequence[Mapping[str, Any]],
    candidate: Sequence[bool],
    baseline: Sequence[bool],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if len(rows) != len(candidate) or len(rows) != len(baseline):
        raise ValueError("cluster inputs must be aligned")
    clusters: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        clusters[(str(row["benchmark"]), str(row["video_id"]))].append(index)
    cluster_rows = []
    for key in sorted(clusters):
        indices = clusters[key]
        delta = sum(int(candidate[i]) - int(baseline[i]) for i in indices)
        cluster_rows.append({
            "benchmark": key[0],
            "video_id": key[1],
            "questions": len(indices),
            "correct_delta": delta,
            "accuracy_delta": delta / len(indices),
        })
    positive = sum(row["correct_delta"] > 0 for row in cluster_rows)
    negative = sum(row["correct_delta"] < 0 for row in cluster_rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cluster_rows:
        grouped[str(row["benchmark"])].append(row)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(resamples):
        numerator = 0
        denominator = 0
        for benchmark in sorted(grouped):
            available = grouped[benchmark]
            for _cluster in range(len(available)):
                selected = available[rng.randrange(len(available))]
                numerator += int(selected["correct_delta"])
                denominator += int(selected["questions"])
        bootstrap.append(numerator / denominator)
    per_benchmark = {}
    for benchmark, values in sorted(grouped.items()):
        pos = sum(row["correct_delta"] > 0 for row in values)
        neg = sum(row["correct_delta"] < 0 for row in values)
        per_benchmark[benchmark] = {
            "clusters": len(values),
            "positive": pos,
            "negative": neg,
            "ties": len(values) - pos - neg,
            "positive_minus_negative": pos - neg,
            "cluster_sign_exact_two_sided_p": exact_binomial_two_sided(pos, neg),
            "correct_delta": sum(int(row["correct_delta"]) for row in values),
            "questions": sum(int(row["questions"]) for row in values),
        }
    return {
        "clusters": len(cluster_rows),
        "positive": positive,
        "negative": negative,
        "ties": len(cluster_rows) - positive - negative,
        "positive_minus_negative": positive - negative,
        "cluster_sign_exact_two_sided_p": exact_binomial_two_sided(positive, negative),
        "stratified_cluster_bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "mean": sum(bootstrap) / len(bootstrap),
            "lower_95": percentile(bootstrap, 0.025),
            "upper_95": percentile(bootstrap, 0.975),
        },
        "by_benchmark": per_benchmark,
        "by_cluster": cluster_rows,
    }


def _subset_metrics(
    rows: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, Sequence[bool]],
    indices: Iterable[int],
) -> dict[str, Any]:
    selected = list(indices)
    primary = [bool(vectors["primary"][i]) for i in selected]
    return {
        name: paired_metrics([bool(values[i]) for i in selected], primary)
        for name, values in vectors.items()
    }


def analyze(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if len(rows) != int(protocol["expected_rows"]):
        raise ValueError(f"expected {protocol['expected_rows']} complete rows, got {len(rows)}")
    identities = [(str(row["benchmark"]), str(row["sample_id"])) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate V19 sample identities")
    if any(bool(row.get("runtime_saw_gold_or_official_structure", True)) for row in rows):
        raise ValueError("a V19 runtime branch saw forbidden target supervision")
    contracts = {str(row.get("collection_contract_sha256", "")) for row in rows}
    if len(contracts) != 1 or "" in contracts:
        raise ValueError("V19 collection contract is missing or inconsistent")

    vectors: dict[str, list[bool]] = {
        name: [bool(row[field]) for row in rows]
        for name, field in CONDITION_FIELDS.items()
    }
    masks = build_fixed_control_masks(
        rows,
        shuffled_seed=int(config["control_seeds"]["shuffled_recovery_mask"]),
        marginal_seed=int(config["control_seeds"]["marginal_recovery_mask"]),
    )
    for name, mask in masks.items():
        vectors[name] = correct_from_mask(rows, mask)
    primary = vectors["primary"]
    pooled = {name: paired_metrics(values, primary) for name, values in vectors.items()}
    by_benchmark = {
        benchmark: _subset_metrics(
            rows, vectors,
            (i for i, row in enumerate(rows) if str(row["benchmark"]) == benchmark),
        )
        for benchmark in ("star", "nextqa")
    }
    families = sorted({(str(row["benchmark"]), str(row["family"])) for row in rows})
    by_family = {
        f"{benchmark}:{family}": _subset_metrics(
            rows, vectors,
            (
                i for i, row in enumerate(rows)
                if str(row["benchmark"]) == benchmark and str(row["family"]) == family
            ),
        )
        for benchmark, family in families
    }

    cluster_config = protocol["cluster_inference"]
    authentic_clusters = cluster_metrics(
        rows,
        vectors["authentic_source_compatible_guard"],
        primary,
        resamples=int(cluster_config["resamples"]),
        seed=int(config["control_seeds"]["cluster_bootstrap"]),
    )
    controls = [
        "always_cross_model_proof",
        "unrestricted_typed_proof_guard",
        "inverted_source_applicability",
        "within_benchmark_family_shuffled_recovery_mask",
        "same_rate_marginal_recovery_mask",
    ]
    authentic_vs_controls = {
        name: {
            "paired": paired_metrics(
                vectors["authentic_source_compatible_guard"], vectors[name]
            ),
            "cluster": cluster_metrics(
                rows,
                vectors["authentic_source_compatible_guard"],
                vectors[name],
                resamples=int(cluster_config["resamples"]),
                seed=int(config["control_seeds"]["cluster_bootstrap"]) + offset,
            ),
        }
        for offset, name in enumerate(controls, start=1)
    }

    authentic = pooled["authentic_source_compatible_guard"]
    gates = config["formal_gates"]
    frozen_gate_results = {
        "minimum_authentic_net_wins_pooled": (
            authentic["net_wins"] >= int(gates["minimum_authentic_net_wins_pooled"])
        ),
        "minimum_authentic_net_wins_each_benchmark": all(
            by_benchmark[benchmark]["authentic_source_compatible_guard"]["net_wins"]
            >= int(gates["minimum_authentic_net_wins_each_benchmark"])
            for benchmark in by_benchmark
        ),
        "maximum_question_level_exact_two_sided_p": (
            authentic["exact_two_sided_p"]
            <= float(gates["maximum_question_level_exact_two_sided_p"])
        ),
        "minimum_positive_video_clusters": (
            authentic_clusters["positive"] >= int(gates["minimum_positive_video_clusters"])
        ),
        "authentic_strictly_above_primary": (
            authentic["correct"] > pooled["primary"]["correct"]
        ),
        "authentic_strictly_above_always_proof": (
            authentic["correct"] > pooled["always_cross_model_proof"]["correct"]
        ),
        "authentic_strictly_above_unrestricted_guard": (
            authentic["correct"] > pooled["unrestricted_typed_proof_guard"]["correct"]
        ),
        "authentic_strictly_above_inverted_applicability": (
            authentic["correct"] > pooled["inverted_source_applicability"]["correct"]
        ),
        "authentic_strictly_above_fixed_shuffled_and_marginal_controls": all(
            authentic["correct"] > pooled[name]["correct"]
            for name in (
                "within_benchmark_family_shuffled_recovery_mask",
                "same_rate_marginal_recovery_mask",
            )
        ),
    }
    additional_gate_results = {
        "pooled_authentic_minus_primary_cluster_bootstrap_lower_bound_must_exceed": (
            authentic_clusters["stratified_cluster_bootstrap"]["lower_95"]
            > float(protocol["additional_preoutcome_claim_gates"][
                "pooled_authentic_minus_primary_cluster_bootstrap_lower_bound_must_exceed"
            ])
        ),
        "minimum_positive_minus_negative_video_clusters_each_benchmark": all(
            values["positive_minus_negative"]
            >= int(protocol["additional_preoutcome_claim_gates"][
                "minimum_positive_minus_negative_video_clusters_each_benchmark"
            ])
            for values in authentic_clusters["by_benchmark"].values()
        ),
    }
    all_gates = {**frozen_gate_results, **additional_gate_results}
    mask_audit = {}
    authentic_mask = [bool(row["authentic_recover"]) for row in rows]
    for name, mask in masks.items():
        mask_audit[name] = {
            "recoveries": sum(mask),
            "authentic_recoveries": sum(authentic_mask),
            "hamming_distance_from_authentic": sum(a != b for a, b in zip(mask, authentic_mask)),
            "selected_sample_ids_sha256": content_sha256([
                str(row["sample_id"]) for row, selected in zip(rows, mask) if selected
            ]),
        }
    return {
        "schema_version": 19,
        "status": "PASS" if all(all_gates.values()) else "FAIL",
        "claim": (
            "prospective natural-video success-rate transfer validated"
            if all(all_gates.values())
            else "prospective natural-video success-rate transfer not validated"
        ),
        "claim_boundary": config["claim_boundary"],
        "rows": len(rows),
        "video_clusters": len({
            (str(row["benchmark"]), str(row["video_id"])) for row in rows
        }),
        "collection_contract_sha256": next(iter(contracts)),
        "condition_metrics_vs_primary": pooled,
        "by_benchmark_vs_primary": by_benchmark,
        "by_family_vs_primary": by_family,
        "authentic_video_cluster_inference": authentic_clusters,
        "authentic_vs_controls": authentic_vs_controls,
        "control_mask_audit": mask_audit,
        "frozen_gate_results": frozen_gate_results,
        "additional_preoutcome_gate_results": additional_gate_results,
        "all_claim_gates": all_gates,
        "all_claim_gates_passed": all(all_gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rows = json.loads(args.receipts.read_text(encoding="utf-8"))
    report = analyze(rows, config, protocol)
    report["artifacts"] = {
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol),
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
        "rows": report["rows"],
        "video_clusters": report["video_clusters"],
        "authentic_vs_primary": report["condition_metrics_vs_primary"][
            "authentic_source_compatible_guard"
        ],
        "cluster_bootstrap": report["authentic_video_cluster_inference"][
            "stratified_cluster_bootstrap"
        ],
        "failed_gates": [name for name, passed in report["all_claim_gates"].items() if not passed],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
