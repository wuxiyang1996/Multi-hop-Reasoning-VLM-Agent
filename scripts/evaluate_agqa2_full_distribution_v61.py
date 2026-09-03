#!/usr/bin/env python3
"""Evaluate the preregistered video-clustered AGQA V61 formal endpoint."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_query_object_source_specific import (  # noqa: E402
    exact_one_sided_pvalue,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    body = dict(payload)
    claimed = body.pop(field)
    if stable_hash(body) != claimed:
        raise ValueError(f"content hash mismatch: {path}")
    return payload


def evaluate(
    *, config_path: Path, base_report_path: Path, output_path: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    formal = config["full_distribution_evaluation"]
    if _sha256(Path(__file__)) != formal["evaluator_file_sha256"]:
        raise ValueError("V61 evaluator differs from preregistration")
    prereg_path = REPO_ROOT / config["preregistration"]
    if _sha256(prereg_path) != config["preregistration_file_sha256"]:
        raise ValueError("V61 preregistration file hash mismatch")
    prereg = json.loads(prereg_path.read_text())
    if prereg["status"] != "FROZEN_BEFORE_ANY_V61_PROVIDER_OR_OUTCOME_CALL":
        raise ValueError("V61 preregistration is not frozen")
    qualification_path = REPO_ROOT / formal["qualification_report"]
    if _sha256(qualification_path) != formal[
        "qualification_report_file_sha256"
    ]:
        raise ValueError("V61 qualification dependency changed")
    qualification = _verified(qualification_path, "report_sha256")
    if not qualification.get("grounder_qualified") or not all(
        qualification.get("qualification_gates", {}).values()
    ):
        raise ValueError("V61 formal requires passed V60 qualification")

    base = _verified(base_report_path, "report_sha256")
    if base["grounder_sha256"] != qualification["grounder_sha256"]:
        raise ValueError("V61 changed the V60-qualified grounder")
    manifest = _verified(REPO_ROOT / config["manifest"], "manifest_sha256")
    if base["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("V61 base report/manifest mismatch")
    rows = base["rows"]
    by_video: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[str(row["video_id"])].append(row)

    positive = negative = tied = 0
    video_details = []
    for video_id, video_rows in sorted(by_video.items()):
        source_correct = sum(bool(row["source_correct"]) for row in video_rows)
        direct_correct = sum(bool(row["direct_correct"]) for row in video_rows)
        delta = source_correct - direct_correct
        positive += delta > 0
        negative += delta < 0
        tied += delta == 0
        video_details.append({
            "video_id_sha256": stable_hash(video_id),
            "rows": len(video_rows),
            "source_correct": source_correct,
            "direct_correct": direct_correct,
            "delta": delta,
        })
    cluster = {
        "positive_video_clusters": positive,
        "negative_video_clusters": negative,
        "tied_video_clusters": tied,
        "net_positive_video_clusters": positive - negative,
        "exact_one_sided_sign_pvalue": exact_one_sided_pvalue(
            source_wins=positive, source_losses=negative,
        ),
    }
    row_metrics = dict(base["source_vs_direct"])
    applicability_by_operator = Counter(
        str(row["public_plan"]["temporal_operator"])
        for row in rows if row.get("public_plan") is not None
    )
    unsupported_preserved = all(
        row["source_prediction"] == row["direct_response"]
        for row in rows if not row["source_executor_authorized"]
    )
    gate = formal["formal_gates"]
    gates = {
        "qualification_dependency_passed": bool(
            qualification["grounder_qualified"]
        ),
        "same_frozen_grounder_as_qualification": (
            base["grounder_sha256"] == qualification["grounder_sha256"]
        ),
        "required_rows": len(rows) == int(gate["required_rows"]),
        "required_unique_videos": len(by_video)
        == int(gate["required_unique_videos"]),
        "required_row_count_histogram": {
            str(count): frequency
            for count, frequency in sorted(Counter(
                len(video_rows) for video_rows in by_video.values()
            ).items())
        } == gate["required_row_count_histogram"],
        "minimum_applicable_rows": base["applicable_rows"]
        >= int(gate["minimum_applicable_rows"]),
        "minimum_source_authorizations": base["source_executor_authorizations"]
        >= int(gate["minimum_source_authorizations"]),
        "minimum_row_wins": row_metrics["wins"]
        >= int(gate["minimum_row_wins"]),
        "maximum_row_losses": row_metrics["losses"]
        <= int(gate["maximum_row_losses"]),
        "minimum_row_net_gain": row_metrics["net_gain"]
        >= int(gate["minimum_row_net_gain"]),
        "maximum_row_exact_pvalue": row_metrics["exact_one_sided_pvalue"]
        <= float(gate["maximum_row_exact_pvalue"]),
        "minimum_positive_video_clusters": positive
        >= int(gate["minimum_positive_video_clusters"]),
        "maximum_negative_video_clusters": negative
        <= int(gate["maximum_negative_video_clusters"]),
        "maximum_cluster_exact_pvalue": cluster["exact_one_sided_sign_pvalue"]
        <= float(gate["maximum_cluster_exact_pvalue"]),
        "unsupported_or_abstained_rows_preserve_identical_direct": (
            unsupported_preserved
        ),
        "all_runtime_rows_outcome_blind": all(
            row.get(field) is False
            for row in rows
            for field in (
                "runtime_answer_read", "runtime_functional_program_read",
                "runtime_scene_graph_read", "runtime_source_identity_read",
            )
        ),
        "provider_cost_within_cap": float(base["reported_provider_cost_usd"])
        <= float(gate["maximum_reported_provider_cost_usd"]),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "agqa2-full-distribution-formal-report-v61",
        "status": (
            "AGQA2_FULL_DISTRIBUTION_V61_FORMAL_QUALIFIED" if passed
            else "AGQA2_FULL_DISTRIBUTION_V61_FORMAL_NOT_QUALIFIED"
        ),
        "confirmatory_claim": passed,
        "claim_boundary": config["claim_boundary"],
        "sample_count": len(rows),
        "unique_video_count": len(by_video),
        "rows_per_video": sorted({len(value) for value in by_video.values()}),
        "selection_without_operator_filter": True,
        "applicable_rows": base["applicable_rows"],
        "applicability_by_temporal_operator": dict(
            sorted(applicability_by_operator.items())
        ),
        "source_executor_authorizations": base["source_executor_authorizations"],
        "source_vs_direct_rows": row_metrics,
        "source_vs_direct_video_clusters": cluster,
        "formal_gates": gates,
        "grounder_sha256": base["grounder_sha256"],
        "qualification_report_sha256": qualification["report_sha256"],
        "base_report_sha256": base["report_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "reported_provider_cost_usd": base["reported_provider_cost_usd"],
        "interpretation": {
            "selective_cross_domain_transfer_on_full_agqa_distribution": passed,
            "all_agqa_question_families_solved": False,
            "full_agqa_distribution_solved": False,
            "source_induced_primitives_plus_target_native_composition": True,
            "source_provenance_necessary": False,
            "generic_or_target_written_equivalent_beaten": False,
        },
        "video_cluster_details": video_details,
    }
    result = body | {"report_sha256": stable_hash(body)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(
        config_path=args.config.resolve(),
        base_report_path=args.base_report.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps({key: result[key] for key in (
        "status", "sample_count", "unique_video_count", "applicable_rows",
        "source_executor_authorizations", "source_vs_direct_rows",
        "source_vs_direct_video_clusters", "formal_gates",
        "reported_provider_cost_usd", "report_sha256",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
