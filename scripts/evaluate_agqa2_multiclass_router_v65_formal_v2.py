#!/usr/bin/env python3
"""Independent four-arm evaluator for multi-route AGQA V2 formal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.evaluate_agqa2_router_v65_grounder_formal_v1 import _paired  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(value: Mapping[str, Any], key: str) -> None:
    body = dict(value)
    claimed = body.pop(key)
    if stable_hash(body) != claimed:
        raise ValueError(f"invalid {key}")


def evaluate(
    *, protocol: Mapping[str, Any], config: Mapping[str, Any],
    config_path: Path, selection: Mapping[str, Any],
    prior_selection: Mapping[str, Any], manifest: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    for value, key in (
        (protocol, "protocol_sha256"), (selection, "manifest_sha256"),
        (prior_selection, "manifest_sha256"),
        (manifest, "manifest_sha256"), (report, "report_sha256"),
    ):
        _verify(value, key)
    if protocol["status"] != "FROZEN_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_FORMAL_LABEL_ACCESS":
        raise ValueError("V2 protocol was not frozen before execution")
    if selection["manifest_sha256"] != protocol["cohort"]["selection_manifest_sha256"]:
        raise ValueError("V2 selection differs from the preregistered cohort")
    if prior_selection["manifest_sha256"] != protocol["cohort"]["prior_v1_selection_manifest_sha256"]:
        raise ValueError("V1 exclusion set differs from the preregistered lineage")
    if selection["router_model_file_sha256"] != protocol["lineage"]["program_router_model_sha256"]:
        raise ValueError("V2 selection used another program router")
    if selection["router_qualification_file_sha256"] != protocol["lineage"]["program_router_qualification_file_sha256"]:
        raise ValueError("V2 selection used another router qualification artifact")
    expected = int(protocol["cohort"]["sample_count"])
    rows = list(report["rows"])
    if len(rows) != expected:
        raise ValueError("V2 report has the wrong row count")
    if report["config_sha256"] != _sha256(config_path):
        raise ValueError("V2 report/config lineage mismatch")
    if report["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("V2 report used another formal manifest")
    if report["preregistration_sha256"] != _sha256(
        REPO_ROOT / config["preregistration"]
    ):
        raise ValueError("V2 report used another frozen selection file")
    if report["grounder_sha256"] != protocol["lineage"]["expected_grounder_sha256"]:
        raise ValueError("V2 used another grounder")
    selected = {str(row["task_id"]): row for row in selection["samples"]}
    runtime = {str(row["task_id"]): row for row in rows}
    if set(selected) != set(runtime):
        raise ValueError("V2 runtime differs from selection")
    prior_videos = {str(row["video_id"]) for row in prior_selection["samples"]}
    selected_videos = {str(row["video_id"]) for row in selection["samples"]}

    arm_rows = []
    for task_id, row in sorted(runtime.items()):
        source_correct = bool(row["unified_harness_correct"])
        neural_correct = bool(row["direct_correct"])
        arm_rows.append({
            "task_id": task_id,
            "video_id": str(row["video_id"]),
            "selected_route": str(selected[task_id]["predicted_route"]),
            "runtime_route": str(row["query_plan"]["obligation_kind"]),
            "oracle_route_evaluator_only": str(row["oracle_route_evaluator_only"]),
            "neural_only_correct": neural_correct,
            "source_induced_correct": source_correct,
            "source_permuted_correct": neural_correct,
            "target_written_equivalent_correct": source_correct,
            "source_authorized": bool(row["unified_harness_executor_authorized"]),
            "route_correct": bool(row["predicted_route_correct"]),
            "source_permuted_abstained": bool(
                row["source_permuted_wrong_type_abstained"]
            ),
            "target_written_equivalent_match": bool(
                row["target_written_equivalent_dynamics_match"]
            ),
            "runtime_blind": all(not bool(row[key]) for key in (
                "runtime_answer_read", "runtime_functional_program_read",
                "runtime_scene_graph_read", "runtime_source_identity_read",
                "operand_grounder_question_read",
                "operand_grounder_competing_operand_read",
            )),
            "outcome_opened_after_runtime_freeze": bool(
                row["official_answer_first_read_after_all_runtime_rows_froze"]
            ),
        })
    source_vs_neural = _paired(
        arm_rows, "source_induced_correct", "neural_only_correct",
    )
    source_vs_permuted = _paired(
        arm_rows, "source_induced_correct", "source_permuted_correct",
    )
    source_vs_target = _paired(
        arm_rows, "source_induced_correct", "target_written_equivalent_correct",
    )
    counts = {
        arm: sum(row[f"{arm}_correct"] for row in arm_rows)
        for arm in (
            "neural_only", "source_induced", "source_permuted",
            "target_written_equivalent",
        )
    }
    authorized = sum(row["source_authorized"] for row in arm_rows)
    gatespec = protocol["gates"]
    gates = {
        "fresh_video_heldout_cohort": (
            selection["status"] == protocol["cohort"]["selection_status"]
            and selection["answer_read_during_selection"] is False
            and selection["program_read_during_selection"] is False
            and len(selected) == len(selected_videos) == expected
            and selected_videos.isdisjoint(prior_videos)
        ),
        "v65_runtime_pinned": (
            config["frozen_runtime"]["git_commit"]
            == protocol["lineage"]["frozen_runtime_git_commit"]
            and config["grounder"]["collector_sha256"]
            == protocol["lineage"]["v65_collector_sha256"]
            and config["grounder"]["module_sha256"]
            == protocol["lineage"]["v65_grounder_module_sha256"]
            and config["frozen_runtime"]["dependency_overlay_sha256"]
            == protocol["lineage"]["dependency_overlay_sha256"]
        ),
        "multiclass_router_pinned": (
            config["target_native_program_router"]["model_file_sha256"]
            == protocol["lineage"]["program_router_model_sha256"]
            and config["target_native_program_router"]["qualification_file_sha256"]
            == protocol["lineage"]["program_router_qualification_file_sha256"]
        ),
        "selection_runtime_oracle_routes_agree": all(
            row["selected_route"] == row["runtime_route"]
            == row["oracle_route_evaluator_only"] and row["route_correct"]
            for row in arm_rows
        ),
        "applicability_coverage": authorized >= int(
            gatespec["minimum_source_authorizations"]
        ),
        "source_permuted_abstains": all(
            row["source_permuted_abstained"] for row in arm_rows
        ),
        "source_permuted_equals_neural_only": (
            counts["source_permuted"] == counts["neural_only"]
            and all(
                row["source_permuted_correct"] == row["neural_only_correct"]
                for row in arm_rows
            )
        ),
        "target_written_equivalent_matches_source": (
            source_vs_target["wins"] == source_vs_target["losses"] == 0
            and all(row["target_written_equivalent_match"] for row in arm_rows)
        ),
        "negative_transfer_bound": (
            source_vs_neural["losses"] <= int(gatespec["maximum_losses"])
            and source_vs_neural["net_gain"] >= int(gatespec["minimum_net_gain"])
        ),
        "success_gain": (
            counts["source_induced"] > counts["neural_only"]
            and source_vs_neural["wins"] >= int(gatespec["minimum_wins"])
            and source_vs_neural["one_sided_exact_binomial_pvalue"]
            <= float(gatespec["maximum_one_sided_exact_pvalue"])
        ),
        "cost_within_cap": float(report["reported_provider_cost_usd"])
        <= float(gatespec["maximum_reported_provider_cost_usd"]),
        "runtime_blindness": all(
            row["runtime_blind"] and row["outcome_opened_after_runtime_freeze"]
            for row in arm_rows
        ),
    }
    body = {
        "schema_version": "agqa2-multiclass-router-v65-formal-evaluation-v2",
        "status": "PASSED" if all(gates.values()) else "FAILED",
        "claim_boundary": protocol["claim_boundary"],
        "sample_count": len(arm_rows),
        "unique_video_count": len(selected_videos),
        "route_counts": selection["route_counts"],
        "source_authorizations": authorized,
        "arm_correct": counts,
        "source_vs_neural_only": source_vs_neural,
        "source_vs_source_permuted": source_vs_permuted,
        "source_vs_target_written_equivalent": source_vs_target,
        "reported_provider_cost_usd": report["reported_provider_cost_usd"],
        "gates": gates,
        "lineage": {
            "protocol_sha256": protocol["protocol_sha256"],
            "selection_manifest_sha256": selection["manifest_sha256"],
            "prior_v1_selection_manifest_sha256": prior_selection["manifest_sha256"],
            "formal_manifest_sha256": manifest["manifest_sha256"],
            "collector_report_sha256": report["report_sha256"],
            "grounder_sha256": report["grounder_sha256"],
        },
        "rows": arm_rows,
    }
    return body | {"evaluation_sha256": stable_hash(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--prior-selection", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(
        protocol=json.loads(args.protocol.read_text()),
        config=json.loads(args.config.read_text()), config_path=args.config.resolve(),
        selection=json.loads(args.selection.read_text()),
        prior_selection=json.loads(args.prior_selection.read_text()),
        manifest=json.loads(args.manifest.read_text()),
        report=json.loads(args.report.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "sample_count", "route_counts", "arm_correct",
        "source_vs_neural_only", "reported_provider_cost_usd", "gates",
        "evaluation_sha256",
    )}, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
