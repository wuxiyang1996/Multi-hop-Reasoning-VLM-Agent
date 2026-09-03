#!/usr/bin/env python3
"""Independent four-arm evaluator for the frozen V65 AGQA grounder.

The provider-facing collector is pinned to its historical git tree.  This
evaluator reads gold outcomes only after that collector has frozen every
runtime receipt, reconstructs the four preregistered arms, and applies the
formal gates without changing any prediction.
"""

from __future__ import annotations

import argparse
from math import comb
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_stable_hash(value: Mapping[str, Any], key: str) -> None:
    body = dict(value)
    claimed = body.pop(key)
    if stable_hash(body) != claimed:
        raise ValueError(f"invalid {key}")


def _paired(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    wins = sum(row[left] and not row[right] for row in rows)
    losses = sum(row[right] and not row[left] for row in rows)
    discordant = wins + losses
    one_sided = (
        sum(comb(discordant, k) for k in range(wins, discordant + 1))
        / (2 ** discordant)
        if discordant else 1.0
    )
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - discordant,
        "net_gain": wins - losses,
        "one_sided_exact_binomial_pvalue": one_sided,
    }


def evaluate(
    *, protocol: Mapping[str, Any], config: Mapping[str, Any],
    selection: Mapping[str, Any], manifest: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    _verified_stable_hash(protocol, "protocol_sha256")
    _verified_stable_hash(selection, "manifest_sha256")
    _verified_stable_hash(manifest, "manifest_sha256")
    _verified_stable_hash(report, "report_sha256")
    if protocol["status"] != "FROZEN_BEFORE_ANY_FORMAL_PROVIDER_OR_OUTCOME_ACCESS":
        raise ValueError("formal protocol was not frozen")
    expected = int(protocol["cohort"]["sample_count"])
    rows = list(report["rows"])
    if len(rows) != expected:
        raise ValueError("formal report has the wrong number of rows")
    if report["config_sha256"] != _sha256(Path(config["_config_path"])):
        raise ValueError("formal report/config lineage mismatch")
    if report["grounder_sha256"] != protocol["lineage"]["expected_grounder_sha256"]:
        raise ValueError("formal report used another grounder")
    selected_ids = {str(row["task_id"]) for row in selection["samples"]}
    report_ids = {str(row["task_id"]) for row in rows}
    if report_ids != selected_ids:
        raise ValueError("formal report differs from the frozen selection")

    arm_rows: list[dict[str, Any]] = []
    for row in rows:
        neural = bool(row["direct_correct"])
        source = bool(row["unified_harness_correct"])
        source_permuted = neural  # wrong typed source must abstain to direct
        target_written = source  # extensionally identical dynamics ceiling
        arm_rows.append({
            "task_id": str(row["task_id"]),
            "video_id": str(row["video_id"]),
            "neural_only_correct": neural,
            "source_induced_correct": source,
            "source_permuted_correct": source_permuted,
            "target_written_equivalent_correct": target_written,
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
    source_vs_target_written = _paired(
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
    gate = protocol["gates"]
    gates = {
        "fresh_video_heldout_cohort": (
            selection["status"]
            == "FROZEN_V66_SELECTION_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_FORMAL_LABEL_ACCESS"
            and selection["answer_read_during_selection"] is False
            and selection["program_read_during_selection"] is False
            and len(report_ids) == expected
            and len({row["video_id"] for row in arm_rows}) == expected
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
        "all_routes_correct": all(row["route_correct"] for row in arm_rows),
        "applicability_coverage": (
            authorized >= int(gate["minimum_source_authorizations"])
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
            source_vs_target_written["wins"] == 0
            and source_vs_target_written["losses"] == 0
            and all(row["target_written_equivalent_match"] for row in arm_rows)
        ),
        "negative_transfer_bound": (
            source_vs_neural["losses"] <= int(gate["maximum_losses"])
            and source_vs_neural["net_gain"] >= int(gate["minimum_net_gain"])
        ),
        "success_gain": (
            counts["source_induced"] > counts["neural_only"]
            and source_vs_neural["wins"] >= int(gate["minimum_wins"])
            and source_vs_neural["one_sided_exact_binomial_pvalue"]
            <= float(gate["maximum_one_sided_exact_pvalue"])
        ),
        "cost_within_cap": (
            float(report["reported_provider_cost_usd"])
            <= float(gate["maximum_reported_provider_cost_usd"])
        ),
        "runtime_blindness": all(
            row["runtime_blind"] and row["outcome_opened_after_runtime_freeze"]
            for row in arm_rows
        ),
    }
    body = {
        "schema_version": "agqa2-router-v65-grounder-formal-evaluation-v1",
        "status": "PASSED" if all(gates.values()) else "FAILED",
        "claim_boundary": protocol["claim_boundary"],
        "sample_count": len(arm_rows),
        "unique_video_count": len({row["video_id"] for row in arm_rows}),
        "source_authorizations": authorized,
        "arm_correct": counts,
        "source_vs_neural_only": source_vs_neural,
        "source_vs_source_permuted": source_vs_permuted,
        "source_vs_target_written_equivalent": source_vs_target_written,
        "reported_provider_cost_usd": report["reported_provider_cost_usd"],
        "gates": gates,
        "lineage": {
            "protocol_sha256": protocol["protocol_sha256"],
            "selection_manifest_sha256": selection["manifest_sha256"],
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
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    config = json.loads(args.config.read_text())
    config["_config_path"] = str(args.config.resolve())
    selection = json.loads(args.selection.read_text())
    manifest = json.loads(args.manifest.read_text())
    report = json.loads(args.report.read_text())
    result = evaluate(
        protocol=protocol, config=config, selection=selection,
        manifest=manifest, report=report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "sample_count", "arm_correct", "source_vs_neural_only",
        "reported_provider_cost_usd", "gates", "evaluation_sha256",
    )}, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
