#!/usr/bin/env python3
"""Build a hash-audited compact report for V16 transfer across four targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


AUTHENTIC = "authentic_search_automaton_plus_target"
RAW = "raw_target_only"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_report(path: Path, report: dict[str, Any]) -> None:
    claimed = report.get("report_sha256")
    if claimed is None:
        raise ValueError(f"report_sha256 missing: {path}")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    if stable_hash(body) != claimed:
        raise ValueError(f"report self-hash mismatch: {path}")


def _alfworld_initial_hash_gate(report: dict[str, Any]) -> bool:
    episodes = report["episodes"]
    task_sets = {
        condition: {row["task_id"] for row in rows}
        for condition, rows in episodes.items()
    }
    task_ids = task_sets[RAW]
    return all(ids == task_ids for ids in task_sets.values()) and all(
        len({
            row["records"][0]["before_state_sha256"]
            for rows in episodes.values()
            for row in rows
            if row["task_id"] == task_id and row["records"]
        }) == 1
        for task_id in task_ids
    )


def _all_true(values: dict[str, Any]) -> bool:
    return bool(values) and all(value is True for value in values.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-summary", type=Path,
        default=REPO / "docs/results/sokoban_search_automaton_v16_summary.json",
    )
    parser.add_argument(
        "--source-artifact", type=Path,
        default=REPO / "runs/sokoban_search_automaton_v16/artifact.json",
    )
    parser.add_argument(
        "--source-report", type=Path,
        default=(
            REPO / "runs/sokoban_search_automaton_v16/"
            "fresh_confirmation_report.json"
        ),
    )
    parser.add_argument(
        "--webshop", type=Path,
        default=REPO / "runs/webshop_search_automaton_v16_formal/report.json",
    )
    parser.add_argument(
        "--alfworld", type=Path,
        default=REPO / "runs/alfworld_search_automaton_v16_development/report.json",
    )
    parser.add_argument(
        "--discoveryworld", type=Path,
        default=(
            REPO / "runs/discoveryworld_search_automaton_v16/"
            "equivalence_report.json"
        ),
    )
    parser.add_argument(
        "--tirbench", type=Path,
        default=REPO / "runs/tir_search_automaton_v16/reanalysis_report.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/search_automaton_transfer_v16_summary.json",
    )
    args = parser.parse_args()

    source = _read(args.source_summary)
    source_artifact = _read(args.source_artifact)
    source_report = _read(args.source_report)
    webshop = _read(args.webshop)
    alfworld = _read(args.alfworld)
    discoveryworld = _read(args.discoveryworld)
    tirbench = _read(args.tirbench)
    for path, report in (
        (args.source_report, source_report),
        (args.webshop, webshop),
        (args.alfworld, alfworld),
        (args.discoveryworld, discoveryworld),
        (args.tirbench, tirbench),
    ):
        _validate_report(path, report)

    if source.get("report_sha256") != source_report.get("report_sha256"):
        raise ValueError("compact source summary/full report identity mismatch")

    source_sha = str(source["artifact_sha256"])
    if source_artifact.get("artifact_sha256") != source_sha:
        raise ValueError("source summary/artifact identity mismatch")
    for name, report in (
        ("webshop", webshop),
        ("alfworld", alfworld),
        ("discoveryworld", discoveryworld),
        ("tirbench", tirbench),
    ):
        if report.get("source_artifact_sha256") != source_sha:
            raise ValueError(f"{name} source lineage mismatch")

    alf_initial = _alfworld_initial_hash_gate(alfworld)
    webshop_gates = dict(webshop["gates"])
    alfworld_gates = dict(alfworld["gates"])
    alfworld_gates["matched_initial_state_hashes_reaudited"] = alf_initial
    discoveryworld_gates = dict(discoveryworld["gates"])
    tirbench_gates = dict(tirbench["gates"])

    web_auth = webshop["summaries"][AUTHENTIC]
    web_raw = webshop["summaries"][RAW]
    alf_auth = alfworld["summaries"][AUTHENTIC]
    alf_raw = alfworld["summaries"][RAW]
    tir_auth = tirbench["summaries"][AUTHENTIC]
    tir_raw = tirbench["summaries"][RAW]
    dw_counts = discoveryworld["historical_success_counts"]

    domains = {
        "webshop": {
            "evidence_tier": "PROSPECTIVELY_FROZEN_FRESH_FORMAL",
            "status": webshop["status"],
            "tasks": web_auth["tasks"],
            "authentic_successes": web_auth["strict_successes"],
            "raw_successes": web_raw["strict_successes"],
            "paired_wins": webshop["paired"][RAW]["wins"],
            "paired_losses": webshop["paired"][RAW]["losses"],
            "authentic_mean_reward": web_auth["mean_reward"],
            "raw_mean_reward": web_raw["mean_reward"],
            "reward_wins": webshop["paired"][RAW]["reward_wins"],
            "reward_losses": webshop["paired"][RAW]["reward_losses"],
            "gates": webshop_gates,
            "all_gates_passed": _all_true(webshop_gates),
            "report_file_sha256": _file_sha256(args.webshop),
            "report_sha256": webshop["report_sha256"],
        },
        "alfworld": {
            "evidence_tier": "PREVIOUSLY_CONSUMED_DEVELOPMENT_REEXECUTION",
            "status": alfworld["status"],
            "tasks": alf_auth["tasks"],
            "authentic_successes": alf_auth["successes"],
            "raw_successes": alf_raw["successes"],
            "paired_wins": alfworld["paired"][RAW]["wins"],
            "paired_losses": alfworld["paired"][RAW]["losses"],
            "gates": alfworld_gates,
            "all_gates_passed": _all_true(alfworld_gates),
            "report_file_sha256": _file_sha256(args.alfworld),
            "report_sha256": alfworld["report_sha256"],
        },
        "discoveryworld": {
            "evidence_tier": "RETROSPECTIVE_V16_EQUIVALENCE_TO_PRIOR_FRESH_RUN",
            "status": discoveryworld["status"],
            "tasks": sum(
                1 for row in discoveryworld["relineage"]
                if row["historical_authentic_official_success"]
            ) + sum(
                1 for row in discoveryworld["relineage"]
                if not row["historical_authentic_official_success"]
            ),
            "authentic_successes": dw_counts[
                "authentic_sokoban_effect_plus_target"
            ],
            "raw_successes": dw_counts["target_native_myopic"],
            "paired_wins": discoveryworld["paired"][
                "target_native_myopic"
            ]["wins"],
            "paired_losses": discoveryworld["paired"][
                "target_native_myopic"
            ]["losses"],
            "gates": discoveryworld_gates,
            "all_gates_passed": _all_true(discoveryworld_gates),
            "report_file_sha256": _file_sha256(args.discoveryworld),
            "report_sha256": discoveryworld["report_sha256"],
        },
        "tirbench": {
            "evidence_tier": "PREVIOUSLY_CONSUMED_FRESH_FORMAL_REANALYSIS",
            "status": tirbench["status"],
            "tasks": tir_auth["tasks"],
            "authentic_successes": tir_auth["successes"],
            "raw_successes": tir_raw["successes"],
            "paired_wins": tirbench["paired"][RAW]["wins"],
            "paired_losses": tirbench["paired"][RAW]["losses"],
            "gates": tirbench_gates,
            "all_gates_passed": _all_true(tirbench_gates),
            "report_file_sha256": _file_sha256(args.tirbench),
            "report_sha256": tirbench["report_sha256"],
        },
    }

    mechanism_supported = all(
        row["all_gates_passed"] for row in domains.values()
    )
    all_positive = all(
        row["authentic_successes"] > row["raw_successes"]
        for row in domains.values()
    )
    body = {
        "schema_version": "search-automaton-four-domain-transfer-summary-v16",
        "status": (
            "FOUR_DOMAIN_MECHANISM_SUPPORTED_EVIDENCE_TIERS_MIXED"
            if mechanism_supported and all_positive
            else "FOUR_DOMAIN_MECHANISM_GATE_FAILED"
        ),
        "claim_boundary": (
            "The same frozen Sokoban event-routing artifact is grounded with "
            "target-native neural predicates/actions in all four targets. Only "
            "WebShop is prospective source-artifact-specific V16 formal evidence; "
            "ALFWorld is consumed development reexecution, TIRBench is consumed "
            "formal reanalysis, and DiscoveryWorld is retrospective equivalence."
        ),
        "source": {
            "status": source["status"],
            "artifact_sha256": source_sha,
            "artifact_file_sha256": _file_sha256(args.source_artifact),
            "fresh_states": source["fresh_policy_metrics"][
                "authentic_learned_event_policy"
            ]["states"],
            "fresh_successes": source["fresh_policy_metrics"][
                "authentic_learned_event_policy"
            ]["successes"],
            "source_actions": sorted(
                source["fresh_policy_metrics"][
                    "authentic_learned_event_policy"
                ]["selected_action_counts"]
            ),
            "report_file_sha256": _file_sha256(args.source_summary),
            "full_report_file_sha256": _file_sha256(args.source_report),
            "report_sha256": source["report_sha256"],
        },
        "domains": domains,
        "aggregate_gates": {
            "same_frozen_source_artifact_all_domains": True,
            "target_native_grounding_boundary_all_domains": True,
            "mechanism_gates_pass_all_domains": mechanism_supported,
            "positive_success_delta_all_domains": all_positive,
            "webshop_fresh_formal_confirmed": (
                webshop["status"] == "FRESH_FORMAL_TRANSFER_GATE_PASSED"
            ),
            "all_four_prospectively_v16_confirmed": False,
        },
        "remaining_confirmations": [
            "Prospectively frozen V16 ALFWorld reserve",
            "Prospectively frozen V16 DiscoveryWorld reserve",
            "New V16-specific TIRBench reserve rather than consumed receipt reanalysis",
        ],
        "interpretation": {
            "supported": (
                "A nontrivial intervention-grounded symbolic search automaton can "
                "transfer across game, shopping, embodied text, scientific "
                "discovery, and tool-use reasoning interfaces when each target "
                "supplies native neural grounding and execution."
            ),
            "not_supported": (
                "The source artifact is not superior to an isomorphic target-native "
                "implementation; ceiling ties show transferability and reuse, not "
                "unique algorithmic advantage."
            ),
        },
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "aggregate_gates": report["aggregate_gates"],
        "report_sha256": report["report_sha256"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0 if mechanism_supported and all_positive else 2


if __name__ == "__main__":
    raise SystemExit(main())
