#!/usr/bin/env python3
"""Apply the frozen aggregate gate to the four independent replications."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_counts(authentic: Mapping[str, bool], target: Mapping[str, bool]) -> dict[str, int]:
    if set(authentic) != set(target):
        raise ValueError("paired task coverage differs")
    deltas = [int(authentic[key]) - int(target[key]) for key in authentic]
    wins = sum(value > 0 for value in deltas)
    losses = sum(value < 0 for value in deltas)
    return {"wins": wins, "losses": losses, "ties": len(deltas) - wins - losses}


def exact_sign_p(wins: int, losses: int) -> float:
    total = wins + losses
    if not total:
        return 1.0
    tail = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(total, i) for i in range(tail + 1)) / 2**total)


def domain_row(
    *, tasks: int, authentic_successes: int, target_successes: int,
    paired: Mapping[str, int], evidence_status: str,
) -> dict[str, Any]:
    delta = (authentic_successes - target_successes) / tasks
    return {
        "tasks": tasks,
        "authentic_successes": authentic_successes,
        "target_successes": target_successes,
        "success_rate_delta": delta,
        "paired": {
            **paired,
            "exact_two_sided_p": exact_sign_p(paired["wins"], paired["losses"]),
        },
        "evidence_status": evidence_status,
        "gates": {
            "paired_wins_exceed_losses": paired["wins"] > paired["losses"],
            "nonnegative_success_delta": delta >= 0.0,
            "strict_positive_success_delta": delta > 0.0,
        },
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read(args.manifest)
    tir = read(args.tir)
    alf = read(args.alfworld)
    web = read(args.webshop)
    discovery = read(args.discoveryworld)

    tir_auth = tir["summaries"]["authentic_sokoban_topology_plus_target"]
    tir_target = tir["summaries"]["raw_target_only"]
    tir_pair = tir["paired"]["raw_target_only"]

    alf_auth_rows = {
        row["task_id"]: bool(row["official_success"])
        for row in alf["episodes"]["authentic_source_plus_target"]
    }
    alf_target_rows = {
        row["task_id"]: bool(row["official_success"])
        for row in alf["episodes"]["target_only"]
    }
    alf_pair = paired_counts(alf_auth_rows, alf_target_rows)
    alf_auth = alf["summaries"]["authentic_source_plus_target"]
    alf_target = alf["summaries"]["target_only"]

    web_pair = next(
        row for row in web["comparisons"] if row["comparator"] == "target_only"
    )
    web_auth = web["conditions"]["authentic_sokoban_effect_plus_target"]
    web_target = web["conditions"]["target_only"]

    discovery_auth_rows = {
        row["task_id"]: bool(row["authentic_sokoban_effect_plus_target"])
        for row in discovery["per_task"]
    }
    discovery_target_rows = {
        row["task_id"]: bool(row["target_native_myopic"])
        for row in discovery["per_task"]
    }
    discovery_pair = paired_counts(discovery_auth_rows, discovery_target_rows)
    discovery_counts = discovery["success_counts"]

    domains = {
        "tir": domain_row(
            tasks=int(tir_auth["tasks"]),
            authentic_successes=int(tir_auth["successes"]),
            target_successes=int(tir_target["successes"]),
            paired={key: int(tir_pair[key]) for key in ("wins", "losses", "ties")},
            evidence_status=str(tir["status"]),
        ),
        "alfworld": domain_row(
            tasks=int(alf_auth["tasks"]),
            authentic_successes=int(alf_auth["successes"]),
            target_successes=int(alf_target["successes"]),
            paired=alf_pair,
            evidence_status=str(alf["status"]),
        ),
        "webshop": domain_row(
            tasks=int(web["tasks"]),
            authentic_successes=int(web_auth["strict_successes"]),
            target_successes=int(web_target["strict_successes"]),
            paired={
                "wins": int(web_pair["strict_wins"]),
                "losses": int(web_pair["strict_losses"]),
                "ties": int(web_pair["strict_ties"]),
            },
            evidence_status=str(web["scientific_status"]),
        ),
        "discoveryworld": domain_row(
            tasks=int(discovery["eligible_forks"]),
            authentic_successes=int(discovery_counts["authentic_sokoban_effect_plus_target"]),
            target_successes=int(discovery_counts["target_native_myopic"]),
            paired=discovery_pair,
            evidence_status=str(discovery["status"]),
        ),
    }
    pooled_wins = sum(row["paired"]["wins"] for row in domains.values())
    pooled_losses = sum(row["paired"]["losses"] for row in domains.values())
    pooled_ties = sum(row["paired"]["ties"] for row in domains.values())
    strict_positive = sum(row["success_rate_delta"] > 0.0 for row in domains.values())
    gates = {
        "every_domain_paired_wins_exceed_losses": all(
            row["gates"]["paired_wins_exceed_losses"] for row in domains.values()
        ),
        "at_least_three_domains_strict_positive_success_delta": strict_positive >= 3,
        "no_domain_negative_success_delta": all(
            row["gates"]["nonnegative_success_delta"] for row in domains.values()
        ),
        "exact_four_domain_coverage": set(domains) == set(manifest["domains"]),
    }
    passed = all(gates.values())
    report = {
        "schema_version": "four-domain-independent-replication-summary-v1",
        "status": (
            "FOUR_DOMAIN_INDEPENDENT_REPLICATION_VALIDATED"
            if passed else "FOUR_DOMAIN_INDEPENDENT_REPLICATION_NOT_VALIDATED"
        ),
        "claim_boundary": manifest["aggregate_estimand"]["primary"],
        "domains": domains,
        "aggregate": {
            "strict_positive_delta_domains": strict_positive,
            "equal_domain_weight_mean_success_rate_delta": sum(
                row["success_rate_delta"] for row in domains.values()
            ) / len(domains),
            "pooled_paired": {
                "wins": pooled_wins,
                "losses": pooled_losses,
                "ties": pooled_ties,
                "exact_two_sided_p": exact_sign_p(pooled_wins, pooled_losses),
                "descriptive_only": True,
            },
        },
        "gates": gates,
        "all_predeclared_gates_passed": passed,
        "integrity": {
            "manifest_file_sha256": file_sha256(args.manifest),
            "input_file_sha256": {
                "tir": file_sha256(args.tir),
                "alfworld": file_sha256(args.alfworld),
                "webshop": file_sha256(args.webshop),
                "discoveryworld": file_sha256(args.discoveryworld),
            },
        },
        "limitations": [
            "Task units and success semantics differ by domain; pooled statistics are descriptive.",
            "This validates registered exact routes, not arbitrary source-game or arbitrary-target transfer.",
        ],
    }
    report["report_sha256"] = stable_hash(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tir", type=Path, required=True)
    parser.add_argument("--alfworld", type=Path, required=True)
    parser.add_argument("--webshop", type=Path, required=True)
    parser.add_argument("--discoveryworld", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
