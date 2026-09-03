#!/usr/bin/env python3
"""Relineage four validated V16 targets to the six-game common artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase1_target_relineage import (  # noqa: E402
    summarize_domain_relineage,
)
from motif_transfer.search_automaton_transfer_v16 import (  # noqa: E402
    SourceSearchAutomaton,
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-source", type=Path,
        default=REPO / "runs/sokoban_search_automaton_v16/artifact.json",
    )
    parser.add_argument(
        "--new-source", type=Path,
        default=(
            REPO / "runs/phase1_common_search_ir_formal_v1/"
            "common_search_automaton_artifact.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=(
            REPO / "runs/phase1_common_search_ir_formal_v1/"
            "four_target_relineage_report.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    old_source = SourceSearchAutomaton(_read(args.old_source))
    new_source = SourceSearchAutomaton(_read(args.new_source))
    specifications = {
        "webshop": {
            "report": REPO / "runs/webshop_search_automaton_v16_formal/report.json",
            "supplemental": sorted((
                REPO / "runs/webshop_search_automaton_v16_formal"
            ).glob("webshop.*.*.json")),
            "tier": "HISTORICAL_PROSPECTIVELY_FROZEN_FRESH_FORMAL",
        },
        "alfworld": {
            "report": REPO / "runs/alfworld_search_automaton_v16_development/report.json",
            "supplemental": [],
            "tier": "HISTORICAL_CONSUMED_DEVELOPMENT_REEXECUTION",
        },
        "discoveryworld": {
            "report": REPO / "runs/discoveryworld_search_automaton_v16/equivalence_report.json",
            "supplemental": [],
            "tier": "HISTORICAL_RETROSPECTIVE_EQUIVALENCE",
        },
        "tirbench": {
            "report": REPO / "runs/tir_search_automaton_v16/reanalysis_report.json",
            "supplemental": [],
            "tier": "HISTORICAL_CONSUMED_FRESH_FORMAL_REANALYSIS",
        },
    }
    domains = {}
    for domain, spec in specifications.items():
        supplemental_paths = list(spec["supplemental"])
        domains[domain] = summarize_domain_relineage(
            domain=domain,
            target_report=_read(spec["report"]),
            target_report_path=spec["report"],
            supplemental_receipts=[_read(path) for path in supplemental_paths],
            supplemental_paths=supplemental_paths,
            old_source=old_source,
            new_source=new_source,
            evidence_tier=str(spec["tier"]),
        )
    passed = all(
        all(report["gates"].values()) for report in domains.values()
    )
    body = {
        "schema_version": "phase1-six-game-four-target-relineage-v1",
        "status": (
            "FOUR_TARGET_PROGRAM_EQUIVALENCE_PASSED"
            if passed else "FOUR_TARGET_PROGRAM_EQUIVALENCE_FAILED"
        ),
        "old_source_artifact_sha256": old_source.artifact_sha256,
        "new_source_artifact_sha256": new_source.artifact_sha256,
        "domains": domains,
        "gates": {
            "all_four_domains_relineaged": set(domains) == {
                "webshop", "alfworld", "discoveryworld", "tirbench"
            },
            "all_domain_gates_passed": passed,
            "new_source_policy_identical_to_target_validated_policy": (
                old_source.policy == new_source.policy
            ),
        },
        "claim_boundary": (
            "SOURCE_IS_NEW_SIX_GAME_FORMAL_EVIDENCE;TARGET_OUTCOMES_ARE_"
            "HISTORICAL_AND_NOT_NEW_PROSPECTIVE_RUNS;EVERY_RECORDED_ROUTE_"
            "IS_EXHAUSTIVELY_RELINEAGED"
        ),
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "new_source_artifact_sha256": new_source.artifact_sha256,
        "domain_decisions": {
            domain: value["routed_decisions"]
            for domain, value in domains.items()
        },
        "report_sha256": report["report_sha256"],
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
