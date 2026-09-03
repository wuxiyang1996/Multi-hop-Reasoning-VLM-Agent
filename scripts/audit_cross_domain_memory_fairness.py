#!/usr/bin/env python3
"""Audit a frozen four-domain memory suite before outcome comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.cross_domain_fairness import audit_target_bound_suite  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    expected_count = int(protocol["shared_inputs"]["source_episode_count"])
    fidelity = str(protocol["implementation_fidelity"]["level"])
    target_cohort_ready = (
        protocol.get("formal_target_cohort", {}).get("status")
        == "FROZEN_BEFORE_ANY_FORMAL_OUTCOME"
    )
    domains = ("webshop", "alfworld", "discoveryworld", "tirbench")
    methods = ("expel", "awm", "reasoning_bank")
    audits = {}
    for domain in domains:
        artifacts = {
            method: json.loads(
                (args.bound_root / domain / f"{method}.json").read_text(encoding="utf-8")
            )
            for method in methods
        }
        audits[domain] = asdict(audit_target_bound_suite(
            artifacts,
            target_domain=domain,
            expected_source_episodes=expected_count,
            implementation_fidelity=fidelity,
        ))
    body = {
        "schema_version": 1,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": stable_hash(protocol),
        "comparison_label": "clean-room style memory mechanisms",
        "artifact_suites_ready": all(row["formal_ready"] for row in audits.values()),
        "formal_target_cohort_ready": target_cohort_ready,
        "all_domains_formal_ready": (
            target_cohort_ready and all(row["formal_ready"] for row in audits.values())
        ),
        "exact_upstream_baseline_claim_allowed": all(
            row["exact_baseline_ready"] for row in audits.values()
        ),
        "domains": audits,
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_domains_formal_ready": report["all_domains_formal_ready"],
        "exact_upstream_baseline_claim_allowed": report["exact_upstream_baseline_claim_allowed"],
        "blockers": {domain: row["blockers"] for domain, row in audits.items()},
        "global_blockers": (
            [] if target_cohort_ready else ["formal target cohort is not frozen"]
        ),
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0 if report["all_domains_formal_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
