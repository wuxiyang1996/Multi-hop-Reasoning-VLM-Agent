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
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--bound-root", type=Path)
    parser.add_argument("--methods", nargs="+")
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
    methods = tuple(args.methods or protocol["comparison_methods"])
    panel_roots = {"raw": args.raw_root, "gated": args.bound_root}
    if not any(panel_roots.values()):
        raise SystemExit("at least one of --raw-root or --bound-root is required")
    panels = {}
    for panel_name, root in panel_roots.items():
        if root is None:
            continue
        audits = {}
        for domain in domains:
            artifacts = {
                method: json.loads(
                    ((root / f"{method}.json") if panel_name == "raw" else
                     (root / domain / f"{method}.json")).read_text(encoding="utf-8")
                )
                for method in methods
            }
            audits[domain] = asdict(audit_target_bound_suite(
                artifacts, target_domain=domain,
                expected_source_episodes=expected_count,
                implementation_fidelity=fidelity,
                transfer_panel=panel_name,
                expected_methods=methods,
            ))
        panels[panel_name] = {
            "formal_ready": all(row["formal_ready"] for row in audits.values()),
            "domains": audits,
        }
    body = {
        "schema_version": 1,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": stable_hash(protocol),
        "comparison_labels": protocol["implementation_fidelity"]["required_result_labels"],
        "comparison_footnote": protocol["implementation_fidelity"]["required_table_footnote"],
        "artifact_suites_ready": (
            set(panels) == {"raw", "gated"}
            and all(row["formal_ready"] for row in panels.values())
        ),
        "formal_target_cohort_ready": target_cohort_ready,
        "all_domains_formal_ready": (
            target_cohort_ready and set(panels) == {"raw", "gated"}
            and all(row["formal_ready"] for row in panels.values())
        ),
        "exact_upstream_baseline_claim_allowed": all(
            domain["exact_baseline_ready"]
            for panel in panels.values() for domain in panel["domains"].values()
        ),
        "panels": panels,
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_domains_formal_ready": report["all_domains_formal_ready"],
        "exact_upstream_baseline_claim_allowed": report["exact_upstream_baseline_claim_allowed"],
        "blockers": {
            panel_name: {
                domain: row["blockers"]
                for domain, row in panel["domains"].items()
            }
            for panel_name, panel in panels.items()
        },
        "global_blockers": (
            [] if target_cohort_ready else ["formal target cohort is not frozen"]
        ),
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0 if report["all_domains_formal_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
