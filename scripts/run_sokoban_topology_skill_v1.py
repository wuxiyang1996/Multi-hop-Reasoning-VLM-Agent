#!/usr/bin/env python3
"""Build and fresh-confirm the source-only Sokoban topology executor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.sokoban_commit_skill import build_fresh_confirmation_plan  # noqa: E402
from motif_transfer.sokoban_topology_skill import (  # noqa: E402
    build_topology_artifact,
    confirm_topology_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discovery-plan", type=Path,
        default=REPO / "runs/sokoban_effect_program_v2/fresh_confirmation_plan.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "runs/sokoban_topology_skill_v1",
    )
    args = parser.parse_args()
    discovery = json.loads(args.discovery_plan.read_text(encoding="utf-8"))
    artifact = build_topology_artifact(discovery)
    # This plan is serialized before artifact prediction and uses disjoint new
    # procedural seeds.  Writing it before qualification preserves the audit.
    fresh = build_fresh_confirmation_plan(
        seeds=range(97001, 97025), snapshots_per_episode=4,
    )
    report = confirm_topology_artifact(artifact, fresh)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("discovery_artifact.json", artifact),
        ("fresh_confirmation_plan.json", fresh),
        ("fresh_confirmation_report.json", report),
    ):
        (args.output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "artifact_sha256": artifact["artifact_sha256"],
        "status": report["status"],
        "eligible_examples": report["eligible_examples"],
        "metrics": report["condition_metrics"],
        "gates": report["gates"],
        "report_sha256": report["report_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
