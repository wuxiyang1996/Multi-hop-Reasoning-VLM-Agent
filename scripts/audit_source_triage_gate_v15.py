#!/usr/bin/env python3
"""Audit whether source evidence authorizes the WebShop V15 tri-controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.source_triage_gate_v15 import (  # noqa: E402
    build_source_triage_report,
    file_sha256,
    read_json_object,
    read_jsonl_objects,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sokoban-artifact",
        type=Path,
        default=REPO / "runs/sokoban_effect_program_v2/discovery_artifact.json",
    )
    parser.add_argument(
        "--sokoban-confirmation",
        type=Path,
        default=REPO / "runs/sokoban_effect_program_v2/fresh_confirmation_report.json",
    )
    parser.add_argument(
        "--microcontroller-summary",
        type=Path,
        default=REPO / "docs/results/source_microcontroller_v1_summary.json",
    )
    parser.add_argument(
        "--microcontroller-rows",
        type=Path,
        default=(
            REPO
            / "runs/source_microcontroller_v1_gymv/execution/"
            "microcontroller_rows.jsonl"
        ),
    )
    parser.add_argument(
        "--topology-artifact",
        type=Path,
        default=REPO / "runs/sokoban_topology_skill_v1/discovery_artifact.json",
    )
    parser.add_argument(
        "--topology-confirmation",
        type=Path,
        default=(
            REPO / "runs/sokoban_topology_skill_v1/fresh_confirmation_report.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "docs/results/source_triage_gate_v15.json",
    )
    args = parser.parse_args()

    paths = {
        "sokoban_artifact": args.sokoban_artifact,
        "sokoban_confirmation": args.sokoban_confirmation,
        "microcontroller_summary": args.microcontroller_summary,
        "microcontroller_rows": args.microcontroller_rows,
        "topology_artifact": args.topology_artifact,
        "topology_confirmation": args.topology_confirmation,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing source evidence: {missing}")

    report = build_source_triage_report(
        sokoban_artifact=read_json_object(args.sokoban_artifact),
        sokoban_confirmation=read_json_object(args.sokoban_confirmation),
        microcontroller_summary=read_json_object(args.microcontroller_summary),
        microcontroller_rows=read_jsonl_objects(args.microcontroller_rows),
        topology_artifact=read_json_object(args.topology_artifact),
        topology_confirmation=read_json_object(args.topology_confirmation),
        input_provenance={
            name: {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "failed_branches": report["failed_branches"],
        "target_decision": report["target_execution"]["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
