#!/usr/bin/env python3
"""Induce acquisition structure on discovery and confirm on frozen fresh source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.source_goal_acquisition_induction import (  # noqa: E402
    build_goal_acquisition_dataset,
    confirm_goal_acquisition_program,
    induce_goal_acquisition_program,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(payload: dict, field: str) -> None:
    body = dict(payload)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/sokoban_goal_acquisition_v1_frozen/manifest.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "runs/sokoban_goal_acquisition_v1",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    _self_hash(manifest, "manifest_sha256")
    if manifest.get("status") != "FROZEN_BEFORE_ACQUISITION_CONFIRMATION":
        raise SystemExit("source acquisition manifest is not frozen")
    dependencies = {
        "induction_module_file_sha256": (
            REPO / "src/motif_transfer/source_goal_acquisition_induction.py"
        ),
        "runner_file_sha256": Path(__file__).resolve(),
        "discovery_plan_file_sha256": REPO / manifest["discovery_plan_path"],
        "fresh_source_plan_file_sha256": REPO / manifest[
            "fresh_source_plan_path"
        ],
        "relation_artifact_file_sha256": REPO / manifest[
            "relation_artifact_path"
        ],
    }
    for field, path in dependencies.items():
        if _sha256(path) != manifest[field]:
            raise SystemExit(f"frozen source acquisition dependency changed: {path}")
    discovery_plan = json.loads(
        dependencies["discovery_plan_file_sha256"].read_text(encoding="utf-8")
    )
    fresh_plan = json.loads(
        dependencies["fresh_source_plan_file_sha256"].read_text(encoding="utf-8")
    )
    relation = json.loads(
        dependencies["relation_artifact_file_sha256"].read_text(encoding="utf-8")
    )
    discovery = build_goal_acquisition_dataset(discovery_plan, relation)
    fresh = build_goal_acquisition_dataset(fresh_plan, relation)
    artifact = induce_goal_acquisition_program(discovery)
    report = confirm_goal_acquisition_program(artifact, fresh)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "discovery_acquisition_interventions.json": discovery,
        "fresh_acquisition_interventions.json": fresh,
        "artifact.json": artifact,
        "fresh_confirmation_report.json": report,
    }
    for name, payload in outputs.items():
        (args.output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    summary_body = {
        "schema_version": "sokoban-goal-acquisition-summary-v1",
        "status": str(report["status"]),
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "artifact_sha256": str(artifact["artifact_sha256"]),
        "fresh_report_sha256": str(report["report_sha256"]),
        "program": artifact["program"],
        "metrics": report["metrics"],
        "gates": report["gates"],
    }
    summary = summary_body | {"summary_sha256": stable_hash(summary_body)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if report["source_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
