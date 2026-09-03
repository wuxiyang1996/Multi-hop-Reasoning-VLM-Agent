#!/usr/bin/env python3
"""Induce on consumed source rollouts and confirm on frozen fresh rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.relational_structural_induction import (  # noqa: E402
    build_source_intervention_dataset,
    confirm_relational_structural_program,
    induce_relational_structural_program,
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
        default=REPO / "configs/sokoban_relational_structural_v2_frozen/manifest.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "runs/sokoban_relational_structural_v2",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    _self_hash(manifest, "manifest_sha256")
    if manifest.get("status") != "FROZEN_BEFORE_SOURCE_PROGRAM_CONFIRMATION":
        raise SystemExit("source relational manifest is not frozen")
    module_path = REPO / "src/motif_transfer/relational_structural_induction.py"
    if _sha256(module_path) != manifest["induction_module_file_sha256"]:
        raise SystemExit("source relational induction code changed after freeze")
    discovery_path = REPO / manifest["discovery_plan_path"]
    fresh_path = REPO / manifest["fresh_source_plan_path"]
    for path, field in (
        (discovery_path, "discovery_plan_file_sha256"),
        (fresh_path, "fresh_source_plan_file_sha256"),
    ):
        if _sha256(path) != manifest[field]:
            raise SystemExit(f"frozen source dependency changed: {path}")
    discovery_plan = json.loads(discovery_path.read_text(encoding="utf-8"))
    fresh_plan = json.loads(fresh_path.read_text(encoding="utf-8"))
    discovery_dataset = build_source_intervention_dataset(discovery_plan)
    fresh_dataset = build_source_intervention_dataset(fresh_plan)
    artifact = induce_relational_structural_program(discovery_dataset)
    report = confirm_relational_structural_program(artifact, fresh_dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "discovery_interventions.json": discovery_dataset,
        "fresh_interventions.json": fresh_dataset,
        "artifact.json": artifact,
        "fresh_confirmation_report.json": report,
    }
    for name, payload in outputs.items():
        (args.output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    summary_body = {
        "schema_version": "sokoban-relational-structural-summary-v2",
        "status": report["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "artifact_sha256": artifact["artifact_sha256"],
        "fresh_report_sha256": report["report_sha256"],
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
