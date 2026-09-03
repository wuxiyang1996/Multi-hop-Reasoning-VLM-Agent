#!/usr/bin/env python3
"""Freeze untouched procedural source episodes for acquisition confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.sokoban_commit_skill import (  # noqa: E402
    build_fresh_confirmation_plan,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discovery-plan", type=Path,
        default=REPO / "runs/sokoban_effect_program_v2/fresh_confirmation_plan.json",
    )
    parser.add_argument(
        "--relation-artifact", type=Path,
        default=REPO / "runs/sokoban_goal_relation_macro_v3/artifact.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "configs/sokoban_goal_acquisition_v1_frozen",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_module = (
        REPO / "src/motif_transfer/source_goal_acquisition_induction.py"
    )
    runner = REPO / "scripts/induce_sokoban_goal_acquisition_v1.py"
    fresh = build_fresh_confirmation_plan(
        seeds=range(299001, 299025), snapshots_per_episode=4,
    )
    fresh_path = args.output_dir / "fresh_source_plan.json"
    fresh_path.write_text(
        json.dumps(fresh, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    body = {
        "schema_version": "sokoban-goal-acquisition-freeze-v1",
        "status": "FROZEN_BEFORE_ACQUISITION_CONFIRMATION",
        "claim_boundary": "SOURCE_ONLY;NO_TARGET_DATA",
        "discovery_plan_path": str(args.discovery_plan.relative_to(REPO)),
        "discovery_plan_file_sha256": _sha256(args.discovery_plan),
        "fresh_source_plan_path": str(fresh_path.relative_to(REPO)),
        "fresh_source_plan_file_sha256": _sha256(fresh_path),
        "fresh_seed_range": [299001, 299024],
        "relation_artifact_path": str(args.relation_artifact.relative_to(REPO)),
        "relation_artifact_file_sha256": _sha256(args.relation_artifact),
        "induction_module_file_sha256": _sha256(source_module),
        "runner_file_sha256": _sha256(runner),
        "target_data_read": False,
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
