#!/usr/bin/env python3
"""Freeze the deterministic Phase 16 composite gap-closure audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen config: {output}")
    paths = {
        "alfworld_acquisition_report": (
            "docs/results/alfworld_matched_acquisition_cost_v25.json"
        ),
        "discoveryworld_acquisition_report": (
            "docs/results/put_near_discoveryworld_acquisition_v27.json"
        ),
        "cyclic_source_report": (
            "docs/results/tetris_cyclic_source_induction_v28.json"
        ),
        "target_synthesis_report": (
            "runs/target_schema_synthesis_v29/report.json"
        ),
        "alfworld_target_written_report": (
            "runs/alfworld_target_written_provenance_v15/report.json"
        ),
        "source_plan": (
            "runs/sokoban_effect_program_v2/fresh_confirmation_plan.json"
        ),
        "source_relation_dataset": (
            "runs/sokoban_goal_relation_macro_v3/"
            "discovery_macro_interventions.json"
        ),
        "source_acquisition_dataset": (
            "runs/sokoban_goal_acquisition_v1/"
            "discovery_acquisition_interventions.json"
        ),
        "source_cost_runtime": "src/motif_transfer/source_fork_cost.py",
        "source_primitive_runtime": (
            "src/motif_transfer/relational_structural_induction.py"
        ),
        "target_synthesis_runtime": (
            "src/motif_transfer/target_schema_synthesis.py"
        ),
        "analyzer": "scripts/analyze_phase16_gap_closure_v30.py",
    }
    body = {
        "schema_version": "phase16-gap-closure-v30-protocol",
        "status": "FROZEN_COMPOSITE_AUDIT",
        "role": "deterministic_audit_of_pre_frozen_component_results",
        "claim_boundary": (
            "This composite audit adds no provider call, source collection, "
            "target interaction, or success outcome. V28 and V29 were frozen "
            "before their respective source collection and model calls. V30 "
            "reconstructs omitted primitive source-fork costs and combines the "
            "hash-bound results without changing any component gate."
        ),
        **paths,
        **{
            f"{field}_file_sha256": _sha(REPO / path)
            for field, path in paths.items()
        },
        "dependency_fields": {
            f"{field}_file_sha256": field for field in paths
        },
        "source_primary_order_namespace": (
            "phase14-matched-acquisition-order-v1"
        ),
        "output": "docs/results/phase16_gap_closure_v30.json",
    }
    config = body | {"config_sha256": stable_hash(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "configs/phase16_gap_closure_v30.json",
    )
    args = parser.parse_args()
    print(json.dumps(freeze(args.output), indent=2))


if __name__ == "__main__":
    main()
