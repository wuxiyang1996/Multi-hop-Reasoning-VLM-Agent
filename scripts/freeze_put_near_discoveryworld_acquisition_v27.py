#!/usr/bin/env python3
"""Freeze the retrospective second-program-family acquisition audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


DEFAULT_OUTPUT = (
    REPO / "configs/put_near_discoveryworld_acquisition_v27.json"
)


PATHS = {
    "source_portable_receipt": (
        "docs/results/put_near_source_induction_receipts_v26.json"
    ),
    "source_program": (
        "configs/source_structural_v5c_frozen/programs/put_near.json"
    ),
    "wrong_family_program": (
        "configs/source_structural_v5c_frozen/programs/unlock_pickup.json"
    ),
    "source_fresh_report": "runs/source_structural_v5c_fresh/report.json",
    "source_inducer": "src/motif_transfer/structural_delta_induction.py",
    "target_development_report": (
        "runs/discoveryworld_structural_grounder_v1_development/report_v2.json"
    ),
    "target_inducer": "src/motif_transfer/target_structural_induction.py",
    "target_formal_report": (
        "runs/discoveryworld_structural_transfer_v1_matched/report.json"
    ),
    "phase9_heterogeneity_report": (
        "docs/results/phase9_source_program_heterogeneity_v1.json"
    ),
    "phase14_acquisition_report": (
        "docs/results/alfworld_matched_acquisition_cost_v25.json"
    ),
    "compactor": (
        "scripts/compact_put_near_source_induction_receipts_v26.py"
    ),
    "analyzer": (
        "scripts/analyze_put_near_discoveryworld_acquisition_v27.py"
    ),
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_self_hash(path: Path) -> None:
    value = _read(path)
    candidates = (
        "receipt_sha256", "program_sha256", "report_sha256", "summary_sha256"
    )
    field = next((key for key in candidates if key in value), None)
    if field is None:
        raise ValueError(f"no self hash in dependency: {path}")
    body = dict(value)
    claimed = str(body.pop(field))
    if stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}: {path}")


def freeze() -> dict[str, Any]:
    for name, relative in PATHS.items():
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if name not in {"source_inducer", "target_inducer", "compactor", "analyzer"}:
            _validate_self_hash(path)
    hash_fields = {
        f"{name}_file_sha256": _sha(REPO / relative)
        for name, relative in PATHS.items()
    }
    body = {
        "schema_version": (
            "put-near-discoveryworld-acquisition-v27-protocol"
        ),
        "status": "FROZEN_RETROSPECTIVE_SECOND_FAMILY_PROTOCOL",
        "role": "retrospective_second_program_family_acquisition_audit",
        "claim_boundary": (
            "Uses already-consumed MiniGrid source intervention receipts, "
            "DiscoveryWorld development/qualification structural sequences, "
            "and an already-completed prospective formal report. It resets no "
            "environment, performs no model call, and adds no prospective "
            "success claim. Source exploration collections and target complete "
            "trajectories are different cost units and are reported separately."
        ),
        "estimand": (
            "Whether source acquisition value and target K=0/K=1 recovery "
            "replicate for a finite anonymous graph-edit sequence that is "
            "structurally distinct from the recurrent ALFWorld program."
        ),
        **PATHS,
        **hash_fields,
        "dependency_fields": {
            f"{name}_file_sha256": name for name in PATHS
        },
        "source_order_namespace": "phase15-put-near-source-order-v1",
        "frozen_gates": {
            "source_k0_abstain": True,
            "source_k1_below_minimum": True,
            "source_k2_matches_frozen_normal_form": True,
            "source_fresh_success_paths": 4,
            "target_k0_abstain": True,
            "every_target_k1_path_recovers": True,
            "target_single_demo_paths": 3,
            "require_zero_reversed_permuted_wrong_family_support": True,
            "require_existing_formal_source_above_neural_and_permuted": True,
            "require_distinct_from_recurrent_phase14_program": True,
            "forbid_new_target_success_claim": True,
        },
        "output": (
            "docs/results/put_near_discoveryworld_acquisition_v27.json"
        ),
    }
    return body | {"config_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else REPO / args.output
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen protocol: {output}")
    config = freeze()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": config["status"],
        "config_sha256": config["config_sha256"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
