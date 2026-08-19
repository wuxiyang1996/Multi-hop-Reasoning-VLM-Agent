#!/usr/bin/env python3
"""Freeze the retrospective ALFWorld matched-acquisition audit."""

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


DEFAULT_OUTPUT = REPO / "configs/alfworld_matched_acquisition_cost_v25.json"


PATHS = {
    "source_discovery_acquisition": (
        "runs/sokoban_goal_acquisition_v1/"
        "discovery_acquisition_interventions.json"
    ),
    "source_fresh_acquisition": (
        "runs/sokoban_goal_acquisition_v1/"
        "fresh_acquisition_interventions.json"
    ),
    "source_discovery_relation": (
        "runs/sokoban_goal_relation_macro_v3/"
        "discovery_macro_interventions.json"
    ),
    "source_fresh_relation": (
        "runs/sokoban_goal_relation_macro_v3/"
        "fresh_macro_interventions.json"
    ),
    "source_causal_inducer": (
        "src/motif_transfer/source_goal_relation_causal_budget.py"
    ),
    "source_relation_inducer": (
        "src/motif_transfer/source_goal_relation_induction.py"
    ),
    "source_acquisition_inducer": (
        "src/motif_transfer/source_goal_acquisition_induction.py"
    ),
    "target_recurrent_inducer": (
        "src/motif_transfer/alfworld_target_recurrent_induction.py"
    ),
    "target_v13_report": (
        "runs/alfworld_unified_goal_acquisition_v13_formal/report.json"
    ),
    "target_v14_report": (
        "runs/alfworld_program_driven_policy_v14_formal/report.json"
    ),
    "phase13_acquisition_report": (
        "docs/results/alfworld_target_acquisition_value_v16_qualification.json"
    ),
    "analyzer": "scripts/analyze_alfworld_matched_acquisition_cost_v25.py",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def freeze(output: Path) -> dict[str, Any]:
    for relative in PATHS.values():
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(path)
    v13 = _read(REPO / PATHS["target_v13_report"])
    v14 = _read(REPO / PATHS["target_v14_report"])
    phase13 = _read(REPO / PATHS["phase13_acquisition_report"])
    for value in (v13, v14, phase13):
        body = dict(value)
        claimed = str(body.pop("report_sha256", ""))
        if not claimed or stable_hash(body) != claimed:
            raise ValueError("input report self-hash mismatch")

    hash_fields = {
        f"{name}_file_sha256": _sha(REPO / relative)
        for name, relative in PATHS.items()
    }
    dependency_fields = {
        f"{name}_file_sha256": name for name in PATHS
    }
    body = {
        "schema_version": "alfworld-matched-acquisition-cost-v25-protocol",
        "status": "FROZEN_RETROSPECTIVE_MECHANISM_PROTOCOL",
        "role": "retrospective_source_vs_target_acquisition_audit",
        "claim_boundary": (
            "Uses already-consumed source discovery/held-out receipts and "
            "already-consumed V13/V14 target trajectories. It resets no "
            "environment, reads no new target outcome, and adds no prospective "
            "success claim. Source episode counts, retained successful-path "
            "primitive transitions, and target complete-trajectory transitions "
            "are reported separately; they are not asserted to be a common "
            "sample-cost unit."
        ),
        "estimand": (
            "Evidence required to recover the validated ALFWorld recurrence "
            "normal form from source-only intervention tuples versus complete "
            "target-only successful trajectories."
        ),
        **PATHS,
        **hash_fields,
        "dependency_fields": dependency_fields,
        "source_budget": {
            "primary_order_namespace": (
                "phase14-matched-acquisition-order-v1"
            ),
            "robustness_namespace_prefix": (
                "phase14-matched-acquisition-order-v1/repeat/"
            ),
            "robustness_repetitions": 64,
            "maximum_episodes": 16,
            "qualification": (
                "MATCH_FULL_SOURCE_EXECUTION_NORMAL_FORM_AND_PASS_BOTH_"
                "UNCHANGED_FRESH_SOURCE_CONFIRMATION_GATES"
            ),
        },
        "target_budget": {
            "complete_successful_trajectory_budgets": [0, 1],
            "single_demo_robustness": (
                "EACH_ELIGIBLE_V13_DEMONSTRATION_INDEPENDENTLY"
            ),
            "heldout_pool": "LATER_V14_RAW_TARGET_ONLY_SUCCESS_PATHS",
        },
        "frozen_gates": {
            "source_primary_maximum_episodes": 16,
            "source_robustness_repetitions": 64,
            "require_all_source_orders_recover": True,
            "require_every_target_single_demo_match": True,
            "require_every_target_single_demo_support_all_heldout": True,
            "require_zero_shuffled_and_permuted_support": True,
            "require_source_target_data_read_false": True,
            "forbid_common_cost_unit_claim": True,
        },
        "output": "docs/results/alfworld_matched_acquisition_cost_v25.json",
    }
    return body | {"config_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else REPO / args.output
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen protocol: {output}")
    config = freeze(output)
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
