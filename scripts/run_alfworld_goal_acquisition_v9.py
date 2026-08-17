#!/usr/bin/env python3
"""Run source-induced acquisition on the already-consumed ALFWorld dev set."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_alfworld_goal_relation_macro_v3 as v3  # noqa: E402
from motif_transfer.alfworld_goal_acquisition_v9 import (  # noqa: E402
    TargetAcquisitionExecutionState,
    choose_goal_relation_action,
    configure_source_acquisition,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(payload: dict, field: str) -> None:
    body = dict(payload)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise SystemExit(f"invalid {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/alfworld_goal_acquisition_v9_development.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _self_hash(config, "config_sha256")
    if config.get("experiment_version") != (
        "SOURCE_INDUCED_ACQUISITION_CONSUMED_DEVELOPMENT_V9"
    ):
        raise SystemExit("not a V9 consumed-development acquisition config")
    dependencies = {
        "v9_runner_file_sha256": Path(__file__).resolve(),
        "v9_target_runtime_file_sha256": (
            REPO / "src/motif_transfer/alfworld_goal_acquisition_v9.py"
        ),
        "source_acquisition_artifact_file_sha256": REPO / config[
            "source_acquisition_artifact"
        ],
        "source_acquisition_confirmation_file_sha256": REPO / config[
            "source_acquisition_confirmation"
        ],
    }
    for field, path in dependencies.items():
        if _sha256(path) != config.get(field):
            raise SystemExit(f"frozen V9 dependency changed: {path}")
    acquisition = json.loads(
        dependencies["source_acquisition_artifact_file_sha256"].read_text(
            encoding="utf-8"
        )
    )
    confirmation = json.loads(
        dependencies[
            "source_acquisition_confirmation_file_sha256"
        ].read_text(encoding="utf-8")
    )
    configure_source_acquisition(acquisition, confirmation)
    v3.choose_goal_relation_action = choose_goal_relation_action
    v3.TargetRelationExecutionState = TargetAcquisitionExecutionState
    sys.argv = [str(Path(v3.__file__).resolve()), "--config", str(args.config)]
    _ = v3.main()

    output = REPO / config["output"]
    report = json.loads(output.read_text(encoding="utf-8"))
    report.pop("report_sha256", None)
    diagnostics = {
        condition: dict(Counter(
            str(record["diagnostic"])
            for episode in episodes
            for record in episode["records"]
        ))
        for condition, episodes in report["episodes"].items()
    }
    authentic_count = diagnostics[
        "authentic_source_goal_relation_macro"
    ].get("SOURCE_INDUCED_ACQUISITION_OPERATOR_GROUNDED", 0)
    ceiling_count = diagnostics[
        "target_native_recurrent_relation_ceiling"
    ].get("SOURCE_INDUCED_ACQUISITION_OPERATOR_GROUNDED", 0)
    control_count = sum(
        diagnostics[name].get(
            "SOURCE_INDUCED_ACQUISITION_OPERATOR_GROUNDED", 0,
        )
        for name in (
            "source_cardinality_exactly_one_control",
            "source_effect_binding_permuted_control",
            "generic_single_relation_scaffold",
        )
    )
    report["gates"] |= {
        "source_acquisition_fresh_confirmation_passed": bool(
            confirmation["source_gate_passed"]
        ),
        "authentic_executes_source_induced_acquisition": (
            authentic_count >= int(config["gates"][
                "minimum_source_acquisition_groundings"
            ])
        ),
        "source_acquisition_control_isolation": control_count == 0,
        "authentic_matches_target_native_acquisition_execution": (
            authentic_count == ceiling_count
        ),
    }
    passed = all(report["gates"].values())
    report |= {
        "schema_version": "alfworld-goal-acquisition-development-v9",
        "status": (
            "CONSUMED_DEVELOPMENT_ACQUISITION_GATE_PASSED" if passed
            else "CONSUMED_DEVELOPMENT_ACQUISITION_GATE_FAILED"
        ),
        "experiment_version": str(config["experiment_version"]),
        "source_acquisition_artifact_sha256": str(
            acquisition["artifact_sha256"]
        ),
        "source_acquisition_confirmation_sha256": str(
            confirmation["report_sha256"]
        ),
        "acquisition_diagnostics": diagnostics,
        "acquisition_groundings": {
            "authentic": authentic_count,
            "target_native_ceiling": ceiling_count,
            "source_controls": control_count,
        },
        "claim_boundary": str(config["claim_boundary"]),
    }
    report["report_sha256"] = stable_hash(report)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "v9_status": report["status"],
        "summaries": report["summaries"],
        "paired": report["paired"],
        "acquisition_groundings": report["acquisition_groundings"],
        "gates": report["gates"],
        "v9_report_sha256": report["report_sha256"],
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
