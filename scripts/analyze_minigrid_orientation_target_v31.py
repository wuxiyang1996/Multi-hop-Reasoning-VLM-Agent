#!/usr/bin/env python3
"""Summarize the frozen prospective V31 target reserve without rerunning it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.minigrid_neural_grounder import (  # noqa: E402
    validate_grounder_artifact,
)


DEFAULT_CONFIG = REPO / "configs/minigrid_orientation_target_v31.json"
DEFAULT_OUTPUT = REPO / "docs/results/minigrid_orientation_target_v31_summary.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": str(report["status"]),
        "tasks": int(report["metrics"]["source_induced"]["tasks"]),
        "grounder_panel_correct": int(report["grounding"]["panel_correct"]),
        "grounder_panel_total": int(report["grounding"]["panel_total"]),
        "exact_effect_binding_tasks": int(
            report["grounding"]["effect_binding_exact_tasks"]
        ),
        "source_induced_success": int(
            report["metrics"]["source_induced"]["native_success"]
        ),
        "neural_only_success": int(
            report["metrics"]["neural_only_direct"]["native_success"]
        ),
        "target_written_isomorphic_success": int(
            report["metrics"]["target_written_isomorphic"]["native_success"]
        ),
        "destructive_control_success": {
            name: int(report["metrics"][name]["native_success"])
            for name in (
                "copy_effect_control", "fixed_token_control",
                "shuffled_binding_control",
            )
        },
        "source_vs_neural_only": dict(
            report["paired_source"]["neural_only_direct"]
        ),
        "all_frozen_gates_pass": all(report["gates"].values()),
        "report_sha256": str(report["report_sha256"]),
    }


def analyze(config_path: Path) -> dict[str, Any]:
    config = _read(config_path)
    _self_hash(config, "config_sha256")
    artifact_path = REPO / str(config["outputs"]["grounder_artifact"])
    artifact = _read(artifact_path)
    validate_grounder_artifact(artifact)
    reports = {
        split: _read(REPO / str(config["outputs"][f"{split}_report"]))
        for split in ("development", "qualification", "formal_reserve")
    }
    for report in reports.values():
        _self_hash(report, "report_sha256")
        if report["grounder_artifact_sha256"] != artifact["artifact_sha256"]:
            raise ValueError("stage report used another grounder artifact")
        if report["program_sha256"] != config["source_program_sha256"]:
            raise ValueError("stage report used another source program")
    stages = {split: _stage(report) for split, report in reports.items()}
    formal = stages["formal_reserve"]
    qualification = stages["qualification"]
    gates = {
        "development_passed_before_qualification": stages["development"][
            "status"
        ] == "CONSUMED_DEVELOPMENT_MINIGRID_CYCLIC_TRANSFER_PASSED",
        "qualification_passed_before_formal_reserve": qualification["status"]
        == "FRESH_QUALIFICATION_MINIGRID_CYCLIC_TRANSFER_PASSED",
        "formal_reserve_passed": formal["status"]
        == "UNTOUCHED_FORMAL_RESERVE_MINIGRID_CYCLIC_TRANSFER_PASSED",
        "all_stage_frozen_gates_pass": all(
            stage["all_frozen_gates_pass"] for stage in stages.values()
        ),
        "formal_grounding_exact": (
            formal["grounder_panel_correct"] == formal["grounder_panel_total"]
            and formal["exact_effect_binding_tasks"] == formal["tasks"]
        ),
        "formal_source_native_success_complete": (
            formal["source_induced_success"] == formal["tasks"]
        ),
        "formal_source_above_neural_only": (
            formal["source_induced_success"] > formal["neural_only_success"]
            and formal["source_vs_neural_only"]["wins"]
            > formal["source_vs_neural_only"]["losses"]
            and formal["source_vs_neural_only"]["exact_two_sided_p"] <= 0.05
        ),
        "formal_source_strictly_above_destructive_controls": all(
            formal["source_induced_success"] > value
            for value in formal["destructive_control_success"].values()
        ),
        "target_written_isomorphic_matches_source": (
            formal["target_written_isomorphic_success"]
            == formal["source_induced_success"]
        ),
        "grounder_acquisition_read_zero_success_and_source": (
            artifact["training"]["target_native_success_or_reward_read"] == 0
            and artifact["training"]["complete_target_trajectories_read"] == 0
            and artifact["training"]["source_program_or_identity_read"] is False
        ),
    }
    status = (
        "PROSPECTIVE_V28_TARGET_RESERVE_VALIDATED"
        if all(gates.values())
        else "V31_TARGET_RESERVE_GATE_FAILED"
    )
    body = {
        "schema_version": "minigrid-orientation-target-summary-v31",
        "status": status,
        "config_sha256": str(config["config_sha256"]),
        "config_file_sha256": _sha(config_path),
        "source_program_sha256": str(config["source_program_sha256"]),
        "grounder_artifact_sha256": str(artifact["artifact_sha256"]),
        "grounder_artifact_file_sha256": _sha(artifact_path),
        "program_family": "ALGEBRAIC_CYCLIC_IDENTITY_RECOVERY",
        "target": "MiniGrid-Empty-Random-6x6-v0 custom orientation-recovery MDP",
        "splits": {key: len(value) for key, value in config["splits"].items()},
        "stages": stages,
        "gates": gates,
        "resource_accounting": {
            "provider_calls": 0,
            "formal_provider_cost_usd": 0.0,
            "target_development_tasks": int(
                artifact["training"]["development_tasks"]
            ),
            "target_development_orientation_labels": int(
                artifact["training"]["target_native_orientation_labels_read"]
            ),
            "target_development_recovery_labels_for_neural_only": int(
                artifact["training"]["target_native_recovery_labels_read"]
            ),
            "complete_target_trajectories_for_source_program": 0,
        },
        "interpretation": (
            "The exact V28 source-induced identity-recovery content transfers "
            "prospectively through a target-native neural grounder and improves "
            "native formal success over a target-only MLP trained on 64 recovery "
            "labels. Destructive relation/binding controls fail. An independently "
            "written isomorphic target executor matches source exactly, confirming "
            "that content rather than source provenance causes execution success."
        ),
        "claim_boundary": str(config["claim_boundary"]),
        "excluded_pilot_boundary": (
            "Only 740xxx/750xxx/751xxx pilot namespaces informed design. No "
            "760xxx protocol seed was rendered before config freeze. The two "
            "OpenRouter visual pilots were excluded from every result and cost "
            "approximately $0.0068624 in total."
        ),
    }
    return body | {"summary_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = analyze(args.config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "formal": report["stages"]["formal_reserve"],
        "gates": report["gates"], "summary_sha256": report["summary_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
