#!/usr/bin/env python3
"""Freeze the 60-step V10 candidate before its fresh adaptation reset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from motif_transfer.alfworld_masked_effect_grounder import (
    validate_artifact as validate_target_artifact,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.parameterized_alfworld_harness import (
    validate_property_router,
)
from motif_transfer.slot_aware_alfworld_harness import (
    validate_slot_source_ir,
)
from motif_transfer.slot_aware_alfworld_harness_v10 import (
    CONDITION_SEMANTICS,
    compile_source_effect_graph,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_hash(value: dict[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if stable_hash(body) != claimed:
        raise SystemExit(f"invalid frozen artifact hash: {field}")


def _receipt(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v9-candidate", type=Path, required=True)
    parser.add_argument("--v9-gate-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v10-harness-code", type=Path, required=True)
    parser.add_argument("--v9-graph-code", type=Path, required=True)
    parser.add_argument("--slot-ledger-code", type=Path, required=True)
    parser.add_argument("--v10-runner-code", type=Path, required=True)
    parser.add_argument("--shared-v9-runner-code", type=Path, required=True)
    parser.add_argument("--shared-v8-runner-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--runner-seed", type=int, default=99501)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V10 candidate: {args.output}")
    if args.max_steps != 60:
        raise SystemExit("V10 preregisters an exact 60-step budget")
    v9 = _read(args.v9_candidate)
    _validate_hash(v9, "candidate_sha256")
    if v9.get("schema_version") != (
        "executable-source-graph-alfworld-candidate-v9"
    ):
        raise SystemExit("wrong V9 parent candidate")
    gate = _read(args.v9_gate_report)
    _validate_hash(gate, "report_sha256")
    if gate.get("status") != "ADAPTATION_GATE_FAILED_STOP":
        raise SystemExit("V10 requires the stopped V9 development gate")
    if gate.get("existing_valid_unseen_heldout_read"):
        raise SystemExit("V9 crossed the existing heldout boundary")
    if gate["gates"].get("authentic_success_superior_to_edge_control"):
        raise SystemExit("V9 did not expose the expected budget diagnosis")
    manifest = _read(args.manifest)
    _validate_hash(manifest, "manifest_sha256")
    manifest_schema = (
        "budgeted-executable-source-graph-alfworld-manifest-v10"
    )
    manifest_status = "FROZEN_BEFORE_ANY_V10_ADAPTATION_RESET"
    if manifest.get("schema_version") != manifest_schema:
        raise SystemExit("wrong V10 manifest schema")
    if manifest.get("status") != manifest_status:
        raise SystemExit("V10 manifest was not frozen")
    if manifest.get("selection_used_target_rollout_outcomes"):
        raise SystemExit("V10 task selection used target outcomes")
    target_path = Path(str(v9["target_grounder"]["path"]))
    if _sha256(target_path) != v9["target_grounder"]["file_sha256"]:
        raise SystemExit("target grounder artifact changed")
    target = _read(target_path)
    validate_target_artifact(target)
    router = dict(v9["property_router"])
    validate_property_router(router)
    source_ir = dict(v9["slot_source_ir"])
    validate_slot_source_ir(source_ir)
    authentic_graph = compile_source_effect_graph(
        source_ir, condition="authentic_slot_ir"
    )
    node_control_graph = compile_source_effect_graph(
        source_ir, condition="edge_permuted_ir"
    )
    body = {
        "schema_version": (
            "budgeted-executable-source-graph-alfworld-candidate-v10"
        ),
        "experiment_version": "v10",
        "status": "ADAPTATION_GATE_ONLY",
        "claim_boundary": (
            "SIXTY_STEP_ENDPOINT_SELECTED_FROM_CONSUMED_V9_GATE; FROZEN_"
            "BEFORE_FRESH_V10_ADAPTATION; PRESERVED_CONFIRMATION_"
            "FORBIDDEN_UNTIL_GATE_PASSES; EXISTING_VALID_UNSEEN_FORBIDDEN"
        ),
        "v9_parent_candidate": _receipt(args.v9_candidate) | {
            "candidate_sha256": v9["candidate_sha256"]
        },
        "consumed_v9_gate_report": _receipt(args.v9_gate_report) | {
            "report_sha256": gate["report_sha256"],
            "use_authority": (
                "DEVELOPMENT_ONLY_FOR_ENDPOINT_AND_CONTROL_DESIGN"
            ),
            "authentic_successes": gate["summaries"][
                "authentic_slot_ir"
            ]["successes"],
            "edge_control_successes": gate["summaries"][
                "edge_permuted_ir"
            ]["successes"],
            "authentic_mean_steps": gate["summaries"][
                "authentic_slot_ir"
            ]["mean_steps"],
            "edge_control_mean_steps": gate["summaries"][
                "edge_permuted_ir"
            ]["mean_steps"],
        },
        "slot_source_ir": source_ir,
        "compiled_source_graphs": {
            "authentic": authentic_graph,
            "node_only_control": node_control_graph,
            "execution_contract": (
                "TARGET_STATE_SELECTS_GUARD; MATCHED_BOUND_SOURCE_EDGE_"
                "SUPPLIES_NEXT_EFFECT"
            ),
        },
        "target_grounder": _receipt(target_path) | {
            "artifact_sha256": target["artifact_sha256"]
        },
        "property_router": router,
        "manifest": _receipt(args.manifest) | {
            "manifest_sha256": manifest["manifest_sha256"]
        },
        "manifest_schema": manifest_schema,
        "manifest_status": manifest_status,
        "implementation": {
            "v10_harness": _receipt(args.v10_harness_code),
            "v9_graph_executor": _receipt(args.v9_graph_code),
            "slot_ledger": _receipt(args.slot_ledger_code),
            "v10_runner": _receipt(args.v10_runner_code),
            "shared_v9_runner": _receipt(args.shared_v9_runner_code),
            "shared_v8_runner_helpers": _receipt(
                args.shared_v8_runner_code
            ),
        },
        "experiment_parameters": {
            "primary_endpoint": "OFFICIAL_SUCCESS_WITHIN_STEP_BUDGET",
            "max_steps": args.max_steps,
            "runner_seed": args.runner_seed,
            "endpoint_selection_authority": (
                "CHOSEN_AFTER_CONSUMED_V9_ADAPTATION_AND_BEFORE_ANY_V10_"
                "SELECTED_RESET"
            ),
        },
        "condition_semantics": CONDITION_SEMANTICS,
        "thresholds": dict(v9["thresholds"]),
        "transfer_scope": dict(v9["transfer_scope"]) | {
            "name": "BUDGETED_EXECUTED_BIND_RELATE_SOURCE_GRAPH"
        },
        "adaptation_gates": {
            "authentic_success_noninferior_to_target_only": True,
            "authentic_success_superior_to_edge_control": True,
            "paired_net_win_nonnegative": True,
            "changed_effects_each_claimed_effect": 1,
            "changed_tasks": 4,
            "changed_source_edges": 2,
            "authentic_changes_exceed_edge_control": True,
            "source_admission_rate_range": [0.005, 0.30],
            "reopened_completed_slots": 0,
            "failed_selected_postconditions": 0,
        },
        "permissions": {
            "source_ir": [
                "SUPPLY_START_EFFECT_NODE",
                "SUPPLY_GUARDED_SUCCESSOR_EFFECT",
            ],
            "target_native": [
                "PARSE_GOAL_ROLES_AND_EXPLICIT_OPERATOR",
                "SELECT_SOURCE_GUARD_FROM_OBSERVED_SLOT_STATE",
                "NEURALLY_GROUND_EFFECT_TO_NATIVE_ACTION",
                "UPDATE_LEDGER_FROM_OBSERVED_POSTCONDITIONS",
                "SHIELD_OBSERVED_COMPLETED_SLOTS",
                "ABSTAIN_TO_EXACT_TARGET_POLICY",
            ],
            "forbidden": [
                "TARGET_CODED_SOURCE_EFFECT_SUCCESSOR",
                "OFFICIAL_SUCCESS_FOR_ACTION_SELECTION",
                "V10_CONFIRMATION_BEFORE_GATE",
                "CHANGE_STEP_BUDGET_AFTER_FREEZE",
                "EXISTING_VALID_UNSEEN_RESET",
            ],
        },
    }
    result = body | {"candidate_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidate_sha256": result["candidate_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "max_steps": args.max_steps,
        "runner_seed": args.runner_seed,
        "condition_semantics": CONDITION_SEMANTICS,
        "confirmation_authorized": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
