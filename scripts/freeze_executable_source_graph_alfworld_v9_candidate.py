#!/usr/bin/env python3
"""Freeze the executable-source-graph V9 candidate before adaptation."""

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
from motif_transfer.slot_aware_alfworld_harness_v9 import (
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


def _file_receipt(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v8-harness", type=Path, required=True)
    parser.add_argument("--v8-negative-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v9-harness-code", type=Path, required=True)
    parser.add_argument("--slot-ledger-code", type=Path, required=True)
    parser.add_argument("--v9-runner-code", type=Path, required=True)
    parser.add_argument(
        "--shared-runner-helper-code", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V9 candidate: {args.output}")
    v8 = _read(args.v8_harness)
    _validate_hash(v8, "harness_sha256")
    if v8.get("status") != "FRESH_CONFIRMATION_AUTHORIZED":
        raise SystemExit("V8 dependency is not a frozen Harness")
    negative = _read(args.v8_negative_report)
    _validate_hash(negative, "report_sha256")
    if negative.get("status") != "FRESH_CONFIRMATION_NEGATIVE_STOP":
        raise SystemExit("V9 requires the consumed V8 negative report")
    if negative.get("existing_valid_unseen_heldout_read"):
        raise SystemExit("V8 crossed the reserved heldout boundary")
    manifest = _read(args.manifest)
    _validate_hash(manifest, "manifest_sha256")
    if manifest.get("schema_version") != (
        "executable-source-graph-alfworld-manifest-v9"
    ):
        raise SystemExit("wrong V9 manifest")
    if manifest.get("status") != (
        "FROZEN_BEFORE_ANY_SELECTED_TASK_RESET"
    ):
        raise SystemExit("V9 manifest was not frozen")
    if manifest.get("selection_used_target_rollout_outcomes"):
        raise SystemExit("V9 selection used target outcomes")
    if set(manifest.get("target_families", ())) != {
        "pick_and_place_simple", "pick_two_obj_and_place"
    }:
        raise SystemExit("V9 manifest is not relation-only")
    target_path = Path(str(v8["target_grounder"]["path"]))
    if _sha256(target_path) != v8["target_grounder"]["file_sha256"]:
        raise SystemExit("frozen target grounder artifact changed")
    target = _read(target_path)
    validate_target_artifact(target)
    router = dict(v8["property_router"])
    validate_property_router(router)
    source_ir = dict(v8["slot_source_ir"])
    validate_slot_source_ir(source_ir)
    authentic_graph = compile_source_effect_graph(
        source_ir, condition="authentic_slot_ir"
    )
    edge_control_graph = compile_source_effect_graph(
        source_ir, condition="edge_permuted_ir"
    )
    inherited = v8["thresholds"]
    thresholds = {
        "selection_authority": (
            "INHERITED_FROM_FROZEN_V8; PROPERTY_CONFIDENCE_CHANGED_TO_"
            "DIAGNOSTIC_ONLY_AFTER_CONSUMED_V8_FAILURE"
        ),
        "minimum_property_confidence_diagnostic_only": float(
            inherited["minimum_property_confidence"]
        ),
        "minimum_role_binding": float(inherited["minimum_role_binding"]),
        "minimum_realization_score": float(
            inherited["minimum_realization_score"]
        ),
        "minimum_target_policy_ratio": float(
            inherited["minimum_target_policy_ratio"]
        ),
    }
    body = {
        "schema_version": (
            "executable-source-graph-alfworld-candidate-v9"
        ),
        "status": "ADAPTATION_GATE_ONLY",
        "claim_boundary": (
            "FROZEN_BEFORE_V9_ADAPTATION_RESET; V8_CONFIRMATION_CONSUMED_"
            "FOR_DEVELOPMENT; V9_CONFIRMATION_FORBIDDEN_UNTIL_GATE_"
            "PASSES; EXISTING_VALID_UNSEEN_HELDOUT_FORBIDDEN"
        ),
        "v8_parent_harness": _file_receipt(args.v8_harness) | {
            "harness_sha256": v8["harness_sha256"]
        },
        "consumed_v8_negative_report": _file_receipt(
            args.v8_negative_report
        ) | {
            "report_sha256": negative["report_sha256"],
            "use_authority": "DEVELOPMENT_DIAGNOSIS_ONLY",
            "successes": negative["summaries"][
                "authentic_slot_ir"
            ]["successes"],
            "changed_effects": negative["summaries"][
                "authentic_slot_ir"
            ]["changed_effect_count"],
        },
        "slot_source_ir": source_ir,
        "compiled_source_graphs": {
            "authentic": authentic_graph,
            "edge_control": edge_control_graph,
            "execution_contract": (
                "TARGET_STATE_SELECTS_GUARD; MATCHED_BOUND_SOURCE_EDGE_"
                "SUPPLIES_NEXT_EFFECT"
            ),
        },
        "target_grounder": _file_receipt(target_path) | {
            "artifact_sha256": target["artifact_sha256"]
        },
        "property_router": router,
        "manifest": _file_receipt(args.manifest) | {
            "manifest_sha256": manifest["manifest_sha256"]
        },
        "implementation": {
            "v9_harness": _file_receipt(args.v9_harness_code),
            "slot_ledger": _file_receipt(args.slot_ledger_code),
            "v9_runner": _file_receipt(args.v9_runner_code),
            "shared_runner_helpers": _file_receipt(
                args.shared_runner_helper_code
            ),
        },
        "thresholds": thresholds,
        "transfer_scope": {
            "name": "EXECUTED_CROSS_ENGINE_BIND_RELATE_SOURCE_GRAPH",
            "allowed_source_effects": ["BIND", "RELATE"],
            "active_required_properties": ["NONE"],
            "claimed_changed_effects": ["BIND", "RELATE"],
            "target_families": [
                "pick_and_place_simple",
                "pick_two_obj_and_place",
            ],
            "excluded_claims": [
                "NO_MUTATE_OR_PROPERTY_TRANSFER_CLAIM",
                "NO_EXISTING_VALID_UNSEEN_OOD_CLAIM",
            ],
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
                "SOURCE_ACTION_OR_COORDINATE_AT_RUNTIME",
                "SOURCE_TASK_ID_FOR_TARGET_ACTION_SELECTION",
                "OFFICIAL_SUCCESS_FOR_ACTION_SELECTION",
                "V9_CONFIRMATION_BEFORE_GATE",
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
        "source_ir_sha256": source_ir["ir_sha256"],
        "authentic_graph_sha256": authentic_graph["graph_sha256"],
        "edge_control_graph_sha256": edge_control_graph[
            "graph_sha256"
        ],
        "confirmation_authorized": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
